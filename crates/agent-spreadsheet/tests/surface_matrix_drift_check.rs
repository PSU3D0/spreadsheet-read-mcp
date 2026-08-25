use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("resolve workspace root")
}

fn run_checker(repo_root: &Path, matrix_override: Option<&Path>) -> Output {
    let script = repo_root.join("scripts/check_surface_matrix_drift.py");
    assert!(
        script.exists(),
        "drift checker script missing at {}",
        script.display()
    );

    let mut cmd = Command::new("python3");
    cmd.arg(&script).current_dir(repo_root);
    if let Some(matrix_path) = matrix_override {
        cmd.env("SURFACE_MATRIX_MD", matrix_path);
    }

    cmd.output().expect("run matrix drift checker")
}

fn write_temp_file(prefix: &str, content: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{stamp}.md"));
    fs::write(&path, content).expect("write temp matrix file");
    path
}

#[test]
fn surface_matrix_drift_check_passes() {
    let root = repo_root();
    let output = run_checker(&root, None);

    if !output.status.success() {
        panic!(
            "surface matrix drift check failed (status: {:?})\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn surface_matrix_drift_check_fails_when_variant_row_missing() {
    let root = repo_root();
    let matrix_path = root.join("docs/architecture/surface-capability-matrix.md");
    let original = fs::read_to_string(&matrix_path).expect("read matrix");

    let modified = original
        .lines()
        .filter(|line| !line.contains("`read export --format grid`"))
        .collect::<Vec<_>>()
        .join("\n");

    let temp_path = write_temp_file("surface-matrix-missing-variant", &modified);
    let output = run_checker(&root, Some(&temp_path));

    assert!(
        !output.status.success(),
        "checker should fail when a required variant row is removed"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Missing CLI entries") || stdout.contains("read export --format grid"),
        "expected missing variant diagnostics, got:\n{stdout}"
    );

    let _ = fs::remove_file(temp_path);
}

#[test]
fn surface_matrix_drift_check_fails_on_stale_row() {
    let root = repo_root();
    let matrix_path = root.join("docs/architecture/surface-capability-matrix.md");
    let original = fs::read_to_string(&matrix_path).expect("read matrix");

    let marker = "| `analyze ref-impact` |";
    let fake_row = "| `fake-command-for-drift-check` | _(none)_ | CLI_ONLY | `adapter-cli.fake` | n/a | test stale row | `crates/agent-spreadsheet/src/cli/mod.rs` | `crates/agent-spreadsheet/tests/surface_matrix_drift_check.rs` |";
    let modified = if let Some(pos) = original.find(marker) {
        let mut out = String::with_capacity(original.len() + fake_row.len() + 2);
        out.push_str(&original[..pos]);
        out.push_str(fake_row);
        out.push('\n');
        out.push_str(&original[pos..]);
        out
    } else {
        panic!("failed to locate insertion marker '{marker}' in matrix");
    };

    let temp_path = write_temp_file("surface-matrix-stale", &modified);
    let output = run_checker(&root, Some(&temp_path));

    assert!(
        !output.status.success(),
        "checker should fail on stale matrix entries"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Stale CLI entries") || stdout.contains("fake-command-for-drift-check"),
        "expected stale row diagnostics, got:\n{stdout}"
    );

    let _ = fs::remove_file(temp_path);
}

#[test]
fn surface_matrix_drift_check_fails_on_missing_required_metadata() {
    let root = repo_root();
    let matrix_path = root.join("docs/architecture/surface-capability-matrix.md");
    let original = fs::read_to_string(&matrix_path).expect("read matrix");

    let target_row = "| `read sheets` | `list_sheets` | ALL | `core.read.list_sheets` | mvp | Shared read primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::list_sheets` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |";
    let broken_row = "| `read sheets` | `list_sheets` | ALL | `core.read.list_sheets` | mvp | Shared read primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::list_sheets` |  |";

    let modified = original.replacen(target_row, broken_row, 1);
    assert_ne!(
        modified, original,
        "failed to rewrite target row for required-metadata test"
    );

    let temp_path = write_temp_file("surface-matrix-metadata", &modified);
    let output = run_checker(&root, Some(&temp_path));

    assert!(
        !output.status.success(),
        "checker should fail when required metadata columns are blank"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("parity test owner must be non-empty")
            || stdout.contains("Matrix validation errors"),
        "expected metadata validation diagnostics, got:\n{stdout}"
    );

    let _ = fs::remove_file(temp_path);
}
