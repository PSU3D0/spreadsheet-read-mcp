//! Determinism, tested rather than assumed.
//!
//! The 5003 bake-off found spike D producing a different column unit — and so a
//! different image width, 1122 px against 1299 px on the same fixture —
//! between runs of the *same binary*, because it inferred the workbook's normal
//! font by frequency over `Worksheet::get_cell_collection()`, whose iteration
//! order is not stable. PNG hash goldens are worthless without a guard against
//! that class of defect, so this suite renders every fixture twice in one
//! process and once more in a freshly spawned process, and requires all three
//! hashes to agree.
//!
//! The fresh process is this same test binary, re-executed with
//! `RENDER_DETERMINISM_CHILD=1`, so no extra dev-dependency and no `cargo`
//! invocation is involved.

mod common;

use std::process::Command;

use common::{FIXTURES, pixel_signature, render_fixture, sha256_hex};

const CHILD_ENV: &str = "RENDER_DETERMINISM_CHILD";

/// Pixel identity plus PNG-byte identity. Within one build the deflate
/// backend is fixed, so both must hold; across builds only the pixels are a
/// promise (see `goldens.rs`).
fn hashes() -> Vec<(String, String)> {
    FIXTURES
        .iter()
        .map(|(name, _)| {
            let (_, output) = render_fixture(name);
            (
                name.to_string(),
                format!(
                    "{} png:{}",
                    pixel_signature(&output),
                    sha256_hex(&output.png)
                ),
            )
        })
        .collect()
}

#[test]
fn two_renders_in_one_process_are_byte_identical() {
    // Covers both the pixels and the encoded PNG bytes.
    let first = hashes();
    let second = hashes();
    assert_eq!(
        first, second,
        "the same fixtures rendered differently twice in one process"
    );
}

#[test]
fn scene_extraction_is_byte_identical_across_repeats() {
    for (name, _) in FIXTURES {
        let (first, _) = render_fixture(name);
        let (second, _) = render_fixture(name);
        assert_eq!(first, second, "{name} extracted a different scene");
        // Geometry specifically: this is what the unstable-iteration defect
        // moved.
        assert_eq!(first.width, second.width, "{name} width");
        assert_eq!(first.height, second.height, "{name} height");
    }
}

/// Emits `name<TAB>hash WxH` lines when re-executed as the child.
#[test]
fn emit_hashes_for_parent() {
    if std::env::var(CHILD_ENV).is_err() {
        // Not the child: nothing to do. The real assertion lives in
        // `a_fresh_process_produces_the_same_hashes`.
        return;
    }
    for (name, line) in hashes() {
        println!("HASH\t{name}\t{line}");
    }
}

#[test]
fn a_fresh_process_produces_the_same_hashes() {
    if std::env::var(CHILD_ENV).is_ok() {
        return; // do not recurse
    }
    let executable = std::env::current_exe().expect("test binary path");
    let output = Command::new(executable)
        .env(CHILD_ENV, "1")
        .args(["--exact", "emit_hashes_for_parent", "--nocapture"])
        .output()
        .expect("re-executing the test binary");
    assert!(
        output.status.success(),
        "child run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let child: Vec<(String, String)> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("HASH\t"))
        .filter_map(|line| {
            let (name, hash) = line.split_once('\t')?;
            Some((name.to_string(), hash.to_string()))
        })
        .collect();
    assert_eq!(
        child.len(),
        FIXTURES.len(),
        "child emitted {} of {} fixtures; stdout was:\n{stdout}",
        child.len(),
        FIXTURES.len()
    );
    assert_eq!(
        child,
        hashes(),
        "a fresh process rendered different bytes from this one"
    );
}
