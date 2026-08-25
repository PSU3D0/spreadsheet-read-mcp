#[cfg(feature = "recalc")]
mod recalc_security {
    use agent_spreadsheet::errors::InvalidParamsError;

    use agent_spreadsheet::fork::{ForkConfig, ForkRegistry};
    use agent_spreadsheet::recalc::macro_uri::{export_screenshot_uri, recalc_and_save_uri};

    #[test]
    fn macro_uri_escapes_quotes_in_sheet_name() {
        let sheet = "Sheet1\";MsgBox(\"pwn\");\"";
        let uri =
            export_screenshot_uri("file:///tmp/workbook.xlsx", "/tmp/out.pdf", sheet, "A1:B2")
                .expect("uri");

        // Quotes must be doubled inside a Basic string literal.
        assert!(uri.contains("Sheet1\"\";MsgBox(\"\"pwn\"\");\"\""));
        // The URI must still be a single macro call.
        assert!(uri.starts_with("macro:///Standard.Module1.ExportScreenshot("));
    }

    #[test]
    fn macro_uri_rejects_control_chars() {
        let err = export_screenshot_uri(
            "file:///tmp/workbook.xlsx",
            "/tmp/out.pdf",
            "bad\nname",
            "A1:B2",
        )
        .expect_err("expected invalid params");
        assert!(err.is::<InvalidParamsError>());
    }

    #[test]
    fn recalc_macro_uri_escapes_quotes_in_file_url() {
        let uri = recalc_and_save_uri("file:///tmp/has\"quote.xlsx").expect("uri");
        assert!(uri.contains("file:///tmp/has\"\"quote.xlsx"));
    }

    #[cfg(unix)]
    #[test]
    fn create_fork_rejects_symlink_escape_outside_workspace() {
        use std::fs;
        use std::os::unix::fs::symlink;

        let workspace = tempfile::tempdir().expect("workspace");
        let outside = tempfile::tempdir().expect("outside");

        let outside_xlsx = outside.path().join("outside.xlsx");
        fs::write(&outside_xlsx, b"fake").expect("write outside xlsx");

        // Symlink inside workspace pointing outside.
        let link = workspace.path().join("linked.xlsx");
        symlink(&outside_xlsx, &link).expect("symlink");

        let fork_dir = tempfile::tempdir().expect("fork_dir");
        let registry = ForkRegistry::new(ForkConfig {
            fork_dir: fork_dir.path().to_path_buf(),
            ..Default::default()
        })
        .expect("registry");

        let err = registry
            .create_fork(&link, workspace.path())
            .expect_err("expected invalid params");
        assert!(err.is::<InvalidParamsError>());
        assert!(err.to_string().contains("base_path"));
    }

    #[cfg(unix)]
    #[test]
    fn save_fork_rejects_dotdot_escape_and_symlink_dir_escape() {
        use std::fs;
        use std::os::unix::fs::symlink;

        let workspace = tempfile::tempdir().expect("workspace");
        let outside = tempfile::tempdir().expect("outside");

        let base = workspace.path().join("base.xlsx");
        fs::write(&base, b"fake").expect("write base xlsx");

        let fork_dir = tempfile::tempdir().expect("fork_dir");
        let registry = ForkRegistry::new(ForkConfig {
            fork_dir: fork_dir.path().to_path_buf(),
            ..Default::default()
        })
        .expect("registry");

        let fork_id = registry
            .create_fork(&base, workspace.path())
            .expect("create fork");

        // `..` escape attempt.
        let target_escape = workspace.path().join("..").join("escaped.xlsx");
        let err = registry
            .save_fork(&fork_id, &target_escape, workspace.path(), true)
            .expect_err("expected invalid params");
        assert!(err.is::<InvalidParamsError>());
        assert!(err.to_string().contains("target_path"));

        // Symlinked directory inside workspace pointing outside.
        let out_dir_link = workspace.path().join("out");
        symlink(outside.path(), &out_dir_link).expect("symlink dir");
        let target_symlinked = out_dir_link.join("saved.xlsx");
        let err = registry
            .save_fork(&fork_id, &target_symlinked, workspace.path(), true)
            .expect_err("expected invalid params");
        assert!(err.is::<InvalidParamsError>());
    }
}
