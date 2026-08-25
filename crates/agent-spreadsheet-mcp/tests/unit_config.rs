use std::fs;

use agent_spreadsheet_mcp::{
    CliArgs, OutputProfile, RecalcBackendKind, ServerConfig, TransportKind,
};
use clap::Parser;

#[test]
fn merges_config_file_and_cli_overrides() {
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let config_dir = tempfile::tempdir().expect("config tempdir");
    let config_path = config_dir.path().join("server.yaml");
    let yaml = format!(
        "workspace_root: {}\ncache_capacity: 3\nvba_enabled: true\nextensions:\n  - xlsx\n  - XLS\n",
        workspace.path().display()
    );
    fs::write(&config_path, yaml).expect("write config");

    let args = CliArgs::parse_from([
        "gridbench-mcp",
        "--config",
        config_path.to_str().unwrap(),
        "--workspace-root",
        workspace.path().to_str().unwrap(),
        "--cache-capacity",
        "12",
        "--extensions",
        "xlsb,.XLSX",
        "--enabled-tools",
        "list_workbooks,sheet_page",
    ]);
    let config = ServerConfig::from_args(args).expect("config");

    assert_eq!(config.workspace_root, workspace.path().to_path_buf());
    assert_eq!(config.cache_capacity, 12);
    assert_eq!(
        config.supported_extensions,
        vec!["xlsb".to_string(), "xlsx".to_string()]
    );
    let mut enabled = config.enabled_tools.expect("enabled set");
    assert!(enabled.remove("list_workbooks"));
    assert!(enabled.remove("sheet_page"));
    assert!(enabled.is_empty());
    assert!(config.vba_enabled);
    assert_eq!(config.transport, TransportKind::Http);
    assert_eq!(
        config.http_bind_address,
        "127.0.0.1:8079".parse().expect("default bind")
    );
}

#[test]
fn empty_extensions_is_error() {
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let args = CliArgs {
        config: None,
        workspace_root: Some(workspace.path().to_path_buf()),
        screenshot_dir: None,
        path_map: None,
        cache_capacity: Some(1),
        extensions: Some(Vec::new()),
        workbook: None,
        enabled_tools: None,
        transport: None,
        http_bind: None,
        recalc_enabled: false,
        recalc_backend: None,
        vba_enabled: false,
        max_concurrent_recalcs: None,
        tool_timeout_ms: None,
        max_response_bytes: None,
        output_profile: None,
        max_payload_bytes: None,
        max_cells: None,
        max_items: None,
        allow_overwrite: false,
        slim_surface: None,
    };
    let err = ServerConfig::from_args(args).expect_err("expected failure");
    assert!(err.to_string().contains("at least one file extension"));
}

#[test]
fn ensure_workspace_root_errors_for_missing_dir() {
    let config = ServerConfig {
        workspace_root: std::path::PathBuf::from("/this/does/not/exist"),
        screenshot_dir: std::path::PathBuf::from("/this/does/not/exist/screenshots"),
        path_mappings: Vec::new(),
        cache_capacity: 2,
        supported_extensions: vec!["xlsx".to_string()],
        single_workbook: None,
        enabled_tools: None,
        transport: TransportKind::Http,
        http_bind_address: "127.0.0.1:8079".parse().unwrap(),
        recalc_enabled: false,
        recalc_backend: RecalcBackendKind::Auto,
        vba_enabled: false,
        max_concurrent_recalcs: 2,
        tool_timeout_ms: Some(30_000),
        max_response_bytes: Some(1_000_000),
        output_profile: OutputProfile::TokenDense,
        max_payload_bytes: Some(65_536),
        max_cells: Some(10_000),
        max_items: Some(500),
        allow_overwrite: false,
        slim_surface: true,
    };
    let err = config.ensure_workspace_root().expect_err("missing dir");
    assert!(
        err.to_string()
            .contains("workspace root \"/this/does/not/exist\"")
    );
}

#[test]
fn single_workbook_sets_default_workspace_root() {
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let workbook = workspace.path().join("focus.xlsx");
    std::fs::write(&workbook, b"fake").expect("write workbook");

    let args = CliArgs::parse_from(["gridbench-mcp", "--workbook", workbook.to_str().unwrap()]);
    let config = ServerConfig::from_args(args).expect("config");

    assert_eq!(config.workspace_root, workspace.path().to_path_buf());
    assert_eq!(
        config
            .single_workbook()
            .expect("single workbook")
            .to_path_buf(),
        workbook
    );
}

#[test]
fn transport_cli_override_parses() {
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let args = CliArgs::parse_from([
        "gridbench-mcp",
        "--workspace-root",
        workspace.path().to_str().unwrap(),
        "--transport",
        "stdio",
    ]);
    let config = ServerConfig::from_args(args).expect("config");

    assert_eq!(config.transport, TransportKind::Stdio);
}

#[test]
fn http_transport_alias_still_parses() {
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let args = CliArgs::parse_from([
        "gridbench-mcp",
        "--workspace-root",
        workspace.path().to_str().unwrap(),
        "--transport",
        "http",
    ]);
    let config = ServerConfig::from_args(args).expect("config");

    assert_eq!(config.transport, TransportKind::Http);
}

#[test]
fn http_bind_override_from_cli() {
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let args = CliArgs::parse_from([
        "gridbench-mcp",
        "--workspace-root",
        workspace.path().to_str().unwrap(),
        "--http-bind",
        "127.0.0.1:0",
    ]);
    let config = ServerConfig::from_args(args).expect("config");

    assert_eq!(config.http_bind_address, "127.0.0.1:0".parse().unwrap());
}

#[test]
fn recalc_backend_override_from_cli() {
    let workspace = tempfile::tempdir().expect("workspace tempdir");
    let args = CliArgs::parse_from([
        "gridbench-mcp",
        "--workspace-root",
        workspace.path().to_str().unwrap(),
        "--recalc-enabled",
        "--recalc-backend",
        "formualizer",
    ]);
    let config = ServerConfig::from_args(args).expect("config");

    assert!(config.recalc_enabled);
    assert_eq!(config.recalc_backend, RecalcBackendKind::Formualizer);
}
