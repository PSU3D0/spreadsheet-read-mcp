//! argv/config-file surface for [`ServerConfig`].
//!
//! Lives behind the `cli` feature so that clap and serde_yaml are not compiled
//! into the wasm32 byte/session adapter, which constructs [`ServerConfig`]
//! directly instead of parsing arguments.

use super::defaults::{
    DEFAULT_CACHE_CAPACITY, DEFAULT_EXTENSIONS, DEFAULT_HTTP_BIND, DEFAULT_MAX_CELLS,
    DEFAULT_MAX_ITEMS, DEFAULT_MAX_PAYLOAD_BYTES, DEFAULT_MAX_RECALCS, DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_TOOL_TIMEOUT_MS,
};
use super::{OutputProfile, PathMapping, RecalcBackendKind, ServerConfig, TransportKind};
use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

impl ServerConfig {
    pub fn from_args(args: CliArgs) -> Result<Self> {
        let CliArgs {
            config,
            workspace_root: cli_workspace_root,
            screenshot_dir: cli_screenshot_dir,
            path_map: cli_path_map,
            cache_capacity: cli_cache_capacity,
            extensions: cli_extensions,
            workbook: cli_single_workbook,
            enabled_tools: cli_enabled_tools,
            transport: cli_transport,
            http_bind: cli_http_bind,
            recalc_enabled: cli_recalc_enabled,
            recalc_backend: cli_recalc_backend,
            vba_enabled: cli_vba_enabled,
            max_concurrent_recalcs: cli_max_concurrent_recalcs,
            tool_timeout_ms: cli_tool_timeout_ms,
            max_response_bytes: cli_max_response_bytes,
            slim_surface: cli_slim_surface,
            output_profile: cli_output_profile,
            max_payload_bytes: cli_max_payload_bytes,
            max_cells: cli_max_cells,
            max_items: cli_max_items,
            allow_overwrite: cli_allow_overwrite,
        } = args;

        let file_config = if let Some(path) = config.as_ref() {
            load_config_file(path)?
        } else {
            PartialConfig::default()
        };

        let PartialConfig {
            workspace_root: file_workspace_root,
            screenshot_dir: file_screenshot_dir,
            path_map: file_path_map,
            cache_capacity: file_cache_capacity,
            extensions: file_extensions,
            single_workbook: file_single_workbook,
            enabled_tools: file_enabled_tools,
            transport: file_transport,
            http_bind: file_http_bind,
            recalc_enabled: file_recalc_enabled,
            recalc_backend: file_recalc_backend,
            vba_enabled: file_vba_enabled,
            max_concurrent_recalcs: file_max_concurrent_recalcs,
            tool_timeout_ms: file_tool_timeout_ms,
            max_response_bytes: file_max_response_bytes,
            slim_surface: file_slim_surface,
            output_profile: file_output_profile,
            max_payload_bytes: file_max_payload_bytes,
            max_cells: file_max_cells,
            max_items: file_max_items,
            allow_overwrite: file_allow_overwrite,
        } = file_config;

        let mut path_mappings = Vec::new();
        for spec in cli_path_map
            .or(file_path_map)
            .unwrap_or_default()
            .into_iter()
            .filter(|s| !s.trim().is_empty())
        {
            path_mappings.push(PathMapping::parse(&spec)?);
        }
        // Prefer longer, more specific prefixes first.
        path_mappings.sort_by_key(|m| std::cmp::Reverse(m.internal_prefix.as_os_str().len()));

        let single_workbook = cli_single_workbook.or(file_single_workbook);

        let workspace_root = cli_workspace_root
            .or(file_workspace_root)
            .or_else(|| {
                single_workbook.as_ref().and_then(|path| {
                    if path.is_absolute() {
                        path.parent().map(|parent| parent.to_path_buf())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_else(|| PathBuf::from("."));

        let screenshot_dir = cli_screenshot_dir
            .or(file_screenshot_dir)
            .map(|p| {
                if p.is_absolute() {
                    p
                } else {
                    workspace_root.join(p)
                }
            })
            .unwrap_or_else(|| workspace_root.join("screenshots"));

        let cache_capacity = cli_cache_capacity
            .or(file_cache_capacity)
            .unwrap_or(DEFAULT_CACHE_CAPACITY)
            .max(1);

        let mut supported_extensions = cli_extensions
            .or(file_extensions)
            .unwrap_or_else(|| {
                DEFAULT_EXTENSIONS
                    .iter()
                    .map(|ext| (*ext).to_string())
                    .collect()
            })
            .into_iter()
            .map(|ext| ext.trim().trim_start_matches('.').to_ascii_lowercase())
            .filter(|ext| !ext.is_empty())
            .collect::<Vec<_>>();

        supported_extensions.sort();
        supported_extensions.dedup();

        anyhow::ensure!(
            !supported_extensions.is_empty(),
            "at least one file extension must be provided"
        );

        let single_workbook = single_workbook.map(|path| {
            if path.is_absolute() {
                path
            } else {
                workspace_root.join(path)
            }
        });

        if let Some(workbook_path) = single_workbook.as_ref() {
            anyhow::ensure!(
                workbook_path.exists(),
                "configured workbook {:?} does not exist",
                workbook_path
            );
            anyhow::ensure!(
                workbook_path.is_file(),
                "configured workbook {:?} is not a file",
                workbook_path
            );
            let allowed = workbook_path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.to_ascii_lowercase())
                .map(|ext| supported_extensions.contains(&ext))
                .unwrap_or(false);
            anyhow::ensure!(
                allowed,
                "configured workbook {:?} does not match allowed extensions {:?}",
                workbook_path,
                supported_extensions
            );
        }

        let enabled_tools = cli_enabled_tools
            .or(file_enabled_tools)
            .map(|tools| {
                tools
                    .into_iter()
                    .map(|tool| tool.to_ascii_lowercase())
                    .filter(|tool| !tool.is_empty())
                    .collect::<HashSet<_>>()
            })
            .filter(|set| !set.is_empty());

        let transport = cli_transport
            .or(file_transport)
            .unwrap_or(TransportKind::Http);

        let http_bind_address = cli_http_bind.or(file_http_bind).unwrap_or_else(|| {
            DEFAULT_HTTP_BIND
                .parse()
                .expect("default bind address valid")
        });

        let recalc_enabled = cli_recalc_enabled || file_recalc_enabled.unwrap_or(false);
        let recalc_backend = cli_recalc_backend
            .or(file_recalc_backend)
            .unwrap_or_default();
        let vba_enabled = cli_vba_enabled || file_vba_enabled.unwrap_or(false);

        let max_concurrent_recalcs = cli_max_concurrent_recalcs
            .or(file_max_concurrent_recalcs)
            .unwrap_or(DEFAULT_MAX_RECALCS)
            .max(1);

        let tool_timeout_ms = cli_tool_timeout_ms
            .or(file_tool_timeout_ms)
            .unwrap_or(DEFAULT_TOOL_TIMEOUT_MS);
        let tool_timeout_ms = if tool_timeout_ms == 0 {
            None
        } else {
            Some(tool_timeout_ms)
        };

        let max_response_bytes = cli_max_response_bytes
            .or(file_max_response_bytes)
            .unwrap_or(DEFAULT_MAX_RESPONSE_BYTES);
        let max_response_bytes = if max_response_bytes == 0 {
            None
        } else {
            Some(max_response_bytes)
        };

        let output_profile = cli_output_profile
            .or(file_output_profile)
            .unwrap_or_default();

        let max_payload_bytes = cli_max_payload_bytes
            .or(file_max_payload_bytes)
            .unwrap_or(DEFAULT_MAX_PAYLOAD_BYTES);
        let max_payload_bytes = if max_payload_bytes == 0 {
            None
        } else {
            Some(max_payload_bytes)
        };

        let max_cells = cli_max_cells
            .or(file_max_cells)
            .unwrap_or(DEFAULT_MAX_CELLS);
        let max_cells = if max_cells == 0 {
            None
        } else {
            Some(max_cells)
        };

        let max_items = cli_max_items
            .or(file_max_items)
            .unwrap_or(DEFAULT_MAX_ITEMS);
        let max_items = if max_items == 0 {
            None
        } else {
            Some(max_items)
        };

        let allow_overwrite = cli_allow_overwrite || file_allow_overwrite.unwrap_or(false);

        let slim_surface = cli_slim_surface.or(file_slim_surface).unwrap_or(true);

        Ok(Self {
            workspace_root,
            screenshot_dir,
            path_mappings,
            cache_capacity,
            supported_extensions,
            single_workbook,
            enabled_tools,
            transport,
            http_bind_address,
            recalc_enabled,
            recalc_backend,
            vba_enabled,
            max_concurrent_recalcs,
            tool_timeout_ms,
            max_response_bytes,
            output_profile,
            max_payload_bytes,
            max_cells,
            max_items,
            allow_overwrite,
            slim_surface,
        })
    }
}

#[derive(Parser, Debug, Default, Clone)]
#[command(
    name = "agent-spreadsheet-mcp",
    about = "Spreadsheet MCP server",
    version
)]
pub struct CliArgs {
    #[arg(
        long,
        value_name = "FILE",
        help = "Path to a configuration file (YAML or JSON)",
        global = true
    )]
    pub config: Option<PathBuf>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_WORKSPACE",
        value_name = "DIR",
        help = "Workspace root containing spreadsheet files"
    )]
    pub workspace_root: Option<PathBuf>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_SCREENSHOT_DIR",
        value_name = "DIR",
        help = "Directory to write screenshot PNGs (default: <workspace_root>/screenshots)"
    )]
    pub screenshot_dir: Option<PathBuf>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_PATH_MAP",
        value_name = "INTERNAL=CLIENT",
        value_delimiter = ',',
        help = "Optional path mapping(s) to include client-visible paths in responses (repeatable; useful for Docker volume mounts)"
    )]
    pub path_map: Option<Vec<String>>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_CACHE_CAPACITY",
        value_name = "N",
        help = "Maximum number of workbooks kept in memory",
        value_parser = clap::value_parser!(usize)
    )]
    pub cache_capacity: Option<usize>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_EXTENSIONS",
        value_name = "EXT",
        value_delimiter = ',',
        help = "Comma-separated list of allowed workbook extensions"
    )]
    pub extensions: Option<Vec<String>>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_WORKBOOK",
        value_name = "FILE",
        help = "Lock the server to a single workbook path"
    )]
    pub workbook: Option<PathBuf>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_ENABLED_TOOLS",
        value_name = "TOOL",
        value_delimiter = ',',
        help = "Restrict execution to the provided tool names"
    )]
    pub enabled_tools: Option<Vec<String>>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_TRANSPORT",
        value_enum,
        value_name = "TRANSPORT",
        help = "Transport to expose (http or stdio)"
    )]
    pub transport: Option<TransportKind>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_HTTP_BIND",
        value_name = "ADDR",
        help = "HTTP bind address when using http transport"
    )]
    pub http_bind: Option<SocketAddr>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_RECALC_ENABLED",
        help = "Enable write/recalc tools (uses the native Formualizer backend by default; see --recalc-backend)"
    )]
    pub recalc_enabled: bool,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_RECALC_BACKEND",
        value_enum,
        value_name = "KIND",
        default_value = "auto",
        help = "Recalc backend preference: auto, formualizer, or libreoffice"
    )]
    pub recalc_backend: Option<RecalcBackendKind>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_VBA_ENABLED",
        help = "Enable VBA introspection tools (read-only)"
    )]
    pub vba_enabled: bool,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_MAX_CONCURRENT_RECALCS",
        help = "Max concurrent LibreOffice instances"
    )]
    pub max_concurrent_recalcs: Option<usize>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_TOOL_TIMEOUT_MS",
        value_name = "MS",
        help = "Tool request timeout in milliseconds (default: 30000; 0 disables)",
        value_parser = clap::value_parser!(u64)
    )]
    pub tool_timeout_ms: Option<u64>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_MAX_RESPONSE_BYTES",
        value_name = "BYTES",
        help = "Max response size in bytes (default: 1000000; 0 disables)",
        value_parser = clap::value_parser!(u64)
    )]
    pub max_response_bytes: Option<u64>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_OUTPUT_PROFILE",
        value_enum,
        value_name = "PROFILE",
        help = "Output profile for tool responses (token_dense or verbose)"
    )]
    pub output_profile: Option<OutputProfile>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_MAX_PAYLOAD_BYTES",
        value_name = "BYTES",
        help = "Max tool payload size in bytes before truncation (default: 65536; 0 disables)",
        value_parser = clap::value_parser!(u64)
    )]
    pub max_payload_bytes: Option<u64>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_MAX_CELLS",
        value_name = "N",
        help = "Max cells per tool payload before truncation (default: 10000; 0 disables)",
        value_parser = clap::value_parser!(u64)
    )]
    pub max_cells: Option<u64>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_MAX_ITEMS",
        value_name = "N",
        help = "Max items per tool payload before truncation (default: 500; 0 disables)",
        value_parser = clap::value_parser!(u64)
    )]
    pub max_items: Option<u64>,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_ALLOW_OVERWRITE",
        help = "Allow save_fork to overwrite original workbook files"
    )]
    pub allow_overwrite: bool,

    #[arg(
        long,
        env = "SPREADSHEET_MCP_SLIM_SURFACE",
        value_name = "BOOL",
        help = "Register only the consolidated write surface (mutate_batch + edit_batch). Set to false to also register the per-family batch tools (default: true)"
    )]
    pub slim_surface: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct PartialConfig {
    workspace_root: Option<PathBuf>,
    screenshot_dir: Option<PathBuf>,
    path_map: Option<Vec<String>>,
    cache_capacity: Option<usize>,
    extensions: Option<Vec<String>>,
    single_workbook: Option<PathBuf>,
    enabled_tools: Option<Vec<String>>,
    transport: Option<TransportKind>,
    http_bind: Option<SocketAddr>,
    recalc_enabled: Option<bool>,
    recalc_backend: Option<RecalcBackendKind>,
    vba_enabled: Option<bool>,
    max_concurrent_recalcs: Option<usize>,
    tool_timeout_ms: Option<u64>,
    max_response_bytes: Option<u64>,
    output_profile: Option<OutputProfile>,
    max_payload_bytes: Option<u64>,
    max_cells: Option<u64>,
    max_items: Option<u64>,
    allow_overwrite: Option<bool>,
    slim_surface: Option<bool>,
}

fn load_config_file(path: &Path) -> Result<PartialConfig> {
    if !path.exists() {
        anyhow::bail!("config file {:?} does not exist", path);
    }
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read config file {:?}", path))?;
    let ext = path
        .extension()
        .and_then(|os| os.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let parsed = match ext.as_str() {
        "yaml" | "yml" => serde_yaml::from_str(&contents)
            .with_context(|| format!("failed to parse YAML config {:?}", path))?,
        "json" => serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse JSON config {:?}", path))?,
        other => anyhow::bail!("unsupported config extension: {other}"),
    };
    Ok(parsed)
}
