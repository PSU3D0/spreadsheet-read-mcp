use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

const DEFAULT_CACHE_CAPACITY: usize = 5;
const DEFAULT_EXTENSIONS: &[&str] = &["xlsx", "xls", "xlsb"];
const DEFAULT_HTTP_BIND: &str = "127.0.0.1:8079";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TransportKind {
    Http,
    Stdio,
}

impl std::fmt::Display for TransportKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportKind::Http => write!(f, "http"),
            TransportKind::Stdio => write!(f, "stdio"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub workspace_root: PathBuf,
    pub cache_capacity: usize,
    pub supported_extensions: Vec<String>,
    pub single_workbook: Option<PathBuf>,
    pub enabled_tools: Option<HashSet<String>>,
    pub transport: TransportKind,
    pub http_bind_address: SocketAddr,
}

impl ServerConfig {
    pub fn from_args(args: CliArgs) -> Result<Self> {
        let CliArgs {
            config,
            workspace_root: cli_workspace_root,
            cache_capacity: cli_cache_capacity,
            extensions: cli_extensions,
            workbook: cli_single_workbook,
            enabled_tools: cli_enabled_tools,
            transport: cli_transport,
            http_bind: cli_http_bind,
        } = args;

        let file_config = if let Some(path) = config.as_ref() {
            load_config_file(path)?
        } else {
            PartialConfig::default()
        };

        let PartialConfig {
            workspace_root: file_workspace_root,
            cache_capacity: file_cache_capacity,
            extensions: file_extensions,
            single_workbook: file_single_workbook,
            enabled_tools: file_enabled_tools,
            transport: file_transport,
            http_bind: file_http_bind,
        } = file_config;

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

        Ok(Self {
            workspace_root,
            cache_capacity,
            supported_extensions,
            single_workbook,
            enabled_tools,
            transport,
            http_bind_address,
        })
    }

    pub fn ensure_workspace_root(&self) -> Result<()> {
        anyhow::ensure!(
            self.workspace_root.exists(),
            "workspace root {:?} does not exist",
            self.workspace_root
        );
        anyhow::ensure!(
            self.workspace_root.is_dir(),
            "workspace root {:?} is not a directory",
            self.workspace_root
        );
        if let Some(workbook) = self.single_workbook.as_ref() {
            anyhow::ensure!(
                workbook.exists(),
                "configured workbook {:?} does not exist",
                workbook
            );
            anyhow::ensure!(
                workbook.is_file(),
                "configured workbook {:?} is not a file",
                workbook
            );
        }
        Ok(())
    }

    pub fn resolve_path<P: AsRef<Path>>(&self, relative: P) -> PathBuf {
        let relative = relative.as_ref();
        if relative.is_absolute() {
            relative.to_path_buf()
        } else {
            self.workspace_root.join(relative)
        }
    }

    pub fn single_workbook(&self) -> Option<&Path> {
        self.single_workbook.as_deref()
    }

    pub fn is_tool_enabled(&self, tool: &str) -> bool {
        match &self.enabled_tools {
            Some(set) => set.contains(&tool.to_ascii_lowercase()),
            None => true,
        }
    }
}

#[derive(Parser, Debug, Default, Clone)]
#[command(name = "spreadsheet-mcp", about = "Spreadsheet MCP server", version)]
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
}

#[derive(Debug, Default, Deserialize)]
struct PartialConfig {
    workspace_root: Option<PathBuf>,
    cache_capacity: Option<usize>,
    extensions: Option<Vec<String>>,
    single_workbook: Option<PathBuf>,
    enabled_tools: Option<Vec<String>>,
    transport: Option<TransportKind>,
    http_bind: Option<SocketAddr>,
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
