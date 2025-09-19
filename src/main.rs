use clap::Parser;
use spreadsheet_read_mcp::{CliArgs, ServerConfig, run_server};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = CliArgs::parse();
    let config = ServerConfig::from_args(cli)?;
    run_server(config).await
}
