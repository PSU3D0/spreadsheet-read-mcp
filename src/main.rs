use clap::Parser;
use spreadsheet_read_mcp::{CliArgs, ServerConfig, run_server};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli = CliArgs::parse();
    let config = ServerConfig::from_args(cli)?;
    run_server(config).await
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}
