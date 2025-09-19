pub mod analysis;
pub mod caps;
pub mod config;
pub mod model;
pub mod server;
pub mod state;
pub mod tools;
pub mod utils;
pub mod workbook;

pub use config::{CliArgs, ServerConfig};
pub use server::SpreadsheetServer;

use anyhow::Result;
use std::sync::Arc;

pub async fn run_server(config: ServerConfig) -> Result<()> {
    let server = SpreadsheetServer::new(Arc::new(config)).await?;
    server.run().await
}
