pub mod analysis;
pub mod caps;
pub mod cli;
pub mod config;
pub mod core;
#[cfg(feature = "recalc")]
pub mod diff;
pub mod errors;
#[cfg(feature = "recalc")]
pub mod fork;
pub mod formula;
pub mod model;
#[cfg(feature = "recalc")]
pub mod recalc;
pub mod repository;
pub mod response_prune;
pub mod rules;
pub mod runtime;
pub mod security;
pub mod server;
pub mod state;
pub mod styles;
pub mod tools;
pub mod utils;
pub mod workbook;

pub use config::{CliArgs, OutputProfile, RecalcBackendKind, ServerConfig, TransportKind};
pub use server::SpreadsheetServer;

use anyhow::Result;
use axum::Router;
use model::WorkbookListResponse;
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};
use state::AppState;
use std::{future::IntoFuture, sync::Arc};
use tokio::{
    net::TcpListener,
    time::{Duration, timeout},
};
use tools::filters::WorkbookFilter;

const HTTP_SERVICE_PATH: &str = "/mcp";

pub async fn run_server(config: ServerConfig) -> Result<()> {
    let config = Arc::new(config);
    config.ensure_workspace_root()?;
    let state = Arc::new(AppState::new(config.clone()));

    tracing::info!(
        transport = %config.transport,
        workspace = %config.workspace_root.display(),
        "starting spreadsheet MCP server",
    );

    match startup_scan(&state) {
        Ok(response) => {
            let count = response.workbooks.len();
            if count == 0 {
                tracing::info!("startup scan complete: no workbooks discovered");
            } else {
                let sample = response
                    .workbooks
                    .iter()
                    .take(3)
                    .filter_map(|descriptor| descriptor.path.as_deref())
                    .collect::<Vec<_>>()
                    .join(", ");
                tracing::info!(
                    workbook_count = count,
                    sample = %sample,
                    "startup scan discovered workbooks"
                );
            }
        }
        Err(error) => {
            tracing::warn!(?error, "startup scan failed");
        }
    }

    match config.transport {
        TransportKind::Stdio => {
            let server = SpreadsheetServer::from_state(state);
            server.run_stdio().await
        }
        TransportKind::Http => run_stream_http_transport(config, state).await,
    }
}

async fn run_stream_http_transport(config: Arc<ServerConfig>, state: Arc<AppState>) -> Result<()> {
    let bind_addr = config.http_bind_address;
    let service_state = state.clone();
    let service = StreamableHttpService::new(
        move || Ok(SpreadsheetServer::from_state(service_state.clone())),
        LocalSessionManager::default().into(),
        Default::default(),
    );

    let router = Router::new().nest_service(HTTP_SERVICE_PATH, service);
    let listener = TcpListener::bind(bind_addr).await?;
    let actual_addr = listener.local_addr()?;
    tracing::info!(transport = "http", bind = %actual_addr, path = HTTP_SERVICE_PATH, "listening" );

    let server_future = axum::serve(listener, router).into_future();
    tokio::pin!(server_future);

    tokio::select! {
        result = server_future.as_mut() => {
            tracing::info!("http transport stopped");
            result.map_err(anyhow::Error::from)?;
            return Ok(());
        }
        ctrl = tokio::signal::ctrl_c() => {
            match ctrl {
                Ok(_) => tracing::info!("shutdown signal received"),
                Err(error) => tracing::warn!(?error, "ctrl_c listener exited unexpectedly"),
            };
        }
    }

    if timeout(Duration::from_secs(5), server_future.as_mut())
        .await
        .is_err()
    {
        tracing::warn!("forcing http transport shutdown after timeout");
        return Ok(());
    }

    server_future.as_mut().await.map_err(anyhow::Error::from)?;
    tracing::info!("http transport stopped");
    Ok(())
}

pub fn startup_scan(state: &Arc<AppState>) -> Result<WorkbookListResponse> {
    state.list_workbooks(WorkbookFilter::default())
}
