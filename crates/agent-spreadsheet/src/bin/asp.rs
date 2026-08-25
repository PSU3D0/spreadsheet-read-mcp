#[cfg(not(target_arch = "wasm32"))]
use anyhow::Result;

#[cfg(not(target_arch = "wasm32"))]
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    agent_spreadsheet::cli::run().await
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("asp is unsupported on wasm32 targets");
    std::process::exit(1);
}
