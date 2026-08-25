use anyhow::{Context, Result, bail};
use std::path::Path;
use tokio::process::Command;
use tokio::sync::OnceCell;

const IMAGE_NAME: &str = "agent-spreadsheet-mcp-full";
const IMAGE_TAG: &str = "test";

static IMAGE_BUILT: OnceCell<String> = OnceCell::const_new();

pub fn image_tag() -> String {
    format!("{}:{}", IMAGE_NAME, IMAGE_TAG)
}

async fn build_image(tag: &str) -> Result<()> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above crate manifest");
    let dockerfile_path = workspace_root.join("Dockerfile.full");

    let output = Command::new("docker")
        .args([
            "build",
            "-f",
            dockerfile_path.to_str().unwrap(),
            "-t",
            tag,
            "-q",
            workspace_root.to_str().unwrap(),
        ])
        .output()
        .await
        .context("failed to run docker build")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("docker build failed: {}", stderr);
    }

    Ok(())
}

pub async fn ensure_image() -> Result<String> {
    IMAGE_BUILT
        .get_or_try_init(|| async {
            let tag = image_tag();
            build_image(&tag).await?;
            Ok(tag)
        })
        .await
        .cloned()
}
