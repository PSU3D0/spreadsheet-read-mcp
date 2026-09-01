pub mod session;
/// Path-bound stateless runtime used by the CLI adapter. Host-filesystem only.
#[cfg(feature = "native-fs")]
pub mod stateless;

/// Run synchronous work away from async executor threads on native targets.
/// wasm32 has no blocking thread pool, so the closure is evaluated inline.
#[cfg(not(target_arch = "wasm32"))]
pub async fn maybe_blocking<F, T>(work: F) -> anyhow::Result<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(work)
        .await
        .map_err(anyhow::Error::from)
}

#[cfg(target_arch = "wasm32")]
pub async fn maybe_blocking<F, T>(work: F) -> anyhow::Result<T>
where
    F: FnOnce() -> T,
{
    Ok(work())
}
