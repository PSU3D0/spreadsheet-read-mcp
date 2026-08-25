//! Domain-specific Docker E2E tests.
//!
//! Going forward, add new Docker tests under `tests/docker/` and list them in
//! `tests/docker/mod.rs`. Existing top-level docker tests remain for now.
#![cfg(feature = "docker-tests")]

mod docker;
mod support;
