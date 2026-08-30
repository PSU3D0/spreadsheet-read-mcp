# Local WASM clock patch

This is `formualizer-workbook` 0.8.4 from crates.io. The only source change is in `src/backends/umya.rs`: wasm32 uses `web_time::Instant` instead of `std::time::Instant`, which panics on `wasm32-unknown-unknown`. Remove this patch when the upstream crate ships the equivalent `js-runtime` fix.
