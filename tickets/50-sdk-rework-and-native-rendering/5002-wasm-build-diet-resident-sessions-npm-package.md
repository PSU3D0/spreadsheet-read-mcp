# Ticket: 5002 WASM Build Diet, Resident Sessions, npm Package

## Depends On
- none

## Why
The dev WASM bundle is 47 MiB and a default release build is 21 MiB. Neither Cargo.toml has a release profile, nothing runs wasm-opt, and the core crate pulls clap, tokio, tracing-subscriber, walkdir, globset, serde_yaml, tempfile, image, png, and rayon into the wasm32 build. Every read in `crates/agent-spreadsheet-wasm/src/lib.rs::execute_operation` also rebuilds a virtual repository and `AppState` from the session bytes, so each call re-parses the workbook. Finally the bindings are not on npm, so `WasmBackend({ bindings })` is unusable outside a repository checkout.

## Owner / Effort / Risk
- Owner: WASM adapter / core crate features
- Effort: M
- Risk: Med (feature gating touches the core crate; resident sessions touch revision invariants)

## Scope

### Dependency diet
- Introduce core-crate features so wasm32 builds compile none of: clap, tracing-subscriber, serde_yaml, walkdir, globset, tempfile, tokio runtime features, image, png, rayon. Suggested: `cli` (clap, tracing-subscriber, serde_yaml, walkdir, globset), `native-fs` (tempfile, tokio fs/process/signal), and image/png only under `recalc-libreoffice` or the future `render` feature.
- Native default features and the `asp`, `agent-spreadsheet`, and `agent-spreadsheet-mcp` binaries must build exactly as today. `cargo test --workspace --locked` stays green.
- Add `[profile.release]` for the WASM crate: opt-level z, lto fat, codegen-units 1, panic abort. Run `wasm-opt -Oz` in the build script. Record before and after sizes in the ticket.
- Add a CI size gate on the release web bundle (raw and brotli) with an explicit ceiling. Start the ceiling at the measured size plus 10 percent and tighten later.

### Resident sessions
- Keep the parsed workbook resident per session instead of rebuilding `virtual_state` on every read. Rebuild only after `write` apply, `recalculate`, or any path that replaces session bytes. Session bytes remain the source of truth for `exportWorkbook` and revision hashing.
- Add a test that a sequence of reads after a write sees the new revision and values, and that `exportWorkbook` bytes equal the pre-resident behaviour byte for byte for the existing fixtures.

### npm package `agent-spreadsheet-wasm`
- New directory `npm/agent-spreadsheet-wasm` publishing the `--target web` build plus a small loader: `createWasmRuntime({ wasmUrl?, wasmBytes? })` that works in Node 18+ (reads the file) and browsers (fetches). It returns the bindings object the SDK consumes. Include the generated `.d.ts`.
- Publish lane `wasm-vX.Y.Z` mirroring `publish-npm-sdk.yml`. Version pinned to the SDK version.
- Do not change the binding function names or semantics in this ticket. 5006 adds render bindings.

## Non-Goals
- Worker mode (5006).
- Any change to canonical operation semantics.

## Tests
- `cargo check -p agent-spreadsheet-wasm --target wasm32-unknown-unknown` with `cargo tree` proof that the listed crates are absent.
- Existing wasm Node and browser integration harnesses pass against the release build, not only `--dev`.
- Size gate runs in CI and fails when exceeded.
- Loader smoke test in Node using the built package.

## Definition of Done
- Release web bundle size is measured, enforced, and materially below 21 MiB.
- `npm install agent-spreadsheet-wasm` gives a working runtime without a checkout.
- Repeated reads do not re-parse the workbook.
