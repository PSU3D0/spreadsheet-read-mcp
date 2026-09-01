# Packaging & Naming Conventions

## Package names

| Surface | Crate name | npm name | Binary name |
| --- | --- | --- | --- |
| Core primitives | `agent-spreadsheet` | — | — |
| WASM adapter/runtime | `agent-spreadsheet-wasm` | `agent-spreadsheet-wasm` (planned distribution package) | — |
| MCP server | `agent-spreadsheet-mcp` | — | `agent-spreadsheet-mcp` |
| CLI | `agent-spreadsheet` | `agent-spreadsheet` | `agent-spreadsheet` |
| TypeScript SDK (local WASM + server `/v1` runtimes) | — | `agent-spreadsheet-sdk` | — |

The workspace umbrella name is **agent-spreadsheet**. The GitHub repo is `PSU3D0/agent-spreadsheet-mcp` (historical — predates the workspace split).

## Versioning

- `agent-spreadsheet` — follows semver independently (currently 0.1.x)
- `agent-spreadsheet-wasm` — follows semver independently, but tracks shared core contract changes closely
- `agent-spreadsheet-mcp` — follows semver independently (currently 0.10.x)
- `agent-spreadsheet` (npm) — published from `cli-v*` tags; package version is derived from tag suffix and must match available `v*` GitHub release assets
- `agent-spreadsheet-sdk` (npm) — published from `sdk-v*` tags; semver independent from CLI/binary cadence

Release ordering for tranche-35 surfaces:

1. publish core crates in dependency order (`agent-spreadsheet` → `agent-spreadsheet-wasm` → `agent-spreadsheet-mcp`)
2. publish npm packages (`agent-spreadsheet`, `agent-spreadsheet-sdk`, and `agent-spreadsheet-wasm` when enabled)
3. run smoke tests against both SDK runtimes (local WASM and the server `/v1` route) before final release promotion

### Tag lanes

- `vX.Y.Z` → Rust release lane (GitHub release assets + crates publish)
- `cli-vX.Y.Z` → npm `agent-spreadsheet` publish lane
- `sdk-vX.Y.Z` → npm `agent-spreadsheet-sdk` publish lane

### npm dist-tag policy

- stable `X.Y.Z` publishes to `latest`
- prerelease `X.Y.Z-rc.N` publishes to `rc`
- prerelease `X.Y.Z-beta.N` publishes to `beta`
- prerelease `X.Y.Z-alpha.N` publishes to `alpha`

### Compatibility notes (SDK/MCP/WASM)

| SDK line | MCP compatibility | WASM compatibility | Notes |
| --- | --- | --- | --- |
| `0.1.x` | compatible with `agent-spreadsheet-mcp` `0.10.x` when required capabilities are present | compatible with tranche-35 `agent-spreadsheet-wasm` exports | Capability checks are the source of truth for mixed-version safety |

Policy:

- Shared core contracts follow semver discipline.
- Adapter-only additions should be additive and non-breaking.
- SDK callers must branch on `backend.getCapabilities()` before backend-specific flows (`supportsForkLifecycle`, `supportsStaging`, etc.).
- Capability removals/deprecations require explicit migration notes and release callouts.

## Release artifacts

GitHub Releases include native binaries for operator/server surfaces:

| Asset pattern | Binary |
| --- | --- |
| `agent-spreadsheet-mcp-{target}` | MCP server |
| `agent-spreadsheet-{target}` | CLI |

Targets: `linux-x86_64`, `macos-x86_64`, `macos-aarch64`, `windows-x86_64(.exe)`

WASM + SDK artifacts are published as package artifacts:

| Artifact | Surface |
| --- | --- |
| crate `agent-spreadsheet-wasm` | Rust/WASM adapter crate |
| npm `agent-spreadsheet-sdk` | TypeScript SDK over the local WASM and server `/v1` runtimes |
| npm `agent-spreadsheet-wasm` (planned) | JS/WASM runtime distribution |

## Default features

`agent-spreadsheet-mcp` ships with `recalc-formualizer` as a default feature. This means:
- `cargo install agent-spreadsheet-mcp` and `cargo install agent-spreadsheet --bin agent-spreadsheet` include the Formualizer recalc engine out of the box
- LibreOffice (`recalc-libreoffice`) is only used in the Docker `:full` image
- To build without recalc: `cargo install agent-spreadsheet-mcp --no-default-features`

## Docker images

Published to `ghcr.io/psu3d0/agent-spreadsheet-mcp`:

| Tag | Contents |
| --- | --- |
| `latest` | Slim read-only image (~15 MB), `agent-spreadsheet-mcp` binary only |
| `full` | Full image (~800 MB), includes LibreOffice + recalc macros |

## `agent-spreadsheet` npm install flow

1. `npm install` triggers `postinstall` → `scripts/install.js`
2. Script resolves platform triple (linux-x64, darwin-x64, darwin-arm64, win32-x64)
3. Downloads `agent-spreadsheet-{asset}` from GitHub Releases `v{version}`
4. Places binary in `vendor/` within the package directory
5. `bin/agent-spreadsheet.js` spawns the vendored binary

Override download source with `AGENT_SPREADSHEET_DOWNLOAD_BASE_URL`. Use a pre-built local binary with `AGENT_SPREADSHEET_LOCAL_BINARY`.

## README structure

| File | Audience | Focus |
| --- | --- | --- |
| Root `README.md` | All users | Umbrella: install, quickstarts, tool surface, deployment, config |
| `crates/agent-spreadsheet/README.md` | Crate consumers | Scope, types, what's excluded |
| `crates/agent-spreadsheet-mcp/README.md` | MCP users | Quickstart configs, feature summary, link to root |
| `crates/agent-spreadsheet/README.md` | CLI users | `agent-spreadsheet` binary usage and command surface |
| `npm/agent-spreadsheet/README.md` | npm CLI users | Install, platform matrix, troubleshooting, env vars |
| `npm/agent-spreadsheet-sdk/README.md` | npm SDK users | Object model, generated types, capabilities, typed errors |
