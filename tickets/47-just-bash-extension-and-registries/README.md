# 47 — just-bash Extension & Registry Follow-ups

Status: planned. The extension is opt-in; registry work follows the first release that ships archives and checksums.

## `@agent-spreadsheet/just-bash`

**Dependency:** complete [ticket 48](../48-canonical-operation-convergence/README.md) first. The adapter is a proof of the canonical operation boundary, not a place to design another spreadsheet surface.

Build an opt-in extension package that registers one `asp` custom command in [vercel-labs/just-bash](https://github.com/vercel-labs/just-bash). just-bash is a pure-TypeScript Bash interpreter for agents: commands cannot spawn native processes and operate against a virtual filesystem.

The command supports only the canonical machine protocol:

```text
asp op <operation> --json <payload>
```

It reads workbook bytes through `ctx.fs`, forwards the operation unchanged to `agent-spreadsheet-sdk`'s canonical `execute()` dispatch using the embedded WASM backend, and writes resulting bytes/output back through `ctx.fs`. It must not carry an operation list, operation-specific argument parser, response normalizer, or spreadsheet behavior in TypeScript. Schemas/help come from the canonical registry described in `docs/architecture/canonical-operation-surface.md`.

Spreadsheet WASM executes in the trusted host-side adapter, not inside just-bash's QuickJS worker. If `js-exec` is enabled, agent-authored JavaScript can invoke the registered `asp` command through just-bash's `child_process.execSync`/`spawnSync` bridge; a second `tools.asp.*` projection is unnecessary for the MVP.

Contract-test the extension against native `asp op` JSON goldens so operation names, payloads, output envelopes, errors, and workbook mutations cannot drift. Target fewer than approximately 150 production lines excluding tests and virtual-filesystem plumbing; exceeding that indicates semantic behavior is leaking out of the core.

Set and document an explicit workbook-size ceiling before loading bytes into WASM. just-bash worker `resourceLimits` do not cap WASM linear memory, so worker limits alone are not a sufficient memory boundary. Reject oversized inputs predictably and cover the boundary in tests.

## Post-0.13 registry follow-ups

1. Submit an aqua-registry PR so users can run plain `mise use -g agent-spreadsheet`. This must wait until a published release exists with the new per-platform archives and `SHA256SUMS`; registry metadata cannot target assets that do not exist yet.
2. Create `PSU3D0/homebrew-tap` and automate formula bumps from each release, using the published archive checksums.
3. Add winget and Scoop manifests later, after the archive/checksum release flow and Windows installation contract have stabilized.

## Acceptance gate

- `@agent-spreadsheet/just-bash` is explicitly installed and registered rather than bundled into just-bash core.
- The command uses only the SDK canonical dispatcher/embedded WASM backend and the just-bash virtual filesystem.
- The adapter has no independent operation taxonomy or operation-specific argument parsing.
- Native `asp op` JSON goldens contract-test the TypeScript shim.
- Oversized workbooks fail before WASM allocation according to a documented ceiling.
- Registry automation consumes published archives and checksums rather than reconstructing release artifacts.

## Status updates (2026-08-29)

- aqua-registry: PR opened — https://github.com/aquaproj/aqua-registry/pull/59654 (argd-scaffolded, container tests pass, dual-command files entry). Once merged, plain `mise use -g PSU3D0/agent-spreadsheet` works via the aqua backend; `ubi:` form already works and is documented.
- Homebrew tap: LIVE — PSU3D0/homebrew-tap with agent-spreadsheet + agent-spreadsheet-mcp formulas at 0.13.0, install/test verified on linuxbrew. Remaining: automate formula bumps from release.yml (needs a tap-scoped PAT or GitHub App secret).
- winget/scoop: still open.
