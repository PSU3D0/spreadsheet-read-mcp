# 47 — just-bash Extension & Registry Follow-ups

Status: planned. The extension is opt-in; registry work follows the first release that ships archives and checksums.

## `@agent-spreadsheet/just-bash`

Build an opt-in extension package that registers `asp` as a custom command in [vercel-labs/just-bash](https://github.com/vercel-labs/just-bash). just-bash is a pure-TypeScript Bash interpreter for agents: commands cannot spawn processes and operate against a virtual filesystem. Its opt-in WASM runtimes, including CPython and sql.js, establish the precedent for embedding a substantial runtime without weakening that sandbox model.

The package should provide a thin TypeScript argv shim over the agent-spreadsheet-sdk embedded WASM backend. The shim reads workbook bytes from the just-bash virtual filesystem, invokes the same SDK operations exposed by the native CLI, and writes resulting workbook bytes and command output back to the virtual filesystem. It must not shell out, access the host filesystem, or duplicate spreadsheet behavior in TypeScript.

Contract-test the extension against the native CLI's JSON goldens so command names, arguments, output envelopes, errors, and workbook mutations cannot drift between native `asp` and the just-bash command. Keep the adapter narrow enough that SDK and CLI contract changes fail tests rather than creating a second compatibility surface.

Set and document an explicit workbook-size ceiling before loading bytes into WASM. just-bash worker `resourceLimits` do not cap WASM linear memory, so worker limits alone are not a sufficient memory boundary. Reject oversized inputs predictably and cover the boundary in tests.

## Post-0.13 registry follow-ups

1. Submit an aqua-registry PR so users can run plain `mise use -g agent-spreadsheet`. This must wait until a published release exists with the new per-platform archives and `SHA256SUMS`; registry metadata cannot target assets that do not exist yet.
2. Create `PSU3D0/homebrew-tap` and automate formula bumps from each release, using the published archive checksums.
3. Add winget and Scoop manifests later, after the archive/checksum release flow and Windows installation contract have stabilized.

## Acceptance gate

- `@agent-spreadsheet/just-bash` is explicitly installed and registered rather than bundled into just-bash core.
- The command uses only the SDK embedded WASM backend and the just-bash virtual filesystem.
- Native CLI JSON goldens contract-test the TypeScript shim.
- Oversized workbooks fail before WASM allocation according to a documented ceiling.
- Registry automation consumes published archives and checksums rather than reconstructing release artifacts.
