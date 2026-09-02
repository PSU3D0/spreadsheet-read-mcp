# 50 - SDK Rework + Native Rendering

Status: implemented on `session/50-integration` (2026-09-01), pending review and the 0.15 release. Follows the 0.14 canonical release.

## Outcome

Make `agent-spreadsheet-sdk` a programmatic-first product surface with real types, and replace the LibreOffice-only screenshot path with a bounded native raster renderer that also runs in WASM.

## Decisions (fixed for this tranche)

1. No canonical artifact-retrieval operation. Image bytes cross adapter boundaries: MCP attaches image content, the HTTP route serves bytes, the WASM binding returns `Uint8Array`.
2. The SDK stops pretending to be a dual MCP/WASM adapter. It has two runtimes, local (WASM) and server (canonical HTTP route on the MCP server process), behind one programmatic object model. The MCP client adapter is dropped; MCP users use an MCP client.
3. The MCP server gains a plain canonical HTTP route beside the MCP service. Same dispatcher, same registry, same in-process fork/checkpoint state.
4. `screenshot_sheet` keeps its name. Output grows additively (`renderer`, `fidelity`, `warnings`, calculation state). Optional `backend` input selects `native` or `libreoffice`.
5. Rendering never recalculates. It renders cached values and reports calculation state.
6. Renderer core has no filesystem, process, tokio, MCP, or Typst dependency. Budget: tiny-skia, png, a shaper, embedded subset fonts.
7. The SDK moves to TypeScript source with a tsc build to CJS and ESM. Legacy backends live under a `compat` subpath for one release.

## Tickets

1. [5001](./5001-canonical-http-route-and-mcp-image-content.md) - canonical HTTP route on the server, artifact bytes route, MCP image content for `screenshot_sheet`
2. [5002](./5002-wasm-build-diet-resident-sessions-npm-package.md) - WASM dependency diet, release profile, size gate, resident sessions, `agent-spreadsheet-wasm` npm package
3. [5003](./5003-renderer-bake-off.md) - renderer bake-off against the corpus with BetterOffice, readany-render, and office2pdf as references
4. [5004](./5004-native-raster-renderer-crate.md) - `agent-spreadsheet-render` crate MVP and canonical wiring
5. [5005](./5005-sdk-rework-typescript-runtimes-object-model.md) - SDK 0.15 rework
6. [5006](./5006-wasm-render-binding-and-sdk-render-sheet.md) - WASM render binding, SDK `renderSheet`, worker mode

## Order

- P1 in parallel: 5001, 5002, 5003
- P2: 5005 once 5001 pins the route shape and 5002 pins the binding shape; 5004 once 5003 reports
- P3: 5006

## Acceptance gate

- A TypeScript user gets compile errors for wrong operation names and wrong input shapes.
- `npm install agent-spreadsheet-sdk agent-spreadsheet-wasm` is enough to open workbook bytes, read, write, recalculate, verify, render, and export with no repository checkout.
- `screenshot_sheet` works on the slim MCP image, on the CLI, and in WASM, with structured fidelity warnings, and LibreOffice is opt-in.
- Release WASM bundle size is enforced in CI.
