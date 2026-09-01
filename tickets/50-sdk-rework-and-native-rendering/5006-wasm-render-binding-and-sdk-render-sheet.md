# Ticket: 5006 WASM Render Binding, SDK renderSheet, Worker Mode

## Depends On
- 5002, 5004, 5005

## Why
Rendering is synchronous CPU work. In the browser it must not block the UI thread, and the SDK must return image bytes without any artifact handle crossing its boundary.

## Owner / Effort / Risk
- Owner: WASM adapter / SDK
- Effort: M
- Risk: Low

## Scope
- WASM crate enables the `render` feature. `screenshot_sheet` through `executeOperation` renders into a bounded per-session artifact slot. New binding `readArtifact(sessionId, handle) -> Uint8Array` and `disposeArtifact(sessionId, handle)`; slots are capped by count and aggregate bytes and are dropped with the session.
- SDK `LocalWorkbook.renderSheet()` calls the operation, reads the bytes, disposes the slot, and returns bytes plus report.
- Worker mode: `createLocalSpreadsheet({ runtime, worker: true })` runs the bindings in a Web Worker or `worker_threads` via a small RPC shim. Default on in browsers when available.
- just-bash: `asp op screenshot_sheet --bind /wb.xlsx --output /shot.png` writes bytes to the VFS. Registry just_bash support for `screenshot_sheet` flips to supported.
- Re-measure the size gate. Renderer plus subset fonts should add under 1.5 MiB raw.

## Tests
- Node and browser integration: render a fixture, hash matches the native golden.
- Worker mode round trip and disposal.
- just-bash VFS output test.

## Definition of Done
- Browser and Node users render without LibreOffice, without blocking the UI thread, and with the same bytes as native.

## Pinned after 5004 landed (2026-09-01)

- The render crate is `crates/agent-spreadsheet-render` behind the core `render` feature (on by default natively). The WASM crate opts out of `native-fs`, so the fs-bound screenshot path (`persist_png_artifact`, `ScreenshotSheetData` in `canonical_optional.rs`) does not compile there. 5006 adds a WASM-only screenshot path: same request and output types, artifact bytes held in the session store instead of a file.
- Add `png_level` (`fast`, `balanced`, `best`) as an optional canonical input on `screenshot_sheet`, additive, so the CLI stops bypassing the operation and the SDK can pass it through.
- Registry flip for `screenshot_sheet` (wasm and just_bash to supported) lands in this ticket together with the binding, never before it.
- Goldens hash decoded pixels, not PNG bytes: the PNG container depends on the flate2 backend chosen by workspace feature unification. The wasm golden test must compare pixel hashes against the native goldens in `crates/agent-spreadsheet-render/tests/goldens/pixels-sha256.txt`.
- Re-measure the size gate after enabling `render` in the WASM crate; the bake-off puts the text stack plus fonts at about 700 KB raw, 275 KB compressed.
