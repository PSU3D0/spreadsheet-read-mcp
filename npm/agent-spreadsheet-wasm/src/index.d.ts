export interface CreateWasmRuntimeOptions {
  /** Explicit location of `agent_spreadsheet_wasm_bg.wasm`. */
  wasmUrl?: string | URL
  /** Preloaded module bytes, e.g. from a bundler asset import. */
  wasmBytes?: BufferSource | WebAssembly.Module
}

/**
 * The wasm-bindgen export surface consumed by `WasmBackend({ bindings })`.
 * See `pkg/agent_spreadsheet_wasm.d.ts` for the generated signatures.
 */
export type WasmRuntime = typeof import("../pkg/agent_spreadsheet_wasm.js")

export function createWasmRuntime(options?: CreateWasmRuntimeOptions): Promise<WasmRuntime>
export default createWasmRuntime
export const rawBindings: WasmRuntime
