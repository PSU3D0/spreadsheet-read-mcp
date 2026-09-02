/** How closely a rendered sheet matches the workbook's presentation. */
export type RenderFidelity = "exact" | "approximate" | "degraded" | "unknown" | (string & {})

/** A structured renderer warning. */
export interface RenderWarning {
  code?: string
  message?: string
  [key: string]: unknown
}

/** PNG encoder effort. Geometry never depends on it. */
export type RenderPngLevel = "fast" | "balanced" | "best"

/** Input for `renderSheet`. */
export interface RenderSheetInput {
  /** Sheet to render. */
  sheet_name: string
  /** Optional A1 range; the adapter picks a bounded default when omitted. */
  range?: string
  /**
   * PNG encoder effort for the in-process raster renderer. `fast` trades bytes
   * for latency, `best` trades latency for bytes. Rejected by the LibreOffice
   * backend, which owns its own encoder.
   */
  png_level?: RenderPngLevel
}

/** The result of rendering one sheet to PNG bytes. */
export interface RenderedSheet {
  /** PNG bytes, fetched across the adapter boundary rather than through a canonical operation. */
  png: Uint8Array
  /** Reported fidelity; `"unknown"` when the running server predates fidelity reporting. */
  fidelity: RenderFidelity
  /** Structured renderer warnings; empty when the renderer reported none. */
  warnings: RenderWarning[]
  /** Calculation state at render time; `null` when the server does not report it. */
  calculation: unknown
  /** The sheet the adapter actually rendered. */
  sheet_name: string
  /** The range the adapter actually rendered. */
  range: string
  /** The content-addressed artifact handle the bytes came from. */
  handle: string
  /** Renderer identity when reported (`native-raster/1`, `libreoffice`). */
  renderer: string | undefined
  /** Image width in device pixels, when the renderer reported it. */
  width: number | undefined
  /** Image height in device pixels, when the renderer reported it. */
  height: number | undefined
  /** The PNG encoder effort the render actually used, when the backend exposes one. */
  png_level: RenderPngLevel | undefined
}
