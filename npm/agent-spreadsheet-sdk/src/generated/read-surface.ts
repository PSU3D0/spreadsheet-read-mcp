// GENERATED FILE - DO NOT EDIT.
// Source: src/generated/canonical-registry.json
// Regenerate: npm run generate:types (read surface)

import type { InputOf, OperationName, OutputOf } from "./operations.js"

/** Distributive `Omit` so union-shaped canonical inputs keep every branch. */
export type OmitResource<T> = T extends unknown ? Omit<T, "resource_id"> : never

/** The canonical input for `K` minus the resource id, which the object injects. */
export type BoundInput<K extends OperationName> = OmitResource<InputOf<K>>

/** Operation names in the shared read surface. */
export const READ_SURFACE_OPERATIONS = [
  "describe_workbook",
  "list_sheets",
  "sheet_overview",
  "read_cells",
  "inspect_cells",
  "read_table",
  "read_layout",
  "export_grid",
  "named_ranges",
  "analyze_styles",
  "search_values",
  "search_formulas",
  "formula_trace",
  "formula_map",
  "profile_table",
  "sheet_statistics",
  "screenshot_sheet",
  "sheetport_manifest",
  "execute_sheetport",
  "inspect_vba"
] as const

/** Operation names available on a client without a bound resource. */
export const CLIENT_SURFACE_OPERATIONS = [
  "list_workbooks"
] as const

export type ReadSurfaceOperation = (typeof READ_SURFACE_OPERATIONS)[number]
export type ClientSurfaceOperation = (typeof CLIENT_SURFACE_OPERATIONS)[number]

/**
 * The generated read surface shared by every workbook-shaped object.
 * Methods are declared statically so editors resolve them without `defineProperty`.
 */
export abstract class GeneratedWorkbookView {
  /** Execute `operation` against this object's bound resource. */
  protected abstract executeBound<K extends OperationName>(
    operation: K,
    input: Record<string, unknown>
  ): Promise<OutputOf<K>>

  /**
   * Return cheap exact workbook metadata, with an opt-in derived summary.
   */
  describeWorkbook(input: BoundInput<"describe_workbook"> = {} as BoundInput<"describe_workbook">): Promise<OutputOf<"describe_workbook">> {
    return this.executeBound("describe_workbook", input as Record<string, unknown>)
  }

  /**
   * List sheets in a bound workbook resource with optional bounds.
   */
  listSheets(input: BoundInput<"list_sheets"> = {} as BoundInput<"list_sheets">): Promise<OutputOf<"list_sheets">> {
    return this.executeBound("list_sheets", input as Record<string, unknown>)
  }

  /**
   * Detect regions, headers, bounds, and notable structure for one sheet.
   */
  sheetOverview(input: BoundInput<"sheet_overview">): Promise<OutputOf<"sheet_overview">> {
    return this.executeBound("sheet_overview", input as Record<string, unknown>)
  }

  /**
   * Read correlated exact ranges or projected row windows with revision-bound continuation.
   */
  readCells(input: BoundInput<"read_cells">): Promise<OutputOf<"read_cells">> {
    return this.executeBound("read_cells", input as Record<string, unknown>)
  }

  /**
   * Inspect bounded sparse cells with values, formulas, formats, styles, and calculation state.
   */
  inspectCells(input: BoundInput<"inspect_cells">): Promise<OutputOf<"inspect_cells">> {
    return this.executeBound("inspect_cells", input as Record<string, unknown>)
  }

  /**
   * Read a header-aware table or detected region with filtering and paging.
   */
  readTable(input: BoundInput<"read_table"> = {} as BoundInput<"read_table">): Promise<OutputOf<"read_table">> {
    return this.executeBound("read_table", input as Record<string, unknown>)
  }

  /**
   * Read a deliberately lossy bounded display/layout projection.
   */
  readLayout(input: BoundInput<"read_layout">): Promise<OutputOf<"read_layout">> {
    return this.executeBound("read_layout", input as Record<string, unknown>)
  }

  /**
   * Export cell content and explicit formatting with coordinates, merges, formats, and styles; implicit presentation defaults are excluded.
   */
  exportGrid(input: BoundInput<"export_grid">): Promise<OutputOf<"export_grid">> {
    return this.executeBound("export_grid", input as Record<string, unknown>)
  }

  /**
   * List workbook- and sheet-scoped named items.
   */
  namedRanges(input: BoundInput<"named_ranges"> = {} as BoundInput<"named_ranges">): Promise<OutputOf<"named_ranges">> {
    return this.executeBound("named_ranges", input as Record<string, unknown>)
  }

  /**
   * Analyze style patterns at explicit workbook or sheet scope.
   */
  analyzeStyles(input: BoundInput<"analyze_styles">): Promise<OutputOf<"analyze_styles">> {
    return this.executeBound("analyze_styles", input as Record<string, unknown>)
  }

  /**
   * Search values while preserving label, direction, region, table, type, header, and context modes.
   */
  searchValues(input: BoundInput<"search_values">): Promise<OutputOf<"search_values">> {
    return this.executeBound("search_values", input as Record<string, unknown>)
  }

  /**
   * Search formula cells or grouped classifications, including actual volatile function names.
   */
  searchFormulas(input: BoundInput<"search_formulas">): Promise<OutputOf<"search_formulas">> {
    return this.executeBound("search_formulas", input as Record<string, unknown>)
  }

  /**
   * Trace bounded precedents or dependents from one formula target.
   */
  formulaTrace(input: BoundInput<"formula_trace">): Promise<OutputOf<"formula_trace">> {
    return this.executeBound("formula_trace", input as Record<string, unknown>)
  }

  /**
   * Map sheet formula topology and repeated formula groups.
   */
  formulaMap(input: BoundInput<"formula_map">): Promise<OutputOf<"formula_map">> {
    return this.executeBound("formula_map", input as Record<string, unknown>)
  }

  /**
   * Profile tabular columns, types, distributions, samples, and data quality.
   */
  profileTable(input: BoundInput<"profile_table"> = {} as BoundInput<"profile_table">): Promise<OutputOf<"profile_table">> {
    return this.executeBound("profile_table", input as Record<string, unknown>)
  }

  /**
   * Compute bounded sheet-level statistics.
   */
  sheetStatistics(input: BoundInput<"sheet_statistics">): Promise<OutputOf<"sheet_statistics">> {
    return this.executeBound("sheet_statistics", input as Record<string, unknown>)
  }

  /**
   * Render a bounded sheet range to a content-addressed PNG artifact without exposing a server path.
   */
  screenshotSheet(input: BoundInput<"screenshot_sheet">): Promise<OutputOf<"screenshot_sheet">> {
    return this.executeBound("screenshot_sheet", input as Record<string, unknown>)
  }

  /**
   * Discover, inspect, validate, normalize, or bind-check portable SheetPort manifest content.
   */
  sheetportManifest(input: BoundInput<"sheetport_manifest">): Promise<OutputOf<"sheetport_manifest">> {
    return this.executeBound("sheetport_manifest", input as Record<string, unknown>)
  }

  /**
   * Execute a portable SheetPort manifest with closed typed inputs, results, errors, and coverage.
   */
  executeSheetport(input: BoundInput<"execute_sheetport">): Promise<OutputOf<"execute_sheetport">> {
    return this.executeBound("execute_sheetport", input as Record<string, unknown>)
  }

  /**
   * Inspect a VBA project summary or bounded module source with revision-bound opaque paging.
   */
  inspectVba(input: BoundInput<"inspect_vba">): Promise<OutputOf<"inspect_vba">> {
    return this.executeBound("inspect_vba", input as Record<string, unknown>)
  }
}

/** The generated client-level surface (operations that take no bound resource). */
export abstract class GeneratedClientSurface {
  /** Execute a resource-free `operation`. */
  protected abstract executeClient<K extends OperationName>(
    operation: K,
    input: Record<string, unknown>
  ): Promise<OutputOf<K>>

  /**
   * Discover workbook resources available to this runtime.
   */
  listWorkbooks(input: InputOf<"list_workbooks"> = {} as InputOf<"list_workbooks">): Promise<OutputOf<"list_workbooks">> {
    return this.executeClient("list_workbooks", input as Record<string, unknown>)
  }
}
