const { CapabilityError, SpreadsheetSdkError } = require("./errors")

/**
 * @typedef {import("./capabilities").BackendCapabilities} BackendCapabilities
 */

/**
 * @typedef {object} SpreadsheetBackend
 * @property {"mcp"|"wasm"} kind
 * @property {() => Readonly<BackendCapabilities>} getCapabilities
 * @property {(input: Record<string, unknown>) => Promise<string[]>} listSheets
 * @property {(input: Record<string, unknown>) => Promise<{ sheetName: string, values: unknown[] }>} rangeValues
 */

/**
 * @param {{ kind: string, getCapabilities: () => Record<string, boolean> }} backend
 * @param {string} capability
 * @param {string} method
 */
function requireCapability(backend, capability, method) {
  const capabilities = backend.getCapabilities()
  if (!capabilities[capability]) {
    throw new CapabilityError({
      backend: backend.kind,
      capability,
      method
    })
  }
}

/**
 * @param {unknown} value
 * @param {string} name
 */
function requiredString(value, name) {
  if (typeof value !== "string" || value.length === 0) {
    throw new SpreadsheetSdkError(`missing required field '${name}'`, {
      code: "INVALID_ARGUMENT"
    })
  }
  return value
}

/**
 * @param {unknown} result
 * @param {string} fallbackSheetName
 */
function normalizeSheetPageResult(result, fallbackSheetName) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid sheetPage response", {
      code: "INVALID_RESPONSE"
    })
  }

  return {
    workbookId: typeof result.workbookId === "string"
      ? result.workbookId
      : typeof result.workbook_id === "string"
        ? result.workbook_id
        : undefined,
    sheetName: typeof result.sheetName === "string"
      ? result.sheetName
      : typeof result.sheet_name === "string"
        ? result.sheet_name
        : fallbackSheetName,
    rows: Array.isArray(result.rows) ? result.rows : [],
    nextStartRow: typeof result.nextStartRow === "number"
      ? result.nextStartRow
      : typeof result.next_start_row === "number"
        ? result.next_start_row
        : undefined,
    headerRow: result.headerRow ?? result.header_row,
    compact: result.compact,
    valuesOnly: result.valuesOnly ?? result.values_only,
    format: result.format
  }
}

/**
 * @param {unknown} result
 */
function normalizeGridExportResult(result) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid gridExport response", {
      code: "INVALID_RESPONSE"
    })
  }
  return {
    sheet: result.sheet,
    anchor: result.anchor,
    columns: result.columns,
    merges: result.merges,
    rows: result.rows
  }
}

/**
 * @param {unknown} result
 */
function normalizeTransformBatchResult(result) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid transformBatch response", {
      code: "INVALID_RESPONSE"
    })
  }
  return {
    opsApplied: result.opsApplied ?? result.ops_applied,
    cellsTouched: result.cellsTouched ?? result.cells_touched,
    cellsValueSet: result.cellsValueSet ?? result.cells_value_set,
    cellsFormulaSet: result.cellsFormulaSet ?? result.cells_formula_set,
    cellsFormulaCleared: result.cellsFormulaCleared ?? result.cells_formula_cleared,
    cellsSkippedKeepFormulas: result.cellsSkippedKeepFormulas ?? result.cells_skipped_keep_formulas,
    formulaParseDiagnostics: result.formulaParseDiagnostics ?? result.formula_parse_diagnostics
  }
}

/**
 * @param {unknown} result
 * @param {string | undefined} fallbackWorkbookId
 */
function normalizeDescribeWorkbookResult(result, fallbackWorkbookId) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid describeWorkbook response", {
      code: "INVALID_RESPONSE"
    })
  }

  return {
    workbookId: result.workbookId ?? result.workbook_id ?? fallbackWorkbookId,
    shortId: result.shortId ?? result.short_id,
    slug: result.slug,
    path: result.path,
    clientPath: result.clientPath ?? result.client_path,
    bytes: result.bytes,
    sheetCount: result.sheetCount ?? result.sheet_count,
    definedNames: result.definedNames ?? result.defined_names,
    tables: result.tables,
    macrosPresent: result.macrosPresent ?? result.macros_present,
    lastModified: result.lastModified ?? result.last_modified,
    revisionId: result.revisionId ?? result.revision_id,
    caps: result.caps
  }
}

/**
 * @param {unknown} result
 * @param {string | undefined} fallbackWorkbookId
 */
function normalizeNamedRangesResult(result, fallbackWorkbookId) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid namedRanges response", {
      code: "INVALID_RESPONSE"
    })
  }

  return {
    workbookId: result.workbookId ?? result.workbook_id ?? fallbackWorkbookId,
    items: Array.isArray(result.items) ? result.items : []
  }
}

/**
 * @param {unknown} result
 * @param {string} fallbackSheetName
 * @param {string | undefined} fallbackWorkbookId
 */
function normalizeSheetOverviewResult(result, fallbackSheetName, fallbackWorkbookId) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid sheetOverview response", {
      code: "INVALID_RESPONSE"
    })
  }

  return {
    workbookId: result.workbookId ?? result.workbook_id ?? fallbackWorkbookId,
    sheetName: result.sheetName ?? result.sheet_name ?? fallbackSheetName,
    narrative: result.narrative,
    regions: Array.isArray(result.regions) ? result.regions : [],
    detectedRegions: result.detectedRegions ?? result.detected_regions ?? [],
    detectedRegionCount: result.detectedRegionCount ?? result.detected_region_count,
    detectedRegionsTruncated: result.detectedRegionsTruncated ?? result.detected_regions_truncated,
    keyRanges: result.keyRanges ?? result.key_ranges ?? [],
    formulaRatio: result.formulaRatio ?? result.formula_ratio,
    notableFeatures: result.notableFeatures ?? result.notable_features ?? [],
    notes: Array.isArray(result.notes) ? result.notes : []
  }
}

/**
 * @param {unknown} result
 * @param {string | undefined} fallbackWorkbookId
 */
function normalizeFindValueResult(result, fallbackWorkbookId) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid findValue response", {
      code: "INVALID_RESPONSE"
    })
  }

  return {
    workbookId: result.workbookId ?? result.workbook_id ?? fallbackWorkbookId,
    matches: Array.isArray(result.matches) ? result.matches : [],
    nextOffset: result.nextOffset ?? result.next_offset
  }
}

/**
 * @param {unknown} result
 * @param {string | undefined} fallbackWorkbookId
 * @param {string | undefined} fallbackSheetName
 */
function normalizeReadTableResult(result, fallbackWorkbookId, fallbackSheetName) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid readTable response", {
      code: "INVALID_RESPONSE"
    })
  }

  return {
    workbookId: result.workbookId ?? result.workbook_id ?? fallbackWorkbookId,
    sheetName: result.sheetName ?? result.sheet_name ?? fallbackSheetName,
    tableName: result.tableName ?? result.table_name,
    warnings: Array.isArray(result.warnings) ? result.warnings : [],
    headers: Array.isArray(result.headers) ? result.headers : [],
    rows: Array.isArray(result.rows) ? result.rows : [],
    values: result.values,
    types: result.types,
    csv: result.csv,
    totalRows: result.totalRows ?? result.total_rows ?? 0,
    nextOffset: result.nextOffset ?? result.next_offset
  }
}

/**
 * @param {unknown} result
 */
function normalizeStructureBatchResult(result) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid structureBatch response", {
      code: "INVALID_RESPONSE"
    })
  }
  return {
    forkId: result.forkId ?? result.fork_id,
    mode: result.mode,
    changeId: result.changeId ?? result.change_id,
    opsApplied: result.opsApplied ?? result.ops_applied,
    summary: result.summary,
    formulaParseDiagnostics: result.formulaParseDiagnostics ?? result.formula_parse_diagnostics,
    impactReport: result.impactReport ?? result.impact_report,
    formulaDeltaPreview: result.formulaDeltaPreview ?? result.formula_delta_preview
  }
}

/**
 * @param {unknown} result
 */
function normalizeReplaceInFormulasResult(result) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid replaceInFormulas response", {
      code: "INVALID_RESPONSE"
    })
  }
  return {
    forkId: result.forkId ?? result.fork_id,
    mode: result.mode,
    changeId: result.changeId ?? result.change_id,
    formulasChecked: result.formulasChecked ?? result.formulas_checked ?? 0,
    formulasChanged: result.formulasChanged ?? result.formulas_changed ?? 0,
    recalcNeeded: result.recalcNeeded ?? result.recalc_needed ?? false,
    samples: Array.isArray(result.samples) ? result.samples : [],
    warnings: Array.isArray(result.warnings) ? result.warnings : [],
    formulaParseDiagnostics: result.formulaParseDiagnostics ?? result.formula_parse_diagnostics
  }
}

/**
 * @param {unknown} result
 * @param {string | undefined} fallbackBaseline
 * @param {string | undefined} fallbackCurrent
 */
function normalizeVerifyWorkbookResult(result, fallbackBaseline, fallbackCurrent) {
  if (!result || typeof result !== "object") {
    throw new SpreadsheetSdkError("invalid verifyWorkbook response", {
      code: "INVALID_RESPONSE"
    })
  }

  const targetClassificationCounts = result.summary?.targetClassificationCounts
    ?? result.summary?.target_classification_counts
    ?? {}

  return {
    baselineRef: result.baselineRef ?? result.baseline_ref ?? result.baseline ?? fallbackBaseline,
    currentRef: result.currentRef ?? result.current_ref ?? result.current ?? fallbackCurrent,
    targetDeltas: Array.isArray(result.targetDeltas)
      ? result.targetDeltas
      : Array.isArray(result.target_deltas)
        ? result.target_deltas.map((item) => ({
            address: item.address,
            before: item.before,
            after: item.after,
            beforeFormula: item.beforeFormula ?? item.before_formula,
            afterFormula: item.afterFormula ?? item.after_formula,
            classification: item.classification,
            changed: item.changed
          }))
        : [],
    newErrors: Array.isArray(result.newErrors)
      ? result.newErrors
      : Array.isArray(result.new_errors)
        ? result.new_errors.map((item) => ({
            address: item.address,
            beforeError: item.beforeError ?? item.before_error,
            afterError: item.afterError ?? item.after_error,
            beforeFormula: item.beforeFormula ?? item.before_formula,
            afterFormula: item.afterFormula ?? item.after_formula
          }))
        : [],
    resolvedErrors: Array.isArray(result.resolvedErrors)
      ? result.resolvedErrors
      : Array.isArray(result.resolved_errors)
        ? result.resolved_errors.map((item) => ({
            address: item.address,
            beforeError: item.beforeError ?? item.before_error,
            afterError: item.afterError ?? item.after_error,
            beforeFormula: item.beforeFormula ?? item.before_formula,
            afterFormula: item.afterFormula ?? item.after_formula
          }))
        : [],
    preexistingErrors: Array.isArray(result.preexistingErrors)
      ? result.preexistingErrors
      : Array.isArray(result.preexisting_errors)
        ? result.preexisting_errors.map((item) => ({
            address: item.address,
            beforeError: item.beforeError ?? item.before_error,
            afterError: item.afterError ?? item.after_error,
            beforeFormula: item.beforeFormula ?? item.before_formula,
            afterFormula: item.afterFormula ?? item.after_formula
          }))
        : [],
    namedRangeDeltas: Array.isArray(result.namedRangeDeltas)
      ? result.namedRangeDeltas
      : Array.isArray(result.named_range_deltas)
        ? result.named_range_deltas.map((item) => ({
            name: item.name,
            scopeKind: item.scopeKind ?? item.scope_kind,
            scopeSheetName: item.scopeSheetName ?? item.scope_sheet_name,
            change: item.change,
            beforeRefersTo: item.beforeRefersTo ?? item.before_refers_to,
            afterRefersTo: item.afterRefersTo ?? item.after_refers_to,
            beforeKind: item.beforeKind ?? item.before_kind,
            afterKind: item.afterKind ?? item.after_kind
          }))
        : [],
    summary: {
      targetCount: result.summary?.targetCount ?? result.summary?.target_count ?? 0,
      changedTargets: result.summary?.changedTargets ?? result.summary?.changed_targets ?? 0,
      newErrorCount: result.summary?.newErrorCount ?? result.summary?.new_error_count ?? 0,
      resolvedErrorCount: result.summary?.resolvedErrorCount ?? result.summary?.resolved_error_count ?? 0,
      preexistingErrorCount: result.summary?.preexistingErrorCount ?? result.summary?.preexisting_error_count ?? 0,
      namedRangeDeltaCount: result.summary?.namedRangeDeltaCount ?? result.summary?.named_range_delta_count ?? 0,
      targetClassificationCounts: {
        unchanged: targetClassificationCounts.unchanged ?? 0,
        directEdit: targetClassificationCounts.directEdit ?? targetClassificationCounts.direct_edit ?? 0,
        recalcResult: targetClassificationCounts.recalcResult ?? targetClassificationCounts.recalc_result ?? 0,
        formulaShift: targetClassificationCounts.formulaShift ?? targetClassificationCounts.formula_shift ?? 0,
        newError: targetClassificationCounts.newError ?? targetClassificationCounts.new_error ?? 0
      }
    }
  }
}

module.exports = {
  requireCapability,
  requiredString,
  normalizeSheetPageResult,
  normalizeGridExportResult,
  normalizeTransformBatchResult,
  normalizeStructureBatchResult,
  normalizeReplaceInFormulasResult,
  normalizeVerifyWorkbookResult,
  normalizeDescribeWorkbookResult,
  normalizeNamedRangesResult,
  normalizeSheetOverviewResult,
  normalizeFindValueResult,
  normalizeReadTableResult
}
