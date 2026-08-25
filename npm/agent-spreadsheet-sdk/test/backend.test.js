const test = require("node:test")
const assert = require("node:assert/strict")

const {
  McpBackend,
  WasmBackend,
  CapabilityError,
  BackendOperationError
} = require("../src")
const { MCP_CAPABILITIES } = require("../src/capabilities")

async function sharedDataFlow(backend) {
  const ctx = { workbookId: "wb-1", sessionId: "session-1", contextId: "ctx-1" }
  const describe = await backend.describeWorkbook(ctx)
  const named = await backend.namedRanges(ctx)
  const sheets = await backend.listSheets(ctx)
  const overview = await backend.sheetOverview({
    ...ctx,
    sheetName: "Sheet1",
    maxRegions: 1,
    maxHeaders: 1,
    includeHeaders: true
  })
  const range = await backend.rangeValues({
    ...ctx,
    sheetName: sheets[0],
    ranges: "A1:B2"
  })
  const page = await backend.sheetPage({
    ...ctx,
    sheetName: sheets[0],
    startRow: 2,
    pageSize: 1,
    columnsByHeader: ["Score"],
    includeFormulas: false,
    includeStyles: false,
    includeHeader: true,
    format: "compact"
  })
  const find = await backend.findValue({
    ...ctx,
    sheetName: sheets[0],
    query: "alpha",
    limit: 10,
    offset: 0
  })
  const table = await backend.readTable({
    ...ctx,
    sheetName: sheets[0],
    range: "A1:B2",
    includeHeaders: true,
    includeTypes: false,
    limit: 10,
    offset: 0
  })
  const grid = await backend.gridExport({
    ...ctx,
    sheetName: sheets[0],
    range: "A1:B2"
  })
  const transform = await backend.transformBatch({
    ...ctx,
    ops: [{ kind: "clear_range", sheet_name: sheets[0], target: { kind: "range", range: "A1" } }],
    mode: "preview"
  })
  return { describe, named, sheets, overview, range, page, find, table, grid, transform }
}

test("switching backends keeps shared data callsites stable", async () => {
  const mcp = new McpBackend({
    transport: {
      async invoke(operation, params) {
        if (operation === "describe_workbook") {
          return {
            workbook_id: "wb-1",
            short_id: "session",
            slug: "session",
            path: "virtual/session.xlsx",
            bytes: 123,
            sheet_count: 1,
            defined_names: 0,
            tables: 0,
            macros_present: false,
            caps: { read: true }
          }
        }
        if (operation === "named_ranges") {
          return { workbook_id: "wb-1", items: [] }
        }
        if (operation === "list_sheets") {
          return { sheets: [{ name: "Sheet1" }] }
        }
        if (operation === "sheet_overview") {
          return {
            workbook_id: "wb-1",
            sheet_name: "Sheet1",
            narrative: "overview",
            regions: [],
            detected_regions: [],
            detected_region_count: 0,
            detected_regions_truncated: false,
            key_ranges: [],
            formula_ratio: 0,
            notable_features: [],
            notes: []
          }
        }
        if (operation === "range_values") {
          return {
            sheet_name: "Sheet1",
            values: [{ range: "A1:B2", rows: [["v1", "v2"]] }]
          }
        }
        if (operation === "sheet_page") {
          return {
            workbook_id: "ctx-1",
            sheet_name: "Sheet1",
            next_start_row: 3,
            format: "compact",
            compact: {
              headers: ["Row", "Score"],
              header_row: ["Score"],
              rows: [[2, 42]]
            }
          }
        }
        if (operation === "find_value") {
          return {
            workbook_id: "wb-1",
            matches: [
              { address: "A2", sheet_name: "Sheet1", value: { kind: "text", value: "alpha" } }
            ],
            next_offset: null
          }
        }
        if (operation === "read_table") {
          return {
            workbook_id: "wb-1",
            sheet_name: "Sheet1",
            table_name: null,
            warnings: [],
            headers: [],
            rows: [],
            csv: "Name,Score\nalpha,42\n",
            total_rows: 1,
            next_offset: null
          }
        }
        if (operation === "grid_export") {
          return {
            sheet: "Sheet1",
            anchor: "A1",
            columns: [],
            merges: [],
            rows: []
          }
        }
        if (operation === "transform_batch") {
          assert.equal(params.mode, "preview")
          assert.equal(params.fork_id, "wb-1")
          return {
            ops_applied: 1,
            cells_touched: 1,
            cells_value_set: 0,
            cells_formula_set: 0,
            cells_formula_cleared: 0,
            cells_skipped_keep_formulas: 0
          }
        }
        throw new Error(`unexpected op ${operation}`)
      }
    }
  })

  const wasm = new WasmBackend({
    bindings: {
      async describeWorkbook(sessionId) {
        return {
          workbook_id: "wb-1",
          short_id: "session",
          slug: "session",
          path: "virtual/session.xlsx",
          bytes: 123,
          sheet_count: 1,
          defined_names: 0,
          tables: 0,
          macros_present: false,
          caps: { read: true }
        }
      },
      async namedRanges(sessionId) {
        return { workbook_id: "wb-1", items: [] }
      },
      async listSheets(sessionId) {
        return ["Sheet1"]
      },
      async sheetOverview(sessionId, params) {
        return {
          workbook_id: "wb-1",
          sheet_name: "Sheet1",
          narrative: "overview",
          regions: [],
          detected_regions: [],
          detected_region_count: 0,
          detected_regions_truncated: false,
          key_ranges: [],
          formula_ratio: 0,
          notable_features: [],
          notes: []
        }
      },
      async rangeValues(sessionId, params) {
        return {
          sheetName: "Sheet1",
          values: [{ range: "A1:B2", rows: [["v1", "v2"]] }]
        }
      },
      async sheetPage(sessionId, params) {
        return {
          workbook_id: "ctx-1",
          sheet_name: "Sheet1",
          next_start_row: 3,
          format: "compact",
          compact: {
            headers: ["Row", "Score"],
            header_row: ["Score"],
            rows: [[2, 42]]
          }
        }
      },
      async findValue(sessionId, params) {
        return {
          workbook_id: "wb-1",
          matches: [
            { address: "A2", sheet_name: "Sheet1", value: { kind: "text", value: "alpha" } }
          ],
          next_offset: null
        }
      },
      async readTable(sessionId, params) {
        return {
          workbook_id: "wb-1",
          sheet_name: "Sheet1",
          table_name: null,
          warnings: [],
          headers: [],
          rows: [],
          csv: "Name,Score\nalpha,42\n",
          total_rows: 1,
          next_offset: null
        }
      },
      async gridExport(sessionId, params) {
        return {
          sheet: "Sheet1",
          anchor: "A1",
          columns: [],
          merges: [],
          rows: []
        }
      },
      async transformBatch(sessionId, ops, options) {
        assert.equal(options.dryRun, true)
        return {
          ops_applied: 1,
          cells_touched: 1,
          cells_value_set: 0,
          cells_formula_set: 0,
          cells_formula_cleared: 0,
          cells_skipped_keep_formulas: 0
        }
      }
    }
  })

  const mcpResult = await sharedDataFlow(mcp)
  const wasmResult = await sharedDataFlow(wasm)
  assert.deepEqual(wasmResult, mcpResult)
})

test("capability misuse returns typed capability errors", async () => {
  const wasm = new WasmBackend({ bindings: {} })

  await assert.rejects(
    () => wasm.createFork({ sessionId: "session-1" }),
    (error) => {
      assert.ok(error instanceof CapabilityError)
      assert.equal(error.code, "UNSUPPORTED_CAPABILITY")
      assert.equal(error.backend, "wasm")
      assert.equal(error.capability, "supportsForkLifecycle")
      return true
    }
  )
})

test("backend failures are normalized", async () => {
  const mcp = new McpBackend({
    transport: {
      async invoke() {
        throw new Error("boom")
      }
    }
  })

  await assert.rejects(
    () => mcp.listSheets({ workbookId: "wb-1" }),
    (error) => {
      assert.ok(error instanceof BackendOperationError)
      assert.equal(error.backend, "mcp")
      assert.equal(error.operation, "list_sheets")
      return true
    }
  )
})

test("mcp createFork normalizes workbook id field", async () => {
  const mcp = new McpBackend({
    transport: {
      async invoke(operation, params) {
        assert.equal(operation, "create_fork")
        assert.equal(params.workbook_or_fork_id, "wb-1")
        return { fork_id: "fork-1" }
      }
    }
  })

  const result = await mcp.createFork({ workbookId: "wb-1" })
  assert.equal(result.fork_id, "fork-1")
})

test("mcp transformBatch normalizes context id to fork_id", async () => {
  const mcp = new McpBackend({
    transport: {
      async invoke(operation, params) {
        assert.equal(operation, "transform_batch")
        assert.equal(params.fork_id, "fork-123")
        return { ops_applied: 1, cells_touched: 0 }
      }
    }
  })

  const result = await mcp.transformBatch({
    contextId: "fork-123",
    ops: []
  })
  assert.equal(result.opsApplied, 1)
})

test("mcp verifyWorkbook sends correct params and normalizes response", async () => {
  const mcp = new McpBackend({
    transport: {
      async invoke(operation, params) {
        assert.equal(operation, "verify_workbook")
        assert.equal(params.baseline_workbook_or_fork_id, "fork-base")
        assert.equal(params.current_workbook_or_fork_id, "fork-current")
        assert.deepEqual(params.targets, ["Summary!B2", "Sheet1!C2"])
        assert.equal(params.sheet_name, "Sheet1")
        assert.equal(params.include_named_range_deltas, true)
        assert.equal(params.errors_only, false)
        assert.equal(params.targets_only, false)
        return {
          baseline: "fork-base",
          current: "fork-current",
          target_deltas: [
            {
              address: "Summary!B2",
              before: { kind: "Text", value: "Ready" },
              after: { kind: "Text", value: "Done" },
              before_formula: null,
              after_formula: null,
              classification: "direct_edit",
              changed: true
            }
          ],
          new_errors: [{ address: "Sheet1!D3", after_error: "#NAME?" }],
          resolved_errors: [{ address: "Sheet1!D4", before_error: "#DIV/0!" }],
          preexisting_errors: [{ address: "Sheet1!D2", before_error: "#DIV/0!", after_error: "#DIV/0!" }],
          named_range_deltas: [{ name: "AmountRef", change: "added", after_refers_to: "'Sheet1'!$B$3" }],
          summary: {
            target_count: 2,
            changed_targets: 1,
            new_error_count: 1,
            resolved_error_count: 1,
            preexisting_error_count: 1,
            named_range_delta_count: 1,
            target_classification_counts: {
              unchanged: 0,
              direct_edit: 1,
              recalc_result: 0,
              formula_shift: 0,
              new_error: 0
            }
          }
        }
      }
    }
  })

  const result = await mcp.verifyWorkbook({
    baselineWorkbookOrForkId: "fork-base",
    currentWorkbookOrForkId: "fork-current",
    targets: ["Summary!B2", "Sheet1!C2"],
    sheetName: "Sheet1",
    includeNamedRangeDeltas: true
  })

  assert.equal(result.baselineRef, "fork-base")
  assert.equal(result.currentRef, "fork-current")
  assert.equal(result.targetDeltas.length, 1)
  assert.equal(result.targetDeltas[0].classification, "direct_edit")
  assert.equal(result.targetDeltas[0].beforeFormula, null)
  assert.equal(result.newErrors.length, 1)
  assert.equal(result.resolvedErrors.length, 1)
  assert.equal(result.preexistingErrors.length, 1)
  assert.equal(result.namedRangeDeltas.length, 1)
  assert.equal(result.summary.targetCount, 2)
  assert.equal(result.summary.resolvedErrorCount, 1)
  assert.equal(result.summary.targetClassificationCounts.directEdit, 1)
})

test("mcp verifyTargets and verifyErrors set scope flags", async () => {
  const seen = []
  const mcp = new McpBackend({
    transport: {
      async invoke(operation, params) {
        seen.push({ operation, params })
        return {
          baseline: params.baseline_workbook_or_fork_id,
          current: params.current_workbook_or_fork_id,
          target_deltas: [],
          new_errors: [],
          resolved_errors: [],
          preexisting_errors: [],
          named_range_deltas: [],
          summary: {
            target_count: 0,
            changed_targets: 0,
            new_error_count: 0,
            resolved_error_count: 0,
            preexisting_error_count: 0,
            named_range_delta_count: 0,
            target_classification_counts: {
              unchanged: 0,
              direct_edit: 0,
              recalc_result: 0,
              formula_shift: 0,
              new_error: 0
            }
          }
        }
      }
    }
  })

  await mcp.verifyTargets({
    baselineId: "base-1",
    currentId: "cur-1",
    targets: ["Sheet1!A1"]
  })
  await mcp.verifyErrors({
    baselineId: "base-2",
    currentId: "cur-2",
    sheetName: "Sheet1"
  })

  assert.equal(seen[0].operation, "verify_workbook")
  assert.equal(seen[0].params.targets_only, true)
  assert.equal(seen[1].operation, "verify_workbook")
  assert.equal(seen[1].params.errors_only, true)
})

test("wasm verify helpers throw unsupported error", async () => {
  const wasm = new WasmBackend({ bindings: {} })

  await assert.rejects(
    () => wasm.verifyWorkbook({ sessionId: "session-1" }),
    (error) => {
      assert.ok(error instanceof CapabilityError)
      assert.equal(error.code, "UNSUPPORTED_CAPABILITY")
      assert.equal(error.backend, "wasm")
      assert.equal(error.capability, "supportsVerification")
      return true
    }
  )

  await assert.rejects(
    () => wasm.verifyTargets({ sessionId: "session-1", targets: ["Sheet1!A1"] }),
    (error) => {
      assert.ok(error instanceof CapabilityError)
      assert.equal(error.capability, "supportsVerification")
      return true
    }
  )
})

test("mcp replaceInFormulas sends correct params and normalizes response", async () => {
  const mcp = new McpBackend({
    transport: {
      async invoke(operation, params) {
        assert.equal(operation, "replace_in_formulas")
        assert.equal(params.fork_id, "fork-abc")
        assert.equal(params.sheet_name, "Sheet1")
        assert.equal(params.find, "C2:C10")
        assert.equal(params.replace, "D2:D20")
        assert.equal(params.regex, false)
        assert.equal(params.case_sensitive, true)
        assert.equal(params.mode, "preview")
        return {
          fork_id: "fork-abc",
          mode: "preview",
          change_id: "chg-123",
          formulas_checked: 5,
          formulas_changed: 2,
          recalc_needed: true,
          samples: [
            { address: "B2", before: "SUM(C2:C10)", after: "SUM(D2:D20)" }
          ],
          warnings: []
        }
      }
    }
  })

  const result = await mcp.replaceInFormulas({
    forkId: "fork-abc",
    sheetName: "Sheet1",
    find: "C2:C10",
    replace: "D2:D20",
    options: { dryRun: true }
  })

  assert.equal(result.forkId, "fork-abc")
  assert.equal(result.mode, "preview")
  assert.equal(result.changeId, "chg-123")
  assert.equal(result.formulasChecked, 5)
  assert.equal(result.formulasChanged, 2)
  assert.equal(result.recalcNeeded, true)
  assert.equal(result.samples.length, 1)
  assert.equal(result.samples[0].address, "B2")
})

test("wasm replaceInFormulas throws unsupported error", async () => {
  const wasm = new WasmBackend({ bindings: {} })

  await assert.rejects(
    () => wasm.replaceInFormulas({ sessionId: "session-1", sheetName: "Sheet1", find: "A", replace: "B" }),
    (error) => {
      assert.ok(error instanceof CapabilityError)
      assert.equal(error.code, "UNSUPPORTED_CAPABILITY")
      assert.equal(error.backend, "wasm")
      assert.equal(error.capability, "supportsReplaceInFormulas")
      return true
    }
  )
})

test("backend-specific no-op methods throw explicit unsupported errors", async () => {
  const mcp = new McpBackend({
    transport: {
      async invoke() {
        throw new Error("not used")
      }
    },
    capabilities: {
      ...MCP_CAPABILITIES,
      supportsSessionLifecycle: true,
      supportsExportWorkbook: true
    }
  })

  await assert.rejects(
    () => mcp.createSession(),
    (error) => {
      assert.equal(error.code, "UNSUPPORTED")
      assert.equal(error.backend, "mcp")
      return true
    }
  )
})

test("mcp structureBatch normalizes impact report fields", async () => {
  const mcp = new McpBackend({
    transport: {
      async invoke(operation, params) {
        assert.equal(operation, "structure_batch")
        assert.equal(params.fork_id, "fork-1")
        assert.equal(params.impact_report, true)
        assert.equal(params.show_formula_delta, true)
        return {
          fork_id: "fork-1",
          mode: "preview",
          change_id: "chg-abc",
          ops_applied: 1,
          summary: { op_kinds: ["structure_batch"], affected_sheets: ["Sheet1"] },
          impact_report: {
            shifted_spans: [{ op_index: 0, sheet_name: "Sheet1", axis: "row", description: "rows 2..∞ shift +1", at: 2, count: 1, direction: "insert" }],
            absolute_ref_warnings: [],
            tokens_affected: 3,
            tokens_unaffected: 1,
            notes: []
          },
          formula_delta_preview: [
            { cell: "Sheet1!B1", before: "A5*2", after: "A6*2", classification: "shifted", warning_code: null }
          ]
        }
      }
    }
  })

  const result = await mcp.structureBatch({
    forkId: "fork-1",
    ops: [{ kind: "insert_rows", sheet_name: "Sheet1", at_row: 2, count: 1 }],
    mode: "preview",
    impactReport: true,
    showFormulaDelta: true
  })

  assert.equal(result.forkId, "fork-1")
  assert.equal(result.mode, "preview")
  assert.equal(result.changeId, "chg-abc")
  assert.equal(result.opsApplied, 1)

  // Impact report fields
  assert.ok(result.impactReport, "impactReport should be present")
  assert.equal(result.impactReport.shifted_spans.length, 1)
  assert.equal(result.impactReport.tokens_affected, 3)

  // Formula delta preview
  assert.ok(result.formulaDeltaPreview, "formulaDeltaPreview should be present")
  assert.equal(result.formulaDeltaPreview.length, 1)
  assert.equal(result.formulaDeltaPreview[0].cell, "Sheet1!B1")
  assert.equal(result.formulaDeltaPreview[0].classification, "shifted")
})

test("mcp structureBatch omits impact fields when not requested", async () => {
  const mcp = new McpBackend({
    transport: {
      async invoke(operation, params) {
        assert.equal(operation, "structure_batch")
        // Backend should not return impact fields
        return {
          fork_id: "fork-1",
          mode: "apply",
          ops_applied: 1,
          summary: {}
        }
      }
    }
  })

  const result = await mcp.structureBatch({
    forkId: "fork-1",
    ops: [],
    mode: "apply"
  })

  assert.equal(result.forkId, "fork-1")
  assert.equal(result.impactReport, undefined)
  assert.equal(result.formulaDeltaPreview, undefined)
})

test("normalizeStructureBatchResult handles snake_case and camelCase", () => {
  const { normalizeStructureBatchResult } = require("../src/backend")

  const snake = normalizeStructureBatchResult({
    fork_id: "f1",
    ops_applied: 2,
    impact_report: { shifted_spans: [] },
    formula_delta_preview: [{ cell: "A1" }]
  })
  assert.equal(snake.forkId, "f1")
  assert.equal(snake.opsApplied, 2)
  assert.ok(snake.impactReport)
  assert.equal(snake.formulaDeltaPreview.length, 1)

  const camel = normalizeStructureBatchResult({
    forkId: "f2",
    opsApplied: 3,
    impactReport: { shifted_spans: [1] },
    formulaDeltaPreview: []
  })
  assert.equal(camel.forkId, "f2")
  assert.equal(camel.opsApplied, 3)
  assert.ok(camel.impactReport)
  assert.equal(camel.formulaDeltaPreview.length, 0)
})

test("mcp defineName normalizes payload without leaking camelCase aliases", async () => {
  const mcp = new McpBackend({
    transport: {
      async invoke(operation, params) {
        assert.equal(operation, "define_name")
        assert.deepEqual(params, {
          fork_id: "fork-1",
          name: "MyRange",
          refers_to: "Sheet1!$A$1:$B$2",
          scope: "sheet",
          scope_sheet_name: "Sheet1"
        })
        return { ok: true }
      }
    }
  })

  await mcp.defineName({
    forkId: "fork-1",
    name: "MyRange",
    refersTo: "Sheet1!$A$1:$B$2",
    scope: "sheet",
    scopeSheetName: "Sheet1"
  })
})

test("mcp updateName allows scope-only updates", async () => {
  const mcp = new McpBackend({
    transport: {
      async invoke(operation, params) {
        assert.equal(operation, "update_name")
        assert.deepEqual(params, {
          fork_id: "fork-1",
          name: "MyRange",
          refers_to: undefined,
          scope: "sheet",
          scope_sheet_name: "Sheet1"
        })
        return { ok: true }
      }
    }
  })

  await mcp.updateName({
    forkId: "fork-1",
    name: "MyRange",
    scope: "sheet",
    scopeSheetName: "Sheet1"
  })
})

test("mcp deleteName normalizes payload", async () => {
  const mcp = new McpBackend({
    transport: {
      async invoke(operation, params) {
        assert.equal(operation, "delete_name")
        assert.deepEqual(params, {
          fork_id: "fork-1",
          name: "MyRange",
          scope: "workbook",
          scope_sheet_name: undefined
        })
        return { deleted: true }
      }
    }
  })

  const result = await mcp.deleteName({
    forkId: "fork-1",
    name: "MyRange",
    scope: "workbook"
  })
  assert.equal(result.deleted, true)
})
