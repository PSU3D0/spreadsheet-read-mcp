/**
 * Type tests. These run under `npm run typecheck` (tsc over src plus this directory).
 *
 * Negative cases use `@ts-expect-error`: tsc fails the build if the line below one
 * stops being an error, so "this must not compile" is enforced, not asserted at runtime.
 */

import { expectTypeOf } from "expect-type"

import {
  connectSpreadsheetServer,
  createLocalSpreadsheet,
  type CanonicalErrorEnvelope,
  type InputOf,
  type OperationName,
  type OutputOf
} from "../../src/index.js"

declare const bytes: Uint8Array

const client = connectSpreadsheetServer({ baseUrl: "http://127.0.0.1:8079" })
const local = createLocalSpreadsheet({ runtime: {} as never })

// --- operation names -------------------------------------------------------

expectTypeOf<"read_cells">().toExtend<OperationName>()
// @ts-expect-error 'read_cell' is not a canonical operation name
expectTypeOf<"read_cell">().toExtend<OperationName>()

// @ts-expect-error 'read_cell' is not a canonical operation name
client.canonical.execute("read_cell", {})

// @ts-expect-error 'read_cell' is not a canonical operation name
local.canonical.execute("read_cell", { resource_id: "session:s1" })

// --- input shapes ----------------------------------------------------------

void client.canonical.execute("read_cells", {
  resource_id: "wb:wb-1",
  sheet_name: "Sheet1",
  selection: { kind: "range", ranges: ["A1:B2"] }
})

const selection = { kind: "range", ranges: ["A1:B2"] } as const
// @ts-expect-error sheet_name must be a string, not a number
void client.canonical.execute("read_cells", { resource_id: "wb:wb-1", sheet_name: 5, selection })

// @ts-expect-error selection is required
void client.canonical.execute("read_cells", {
  resource_id: "wb:wb-1",
  sheet_name: "Sheet1"
})

// @ts-expect-error 'kind' must be one of the closed selection variants
void client.canonical.execute("read_cells", { resource_id: "wb:wb-1", sheet_name: "Sheet1", selection: { kind: "columns", ranges: ["A1:B2"] } })

// @ts-expect-error the input schema is closed: unknown keys are rejected
void client.canonical.execute("read_cells", { resource_id: "wb:wb-1", sheet_name: "Sheet1", selection, unknown_field: true })

expectTypeOf<InputOf<"read_cells">>().toHaveProperty("selection")
expectTypeOf<InputOf<"read_cells">["sheet_name"]>().toEqualTypeOf<string>()

// --- output shapes ---------------------------------------------------------

expectTypeOf<OutputOf<"describe_workbook">>().toExtend<{
  schema_version: "1"
  operation: "describe_workbook"
  resource_id: string
  revision_id: string
  data: unknown
}>()
expectTypeOf<OutputOf<"describe_workbook">["operation"]>().toEqualTypeOf<"describe_workbook">()
expectTypeOf<OutputOf<"describe_workbook">>().toHaveProperty("revision_id")

// `list_workbooks` is a discovery response: no resource id, no revision.
expectTypeOf<OutputOf<"list_workbooks">>().toHaveProperty("data")
// @ts-expect-error discovery envelopes carry no resource_id
expectTypeOf<OutputOf<"list_workbooks">>().toHaveProperty("resource_id")

expectTypeOf<CanonicalErrorEnvelope["error"]["code"]>().toExtend<string>()

// --- object model ----------------------------------------------------------

async function surfaces() {
  const workbook = await local.open(bytes)
  expectTypeOf(workbook.resourceId).toEqualTypeOf<string>()
  expectTypeOf(workbook.revisionId).toEqualTypeOf<string | undefined>()
  expectTypeOf(await workbook.describeWorkbook()).toEqualTypeOf<OutputOf<"describe_workbook">>()
  expectTypeOf(await workbook.exportBytes()).toEqualTypeOf<Uint8Array>()

  // Resource ids are injected: the bound surface does not accept one.
  // @ts-expect-error resource_id is injected by the workbook
  await workbook.readCells({ resource_id: "session:s1", sheet_name: "Sheet1", selection })

  const remote = client.workbook("wb:wb-1")
  const fork = await remote.createFork()
  expectTypeOf(await fork.write({
    mode: "apply",
    ops: [{ kind: "set_cells", sheet_name: "Sheet1", cells: { A1: { kind: "value", value: "e" } } }]
  })).toEqualTypeOf<OutputOf<"write">>()

  // A remote read handle cannot write.
  // @ts-expect-error RemoteWorkbook has no write method
  await remote.write({ mode: "apply", ops: [] })

  // Verification is same-runtime by construction.
  // @ts-expect-error a local workbook cannot be verified against a remote fork
  await workbook.verifyAgainst(fork)

  const rendered = await workbook.renderSheet({ sheet_name: "Sheet1" })
  expectTypeOf(rendered.png).toEqualTypeOf<Uint8Array>()
  expectTypeOf(rendered.warnings).toBeArray()
}

void surfaces
