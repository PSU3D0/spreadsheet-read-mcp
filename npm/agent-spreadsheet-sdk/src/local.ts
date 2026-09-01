import { CapabilityError, TransportError, decodeRejection } from "./errors.js"
import type { OperationName, OutputOf } from "./generated/operations.js"
import { GeneratedClientSurface } from "./generated/read-surface.js"
import { normalizeOperationList } from "./registry.js"
import { CanonicalApi, type CanonicalRuntime, executeCanonical } from "./runtime.js"
import { MutableWorkbookHandle, type VerifyInput } from "./view.js"

/**
 * The binding object exported by `agent-spreadsheet-wasm`
 * (`createWasmRuntime({ wasmUrl?, wasmBytes? })`).
 */
export interface WasmBindings {
  /** Create a resident session from workbook bytes; returns `session:<id>`. */
  createSession(bytes: Uint8Array): string | PromiseLike<string>
  /** Serialized canonical discovery for this runtime. */
  operations(): unknown
  /** Dispatch a canonical operation; rejects with a canonical error envelope string. */
  executeOperation(
    sessionId: string,
    operation: string,
    paramsJson: string
  ): string | PromiseLike<string>
  /** Export the latest applied bytes for a session. */
  exportWorkbook?(sessionId: string): Uint8Array | PromiseLike<Uint8Array>
  /** Release a session. */
  disposeSession?(sessionId: string): unknown
  /** Artifact bytes for a handle produced in this session (ticket 5006). */
  readArtifact?(sessionId: string, handle: string): Uint8Array | PromiseLike<Uint8Array>
}

/** Options for {@link createLocalSpreadsheet}. */
export interface LocalSpreadsheetOptions {
  /** The bindings object, or a promise of it. */
  runtime: WasmBindings | PromiseLike<WasmBindings>
}

function assertBindings(bindings: unknown): asserts bindings is WasmBindings {
  const candidate = bindings as Partial<WasmBindings> | null
  if (!candidate || typeof candidate !== "object") {
    throw new TypeError("createLocalSpreadsheet requires a runtime bindings object")
  }
  // Byte export and session disposal are capability-gated rather than required: a
  // read-only build advertises fewer operations and still drives the read surface.
  for (const method of ["createSession", "operations", "executeOperation"] as const) {
    if (typeof candidate[method] !== "function") {
      throw new TypeError(`runtime bindings are missing ${method}()`)
    }
  }
}

class LocalRuntime implements CanonicalRuntime {
  readonly kind = "local"
  readonly bindings: WasmBindings
  #operations: ReadonlySet<string> | undefined

  constructor(bindings: WasmBindings) {
    this.bindings = bindings
  }

  async operations(): Promise<ReadonlySet<string>> {
    if (!this.#operations) this.#operations = normalizeOperationList(this.bindings.operations())
    return this.#operations
  }

  async dispatch<K extends OperationName>(
    operation: K,
    input: Record<string, unknown>
  ): Promise<OutputOf<K>> {
    const sessionId = typeof input["resource_id"] === "string" ? (input["resource_id"] as string) : ""
    let raw: string
    try {
      raw = await this.bindings.executeOperation(sessionId, operation, JSON.stringify(input))
    } catch (rejection) {
      throw decodeRejection(rejection, { operation, runtime: this.kind })
    }
    return parseEnvelope(raw, operation)
  }

  async artifactBytes(handle: string, resourceId: string): Promise<Uint8Array> {
    if (typeof this.bindings.readArtifact !== "function") {
      throw new CapabilityError({
        capability: "readArtifact",
        message:
          "this WASM runtime cannot return artifact bytes; upgrade to a build with readArtifact()"
      })
    }
    try {
      return await this.bindings.readArtifact(resourceId, handle)
    } catch (rejection) {
      throw decodeRejection(rejection, { operation: "screenshot_sheet", runtime: this.kind })
    }
  }
}

function parseEnvelope<K extends OperationName>(raw: unknown, operation: K): OutputOf<K> {
  if (typeof raw !== "string") return raw as OutputOf<K>
  try {
    return JSON.parse(raw) as OutputOf<K>
  } catch (cause) {
    throw new TransportError(`the local runtime returned invalid JSON for '${operation}'`, {
      operation,
      cause
    })
  }
}

/** A workbook opened from bytes in the local WASM runtime. This handle owns its session. */
export class LocalWorkbook extends MutableWorkbookHandle {
  readonly #local: LocalRuntime
  #disposed = false

  /** @internal */
  constructor(runtime: LocalRuntime, resourceId: string) {
    super(runtime, resourceId)
    this.#local = runtime
  }

  /** True once {@link dispose} has released the session. */
  get disposed(): boolean {
    return this.#disposed
  }

  protected override async executeBound<K extends OperationName>(
    operation: K,
    input: Record<string, unknown>
  ): Promise<OutputOf<K>> {
    this.#assertLive()
    return super.executeBound(operation, input)
  }

  /** Compare this workbook against another local workbook. */
  verifyAgainst(baseline: LocalWorkbook, input: VerifyInput = {}): Promise<OutputOf<"verify_workbook">> {
    return this.verifyAgainstResource(baseline.resourceId, input)
  }

  /** The latest applied or recalculated workbook bytes. */
  async exportBytes(): Promise<Uint8Array> {
    this.#assertLive()
    if (typeof this.#local.bindings.exportWorkbook !== "function") {
      throw new CapabilityError({
        capability: "exportWorkbook",
        message: "this WASM runtime cannot export workbook bytes"
      })
    }
    try {
      return await this.#local.bindings.exportWorkbook(this.resourceId)
    } catch (rejection) {
      throw decodeRejection(rejection, { operation: "export_workbook", runtime: "local" })
    }
  }

  /** Release the underlying WASM session. Safe to call twice. */
  async dispose(): Promise<void> {
    if (this.#disposed) return
    this.#disposed = true
    if (typeof this.#local.bindings.disposeSession !== "function") return
    try {
      await this.#local.bindings.disposeSession(this.resourceId)
    } catch (rejection) {
      throw decodeRejection(rejection, { operation: "dispose_session", runtime: "local" })
    }
  }

  /** `await using workbook = await local.open(bytes)`. */
  async [Symbol.asyncDispose](): Promise<void> {
    await this.dispose()
  }

  #assertLive(): void {
    if (this.#disposed) {
      throw new CapabilityError({
        capability: "session",
        message: `local workbook ${this.resourceId} has been disposed`
      })
    }
  }
}

/** The local (WASM) client. Workbooks are opened from bytes and own their session. */
export class LocalSpreadsheet extends GeneratedClientSurface {
  readonly #bindings: Promise<WasmBindings>
  #runtime: Promise<LocalRuntime> | undefined
  #canonical: CanonicalApi | undefined

  /** @internal */
  constructor(options: LocalSpreadsheetOptions) {
    super()
    if (!options || options.runtime === undefined) {
      throw new TypeError("createLocalSpreadsheet requires { runtime }")
    }
    this.#bindings = Promise.resolve(options.runtime)
  }

  /** The typed canonical escape hatch. Inputs carry their own `resource_id`. */
  get canonical(): CanonicalApi {
    if (!this.#canonical) this.#canonical = new CanonicalApi(lazyRuntime(() => this.#resolve()))
    return this.#canonical
  }

  /** Operation names the live WASM runtime advertises. */
  async capabilities(): Promise<readonly string[]> {
    const runtime = await this.#resolve()
    return [...(await runtime.operations())]
  }

  /** Open workbook bytes as a resident session. */
  async open(bytes: Uint8Array): Promise<LocalWorkbook> {
    if (!(bytes instanceof Uint8Array)) {
      throw new TypeError("local.open requires workbook bytes as a Uint8Array")
    }
    const runtime = await this.#resolve()
    let resourceId: string
    try {
      resourceId = await runtime.bindings.createSession(bytes)
    } catch (rejection) {
      throw decodeRejection(rejection, { operation: "create_session", runtime: "local" })
    }
    if (typeof resourceId !== "string" || resourceId.length === 0) {
      throw new TransportError("the local runtime returned an empty session id")
    }
    return new LocalWorkbook(runtime, resourceId)
  }

  protected override async executeClient<K extends OperationName>(
    operation: K,
    input: Record<string, unknown>
  ): Promise<OutputOf<K>> {
    return executeCanonical(await this.#resolve(), operation, input)
  }

  async #resolve(): Promise<LocalRuntime> {
    if (!this.#runtime) {
      this.#runtime = this.#bindings.then((bindings) => {
        assertBindings(bindings)
        return new LocalRuntime(bindings)
      })
    }
    return this.#runtime
  }
}

function lazyRuntime(resolve: () => Promise<CanonicalRuntime>): CanonicalRuntime {
  return {
    kind: "local",
    async operations() {
      return (await resolve()).operations()
    },
    async dispatch(operation, input) {
      return (await resolve()).dispatch(operation, input)
    },
    async artifactBytes(handle, resourceId) {
      return (await resolve()).artifactBytes(handle, resourceId)
    }
  }
}

/** Create a local spreadsheet client over the `agent-spreadsheet-wasm` bindings. */
export function createLocalSpreadsheet(options: LocalSpreadsheetOptions): LocalSpreadsheet {
  return new LocalSpreadsheet(options)
}
