import { CapabilityError } from "./errors.js"
import type { InputOf, OperationName, OutputOf } from "./generated/operations.js"
import { GeneratedWorkbookView } from "./generated/read-surface.js"
import type {
  RenderPngLevel,
  RenderSheetInput,
  RenderedSheet,
  RenderWarning
} from "./render.js"
import { type CanonicalRuntime, executeCanonical } from "./runtime.js"

export type { BoundInput } from "./generated/read-surface.js"

/** Canonical `write` input with the resource id injected and the revision defaulted. */
export type WriteInput =
  Omit<InputOf<"write">, "resource_id" | "expected_revision"> & { expected_revision?: string }

/** Canonical `recalculate` input with the resource id injected and the revision defaulted. */
export type RecalculateInput =
  Omit<InputOf<"recalculate">, "resource_id" | "expected_revision"> & { expected_revision?: string }

/** Canonical `verify_workbook` input; both resource ids are injected. */
export type VerifyInput =
  Omit<InputOf<"verify_workbook">, "resource_id" | "baseline_resource_id">

interface ScreenshotData {
  sheet_name: string
  range: string
  artifact: { handle: string; hash: string; bytes: number; media_type: string }
  fidelity?: unknown
  warnings?: unknown
  calculation?: unknown
  renderer?: unknown
  width?: unknown
  height?: unknown
  png_level?: unknown
}

/**
 * A canonical resource bound to a runtime.
 *
 * The handle owns its `resource_id` and the `revision_id` it last saw, so callers
 * never thread ids or revisions through call sites.
 */
export abstract class WorkbookHandle extends GeneratedWorkbookView {
  protected readonly runtime: CanonicalRuntime
  /** The opaque canonical resource id this handle is bound to. */
  readonly resourceId: string
  protected trackedRevision: string | undefined

  protected constructor(runtime: CanonicalRuntime, resourceId: string, revisionId?: string) {
    super()
    this.runtime = runtime
    this.resourceId = resourceId
    this.trackedRevision = revisionId
  }

  /** The most recent `revision_id` seen on an envelope for this resource. */
  get revisionId(): string | undefined {
    return this.trackedRevision
  }

  /** Record `resource_id` and `revision_id` from any canonical envelope. */
  protected track(response: unknown): void {
    const envelope = response as { resource_id?: unknown; revision_id?: unknown }
    if (!envelope || typeof envelope !== "object") return
    if (envelope.resource_id !== undefined && envelope.resource_id !== this.resourceId) return
    if (typeof envelope.revision_id === "string") this.trackedRevision = envelope.revision_id
  }

  protected override async executeBound<K extends OperationName>(
    operation: K,
    input: Record<string, unknown>
  ): Promise<OutputOf<K>> {
    const response = await executeCanonical(this.runtime, operation, {
      ...input,
      resource_id: this.resourceId
    })
    this.track(response)
    return response
  }

  /** The tracked revision, fetching one with `describe_workbook` when nothing is tracked. */
  protected async currentRevision(): Promise<string> {
    if (this.trackedRevision === undefined) await this.describeWorkbook()
    if (this.trackedRevision === undefined) {
      throw new CapabilityError({
        capability: "revision_tracking",
        message: `no revision is available for ${this.resourceId}`
      })
    }
    return this.trackedRevision
  }

  /**
   * Render one sheet to PNG bytes.
   *
   * Image bytes cross the adapter boundary rather than a canonical operation: the
   * server runtime fetches `GET /v1/artifacts/{handle}` and the local runtime uses the
   * WASM byte binding.
   */
  async renderSheet(input: RenderSheetInput): Promise<RenderedSheet> {
    const request: Record<string, unknown> = { sheet_name: input.sheet_name }
    if (input.range !== undefined) request["range"] = input.range
    if (input.png_level !== undefined) request["png_level"] = input.png_level
    const response = await this.executeBound("screenshot_sheet", request)
    const data = (response as { data: ScreenshotData }).data
    const png = await this.runtime.artifactBytes(data.artifact.handle, this.resourceId)
    // The bytes have crossed, so nothing needs the slot any more. Releasing here
    // is what keeps a render loop inside one session from evicting its own
    // earlier artifacts.
    if (this.runtime.releaseArtifact) {
      await this.runtime.releaseArtifact(data.artifact.handle, this.resourceId)
    }
    return {
      png,
      fidelity: typeof data.fidelity === "string" ? data.fidelity : "unknown",
      warnings: Array.isArray(data.warnings) ? (data.warnings as RenderWarning[]) : [],
      calculation: data.calculation ?? null,
      sheet_name: data.sheet_name,
      range: data.range,
      handle: data.artifact.handle,
      renderer: typeof data.renderer === "string" ? data.renderer : undefined,
      width: typeof data.width === "number" ? data.width : undefined,
      height: typeof data.height === "number" ? data.height : undefined,
      png_level: typeof data.png_level === "string"
        ? (data.png_level as RenderPngLevel)
        : undefined
    }
  }
}

/** A handle that can also mutate its resource. */
export abstract class MutableWorkbookHandle extends WorkbookHandle {
  /** Apply, preview, or stage a canonical write bundle. */
  async write(input: WriteInput): Promise<OutputOf<"write">> {
    const expected = input.expected_revision ?? (await this.currentRevision())
    return this.executeBound("write", { ...input, expected_revision: expected })
  }

  /** Recalculate the resource in place. */
  async recalculate(input: RecalculateInput = {}): Promise<OutputOf<"recalculate">> {
    const expected = input.expected_revision ?? (await this.currentRevision())
    return this.executeBound("recalculate", { ...input, expected_revision: expected })
  }

  protected verifyAgainstResource(
    baselineResourceId: string,
    input: VerifyInput = {}
  ): Promise<OutputOf<"verify_workbook">> {
    return this.executeBound("verify_workbook", {
      ...input,
      baseline_resource_id: baselineResourceId
    })
  }
}
