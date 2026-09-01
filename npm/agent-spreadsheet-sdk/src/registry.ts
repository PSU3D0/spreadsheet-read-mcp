import {
  canonicalRegistry,
  type CanonicalOperationDescriptor,
  type CanonicalRegistryDocument
} from "./generated/registry-data.js"
import type { OperationName } from "./generated/operations.js"

export { canonicalRegistry }
export type { CanonicalOperationDescriptor, CanonicalRegistryDocument }

/** Canonical protocol schema version this SDK was generated against. */
export const CANONICAL_SCHEMA_VERSION: string = canonicalRegistry.schema_version

/** Every operation name in the checked-in registry. */
export const OPERATION_NAMES: readonly OperationName[] = Object.freeze(
  canonicalRegistry.operations.map((descriptor) => descriptor.name as OperationName)
)

const OPERATION_SET = new Set<string>(OPERATION_NAMES)

const DESCRIPTORS = new Map<string, CanonicalOperationDescriptor>(
  canonicalRegistry.operations.map((descriptor) => [descriptor.name, descriptor])
)

/** True when `value` is a registered canonical operation name. */
export function isOperationName(value: unknown): value is OperationName {
  return typeof value === "string" && OPERATION_SET.has(value)
}

/** The registry descriptor for `operation`, or `undefined`. */
export function descriptorFor(operation: string): CanonicalOperationDescriptor | undefined {
  return DESCRIPTORS.get(operation)
}

/** Operation names an adapter registers as supported in the checked-in registry. */
export function operationsForAdapter(adapter: string): readonly OperationName[] {
  return Object.freeze(
    canonicalRegistry.operations
      .filter((descriptor) => descriptor.adapters[adapter]?.support_status === "supported")
      .map((descriptor) => descriptor.name as OperationName)
  )
}

/**
 * Normalize a live operation list into canonical operation names.
 *
 * Accepts the WASM binding's JSON string, an array of descriptors (the shape
 * `GET /v1/operations` returns), or a bare array of names. Names outside the
 * checked-in registry are dropped: the SDK never advertises what it cannot type.
 */
export function normalizeOperationList(value: unknown): Set<string> {
  let entries: unknown = value
  if (typeof entries === "string") {
    try {
      entries = JSON.parse(entries)
    } catch {
      throw new TypeError("operation discovery returned invalid JSON")
    }
  }
  const container = entries as { operations?: unknown; tools?: unknown } | unknown[]
  if (!Array.isArray(container)) {
    const nested = (container as { operations?: unknown; tools?: unknown })?.operations ??
      (container as { tools?: unknown })?.tools
    if (!Array.isArray(nested)) {
      throw new TypeError("operation discovery must return an array of names or descriptors")
    }
    entries = nested
  }

  const names = (entries as unknown[])
    // A runtime-filtered discovery may carry entries it does not actually serve.
    .filter((entry) => typeof entry === "string" || (entry as { available?: unknown })?.available !== false)
    .map((entry) => (typeof entry === "string" ? entry : (entry as { name?: unknown })?.name))
  if (names.some((name) => typeof name !== "string")) {
    throw new TypeError("operation discovery contains a descriptor without a name")
  }
  return new Set((names as string[]).filter((name) => OPERATION_SET.has(name)))
}
