#!/usr/bin/env node
// Generates the checked-in TypeScript artifacts derived from
// src/generated/canonical-registry.json:
//
//   src/generated/registry-data.ts  - runtime registry (JSON embedded as a string)
//   src/generated/operations.ts     - OperationName, InputOf, OutputOf, error envelope
//   src/generated/read-surface.ts   - static method declarations for the shared surfaces
//
// Run with --check to verify the checked-in files match (used by the drift test).

const fs = require("node:fs")
const path = require("node:path")
const { compile } = require("json-schema-to-typescript")

const packageRoot = path.resolve(__dirname, "..")
const generatedDir = path.join(packageRoot, "src", "generated")
const registryPath = path.join(generatedDir, "canonical-registry.json")

const HEADER = (source) =>
  `// GENERATED FILE - DO NOT EDIT.\n// Source: src/generated/canonical-registry.json\n// Regenerate: npm run generate:types (${source})\n\n`

const COMPILE_OPTIONS = {
  bannerComment: "",
  additionalProperties: false,
  declareExternallyReferenced: true,
  format: false,
  unknownAny: true
}

function camel(name) {
  return name.replace(/_([a-z0-9])/g, (_, letter) => letter.toUpperCase())
}

function pascal(name) {
  const value = camel(name)
  return value.charAt(0).toUpperCase() + value.slice(1)
}

function indent(text) {
  return text
    .split("\n")
    .map((line) => (line.trim().length === 0 ? line : `  ${line}`))
    .join("\n")
}

async function compileNamespace(namespaceName, schema, rootName) {
  const copy = structuredClone(schema)
  delete copy.title
  const body = await compile(copy, rootName, COMPILE_OPTIONS)
  return `export namespace ${namespaceName} {\n${indent(body.trimEnd())}\n}\n`
}

function docComment(text, pad = "") {
  const lines = String(text || "").split("\n")
  return `${pad}/**\n${lines.map((line) => `${pad} * ${line}`).join("\n")}\n${pad} */\n`
}

function registryDataSource(registry) {
  const embedded = JSON.stringify(JSON.stringify(registry))
  return `${HEADER("registry data")}export interface CanonicalAdapterPlan {
  binding_kind: string
  persistence: string
  support_status: string
}

export interface CanonicalOperationDescriptor {
  name: string
  description: string
  schema_version: string
  capability: { name: string; description: string }
  cost: { class: string; bounded_by: string[] }
  risk_ceiling?: string
  adapters: Record<string, CanonicalAdapterPlan>
  input_schema: Record<string, unknown>
  output_schema: Record<string, unknown>
}

export interface CanonicalRegistryDocument {
  schema_version: string
  error_schema: Record<string, unknown>
  operations: CanonicalOperationDescriptor[]
  generated_by?: string
}

export const canonicalRegistry: CanonicalRegistryDocument = JSON.parse(
  ${embedded}
) as CanonicalRegistryDocument
`
}

async function operationsSource(registry) {
  const chunks = []
  for (const descriptor of registry.operations) {
    chunks.push(await compileNamespace(`In${pascal(descriptor.name)}`, descriptor.input_schema, "Input"))
    chunks.push(await compileNamespace(`Out${pascal(descriptor.name)}`, descriptor.output_schema, "Output"))
  }
  const errorNamespace = await compileNamespace("CanonicalErrors", registry.error_schema, "Envelope")

  const inputEntries = registry.operations
    .map((d) => `  ${d.name}: In${pascal(d.name)}.Input`)
    .join("\n")
  const outputEntries = registry.operations
    .map((d) => `  ${d.name}: Out${pascal(d.name)}.Output`)
    .join("\n")

  return `${HEADER("operation types")}/* eslint-disable @typescript-eslint/no-namespace */
${chunks.join("\n")}
${errorNamespace}
/** The canonical error envelope returned by every adapter. */
export type CanonicalErrorEnvelope = CanonicalErrors.Envelope

/** The canonical error code set. */
export type CanonicalErrorCode = CanonicalErrors.CanonicalErrorCode

/** Canonical input object keyed by operation name. */
export interface OperationInputs {
${inputEntries}
}

/** Canonical response envelope keyed by operation name. */
export interface OperationOutputs {
${outputEntries}
}

/** Every operation registered by the canonical dispatcher. */
export type OperationName = keyof OperationInputs

/** The canonical input object for \`K\`. */
export type InputOf<K extends OperationName> = OperationInputs[K]

/** The canonical response envelope for \`K\`. */
export type OutputOf<K extends OperationName> = OperationOutputs[K]
`
}

function surfaceMethod(descriptor, injected) {
  const required = (descriptor.input_schema.required || []).filter((field) => !injected.includes(field))
  const optional = required.length === 0 && !descriptor.input_schema.oneOf
  const method = camel(descriptor.name)
  const bound = `BoundInput<"${descriptor.name}">`
  const parameter = optional
    ? `input: ${bound} = {} as ${bound}`
    : `input: ${bound}`
  return `${docComment(descriptor.description, "  ")}  ${method}(${parameter}): Promise<OutputOf<"${descriptor.name}">> {
    return this.executeBound("${descriptor.name}", input as Record<string, unknown>)
  }
`
}

function clientMethod(descriptor) {
  const required = descriptor.input_schema.required || []
  const method = camel(descriptor.name)
  const type = `InputOf<"${descriptor.name}">`
  const parameter = required.length === 0
    ? `input: ${type} = {} as ${type}`
    : `input: ${type}`
  return `${docComment(descriptor.description, "  ")}  ${method}(${parameter}): Promise<OutputOf<"${descriptor.name}">> {
    return this.executeClient("${descriptor.name}", input as Record<string, unknown>)
  }
`
}

function readSurfaceSource(registry) {
  const reads = registry.operations.filter((d) => d.adapters.mcp.binding_kind === "single_read")
  const clientLevel = registry.operations.filter((d) => d.adapters.mcp.binding_kind === "none")

  return `${HEADER("read surface")}import type { InputOf, OperationName, OutputOf } from "./operations.js"

/** Distributive \`Omit\` so union-shaped canonical inputs keep every branch. */
export type OmitResource<T> = T extends unknown ? Omit<T, "resource_id"> : never

/** The canonical input for \`K\` minus the resource id, which the object injects. */
export type BoundInput<K extends OperationName> = OmitResource<InputOf<K>>

/** Operation names in the shared read surface. */
export const READ_SURFACE_OPERATIONS = [
${reads.map((d) => `  "${d.name}"`).join(",\n")}
] as const

/** Operation names available on a client without a bound resource. */
export const CLIENT_SURFACE_OPERATIONS = [
${clientLevel.map((d) => `  "${d.name}"`).join(",\n")}
] as const

export type ReadSurfaceOperation = (typeof READ_SURFACE_OPERATIONS)[number]
export type ClientSurfaceOperation = (typeof CLIENT_SURFACE_OPERATIONS)[number]

/**
 * The generated read surface shared by every workbook-shaped object.
 * Methods are declared statically so editors resolve them without \`defineProperty\`.
 */
export abstract class GeneratedWorkbookView {
  /** Execute \`operation\` against this object's bound resource. */
  protected abstract executeBound<K extends OperationName>(
    operation: K,
    input: Record<string, unknown>
  ): Promise<OutputOf<K>>

${reads.map((d) => surfaceMethod(d, ["resource_id"])).join("\n")}}

/** The generated client-level surface (operations that take no bound resource). */
export abstract class GeneratedClientSurface {
  /** Execute a resource-free \`operation\`. */
  protected abstract executeClient<K extends OperationName>(
    operation: K,
    input: Record<string, unknown>
  ): Promise<OutputOf<K>>

${clientLevel.map((d) => clientMethod(d)).join("\n")}}
`
}

async function main() {
  const registry = JSON.parse(fs.readFileSync(registryPath, "utf8"))
  const files = {
    "registry-data.ts": registryDataSource(registry),
    "operations.ts": await operationsSource(registry),
    "read-surface.ts": readSurfaceSource(registry)
  }

  const check = process.argv.includes("--check")
  const drift = []
  for (const [name, contents] of Object.entries(files)) {
    const target = path.join(generatedDir, name)
    if (check) {
      const current = fs.existsSync(target) ? fs.readFileSync(target, "utf8") : null
      if (current !== contents) drift.push(name)
      continue
    }
    fs.writeFileSync(target, contents)
    console.log(`wrote src/generated/${name}`)
  }
  if (check && drift.length > 0) {
    console.error(`generated files are stale: ${drift.join(", ")}`)
    process.exit(1)
  }
  if (check) console.log("generated files are up to date")
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
