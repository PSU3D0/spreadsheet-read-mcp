const registry = require("./generated/canonical-registry.json")

const FLAG_NAMES = new Set(["--bind", "--baseline", "--json", "--output"])
const TEXT_ENCODER = new TextEncoder()

function jsonResult(value, exitCode = 0, stderr = false) {
  const text = `${JSON.stringify(value)}\n`
  return { stdout: stderr ? "" : text, stderr: stderr ? text : "", exitCode }
}

function descriptorFor(operation) {
  return registry.operations.find((descriptor) => descriptor.name === operation)
}

function canonicalError(code, message, operation, path) {
  const error = { code, message }
  if (operation !== undefined) error.operation = operation
  if (path !== undefined) error.path = path
  return { schema_version: "1", error }
}

function invalid(message, operation, path = "$.argv") {
  throw canonicalError("INVALID_REQUEST", message, operation, path)
}

function parseOperationArgs(args) {
  if (args[0] !== "op" || !args[1] || args[1].startsWith("--")) {
    invalid("usage: asp op <operation> [--bind PATH] [--baseline PATH] [--json JSON] [--output PATH|--in-place]")
  }
  const operation = args[1]
  const options = { operation, inPlace: false }
  for (let index = 2; index < args.length; index += 1) {
    const flag = args[index]
    if (flag === "--in-place") {
      if (options.inPlace) invalid("duplicate --in-place", operation)
      options.inPlace = true
      continue
    }
    if (!FLAG_NAMES.has(flag)) invalid(`unknown argument '${flag}'`, operation)
    const value = args[++index]
    if (value === undefined) invalid(`${flag} requires a value`, operation, flag)
    const key = flag.slice(2)
    if (options[key] !== undefined) invalid(`duplicate ${flag}`, operation, flag)
    options[key] = value
  }
  return options
}

function utf8Bytes(value) {
  return TEXT_ENCODER.encode(value).byteLength
}

function exampleFromSchema(schema, root = schema, name) {
  if (!schema || typeof schema !== "object") return null
  if (schema.$ref) {
    const target = schema.$ref.split("/").slice(1).reduce((value, part) => value?.[part], root)
    return exampleFromSchema(target, root, name)
  }
  if (schema.const !== undefined) return schema.const
  if (schema.default !== undefined && schema.default !== null) return schema.default
  if (Array.isArray(schema.enum)) return schema.enum[0]
  const variant = schema.oneOf?.[0] || schema.anyOf?.find((entry) => entry.type !== "null")
  if (variant) return exampleFromSchema(variant, root, name)
  if (schema.type === "object" || schema.properties) {
    const required = new Set(schema.required || [])
    return Object.fromEntries(Object.entries(schema.properties || {})
      .filter(([key, value]) => required.has(key) || value.const !== undefined)
      .map(([key, value]) => [key, exampleFromSchema(value, root, key)]))
  }
  if (schema.type === "array") return [exampleFromSchema(schema.items || {}, root)]
  if (schema.type === "integer" || schema.type === "number") return schema.minimum || 0
  if (schema.type === "boolean") return false
  if (schema.type === "null") return null
  if (name === "resource_id") return "session:session-example"
  if (name === "baseline_resource_id") return "session:baseline-example"
  return name || "value"
}

function projectAdapterSchema(descriptor) {
  const inputSchema = JSON.parse(JSON.stringify(descriptor.input_schema))
  const injected = ["resource_id"]
  if (descriptor.name === "verify_workbook") injected.push("baseline_resource_id")
  if (Array.isArray(inputSchema.required)) {
    inputSchema.required = inputSchema.required.filter((field) => !injected.includes(field))
  }
  for (const field of injected) {
    if (inputSchema.properties?.[field]) {
      const flag = field === "baseline_resource_id" ? "--baseline" : "--bind"
      inputSchema.properties[field].description =
        `Injected by the just-bash adapter from ${flag}; omit this field from operation JSON.`
    }
  }
  return inputSchema
}

function discover(args, availableOperations) {
  if (args.length === 1 && args[0] === "operations") {
    return registry.operations
      .filter(({ name, adapters }) => adapters.just_bash.support_status === "supported" &&
        availableOperations.has(name))
      .map(({ input_schema, output_schema, ...descriptor }) => ({ ...descriptor, available: true }))
  }
  if (args.length === 2 && ["schema", "example"].includes(args[0])) {
    const descriptor = descriptorFor(args[1])
    if (!descriptor) throw canonicalError(
      "UNKNOWN_OPERATION", `unknown operation '${args[1]}'`, args[1], "$.operation"
    )
    if (args[0] === "example") return exampleFromSchema(projectAdapterSchema(descriptor))
    return {
      schema_version: descriptor.schema_version,
      name: descriptor.name,
      input_schema: projectAdapterSchema(descriptor),
      output_schema: descriptor.output_schema,
      error_schema: registry.error_schema,
      adapter_binding: {
        bind: "--bind VFS_PATH injects resource_id",
        baseline: descriptor.name === "verify_workbook"
          ? "--baseline VFS_PATH injects baseline_resource_id"
          : undefined,
        persistence: "Mutating operations use --output VFS_PATH or --in-place."
      }
    }
  }
  return null
}

function errorEnvelope(error, operation) {
  if (error?.schema_version === "1" && error.error) return error
  if (error?.code === "UNSUPPORTED_CAPABILITY") {
    return canonicalError("CAPABILITY_UNAVAILABLE", error.message, operation, "adapter")
  }
  return canonicalError(
    error?.aspCode || (error?.aspPath?.startsWith("--") ? "RESOURCE_NOT_FOUND" : "OPERATION_FAILED"),
    error?.message || String(error),
    operation,
    error?.aspPath
  )
}

function assertLimit(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) throw new TypeError(`${name} must be a positive safe integer`)
}

module.exports = {
  assertLimit,
  canonicalError,
  descriptorFor,
  discover,
  errorEnvelope,
  invalid,
  jsonResult,
  parseOperationArgs,
  utf8Bytes
}
