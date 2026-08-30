let tempSequence = 0

function resolveVfsPath(ctx, path) {
  return ctx.fs.resolvePath(ctx.cwd, path)
}

async function readWorkbook(ctx, path, limit, operation, flag) {
  const resolved = resolveVfsPath(ctx, path)
  let stat
  try {
    stat = await ctx.fs.stat(resolved)
  } catch (cause) {
    cause.aspPath = flag
    throw cause
  }
  if (!stat.isFile) throw Object.assign(new Error(`'${path}' is not a file`), { aspPath: flag })
  if (stat.size > limit) {
    const error = new Error(`workbook exceeds the ${limit}-byte adapter limit`)
    error.aspCode = "INVALID_REQUEST"
    error.aspPath = flag
    throw error
  }
  let bytes
  try {
    bytes = await ctx.fs.readFileBuffer(resolved)
  } catch (cause) {
    cause.aspPath = flag
    throw cause
  }
  if (bytes.byteLength > limit) {
    const error = new Error(`workbook exceeds the ${limit}-byte adapter limit`)
    error.aspCode = "INVALID_REQUEST"
    error.aspPath = flag
    throw error
  }
  return { bytes, path: resolved }
}

async function atomicWrite(ctx, target, bytes, replace) {
  const resolved = resolveVfsPath(ctx, target)
  if (!replace && await ctx.fs.exists(resolved)) {
    const error = new Error(`output path '${target}' already exists`)
    error.aspCode = "INVALID_REQUEST"
    error.aspPath = "--output"
    throw error
  }
  let temporary
  do {
    temporary = `${resolved}.asp-tmp-${++tempSequence}`
  } while (await ctx.fs.exists(temporary))
  try {
    await ctx.fs.writeFile(temporary, bytes)
    await ctx.fs.mv(temporary, resolved)
  } catch (cause) {
    try { await ctx.fs.rm(temporary, { force: true }) } catch (_) {}
    cause.aspCode = "OPERATION_FAILED"
    cause.aspPath = "adapter_export"
    throw cause
  }
}

module.exports = { resolveVfsPath, readWorkbook, atomicWrite }
