function resolveVfsPath(ctx, path) {
  return ctx.fs.resolvePath(ctx.cwd, path)
}

async function readWorkbook(ctx, path, limit, flag) {
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
    throw Object.assign(new Error(`workbook exceeds the ${limit}-byte adapter limit`), {
      aspCode: "INVALID_REQUEST", aspPath: flag
    })
  }
  let bytes
  try {
    bytes = await ctx.fs.readFileBuffer(resolved)
  } catch (cause) {
    cause.aspPath = flag
    throw cause
  }
  if (bytes.byteLength > limit) {
    throw Object.assign(new Error(`workbook exceeds the ${limit}-byte adapter limit`), {
      aspCode: "INVALID_REQUEST", aspPath: flag
    })
  }
  return bytes
}

function createVfsWriter() {
  const locks = new Map()
  let tempSequence = 0

  async function withTargetLock(target, task) {
    const previous = locks.get(target) || Promise.resolve()
    let release
    const current = new Promise((resolve) => { release = resolve })
    locks.set(target, current)
    await previous
    try {
      return await task()
    } finally {
      release()
      if (locks.get(target) === current) locks.delete(target)
    }
  }

  async function atomicWrite(ctx, target, bytes, replace) {
    const resolved = resolveVfsPath(ctx, target)
    return withTargetLock(resolved, async () => {
      if (!replace && await ctx.fs.exists(resolved)) {
        throw Object.assign(new Error(`output path '${target}' already exists`), {
          aspCode: "INVALID_REQUEST", aspPath: "--output"
        })
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
    })
  }

  return { atomicWrite }
}

module.exports = { createVfsWriter, readWorkbook, resolveVfsPath }
