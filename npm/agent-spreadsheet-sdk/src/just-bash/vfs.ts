type Json = any

export function resolveVfsPath(ctx: Json, path: string): string {
  return ctx.fs.resolvePath(ctx.cwd, path)
}

export async function readWorkbook(ctx: Json, path: string, limit: number, flag: string): Promise<Uint8Array> {
  const resolved = resolveVfsPath(ctx, path)
  let stat: Json
  try {
    stat = await ctx.fs.stat(resolved)
  } catch (cause: Json) {
    cause.aspPath = flag
    throw cause
  }
  if (!stat.isFile) throw Object.assign(new Error(`'${path}' is not a file`), { aspPath: flag })
  if (stat.size > limit) {
    throw Object.assign(new Error(`workbook exceeds the ${limit}-byte adapter limit`), {
      aspCode: "INVALID_REQUEST", aspPath: flag
    })
  }
  let bytes: Json
  try {
    bytes = await ctx.fs.readFileBuffer(resolved)
  } catch (cause: Json) {
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

export function createVfsWriter(): { atomicWrite: (ctx: Json, target: string, bytes: Uint8Array, replace: boolean) => Promise<void> } {
  const locks = new Map<string, Promise<void>>()
  let tempSequence = 0

  async function withTargetLock(target: string, task: () => Promise<void>): Promise<void> {
    const previous = locks.get(target) || Promise.resolve()
    let release!: () => void
    const current = new Promise<void>((resolve) => { release = resolve })
    locks.set(target, current)
    await previous
    try {
      return await task()
    } finally {
      release()
      if (locks.get(target) === current) locks.delete(target)
    }
  }

  async function atomicWrite(ctx: Json, target: string, bytes: Uint8Array, replace: boolean): Promise<void> {
    const resolved = resolveVfsPath(ctx, target)
    return withTargetLock(resolved, async () => {
      if (!replace && await ctx.fs.exists(resolved)) {
        throw Object.assign(new Error(`output path '${target}' already exists`), {
          aspCode: "INVALID_REQUEST", aspPath: "--output"
        })
      }
      let temporary: string
      do {
        temporary = `${resolved}.asp-tmp-${++tempSequence}`
      } while (await ctx.fs.exists(temporary))
      try {
        await ctx.fs.writeFile(temporary, bytes)
        await ctx.fs.mv(temporary, resolved)
      } catch (cause: Json) {
        try { await ctx.fs.rm(temporary, { force: true }) } catch { /* best effort */ }
        cause.aspCode = "OPERATION_FAILED"
        cause.aspPath = "adapter_export"
        throw cause
      }
    })
  }

  return { atomicWrite }
}
