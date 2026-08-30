# 48 — Canonical Operation Convergence

Status: complete for 0.14. The canonical registry/dispatcher, default MCP projection, native/WASM/SDK adapters, compatibility window, soundness fixes, and just-bash proof are landed.

Design: [`docs/architecture/canonical-operation-surface.md`](../../docs/architecture/canonical-operation-surface.md)

## Outcome

Replace surface-specific operation taxonomies with one canonical Rust operation registry and dispatcher. Project approximately 27 default operations (31 with optional capability groups) consistently through CLI, MCP, WASM, the JavaScript SDK, and the future just-bash adapter.

This is consolidation by durable agent intent, not a three-verb mega-tool design. `sheet_overview`, `inspect_cells`, searches, formula analysis, and table profiling remain distinct. Consolidation targets true overlap: `sheet_page` + `range_values`, write families, related resource CRUD, and duplicated summaries/layout reads.

## Workstreams

1. **Soundness prerequisite** — fix F1: evaluate before proof, report evaluation coverage, and distinguish clean from unevaluated.
2. **Canonical registry/dispatcher** — operation enum, descriptors, risk/capability metadata, schemas, and canonical JSON/error envelope.
3. **Read convergence** — introduce `read_cells`, merge workbook summaries/layout/style reads, and canonicalize search/formula names without changing semantic implementations.
4. **Write convergence** — promote `write` over `mutate_batch`, add `set_cells`, name/import ops, and route CLI writes through the same implementation.
5. **History/lifecycle convergence** — `get_changes`, `checkpoint`, `staged_change`, `export_fork`; remove cache administration from agent tools.
6. **Surface adapters** — `asp op`, canonical MCP router, one WASM dispatch binding, SDK `execute()` plus generated convenience methods.
7. **Compatibility and validation** — legacy wrappers/aliases, generated parity fixtures, SOL traces, Fable contract review, byte/turn measurements.
8. **just-bash proof** — only after convergence: one small custom-command adapter over `asp op`/SDK dispatch and just-bash `ctx.fs`.

## Release strategy

Land additively across small PRs. Keep old MCP tools in explicit compatibility mode and old CLI commands as aliases. Switch the slim default only after parity and trace gates pass. Remove compatibility wrappers only in a later semver-signaled release.

## Exit criteria

See the architecture design's acceptance gates. In particular: one implementation per operation, cross-surface golden parity, complete atomic/CAS `write` op coverage, evaluation soundness, approximately 27/31 MCP tools, and a just-bash adapter containing no spreadsheet behavior.
