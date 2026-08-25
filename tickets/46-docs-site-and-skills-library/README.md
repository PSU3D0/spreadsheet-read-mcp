# 46 — Docs Site & Agent Skills Library

Status: planned. Follows the 0.13 surface-tightening pass (mutate_batch consolidation, slim surface, README refinement).

## Context

agent-spreadsheet now has: a canonical name, published packages (crates.io + npm), a validated MCP registry entry, an audit-hardened tool surface, and cross-linking with the Formualizer engine README. The remaining distribution gaps are (1) a proper docs home and (2) a first-class skills library for agent harnesses.

## Part A — Docs site

Options, in preference order:

1. **Section on formualizer.dev** (`formualizer.dev/agent-spreadsheet` or `/docs/agent-tools`) — one docs infrastructure, engine + tools share the ecosystem narrative, SEO authority pools on one domain. The formualizer docs-site (Next.js) already exists in the engine repo.
2. Standalone site (`agent-spreadsheet.dev`) — only if the tool layer needs an independent brand motion later.

Content inventory (most exists in README/docs, needs restructuring):
- Quickstarts per surface: CLI (agent tool-call + CI recipes), MCP (Claude Desktop/Cursor/generic client configs), SDK (server + embedded WASM backends)
- The core loop as a guide: orient → read → fork/edit → recalculate → verify/diff
- Tool reference generated from `asp schema` / MCP tool list (single source of truth, no drift)
- The audit story as content marketing: "we ran adversarial agent traces against our own tools" writes itself as a launch post and doubles as the citable benchmark piece (competitive-landscape.md: "openpyxl can't calculate; even Anthropic's xlsx skill shells out to LibreOffice")

## Part B — Agent skills library

Seed exists in `skills/` (EXPLORE, SAFE_EDITING, CLI_BATCH_WRITE). Formalize into an installable library:

1. **Canonical skill set** (target 5-6):
   - `spreadsheet-explore` — discovery workflow (exists, refresh for 0.13 surface)
   - `spreadsheet-safe-edit` — fork/preview/recalc/verify loop (merge SAFE_EDITING + CLI_BATCH_WRITE, teach `mutate_batch`)
   - `spreadsheet-verify` — proof/diff workflows; the "prove what changed" narrative
   - `spreadsheet-model-audit` — find errors, trace dependencies, check model health (waits on deferred audit F1: verify must evaluate before this skill can promise soundness)
   - `spreadsheet-sheetport` — manifest lifecycle for typed spreadsheet APIs
2. **Format**: follow the emerging agent-skills conventions (SKILL.md + frontmatter) so the same files work in Claude Code, Pi, and other harnesses. Keep bodies tool-version-pinned and test them: a CI job that runs each skill's commands against fixtures (skills are docs that can rot — treat them as tests).
3. **Distribution**: `skills/` dir in-repo (installable via git), plus listing in skill registries as they stabilize. The npm CLI package could ship them (`agent-spreadsheet skills install`) — decide after format settles.

## Sequencing

1. Skills refresh for 0.13 surface (cheap, immediate — do with or right after mutate_batch lands)
2. Audit-story launch post (needs deferred F1 fixed first to be fully honest about verify)
3. Docs-site section on formualizer.dev
4. Skill registry listings + CI skill-testing job

## Dependencies
- Deferred audit finding F1 (verify/overview must evaluate in memory) gates the model-audit skill and the launch post's strongest claim.
- `mutate_batch` (ticket in 0.13 pass) changes what the safe-edit skill teaches.
