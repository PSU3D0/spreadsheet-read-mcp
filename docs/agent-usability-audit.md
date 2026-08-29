# Agent-Usability Audit — Applied & Deferred Findings

Date: 2026-08-25. Method: five GPT-5.6-sol agents executed real spreadsheet tasks against `asp` (CLI) and `spreadsheet-mcp` (MCP stdio) while logging full traces; two Claude Fable 5 auditors performed read-only judgment reviews of the traces and tool schemas. Traces live in the platform-dev workspace under `scratch/agent-audit/traces/`.

Companion bump: formualizer `0.5.1 → 0.8.4`, `formualizer-parse `1.1.2 → 3.0.0` (same release).

## Applied in this release

### CLI
| Finding | Severity | Fix |
|---|---|---|
| F2: panic (exit 101, umya unwrap) on out-of-range address (`ZZZZ999999=1`) | blocker | A1 bounds validation (A–XFD / 1–1048576) at edit entry points; structured `INVALID_CELL_REFERENCE` error |
| F3: silent exit-0 success on unparseable range (`read values Q1 Total_Q1` returned no values) | blocker | `range_values` fails loudly with `INVALID_RANGE` + named-range guidance; old silent-shape test replaced with loud-failure contract test |
| F11: noise warnings on documented syntax (`WARN_SHORTHAND_EDIT`, `WARN_FORMULA_PREFIX`) training agents to ignore warnings | minor | Removed from shorthand and object-edit paths |
| F6: `verify --help` flattened onto proof, hiding `diff` | major | Proof help now advertises `verify diff` (bare-`verify` = proof remains legacy compat) |

### MCP
| Finding | Severity | Fix |
|---|---|---|
| M-schema: output schemas were ~57% of `tools/list` payload (164KB → descriptions starved at ~86 bytes/tool) | major | `SlimToolRouter` strips `outputSchema`; handshake payload **164KB → 71KB**; responses unchanged (`structuredContent` intact) |
| M-targets: `verify_workbook` rejected bare targets despite schema inviting `sheet_name` + bare address | major | Bare targets are auto-qualified server-side when `sheet_name` is provided |
| M-recalc: partial failure rendered as clean success (`isError:false` + buried `eval_errors`) | minor | Added top-level `status: "completed"|"completed_with_errors"` and `error_count` |

### Evaluation soundness
| Finding | Severity | Fix |
|---|---|---|
| F1: stale or missing formula caches could authorize false-clean verification | blocker | Recalculation now returns canonical state and revision-bound coverage; CLI and MCP verification evaluate both sides with Formualizer before assigning proof status; value reads disclose conservative calculation state |
| F1: formula ratio used cached values as its denominator and could exceed 100% or call formula-only sheets empty | blocker | Formula cells now count as occupied regardless of cache; `formula_ratio` is bounded structural `formula_cells / occupied_cells` |
| F7: spreadsheet errors were exposed as `kind:"Text"` | major | OOXML and Formualizer error tokens, including `#CIRC!`, are emitted as `kind:"Error"` and participate in verification provenance |

## Deferred (do next, ranked)

1. **F4/F10 (major): analysis surfaces fabricate.** `analyze sheet-statistics` and `table-profile` use a buggier region/header pipeline than `read table` and produced materially wrong profiles on two fixtures; three independent header heuristics disagree. Rebase on the working region pipeline or retire/stamp experimental.
2. **M-consolidation (major): nine write tools → one previewable `mutate_batch`.** Collapses ~40KB duplicate schema and the write-side selection failure (agents pick `edit_batch` and lose staging). Add `mode` to `edit_batch` as interim step.
3. **M-health (major): add `check_health`** running fork→recalculate→compare→discard internally; rename `verify_workbook` toward `compare_workbooks` semantics.
4. **F8:** dry-run contradicts apply on `recalc_needed`. Run same dependency check both paths.
5. **F5:** `clone-template-row` rename `confidence` → `structural_confidence`; emit `semantic_followups` for downstream formulas referencing the template row without spanning the insert.
6. **F9:** expose stored cell type (`stored_kind`) alongside coerced value — text-typed numbers are a top real-world defect and currently undetectable.
7. **Naming pass:** `save_fork` → `export_fork` (it writes files!), `checkpoint_fork` → `create_checkpoint`, `rules_batch` fold into unified batch, `formula_trace` cross-sheet ≠ `external`.
8. **State-machine self-documentation:** preview responses should say `ops_staged` not `ops_applied`; restore_checkpoint should return effect counts.

## Verdict quote (Fable, CLI audit)

> "The CLI's mechanics (read, write, recalc, diff) are already agent-grade. Its *claims* — confidence, cleanliness, statistics, headers — are not. Fix the claims before adding any new surface."
