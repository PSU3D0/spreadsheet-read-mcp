# Surface Capability Matrix (CLI / MCP / WASM / SDK)

Status: active migration baseline (canonical registry Wave 3A additive read surface)
Owner: Tranche 35 (tickets/35-js-surface-migration)

This matrix is the planning baseline for cross-surface migration.

## Legend

- **Classification**
  - `ALL` = intended shared capability (CLI + MCP + WASM via shared core)
  - `CLI_ONLY` = host/operator concern (no MCP/WASM parity required)
  - `MCP_ONLY` = agent/session orchestration concern
  - `SHARED_PARTIAL` = shared semantics, but currently only implemented on subset
- **WASM target**
  - `mvp` = planned for initial WASM surface
  - `later` = planned after MVP
  - `n/a` = intentionally not a WASM concern

Boundary contract: `docs/architecture/surface-boundary-rules.md`

---

## A) CLI command catalog

| CLI command/subcommand | MCP equivalent tool(s) | Classification | Core projection target | WASM target | Notes | Implementation module path | Parity test owner |
|---|---|---:|---|---:|---|---|---|
| `read sheets` | `list_sheets` | ALL | `core.read.list_sheets` | mvp | Shared read primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::list_sheets` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `read overview` | `sheet_overview` | ALL | `core.read.sheet_overview` | mvp | Shared read primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::sheet_overview` | `crates/agent-spreadsheet/tests/sheet_overview_truncation.rs` |
| `read values` | `range_values` | ALL | `core.read.range_values` | mvp | Shared read primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::range_values` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `read export --format json/csv` | `range_values` | ALL | `core.read.range_values` + formatter | mvp | CSV serialization shared; CLI handles output path/stdout | `crates/agent-spreadsheet/src/cli/commands/read.rs::range_export` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `read export --format grid` | `grid_export` | ALL | `core.read.grid_export` | mvp | Rich payload export | `crates/agent-spreadsheet/src/cli/commands/read.rs::range_export` | `crates/agent-spreadsheet/tests/unit_grid_roundtrip.rs` |
| `write import --from-grid` | `grid_import` | ALL | `core.write.grid_import` | mvp | Shared grid import semantics | `crates/agent-spreadsheet/src/cli/commands/write.rs::range_import` | `crates/agent-spreadsheet/tests/unit_grid_roundtrip.rs` |
| `write import --from-csv` | _(none today)_ | SHARED_PARTIAL | `core.write.csv_import` | mvp | CLI has path; MCP may add later | `crates/agent-spreadsheet/src/cli/commands/write.rs::range_import` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `read cells` | `inspect_cells` | ALL | `core.read.inspect_cells` | mvp | Strict detail-view: up to 25 cells with full metadata; returns budget object | `crates/agent-spreadsheet/src/cli/commands/read.rs::inspect_cells` | `crates/agent-spreadsheet/tests/read_guardrails.rs` |
| `read page` | `sheet_page` | ALL | `core.read.sheet_page` | mvp | Shared pagination contract | `crates/agent-spreadsheet/src/cli/commands/read.rs::sheet_page` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `read table` | `read_table` | ALL | `core.read.read_table` | mvp | Shared table read primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::read_table` | `crates/agent-spreadsheet/tests/read_table_polish.rs` |
| `analyze find-value` | `find_value` | ALL | `core.analysis.find_value` | mvp | Shared analysis primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::find_value` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `read names` | `named_ranges` | ALL | `core.read.named_ranges` | mvp | Shared read primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::named_ranges` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `write name define` | `define_name` | ALL | `core.write.define_name` | mvp | Named range CRUD (create) | `crates/agent-spreadsheet/src/cli/commands/write.rs::define_name` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `write name update` | `update_name` | ALL | `core.write.update_name` | mvp | Named range CRUD (update) | `crates/agent-spreadsheet/src/cli/commands/write.rs::update_name` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `write name delete` | `delete_name` | ALL | `core.write.delete_name` | mvp | Named range CRUD (delete) | `crates/agent-spreadsheet/src/cli/commands/write.rs::delete_name` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `analyze find-formula` | `find_formula` | ALL | `core.analysis.find_formula` | mvp | Shared analysis primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::find_formula` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `analyze scan-volatiles` | `scan_volatiles` | ALL | `core.analysis.scan_volatiles` | mvp | Shared analysis primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::scan_volatiles` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `analyze sheet-statistics` | `sheet_statistics` | ALL | `core.analysis.sheet_statistics` | mvp | Shared analysis primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::sheet_statistics` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `analyze formula-map` | `sheet_formula_map` | ALL | `core.analysis.sheet_formula_map` | mvp | Shared analysis primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::formula_map` | `crates/agent-spreadsheet/tests/heuristic_scenarios.rs` |
| `analyze formula-trace` | `formula_trace` | ALL | `core.analysis.formula_trace` | later | Shared but heavier graph concerns | `crates/agent-spreadsheet/src/cli/commands/read.rs::formula_trace` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `read workbook` | `describe_workbook` | ALL | `core.read.describe_workbook` | mvp | Contract naming differs by surface | `crates/agent-spreadsheet/src/cli/commands/read.rs::describe` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `analyze table-profile` | `table_profile` | ALL | `core.analysis.table_profile` | mvp | Shared profiling primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::table_profile` | `crates/agent-spreadsheet/tests/read_table_polish.rs` |
| `read layout` | `layout_page` | ALL | `core.read.layout_page` | mvp | Shared layout primitive | `crates/agent-spreadsheet/src/cli/commands/read.rs::layout_page` | `crates/agent-spreadsheet/tests/unit_layout_page.rs` |
| `workbook create` | _(none today)_ | SHARED_PARTIAL | `core.write.create_workbook_bytes` (planned) | later | CLI path-based today | `crates/agent-spreadsheet/src/cli/commands/write.rs::create_workbook` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `workbook copy` | _(none today)_ | CLI_ONLY | `adapter-cli.copy_path` | n/a | Stateless file orchestration | `crates/agent-spreadsheet/src/cli/commands/write.rs::copy` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `write cells` | `edit_batch` | ALL | `core.write.edit_batch` | mvp | CLI shorthand parsing is adapter concern | `crates/agent-spreadsheet/src/cli/commands/write.rs::edit` | `crates/agent-spreadsheet/tests/unit_edit_batch.rs` |
| `write batch transform` | `transform_batch` | ALL | `core.write.transform_batch` | mvp | Shared write primitive | `crates/agent-spreadsheet/src/cli/commands/write.rs::transform_batch` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `write batch style` | `style_batch` | ALL | `core.write.style_batch` | mvp | Shared write primitive | `crates/agent-spreadsheet/src/cli/commands/write.rs::style_batch` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `write batch formula-pattern` | `apply_formula_pattern` | ALL | `core.write.apply_formula_pattern` | later | Shared write primitive | `crates/agent-spreadsheet/src/cli/commands/write.rs::apply_formula_pattern` | `crates/agent-spreadsheet/tests/unit_formula_pattern.rs` |
| `write batch structure` | `structure_batch` | ALL | `core.write.structure_batch` | later | Shared write primitive | `crates/agent-spreadsheet/src/cli/commands/write.rs::structure_batch` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `write batch column-size` | `column_size_batch` | ALL | `core.write.column_size_batch` | later | Shared write primitive | `crates/agent-spreadsheet/src/cli/commands/write.rs::column_size_batch` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `write batch sheet-layout` | `sheet_layout_batch` | ALL | `core.write.sheet_layout_batch` | later | Shared write primitive | `crates/agent-spreadsheet/src/cli/commands/write.rs::sheet_layout_batch` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `write batch rules` | `rules_batch` | ALL | `core.write.rules_batch` | later | Shared write primitive | `crates/agent-spreadsheet/src/cli/commands/write.rs::rules_batch` | `crates/agent-spreadsheet/tests/core_runtime_parity.rs` |
| `write formulas replace` | `replace_in_formulas` | ALL | `core.write.replace_in_formulas` | later | Formula-only find/replace with dry-run | `crates/agent-spreadsheet/src/cli/commands/write.rs::replace_in_formulas` | `crates/agent-spreadsheet/tests/unit_replace_in_formulas.rs` |
| `sheetport manifest candidates` | `get_manifest_stub` | SHARED_PARTIAL | `core.sheetport.manifest_stub` | later | Naming differs | `crates/agent-spreadsheet/src/cli/commands/read.rs::sheetport_manifest_candidates` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `sheetport manifest schema` | _(none today)_ | CLI_ONLY | `adapter-cli.sheetport_schema` | n/a | Local schema print UX | `crates/agent-spreadsheet/src/cli/commands/read.rs::sheetport_manifest_schema` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `sheetport manifest validate` | _(none today)_ | CLI_ONLY | `adapter-cli.sheetport_validate_yaml` | n/a | Local manifest file validation | `crates/agent-spreadsheet/src/cli/commands/read.rs::sheetport_manifest_validate` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `sheetport manifest normalize` | _(none today)_ | CLI_ONLY | `adapter-cli.sheetport_normalize_yaml` | n/a | Local file transform concern | `crates/agent-spreadsheet/src/cli/commands/read.rs::sheetport_manifest_normalize` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `sheetport bind-check` | _(none direct)_ | SHARED_PARTIAL | `core.sheetport.bind_check` | later | Could be unified later | `crates/agent-spreadsheet/src/cli/commands/read.rs::sheetport_bind_check` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `sheetport run` | `execute_manifest` | ALL | `core.sheetport.execute_manifest` | later | Shared core semantics expected | `crates/agent-spreadsheet/src/cli/commands/read.rs::sheetport_run` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `workbook recalculate` | `recalculate` | SHARED_PARTIAL | `core.recalc.recalculate` | later | Backend constraints in WASM | `crates/agent-spreadsheet/src/cli/commands/recalc.rs::recalculate` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `verify proof` | `verify_workbook` | SHARED_PARTIAL | `core.verify.compare_workbooks` | later | Shared proof contract across CLI + MCP; current inputs are file paths in CLI vs workbook/fork ids in MCP; SDK exposes MCP helpers while WASM parity is later | `crates/agent-spreadsheet/src/cli/commands/verify.rs::verify` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `write append` | _(none today)_ | CLI_ONLY | `adapter-cli.append_region` | n/a | Region/table append helper that resolves a detected region or sheet table, accepts JSON rows or CSV rows, supports explicit footer policies, and compiles to `insert_rows` + `write_matrix` | `crates/agent-spreadsheet/src/cli/commands/write.rs::append_region` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `write clone-template-row` | _(none today)_ | CLI_ONLY | `adapter-cli.clone_template_row` | n/a | Preview-first single-row clone helper that compiles to `clone_row`, returns formula/patch targets, and warns on merge-boundary conflicts | `crates/agent-spreadsheet/src/cli/commands/write.rs::clone_template_row` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `write clone-row-band` | _(none today)_ | CLI_ONLY | `adapter-cli.clone_row_band` | n/a | Preview-first contiguous row-band clone helper that inserts repeated blocks, reports formula/patch targets, and warns on merge-boundary conflicts | `crates/agent-spreadsheet/src/cli/commands/write.rs::clone_row_band` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `verify diff` | `get_changeset` (partial overlap) | SHARED_PARTIAL | `core.diff.diff_workbooks` | later | CLI is file-vs-file; MCP is fork-oriented; CLI now projects grouped summary buckets and can suppress `recalc_result` noise | `crates/agent-spreadsheet/src/cli/commands/diff.rs::diff` | `crates/agent-spreadsheet/tests/diff_engine.rs` |
| `analyze ref-impact` | _(none today)_ | CLI_ONLY | `core.analysis.structure_impact` | n/a | Read-only structural impact preflight; uses same engine as `structure-batch --dry-run --impact-report` | `crates/agent-spreadsheet/src/cli/commands/write.rs::check_ref_impact` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `operations` | _(registry discovery)_ | ALL | `operations.operation_registry` | mvp | Lists canonical operation policy/capability metadata; additive discovery/read/search/analysis set is registry-derived | `crates/agent-spreadsheet/src/operations.rs::operation_registry` | `crates/agent-spreadsheet/tests/canonical_operations.rs` |
| `op` | _(canonical dispatcher)_ | ALL | `operations.execute_operation` | mvp | Machine JSON mode; CLI binds --bind path to an ephemeral resource and injects resource_id | `crates/agent-spreadsheet/src/cli/mod.rs::run_machine_operation` | `crates/agent-spreadsheet/tests/canonical_operations.rs` |
| `schema` | _(none today)_ | CLI_ONLY | `adapter-cli.discoverability.schema` | n/a | Global schema discovery for canonical operations, batch write payloads, and session op payloads | `crates/agent-spreadsheet/src/cli/mod.rs::run_schema_command` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `example` | _(none today)_ | CLI_ONLY | `adapter-cli.discoverability.example` | n/a | Global example discovery for batch write payloads and session op payloads | `crates/agent-spreadsheet/src/cli/mod.rs::run_example_command` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `session` | _(none today)_ | CLI_ONLY | `core.session.*` | n/a | Event-sourced session management (start, log, branches, switch, checkout, undo, redo, fork, op, apply, materialize) | `crates/agent-spreadsheet/src/cli/commands/session.rs` | `crates/agent-spreadsheet/tests/cli_integration.rs` |

---

## B) MCP tool catalog

| MCP tool | CLI equivalent | Classification | Core projection target | WASM target | Notes | Implementation module path | Parity test owner |
|---|---|---:|---|---:|---|---|---|
| `list_workbooks` | _(none)_ | MCP_ONLY | `adapter-mcp.workspace.list_workbooks` | n/a | Workspace/repository concern | `crates/agent-spreadsheet/src/tools/mod.rs::list_workbooks` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `describe_workbook` | `read workbook` | ALL | `core.read.describe_workbook` | mvp | Shared read primitive | `crates/agent-spreadsheet/src/tools/mod.rs::describe_workbook` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `workbook_summary` | _(none direct)_ | SHARED_PARTIAL | `core.analysis.workbook_summary` | later | Candidate future CLI command | `crates/agent-spreadsheet/src/tools/mod.rs::workbook_summary` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `list_sheets` | `read sheets` | ALL | `core.read.list_sheets` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::list_sheets` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `sheet_overview` | `read overview` | ALL | `core.read.sheet_overview` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::sheet_overview` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `sheet_page` | `read page` | ALL | `core.read.sheet_page` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::sheet_page` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `find_value` | `analyze find-value` | ALL | `core.analysis.find_value` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::find_value` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `read_table` | `read table` | ALL | `core.read.read_table` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::read_table` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `table_profile` | `analyze table-profile` | ALL | `core.analysis.table_profile` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::table_profile` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `range_values` | `read values` | ALL | `core.read.range_values` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::range_values` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `inspect_cells` | `read cells` | ALL | `core.read.inspect_cells` | mvp | Strict detail-view (≤25 cells); returns budget metadata | `crates/agent-spreadsheet/src/tools/mod.rs::inspect_cells` | `crates/agent-spreadsheet-mcp/tests/read_guardrails_mcp.rs` |
| `sheet_statistics` | `analyze sheet-statistics` | ALL | `core.analysis.sheet_statistics` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::sheet_statistics` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `sheet_formula_map` | `analyze formula-map` | ALL | `core.analysis.sheet_formula_map` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::sheet_formula_map` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `formula_trace` | `analyze formula-trace` | ALL | `core.analysis.formula_trace` | later | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::formula_trace` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `named_ranges` | `read names` | ALL | `core.read.named_ranges` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::named_ranges` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `define_name` | `write name define` | ALL | `core.write.define_name` | mvp | Named range CRUD (create) | `crates/agent-spreadsheet/src/tools/mod.rs::define_name` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `update_name` | `write name update` | ALL | `core.write.update_name` | mvp | Named range CRUD (update) | `crates/agent-spreadsheet/src/tools/mod.rs::update_name` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `delete_name` | `write name delete` | ALL | `core.write.delete_name` | mvp | Named range CRUD (delete) | `crates/agent-spreadsheet/src/tools/mod.rs::delete_name` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `find_formula` | `analyze find-formula` | ALL | `core.analysis.find_formula` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::find_formula` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `scan_volatiles` | `analyze scan-volatiles` | ALL | `core.analysis.scan_volatiles` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::scan_volatiles` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `sheet_styles` | _(none)_ | SHARED_PARTIAL | `core.read.sheet_styles` | later | Candidate future CLI/WASM surface | `crates/agent-spreadsheet/src/tools/mod.rs::sheet_styles` | `crates/agent-spreadsheet-mcp/tests/unit_styles.rs` |
| `layout_page` | `read layout` | ALL | `core.read.layout_page` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::layout_page` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `grid_export` | `read export --format grid` | ALL | `core.read.grid_export` | mvp | Shared | `crates/agent-spreadsheet/src/tools/mod.rs::grid_export` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `workbook_style_summary` | _(none)_ | SHARED_PARTIAL | `core.analysis.workbook_style_summary` | later | Candidate future CLI/WASM surface | `crates/agent-spreadsheet/src/tools/mod.rs::workbook_style_summary` | `crates/agent-spreadsheet-mcp/tests/unit_workbook_style_summary_recalc.rs` |
| `get_manifest_stub` | `sheetport manifest candidates` | SHARED_PARTIAL | `core.sheetport.manifest_stub` | later | Shared semantic target | `crates/agent-spreadsheet/src/tools/mod.rs::get_manifest_stub` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `execute_manifest` | `sheetport run`/`run-manifest` | ALL | `core.sheetport.execute_manifest` | later | Shared semantic target | `crates/agent-spreadsheet/src/tools/mod.rs::execute_manifest` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `close_workbook` | _(none)_ | MCP_ONLY | `adapter-mcp.session.close_workbook` | n/a | MCP resource lifecycle | `crates/agent-spreadsheet/src/tools/mod.rs::close_workbook` | `crates/agent-spreadsheet-mcp/tests/server_smoke.rs` |
| `vba_project_summary` | _(none)_ | SHARED_PARTIAL | `core.vba.project_summary` | later | Parser/runtime constraints for WASM | `crates/agent-spreadsheet/src/tools/vba.rs::vba_project_summary` | `crates/agent-spreadsheet-mcp/tests/unit_vba.rs` |
| `vba_module_source` | _(none)_ | SHARED_PARTIAL | `core.vba.module_source` | later | Same | `crates/agent-spreadsheet/src/tools/vba.rs::vba_module_source` | `crates/agent-spreadsheet-mcp/tests/unit_vba.rs` |
| `create_fork` | _(none)_ | MCP_ONLY | `adapter-mcp.fork.create` | n/a | MCP orchestration | `crates/agent-spreadsheet/src/tools/fork.rs::create_fork` | `crates/agent-spreadsheet-mcp/tests/fork_workflow.rs` |
| `edit_batch` | `write cells` | ALL | `core.write.edit_batch` | mvp | Shared write semantics | `crates/agent-spreadsheet/src/tools/fork.rs::edit_batch` | `crates/agent-spreadsheet-mcp/tests/fork_workflow.rs` |
| `mutate_batch` | _(none)_ | MCP_ONLY | `adapter-mcp.write.mutate_batch` | n/a | Consolidated previewable write surface; routes tagged ops to per-family batch handlers | `crates/agent-spreadsheet/src/tools/mutate_batch.rs::mutate_batch` | `crates/agent-spreadsheet-mcp/tests/unit_mutate_batch.rs` |
| `transform_batch` | `write batch transform` | ALL | `core.write.transform_batch` | mvp | Shared | `crates/agent-spreadsheet/src/tools/fork.rs::transform_batch` | `crates/agent-spreadsheet-mcp/tests/unit_transform_batch.rs` |
| `style_batch` | `write batch style` | ALL | `core.write.style_batch` | mvp | Shared | `crates/agent-spreadsheet/src/tools/fork.rs::style_batch` | `crates/agent-spreadsheet-mcp/tests/unit_style_batch.rs` |
| `grid_import` | `write import --from-grid` | ALL | `core.write.grid_import` | mvp | Shared | `crates/agent-spreadsheet/src/tools/fork.rs::grid_import` | `crates/agent-spreadsheet-mcp/tests/fork_workflow.rs` |
| `column_size_batch` | `write batch column-size` | ALL | `core.write.column_size_batch` | later | Shared | `crates/agent-spreadsheet/src/tools/fork.rs::column_size_batch` | `crates/agent-spreadsheet-mcp/tests/unit_column_size_batch.rs` |
| `sheet_layout_batch` | `write batch sheet-layout` | ALL | `core.write.sheet_layout_batch` | later | Shared | `crates/agent-spreadsheet/src/tools/sheet_layout.rs::sheet_layout_batch` | `crates/agent-spreadsheet-mcp/tests/unit_sheet_layout_batch.rs` |
| `apply_formula_pattern` | `write batch formula-pattern` | ALL | `core.write.apply_formula_pattern` | later | Shared | `crates/agent-spreadsheet/src/tools/fork.rs::apply_formula_pattern` | `crates/agent-spreadsheet-mcp/tests/unit_apply_formula_pattern.rs` |
| `structure_batch` | `write batch structure` | ALL | `core.write.structure_batch` | later | Shared | `crates/agent-spreadsheet/src/tools/fork.rs::structure_batch` | `crates/agent-spreadsheet-mcp/tests/unit_structure_batch.rs` |
| `rules_batch` | `write batch rules` | ALL | `core.write.rules_batch` | later | Shared | `crates/agent-spreadsheet/src/tools/rules_batch.rs::rules_batch` | `crates/agent-spreadsheet-mcp/tests/unit_rules_batch_cf.rs` |
| `replace_in_formulas` | `write formulas replace` | ALL | `core.write.replace_in_formulas` | later | Formula-only find/replace | `crates/agent-spreadsheet/src/tools/fork.rs::replace_in_formulas` | `crates/agent-spreadsheet-mcp/tests/unit_replace_in_formulas.rs` |
| `get_edits` | _(none)_ | MCP_ONLY | `adapter-mcp.fork.edit_log` | n/a | Fork audit trail | `crates/agent-spreadsheet/src/tools/fork.rs::get_edits` | `crates/agent-spreadsheet-mcp/tests/fork_workflow.rs` |
| `verify_workbook` | `verify proof` | SHARED_PARTIAL | `core.verify.compare_workbooks` | later | Shared proof contract; MCP compares workbook/fork ids while CLI compares file paths; SDK exposes MCP helpers, WASM parity is later | `crates/agent-spreadsheet/src/tools/mod.rs::verify_workbook` | `crates/agent-spreadsheet/tests/cli_integration.rs` |
| `get_changeset` | `verify diff` (partial overlap) | SHARED_PARTIAL | `core.diff.get_changeset` + adapter projection | later | MCP is fork diff, CLI is file diff | `crates/agent-spreadsheet/src/tools/fork.rs::get_changeset` | `crates/agent-spreadsheet-mcp/tests/fork_workflow.rs` |
| `recalculate` | `workbook recalculate` | SHARED_PARTIAL | `core.recalc.recalculate` | later | Backend constraints | `crates/agent-spreadsheet/src/tools/fork.rs::recalculate` | `crates/agent-spreadsheet-mcp/tests/unit_recalc_needed.rs` |
| `list_forks` | _(none)_ | MCP_ONLY | `adapter-mcp.fork.list` | n/a | MCP orchestration | `crates/agent-spreadsheet/src/tools/fork.rs::list_forks` | `crates/agent-spreadsheet-mcp/tests/fork_workflow.rs` |
| `discard_fork` | _(none)_ | MCP_ONLY | `adapter-mcp.fork.discard` | n/a | MCP orchestration | `crates/agent-spreadsheet/src/tools/fork.rs::discard_fork` | `crates/agent-spreadsheet-mcp/tests/fork_workflow.rs` |
| `save_fork` | _(none)_ | MCP_ONLY | `adapter-mcp.fork.save` | n/a | MCP orchestration | `crates/agent-spreadsheet/src/tools/fork.rs::save_fork` | `crates/agent-spreadsheet-mcp/tests/fork_workflow.rs` |
| `checkpoint_fork` | _(none)_ | MCP_ONLY | `adapter-mcp.fork.checkpoint_create` | n/a | MCP orchestration | `crates/agent-spreadsheet/src/tools/fork.rs::checkpoint_fork` | `crates/agent-spreadsheet-mcp/tests/fork_workflow.rs` |
| `list_checkpoints` | _(none)_ | MCP_ONLY | `adapter-mcp.fork.checkpoint_list` | n/a | MCP orchestration | `crates/agent-spreadsheet/src/tools/fork.rs::list_checkpoints` | `crates/agent-spreadsheet-mcp/tests/fork_workflow.rs` |
| `restore_checkpoint` | _(none)_ | MCP_ONLY | `adapter-mcp.fork.checkpoint_restore` | n/a | MCP orchestration | `crates/agent-spreadsheet/src/tools/fork.rs::restore_checkpoint` | `crates/agent-spreadsheet-mcp/tests/fork_workflow.rs` |
| `delete_checkpoint` | _(none)_ | MCP_ONLY | `adapter-mcp.fork.checkpoint_delete` | n/a | MCP orchestration | `crates/agent-spreadsheet/src/tools/fork.rs::delete_checkpoint` | `crates/agent-spreadsheet-mcp/tests/fork_workflow.rs` |
| `list_staged_changes` | _(none)_ | MCP_ONLY | `adapter-mcp.fork.staged_list` | n/a | MCP orchestration | `crates/agent-spreadsheet/src/tools/fork.rs::list_staged_changes` | `crates/agent-spreadsheet-mcp/tests/unit_staging.rs` |
| `apply_staged_change` | _(none)_ | MCP_ONLY | `adapter-mcp.fork.staged_apply` | n/a | MCP orchestration | `crates/agent-spreadsheet/src/tools/fork.rs::apply_staged_change` | `crates/agent-spreadsheet-mcp/tests/unit_staging.rs` |
| `discard_staged_change` | _(none)_ | MCP_ONLY | `adapter-mcp.fork.staged_discard` | n/a | MCP orchestration | `crates/agent-spreadsheet/src/tools/fork.rs::discard_staged_change` | `crates/agent-spreadsheet-mcp/tests/unit_staging.rs` |
| `screenshot_sheet` | _(none)_ | MCP_ONLY | `adapter-mcp.render.screenshot` | n/a | Rendering/tooling concern; browser has native rendering paths | `crates/agent-spreadsheet/src/tools/fork.rs::screenshot_sheet` | `crates/agent-spreadsheet-mcp/tests/screenshot_docker.rs` |

---

## C) Canonical registry migration status

Wave 3A registers the complete canonical discovery/read/search/analysis surface without changing the default MCP router. `asp operations`, `asp schema <operation>`, and `asp op <operation>` derive lookup and schemas from the registry.

| Canonical operation | Existing compatibility projection | Dispatcher implementation | Capability | Risk |
|---|---|---|---|---|
| `list_workbooks` | CLI/MCP `list_workbooks` remains legacy data-only | canonical discovery adapter; no request/resource binding | `workbook_discovery` | low |
| `describe_workbook` | legacy metadata and workbook-summary tools remain separate | exact metadata plus opt-in scoped derived summary | `workbook_read` | low |
| `list_sheets`, `sheet_overview`, `read_table` | CLI/MCP wrappers project canonical `data` | shared semantic implementations | `workbook_read` | low |
| `read_cells` | legacy `range_values` and `sheet_page` remain available | correlated range/row engine with revision/request-bound opaque cursor | `workbook_read` | low |
| `inspect_cells`, `named_ranges` | legacy MCP wrappers project canonical `data` | shared semantic implementations | `workbook_read` | low |
| `read_layout` | legacy `layout_page` stays separate | explicitly lossy layout projection | `workbook_read` | low |
| `export_grid` | legacy `grid_export` stays separate | lossless coordinate-preserving paged grid export | `workbook_read` | low |
| `analyze_styles` | legacy sheet/workbook style tools stay separate | closed `scope.kind` union with bounded-count coverage | `workbook_read` | low |
| `search_values` | legacy `find_value` projects canonical `data` | preserves label, direction, scope, type, header, and context options | `workbook_read` | low |
| `search_formulas` | legacy formula search/volatile tools stay separate | closed `result_mode` branches and actual function classifications | `workbook_read` | low |
| `formula_trace`, `formula_map` | legacy MCP wrappers project canonical `data` | shared formula analysis | `workbook_read` | low |
| `profile_table`, `sheet_statistics` | legacy MCP wrappers project canonical `data` | shared bounded analysis | `workbook_read` | low |

Canonical responses use the versioned operation envelope and state reads carry `revision_id`. Value-bearing reads expose calculation state. Merged responses echo branch discriminants. Checked-in full JSON fixtures cover both branches of `describe_workbook`, `read_cells`, `analyze_styles`, and `search_formulas`, plus every unbranched operation.

Compatibility projections are only routed through the dispatcher when the legacy response can be reconstructed without loss. `range_values`/`sheet_page`, workbook summaries, layout/grid, style summaries, and formula-search variants remain separate compatibility implementations rather than claiming false response parity. No default MCP router switch is authorized in Wave 3A.

---

## D) Enforcement hooks

- Boundary contract (non-negotiable): `docs/architecture/surface-boundary-rules.md`
- Matrix drift checker: `scripts/check_surface_matrix_drift.py`
- Local/CI invocation:
  - `python3 scripts/check_surface_matrix_drift.py`
  - `cargo test -p agent-spreadsheet surface_matrix_drift_check`
