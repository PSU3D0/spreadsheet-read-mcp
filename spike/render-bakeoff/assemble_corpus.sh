#!/usr/bin/env bash
# Re-create the non-committed parts of the bake-off corpus from sibling checkouts.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OSS="$(cd "$HERE/../../../.." && pwd)"     # codebase/oss
REPO="$(cd "$HERE/../.." && pwd)"          # spreadsheet-mcp worktree root
C="$HERE/corpus"
mkdir -p "$C"/{fixtures,calamine,synthetic,ironcalc}
cp "$REPO"/crates/agent-spreadsheet/tests/fixtures/f1/*.xlsx "$C/fixtures/"
for f in issue_261_fixed_by_excel issue_261 date date_1904 date_iso temperature-in-middle \
         temperature-table merged_range merge_cells inlineStr_with_value issue127 rph picture pivots; do
  cp "$OSS/calamine/tests/$f.xlsx" "$C/calamine/"
done
for f in repro_chain3 struct_sheet_rename_rebind structural_sheet_recovery cross_sheet_mesh \
         lookup_cross_sheet_dim_fact; do
  cp "$OSS/formualizer/benchmarks/corpus/synthetic/$f.xlsx" "$C/synthetic/"
done
for f in CELL BETA_GAMMA CONCAT arithmetic ARRAYTOTEXT; do
  cp "$OSS/IronCalc/xlsx/tests/calc_tests/$f.xlsx" "$C/ironcalc/"
done
find "$C" -name '*.xlsx' | wc -l
