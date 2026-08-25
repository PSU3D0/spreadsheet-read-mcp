# Scenario benchmark harnesses

This directory stores sanitized, checked-in budget baselines for representative workflow scenarios.

## Current anchor

- `scenario-01-roll-forward`
  - implemented by `crates/agent-spreadsheet/tests/scenario_benchmark_harness.rs`
  - uses a generated workbook fixture (no customer/project workbook dependency)
  - exercises a real CLI workflow: `edit -> recalculate -> verify`

## Budget refresh

When an intentional workflow change improves or legitimately expands the scenario budget, refresh the checked-in baseline explicitly:

```bash
UPDATE_SCENARIO_BUDGETS=1 cargo test -p agent-spreadsheet --test scenario_benchmark_harness -- --nocapture
```

That command rewrites `benchmarks/scenario-01-roll-forward/budget.json` with the latest observed metrics and derived headroom.

## Policy

- correctness assertions stay separate from cost assertions
- scenario fixtures should remain sanitized and industry-agnostic
- budgets should be tight enough to catch regressions, but generous enough to avoid CI flake
