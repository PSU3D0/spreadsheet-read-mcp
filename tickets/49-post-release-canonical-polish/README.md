# 49 - Post-release Canonical Polish

Status: planned after 0.14.

## Outcome

Polish the canonical surface using evidence from the 0.14 convergence work without reopening its registry or compatibility contracts during release stabilization.

## Ranked backlog

1. **`formula_trace` cross-sheet behavior** - close cross-sheet traversal gaps and lock direction/depth semantics with fixtures.
2. **Selection consistency, precedence, and cursor docs** - align overlapping selection forms, state which selector wins, and document request/revision cursor binding in one place.
3. **R1C1 fingerprints** - add stable relative-formula fingerprints for grouping copied formula families.
4. **Deterministic samples** - make sampling seeds/order explicit and reproducible across native, MCP, and WASM adapters.
5. **Brief schema projection** - provide a bounded discoverability view that avoids returning full schemas when an agent needs only required fields and discriminants.
6. **Staged locations** - expose durable staged-change locations/identities without leaking server filesystem paths.
7. **Dynamic risk projection** - derive request-specific risk more precisely while retaining conservative static MCP annotations.
8. **Overview cues** - add cheap structural cues that guide the next read without overstating inferred workbook meaning.
9. **`list_workbooks` revision** - return resource revisions where available and define freshness for discovery results.
10. **CI tool-byte budget** - enforce deterministic byte ceilings for tool discovery/schema payloads as a release regression gate.

## Guardrails

- Preserve the 0.14 canonical operation names and envelopes unless a separately reviewed compatibility plan says otherwise.
- Rank work by measured agent turns/bytes and correctness impact, not surface novelty.
- Add cross-surface contract fixtures for every semantic change.
