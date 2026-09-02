# Tickets: Agent Surface Hardening + Headless Strategy

This directory is the committed planning system for improving `asp` / CLI, MCP, WASM, and JS SDK usability for real spreadsheet-agent work.

## Why this roadmap exists

Recent benchmark work showed that the core architecture is directionally strong, but agent cost is still too high for moderate scenarios like `scenario-01-roll-forward`.

The expensive parts were not spreadsheet reasoning. The expensive parts were:

- command discovery
- payload-shape uncertainty
- recalc/debug ambiguity
- repeated verification
- too many low-level steps for common edits

That is good news: it means we can make meaningful near-term gains by hardening the agent surface rather than waiting on a full long-term rewrite.

## Strategic thesis

UI-driven agents in Excel/LibreOffice currently win on:

- visual locality
- implicit affordances
- easy ambiguity resolution

Headless workflows can win on:

- deterministic edits
- replayability and audit trails
- safe mutation flows
- machine-verifiable postconditions
- CI/runtime portability

The roadmap below is organized to improve the **near-term user experience** while building toward the **long-term headless advantage**.

## Horizons

- **Now** — remove obvious friction and harden contracts for benchmark-grade use.
- **Next** — add workflow-oriented helpers and stronger MCP/SDK ergonomics.
- **Strategic** — build structural understanding and proof/automation layers that can outperform UI agents over time.

## Tranches

| Tranche | Horizon | Focus |
|---|---|---|
| [42-surface-contract-hardening-now](./42-surface-contract-hardening-now/README.md) | now | `asp` obviousness, canonical payloads, explicit result contracts, thin docs |
| [43-verification-and-workflows-now](./43-verification-and-workflows-now/README.md) | now | post-edit proof, grouped change summaries, workflow helpers, benchmark gates |
| [44-mcp-sdk-ergonomics-next](./44-mcp-sdk-ergonomics-next/README.md) | next | task-oriented MCP flows, SDK workflow objects, CLI grouping vnext |
| [45-headless-proof-strategic](./45-headless-proof-strategic/README.md) | strategic | structural model, dependency-cone verification, contract-driven automation |
| [46-docs-site-and-skills-library](./46-docs-site-and-skills-library/README.md) | next | unified docs home and tested agent skills library |
| [47-just-bash-extension-and-registries](./47-just-bash-extension-and-registries/README.md) | next | minimal just-bash adapter after canonical convergence; distribution registries |
| [48-canonical-operation-convergence](./48-canonical-operation-convergence/README.md) | complete | one operation registry/dispatcher across CLI, MCP, WASM, SDK, and adapters |
| [49-post-release-canonical-polish](./49-post-release-canonical-polish/README.md) | next | ranked follow-up polish after the 0.14 canonical release |
| [50-sdk-rework-and-native-rendering](./50-sdk-rework-and-native-rendering/README.md) | complete | SDK 0.15 rework, canonical `/v1` HTTP route, WASM diet, native raster renderer, WASM rendering (released in 0.15.0) |

## Planning principles

1. **One semantic core**
   - Avoid surface drift across CLI, MCP, WASM, and SDK.
2. **Strict over implicit**
   - Ambiguous payloads and empty-result inference should be eliminated.
3. **Proof beats vibes**
   - The headless advantage comes from verification and explainability.
4. **Task flows matter**
   - Agents should solve spreadsheet tasks, not rediscover substrate rules.
5. **Benchmark against real scenarios**
   - `scenario-01-roll-forward` should be a standing regression harness, not a one-off exercise.

## Design doc policy

Anything with meaningful semantic, contract, or architecture complexity should have its own technical design doc.

In this roadmap, design docs are attached for items such as:

- canonical session payload contracts
- post-edit verification and provenance
- region/footer-aware workflow helpers
- SDK workflow object model
- structural workbook model
- dependency-cone verification
- contract/assertion automation

## Suggested execution order

### Immediate execution lane
1. Tranche 42
2. Tranche 43

### Follow-on productization lane
3. Canonical operation convergence (ticket 48), beginning with verification soundness (F1)
4. Tranche 44 SDK/MCP ergonomics projected from the canonical registry
5. Skills/docs refresh (ticket 46)
6. Minimal just-bash adapter (ticket 47)
7. Post-release canonical polish (ticket 49)

### Long-term moat lane
8. Tranche 45

## Acceptance outcome for this roadmap

When the "now" tranches are complete, a moderate spreadsheet workflow should:

- use a discoverable command/tool surface
- fail loudly on malformed payloads
- expose explicit results and warnings
- require fewer primitive edit steps
- provide clear proof of what changed and whether anything newly broke

That is the baseline needed before we can credibly claim the headless workflow is competitive in practice.
