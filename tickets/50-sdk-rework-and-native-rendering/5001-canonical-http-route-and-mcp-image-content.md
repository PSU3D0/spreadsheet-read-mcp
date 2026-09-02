# Ticket: 5001 Canonical HTTP Route + MCP Image Content

## Depends On
- 0.14 canonical router (`crates/agent-spreadsheet-mcp/src/canonical_router.rs`)

## Why
The SDK needs a programmatic way into the stateful server process without speaking MCP. Today the axum app in `crates/agent-spreadsheet-mcp/src/lib.rs` mounts only the rmcp streamable-HTTP service at `/mcp`. Separately, the canonical `screenshot_sheet` route returns only text plus `structuredContent`; only the legacy tool attaches image content, so canonical screenshots are unretrievable over MCP.

## Owner / Effort / Risk
- Owner: MCP / server adapter
- Effort: S-M
- Risk: Low

## Scope

### Canonical HTTP route (transport adapter only)
- `POST /v1/op/{operation}` with a JSON body equal to the canonical input. Response body is the canonical response envelope, status 200.
- Errors return the canonical error envelope with HTTP status mapped from `CanonicalErrorCode`: invalid_request 400, unknown_operation 404, resource_not_found 404, revision_conflict 409, capability_unavailable 501, operation_failed 500. Keep the JSON identical to what MCP `isError` results carry.
- `GET /v1/operations` returns `operations_discovery_for(OperationAdapter::Mcp, capabilities)` for the live process. `GET /v1/registry` returns `registry_projection()`.
- `GET /v1/artifacts/{handle}` streams artifact bytes with the recorded media type for handles produced by `screenshot_sheet` in this process. Reject anything that is not a well-formed `artifact:sha256:<hex>` handle. Never accept or expose a path. Bounded by the existing 16 MiB artifact ceiling.
- Same timeout, response-size, and enabled-tool policy as MCP tool calls. Reuse `execute_canonical_operation` and `canonical_result`. No spreadsheet semantics in the route layer.
- Bind policy unchanged from the MCP HTTP transport. Document that the route has no auth, exactly like `/mcp`.

### MCP image content
- In the canonical router, when the operation is `screenshot_sheet` and the result is not an error, push `Content::image(base64, media_type)` after the text content. Keep the envelope in `structuredContent` unchanged. Enforce the existing response-size limit on the base64 payload.

## Non-Goals
- Workbook upload or session creation over HTTP. The server keeps working on its workspace.
- Auth, TLS, CORS policy beyond a documented default.
- Any registry, schema, or operation change.

## Tests
- Integration test starting the HTTP transport on a loopback port: discovery, `describe_workbook` round trip, a revision-conflict `write` mapped to 409, unknown operation 404, capability_unavailable 501.
- Artifact route: valid handle returns bytes with `image/png`; malformed handle 400; unknown handle 404; a handle from another workspace is not served.
- Canonical MCP `screenshot_sheet` test asserting one text and one image content item plus unchanged `structuredContent` (may be gated behind the docker LibreOffice feature until 5004 lands the native backend; if so, add a unit test using a stub PNG artifact).

## Definition of Done
- SDK server runtime can be written against `/v1/op/{operation}` with no MCP client.
- Canonical MCP screenshot results carry the image.
