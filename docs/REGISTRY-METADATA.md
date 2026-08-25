# MCP Registry Distribution Notes

This repository is prepared for distribution through:

- Official MCP Registry: <https://registry.modelcontextprotocol.io/>
- Docker MCP Catalog: <https://mcp.docker.com/>

## Official MCP Registry

Canonical server name:

```text
io.github.PSU3D0/agent-spreadsheet-mcp
```

Metadata file:

```text
server.json
```

The registry package points at the full GHCR image because that image exposes the write/recalc-capable MCP surface:

```text
ghcr.io/psu3d0/agent-spreadsheet-mcp:0.10.1-full
```

The package is configured for stdio transport and expects clients/runtimes to mount a host workbook directory at `/data`.

Local validation:

```bash
python3 scripts/validate_registry_metadata.py
curl --fail --show-error --silent \
  --header 'Content-Type: application/json' \
  --data @server.json \
  https://registry.modelcontextprotocol.io/v0.1/validate
```

Publishing requires the official `mcp-publisher` CLI and authentication with the namespace owner.

```bash
mcp-publisher login github
mcp-publisher publish server.json
```

## Docker MCP Catalog

Dockerfiles include the required OCI ownership label:

```dockerfile
LABEL io.modelcontextprotocol.server.name="io.github.PSU3D0/agent-spreadsheet-mcp"
```

Docker Catalog submission happens in the Docker registry repo:

```text
https://github.com/docker/mcp-registry
```

Submission is normally a PR adding `servers/<server-name>/server.yaml`. Use Docker's helper flow from that repo:

```bash
task create -- --category productivity \
  --image ghcr.io/psu3d0/agent-spreadsheet-mcp:0.10.1-full \
  https://github.com/PSU3D0/agent-spreadsheet
```

Suggested category/tags:

- category: `productivity` or `data`
- tags: `spreadsheet`, `excel`, `xlsx`, `analysis`, `automation`, `agents`, `recalc`

Suggested title:

```text
Spreadsheet Kit
```

Suggested description:

```text
Agent-safe Excel workbook analysis, editing, recalculation, and verification tools.
```

The Docker Catalog PR may require a reviewer-accessible test fixture or instructions because the server expects a mounted workbook directory at `/data`.
