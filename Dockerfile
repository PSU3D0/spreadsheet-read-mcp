FROM rust:1.91.1-alpine AS builder

RUN apk add --no-cache musl-dev

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN cargo build --release --locked --no-default-features -p agent-spreadsheet-mcp

FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /build/target/release/agent-spreadsheet-mcp /usr/local/bin/agent-spreadsheet-mcp

WORKDIR /data

# Defaults so override/stdio runs still see mounted workspace.
ENV SPREADSHEET_MCP_WORKSPACE=/data

LABEL org.opencontainers.image.source="https://github.com/PSU3D0/agent-spreadsheet"
LABEL org.opencontainers.image.description="MCP server for spreadsheet analysis and editing"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL io.modelcontextprotocol.server.name="io.github.PSU3D0/agent-spreadsheet"

ENTRYPOINT ["agent-spreadsheet-mcp"]
CMD ["--workspace-root", "/data", "--transport", "http", "--http-bind", "0.0.0.0:8079"]
