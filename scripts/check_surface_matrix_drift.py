#!/usr/bin/env python3
"""Validate CLI/MCP surface coverage + required matrix metadata.

The checker enforces:
- every discovered CLI command is represented in matrix section A
- every discovered MCP tool is represented in matrix section B
- no stale matrix-only entries remain
- required matrix columns are populated for every row
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_MOD_RS = Path(
    os.environ.get(
        "SURFACE_CLI_MOD_RS",
        str(REPO_ROOT / "crates/agent-spreadsheet/src/cli/mod.rs"),
    )
)
MCP_SERVER_RS = Path(
    os.environ.get(
        "SURFACE_MCP_SERVER_RS",
        str(REPO_ROOT / "crates/agent-spreadsheet-mcp/src/server.rs"),
    )
)
MATRIX_MD = Path(
    os.environ.get(
        "SURFACE_MATRIX_MD",
        str(REPO_ROOT / "docs/architecture/surface-capability-matrix.md"),
    )
)
CANONICAL_OPERATIONS_RS = Path(
    os.environ.get(
        "SURFACE_CANONICAL_OPERATIONS_RS",
        str(REPO_ROOT / "crates/agent-spreadsheet/src/operations.rs"),
    )
)

VALID_CLASSIFICATIONS = {"ALL", "CLI_ONLY", "MCP_ONLY", "SHARED_PARTIAL"}
VALID_WASM_TARGETS = {"mvp", "later", "n/a", "host-owned", "⛔/optional"}

# Variant rows represented separately in the matrix and enforced as separate entries.
CLI_VARIANT_ROWS: dict[str, set[str]] = {
    "read export": {
        "read export --format json/csv",
        "read export --format grid",
    },
    "write import": {
        "write import --from-grid",
        "write import --from-csv",
    },
}


@dataclass
class MatrixRow:
    kind: str  # cli|mcp
    key: str
    classification: str
    core_projection: str
    wasm_target: str
    impl_path: str
    parity_owner: str
    line_number: int


def camel_to_kebab(name: str) -> str:
    parts: list[str] = []
    for ch in name:
        if ch.isupper() and parts:
            parts.append("-")
        parts.append(ch.lower())
    return "".join(parts)


def discover_cli_commands(cli_source: str) -> set[str]:
    top_level = set(re.findall(r"SurfaceCommands::([A-Za-z0-9_]+)", cli_source))
    read = set(re.findall(r"SurfaceReadCommands::([A-Za-z0-9_]+)", cli_source))
    analyze = set(re.findall(r"SurfaceAnalyzeCommands::([A-Za-z0-9_]+)", cli_source))
    workbook = set(re.findall(r"SurfaceWorkbookCommands::([A-Za-z0-9_]+)", cli_source))
    verify = set(re.findall(r"SurfaceVerifyCommands::([A-Za-z0-9_]+)", cli_source))
    write_top = set(re.findall(r"SurfaceWriteCommands::([A-Za-z0-9_]+)", cli_source))
    write_formula = set(re.findall(r"SurfaceWriteFormulaCommands::([A-Za-z0-9_]+)", cli_source))
    write_name = set(re.findall(r"SurfaceWriteNameCommands::([A-Za-z0-9_]+)", cli_source))
    write_batch = set(re.findall(r"SurfaceWriteBatchCommands::([A-Za-z0-9_]+)", cli_source))
    sheetport = set(re.findall(r"SheetportCommands::([A-Za-z0-9_]+)", cli_source))
    manifest = set(re.findall(r"SheetportManifestCommands::([A-Za-z0-9_]+)", cli_source))

    commands: set[str] = set()

    for name in read:
        commands.add(f"read {camel_to_kebab(name)}")

    for name in analyze:
        commands.add(f"analyze {camel_to_kebab(name)}")

    for name in workbook:
        commands.add(f"workbook {camel_to_kebab(name)}")

    for name in verify:
        commands.add(f"verify {camel_to_kebab(name)}")

    for name in write_top:
        if name in {"Batch", "Name", "Formulas"}:
            continue
        commands.add(f"write {camel_to_kebab(name)}")

    for name in write_formula:
        commands.add(f"write formulas {camel_to_kebab(name)}")

    for name in write_name:
        commands.add(f"write name {camel_to_kebab(name)}")

    for name in write_batch:
        commands.add(f"write batch {camel_to_kebab(name)}")

    if "Operations" in top_level:
        commands.add("operations")
    if "Op" in top_level:
        commands.add("op")
    if "Schema" in top_level:
        commands.add("schema")
    if "Example" in top_level:
        commands.add("example")
    if "Session" in top_level:
        commands.add("session")

    for name in sheetport:
        if name == "Manifest":
            continue
        commands.add(f"sheetport {camel_to_kebab(name)}")

    for name in manifest:
        commands.add(f"sheetport manifest {camel_to_kebab(name)}")

    expanded: set[str] = set()
    for command in commands:
        if command in CLI_VARIANT_ROWS:
            expanded.update(CLI_VARIANT_ROWS[command])
        else:
            expanded.add(command)

    return expanded


def discover_mcp_tools(server_source: str) -> set[str]:
    # Compatibility handlers are static attributes; the default MCP router is
    # projected dynamically from the canonical registry.
    tool_blocks = re.findall(r"#\[tool\((.*?)\)\]", server_source, flags=re.S)
    discovered: set[str] = set()
    for block in tool_blocks:
        match = re.search(r"\bname\s*=\s*\"([a-z0-9_]+)\"", block)
        if match:
            discovered.add(match.group(1))

    if "canonical_tool_router" in server_source:
        operations = CANONICAL_OPERATIONS_RS.read_text(encoding="utf-8")
        registry = operations.split("static REGISTRY:", 1)[1].split(
            "pub fn operation_registry", 1
        )[0]
        discovered.update(
            re.findall(r"descriptor!\(\s*\"([a-z0-9_]+)\"", registry)
        )
        discovered.update(re.findall(r"\bname:\s*\"([a-z0-9_]+)\"", registry))
    return discovered


def strip_ticks(value: str) -> str:
    value = value.strip()
    if value.startswith("`") and value.endswith("`") and len(value) >= 2:
        return value[1:-1]
    return value


def normalize_cli_matrix_entry(raw: str) -> str:
    value = strip_ticks(raw).replace("`", "")
    value = re.sub(r"\s+", " ", value).strip().lower()
    # Remove suffix hints like "(deprecated)".
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
    return value


def normalize_mcp_matrix_entry(raw: str) -> str:
    return strip_ticks(raw).strip().lower()


def normalize_cell(raw: str) -> str:
    return strip_ticks(raw).replace("`", "").strip()


def parse_matrix_rows(matrix_text: str) -> tuple[dict[str, MatrixRow], dict[str, MatrixRow], list[str]]:
    mode: str | None = None
    cli_rows: dict[str, MatrixRow] = {}
    mcp_rows: dict[str, MatrixRow] = {}
    errors: list[str] = []

    for line_number, line in enumerate(matrix_text.splitlines(), start=1):
        if line.startswith("## A) CLI command catalog"):
            mode = "cli"
            continue
        if line.startswith("## B) MCP tool catalog"):
            mode = "mcp"
            continue
        if line.startswith("## ") and not line.startswith("## A)") and not line.startswith("## B)"):
            mode = None
            continue

        if mode not in {"cli", "mcp"}:
            continue
        if not line.startswith("|"):
            continue

        trimmed = line.strip()
        if re.match(r"^\|\s*[-:|\s]+\|$", trimmed):
            continue

        cells = [cell.strip() for cell in trimmed.strip("|").split("|")]
        if not cells:
            continue

        first_col = cells[0]
        if (
            not first_col
            or first_col.lower().startswith("cli command")
            or first_col.lower().startswith("mcp tool")
        ):
            continue

        if len(cells) < 8:
            errors.append(
                f"line {line_number}: expected >=8 columns in {mode.upper()} table row, got {len(cells)}"
            )
            continue

        key = (
            normalize_cli_matrix_entry(first_col)
            if mode == "cli"
            else normalize_mcp_matrix_entry(first_col)
        )

        row = MatrixRow(
            kind=mode,
            key=key,
            classification=normalize_cell(cells[2]),
            core_projection=normalize_cell(cells[3]),
            wasm_target=normalize_cell(cells[4]).lower(),
            impl_path=normalize_cell(cells[6]),
            parity_owner=normalize_cell(cells[7]),
            line_number=line_number,
        )

        if row.classification not in VALID_CLASSIFICATIONS:
            errors.append(
                f"line {line_number} ({key}): invalid classification '{row.classification}'"
            )

        if not row.core_projection:
            errors.append(
                f"line {line_number} ({key}): core projection target must be non-empty"
            )

        if row.wasm_target not in VALID_WASM_TARGETS:
            errors.append(
                f"line {line_number} ({key}): invalid wasm target '{row.wasm_target}'"
            )

        if not row.impl_path:
            errors.append(
                f"line {line_number} ({key}): implementation module path must be non-empty"
            )

        if not row.parity_owner:
            errors.append(
                f"line {line_number} ({key}): parity test owner must be non-empty"
            )

        target = cli_rows if mode == "cli" else mcp_rows
        if key in target:
            errors.append(
                f"line {line_number}: duplicate {mode.upper()} matrix key '{key}' (first seen at line {target[key].line_number})"
            )
            continue
        target[key] = row

    return cli_rows, mcp_rows, errors


def main() -> int:
    cli_source = CLI_MOD_RS.read_text(encoding="utf-8")
    mcp_source = MCP_SERVER_RS.read_text(encoding="utf-8")
    matrix_text = MATRIX_MD.read_text(encoding="utf-8")

    discovered_cli = discover_cli_commands(cli_source)
    discovered_mcp = discover_mcp_tools(mcp_source)
    matrix_cli_rows, matrix_mcp_rows, matrix_errors = parse_matrix_rows(matrix_text)
    staged_risk_contract = (
        "| `staged_change` | three legacy staging tools remain unchanged |"
        " action union; canonical bundles bind applicability to base content revision,"
        " replay through the write dispatcher, and are consumed only after success |"
        " `workbook_write` | destructive ceiling; request-aware low/moderate/destructive |"
    )
    if staged_risk_contract not in matrix_text:
        matrix_errors.append(
            "canonical staged_change risk contract must remain destructive-ceiling"
        )

    matrix_cli = set(matrix_cli_rows.keys())
    matrix_mcp = set(matrix_mcp_rows.keys())

    missing_cli = sorted(command for command in discovered_cli if command not in matrix_cli)
    missing_mcp = sorted(tool for tool in discovered_mcp if tool not in matrix_mcp)

    stale_cli = sorted(entry for entry in matrix_cli if entry not in discovered_cli)
    stale_mcp = sorted(entry for entry in matrix_mcp if entry not in discovered_mcp)

    print("Surface capability drift check")
    print(f"  CLI discovered:   {len(discovered_cli)}")
    print(f"  CLI in matrix:    {len(matrix_cli)}")
    print(f"  MCP discovered:   {len(discovered_mcp)}")
    print(f"  MCP in matrix:    {len(matrix_mcp)}")

    if matrix_errors:
        print("\n[error] Matrix validation errors:")
        for error in matrix_errors:
            print(f"  - {error}")

    if missing_cli:
        print("\n[error] Missing CLI entries in matrix:")
        for command in missing_cli:
            print(f"  - {command}")

    if missing_mcp:
        print("\n[error] Missing MCP tool entries in matrix:")
        for tool in missing_mcp:
            print(f"  - {tool}")

    if stale_cli:
        print("\n[error] Stale CLI entries in matrix (no longer discoverable):")
        for entry in stale_cli:
            print(f"  - {entry}")

    if stale_mcp:
        print("\n[error] Stale MCP entries in matrix (no longer discoverable):")
        for entry in stale_mcp:
            print(f"  - {entry}")

    if matrix_errors or missing_cli or missing_mcp or stale_cli or stale_mcp:
        return 1

    print("\nOK: Matrix entries, required metadata, and discovered CLI/MCP surfaces are aligned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
