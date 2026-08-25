# Spreadsheet Recalc & Write Architecture

This document describes the "Fork & Recalc" architecture for `agent-spreadsheet-mcp`, enabling safe "what-if" analysis.

## Overview

The system allows users to create isolated copies (forks) of a spreadsheet, apply edits, recalculate formulas using LibreOffice, and inspect the resulting changes (diffs).

## Core Concepts

### 1. Forks
A **Fork** is a temporary, writable session based on an existing `.xlsx` file.
- Stored in `/tmp/mcp-forks/{fork_id}.xlsx`.
- Ephemeral: cleaned up after 1 hour (TTL) or when discarded.
- Isolated: edits do not affect the original file until `save_fork` is called.

### 2. Recalculation Engine
We use **LibreOffice (headless)** to evaluate formulas.
- **V1 Implementation:** "Fire-and-forget" model. Spawns a fresh `soffice` process for each recalculation to ensure clean state and avoid memory leaks.
- **Concurrency:** Limited by a global semaphore (default: 2 concurrent processes) to prevent resource exhaustion.
- **Macros:** A custom Basic macro (`RecalculateAndSave`) is injected into the Docker image to trigger `calculateAll()` and save the result.

### 3. Diff Engine (`get_changeset`)
To show the impact of changes, we compare the "Fork" against the "Base" workbook.
- **Streaming Diff:** We parse the underlying XML (`xl/worksheets/sheet*.xml`) using `quick-xml` to avoid loading the entire workbook into memory.
- **Scope:**
    - **Cells:** Value changes, formula changes, recalc results.
    - **Tables:** Detects resized, added, or deleted tables.
    - **Defined Names:** Detects modified named ranges.

## Docker Requirements

The `agent-spreadsheet-mcp:full` image includes:
- `libreoffice-calc`
- `default-jre-headless` (for macros)
- `fonts-liberation`

The standard `agent-spreadsheet-mcp:latest` image is read-only and much smaller (~15MB vs ~600MB).

## Usage Workflow

1.  **Create Fork:** `create_fork(workbook_id)` -> returns `fork_id`.
2.  **Edit:** `edit_batch(fork_id, sheet_name, edits: [{address: "A1", value: "100"}])`.
3.  **Recalc:** `recalculate(fork_id)` -> triggers LibreOffice.
4.  **Review:** `get_changeset(fork_id)` -> JSON diff of values/formulas/tables.
5.  **Commit/Discard:** `save_fork(fork_id, target_path)` or `discard_fork(fork_id)`.

### Docker paths (what’s persisted)

When running in Docker with `--workspace-root /data` and a host mount like `-v /path/to/workbooks:/data`:

- Fork working files are stored under `/tmp/mcp-forks` inside the container and are ephemeral.
- To persist a fork back to the host, call `save_fork` with a `target_path` under `/data` (or a relative path).
- Screenshots from `screenshot_sheet` are written under `/data/screenshots/` (host sees `/path/to/workbooks/screenshots/`).

## Configuration

*   `SPREADSHEET_MCP_RECALC_ENABLED=true`: Enables write tools.
*   `SPREADSHEET_MCP_MAX_CONCURRENT_RECALCS=2`: Limits `soffice` instances.
*   `SPREADSHEET_MCP_CACHE_CAPACITY=5`: LRU cache for base workbooks.
