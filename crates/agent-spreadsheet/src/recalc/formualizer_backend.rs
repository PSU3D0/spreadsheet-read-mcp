use super::RecalcResult;
use crate::recalc::RecalcBackend;
use crate::utils::{column_number_to_name, hash_bytes_sha256_hex, hash_file_sha256_hex};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use formualizer::common::PackedSheetCell;
use formualizer::eval::engine::ingest::EngineLoadStream;
use formualizer::eval::engine::{Engine, EvalConfig, FormulaParsePolicy};
use formualizer::workbook::workbook::WBResolver;
use formualizer::workbook::{
    FormulaCacheUpdate, LiteralValue, SpreadsheetReader, SpreadsheetWriter, UmyaAdapter,
};
use std::collections::HashSet;
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use web_time::Instant;

pub struct FormualizerBackend;

#[async_trait]
impl RecalcBackend for FormualizerBackend {
    async fn recalculate(
        &self,
        fork_work_path: &Path,
        timeout_ms: Option<u64>,
    ) -> Result<RecalcResult> {
        let path = fork_work_path.to_path_buf();
        // Use a dedicated thread with a 32 MiB stack instead of
        // tokio::task::spawn_blocking (which uses 2 MiB by default).
        // Deep formula chains (e.g. 30k cascading rows) can exceed 2 MiB
        // in debug builds.
        let (tx, rx) = tokio::sync::oneshot::channel();
        std::thread::Builder::new()
            .name("formualizer-recalc".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let _ = tx.send(recalc_sync(&path, timeout_ms));
            })
            .map_err(|e| anyhow!("failed to spawn recalc thread: {e}"))?;
        rx.await.map_err(|_| anyhow!("recalc thread panicked"))?
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "formualizer"
    }
}

type FormualizerEngine = Engine<WBResolver>;

fn recalc_sync(path: &Path, timeout_ms: Option<u64>) -> Result<RecalcResult> {
    let start = Instant::now();
    let open_start = Instant::now();
    let adapter = UmyaAdapter::open_path(path)
        .map_err(|e| anyhow!("failed to open workbook adapter {:?}: {e}", path))?;
    let open_ms = open_start.elapsed().as_millis() as u64;
    let (result, _) = recalculate_adapter_sync(
        adapter,
        timeout_ms,
        start,
        open_ms,
        RecalcPersistence::Path(path),
    )?;
    Ok(result)
}

pub fn recalculate_bytes_sync(
    bytes: &[u8],
    timeout_ms: Option<u64>,
) -> Result<(RecalcResult, Vec<u8>)> {
    let start = Instant::now();
    let open_start = Instant::now();
    let adapter = UmyaAdapter::open_bytes(bytes.to_vec())
        .map_err(|e| anyhow!("failed to open workbook adapter from bytes: {e}"))?;
    let open_ms = open_start.elapsed().as_millis() as u64;
    let (result, evaluated) = recalculate_adapter_sync(
        adapter,
        timeout_ms,
        start,
        open_ms,
        RecalcPersistence::Bytes,
    )?;
    Ok((result, evaluated.expect("bytes persistence returns bytes")))
}

enum RecalcPersistence<'a> {
    Path(&'a Path),
    Bytes,
}

fn recalculate_adapter_sync(
    mut adapter: UmyaAdapter,
    timeout_ms: Option<u64>,
    start: Instant,
    open_ms: u64,
    persistence: RecalcPersistence<'_>,
) -> Result<(RecalcResult, Option<Vec<u8>>)> {
    let formula_cells = adapter.formula_cells();
    let formula_cells_len = formula_cells.len();

    // Fast recalc path by default for agentic/stateless workflows:
    // - defer graph building during ingest (dramatically reduces load time)
    // - coerce malformed formulas to errors so one bad sheet doesn't abort the full run
    let eval_config = EvalConfig {
        defer_graph_building: true,
        formula_parse_policy: FormulaParsePolicy::CoerceToError,
        ..Default::default()
    };

    let mut engine = FormualizerEngine::new(WBResolver::default(), eval_config);

    let stream_start = Instant::now();
    adapter
        .stream_into_engine(&mut engine)
        .map_err(|e| anyhow!("failed to ingest workbook into formualizer engine: {e}"))?;
    let stream_ms = stream_start.elapsed().as_millis() as u64;

    let eval_start = Instant::now();
    let (cells_evaluated, cycle_errors, changed_cells, incomplete, interruption) =
        match evaluate_with_optional_timeout(&mut engine, timeout_ms) {
            Ok((cells, cycles, changed)) => (cells, cycles, changed, false, None),
            Err(error) if timeout_ms.is_some() => (
                0,
                0,
                None,
                true,
                Some(format!(
                    "evaluation interrupted before complete coverage: {error}"
                )),
            ),
            Err(error) => return Err(anyhow!("formualizer evaluate_all failed: {error}")),
        };
    let evaluate_ms = eval_start.elapsed().as_millis() as u64;

    let mut eval_errors = Vec::new();
    if let Some(interruption) = interruption {
        eval_errors.push(interruption);
    }
    if cycle_errors > 0 {
        eval_errors.push(format!(
            "Detected {} circular reference cycle(s). Cells in cycles are reported as #CIRC! by this backend; workbooks built with Excel's iterative calculation (common in financial models with interest/cash-sweep circularity) need a backend that iterates. If LibreOffice is installed, retry with SPREADSHEET_MCP_RECALC_BACKEND=libreoffice. Do not try to 'fix' intentional circular references.",
            cycle_errors
        ));
    }

    let build_updates_start = Instant::now();
    let date_system = engine.config.date_system;
    let changed_filter = changed_cells.as_ref();
    let mut cache_updates = Vec::with_capacity(formula_cells_len);
    let mut error_formula_cells = 0u64;
    for (sheet_name, row, col) in formula_cells {
        let value = engine
            .get_cell_value(&sheet_name, row, col)
            .unwrap_or(LiteralValue::Empty);

        if let LiteralValue::Error(err) = &value {
            error_formula_cells += 1;
            if eval_errors.len() < 200 {
                let addr = format!("{}{}", column_number_to_name(col), row);
                eval_errors.push(format!("{}!{}: {}", sheet_name, addr, err));
            }
        }

        let should_write = if incomplete {
            false
        } else if let Some(changed) = changed_filter {
            match engine
                .sheet_id(&sheet_name)
                .and_then(|sid| PackedSheetCell::try_from_excel_1based(sid, row, col))
            {
                Some(packed) => changed.contains(&packed),
                None => true,
            }
        } else {
            true
        };

        if should_write {
            cache_updates.push(FormulaCacheUpdate {
                sheet: sheet_name,
                row,
                col,
                value,
            });
        }
    }
    let build_updates_ms = build_updates_start.elapsed().as_millis() as u64;

    let updates_len = cache_updates.len();

    let mut write_formula_caches_batch_ms = 0u64;

    if !cache_updates.is_empty() {
        let write_start = Instant::now();
        adapter
            .write_formula_caches_batch(&cache_updates, date_system)
            .map_err(|e| anyhow!("failed to write formula caches in batch: {e}"))?;
        write_formula_caches_batch_ms = write_start.elapsed().as_millis() as u64;
    }
    let save_start = Instant::now();
    let (evaluated_bytes, revision_id) = match persistence {
        RecalcPersistence::Path(path) => {
            if !cache_updates.is_empty() {
                adapter
                    .save_as_path(path)
                    .map_err(|e| anyhow!("failed to save recalculated workbook {:?}: {e}", path))?;
            }
            (None, hash_file_sha256_hex(path)?)
        }
        RecalcPersistence::Bytes => {
            let bytes = adapter
                .save_to_bytes()
                .map_err(|e| anyhow!("failed to serialize recalculated workbook: {e}"))?;
            let revision = hash_bytes_sha256_hex(&bytes);
            (Some(bytes), revision)
        }
    };
    let save_as_path_ms = save_start.elapsed().as_millis() as u64;

    let total_ms = start.elapsed().as_millis() as u64;

    tracing::trace!(
        target: "asp::recalc::timing",
        open_ms,
        stream_into_engine_ms = stream_ms,
        evaluate_ms,
        build_updates_ms,
        write_formula_caches_batch_ms,
        save_as_path_ms,
        formula_cells_len,
        updates_len,
        total_ms,
        "formualizer recalc timing"
    );

    Ok((
        RecalcResult {
            duration_ms: total_ms,
            was_warm: true,
            backend_name: "formualizer",
            cells_evaluated: Some(cells_evaluated),
            eval_errors: if eval_errors.is_empty() {
                None
            } else {
                Some(eval_errors)
            },
            evaluation_coverage: crate::model::EvaluationCoverage {
                formula_cells: formula_cells_len as u64,
                evaluated_formula_cells: if incomplete {
                    0
                } else {
                    formula_cells_len as u64
                },
                unsupported_formula_cells: 0,
                error_formula_cells,
                source: crate::model::EvaluationSource::Formualizer,
                freshness: crate::model::EvaluationFreshness::CurrentRevision,
                revision_id,
            },
            incomplete,
        },
        evaluated_bytes,
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn evaluate_with_optional_timeout(
    engine: &mut FormualizerEngine,
    timeout_ms: Option<u64>,
) -> Result<(u64, u64, Option<HashSet<PackedSheetCell>>)> {
    let Some(timeout_ms) = timeout_ms else {
        let (eval, delta) = engine.evaluate_all_with_delta()?;
        let changed = delta.changed_cells.into_iter().collect::<HashSet<_>>();
        return Ok((
            eval.computed_vertices as u64,
            eval.cycle_errors as u64,
            Some(changed),
        ));
    };

    let cancel_flag = Arc::new(AtomicBool::new(false));
    let done_flag = Arc::new(AtomicBool::new(false));
    let cancel_for_thread = cancel_flag.clone();
    let done_for_thread = done_flag.clone();

    let handle = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        // Relaxed is sufficient: flag is monotonic false->true, no data synchronized.
        while !done_for_thread.load(Ordering::Relaxed) {
            if Instant::now() >= deadline {
                cancel_for_thread.store(true, Ordering::Relaxed);
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
    });

    let result = engine.evaluate_all_cancellable(cancel_flag.into());
    done_flag.store(true, Ordering::Relaxed);
    let _ = handle.join();

    let eval = result?;
    Ok((
        eval.computed_vertices as u64,
        eval.cycle_errors as u64,
        None,
    ))
}

#[cfg(target_arch = "wasm32")]
fn evaluate_with_optional_timeout(
    engine: &mut FormualizerEngine,
    _timeout_ms: Option<u64>,
) -> Result<(u64, u64, Option<HashSet<PackedSheetCell>>)> {
    // Browser and Node wasm32 have no portable preemptive thread primitive. Evaluation remains
    // synchronous and in-memory; callers can terminate a Web Worker for a hard deadline.
    let (eval, delta) = engine.evaluate_all_with_delta()?;
    let changed = delta.changed_cells.into_iter().collect::<HashSet<_>>();
    Ok((
        eval.computed_vertices as u64,
        eval.cycle_errors as u64,
        Some(changed),
    ))
}
