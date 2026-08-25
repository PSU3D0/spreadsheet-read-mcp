use crate::cli::{OutputFormat, OutputShape};
use crate::response_prune::prune_non_structural_empties;
use anyhow::{Result, bail};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactProjectionTarget {
    None,
    RangeValues,
    ReadTable,
    SheetPage,
    FormulaTrace,
}

pub fn emit_value(
    value: &Value,
    format: OutputFormat,
    shape: OutputShape,
    projection_target: CompactProjectionTarget,
    compact: bool,
    quiet: bool,
) -> Result<()> {
    if matches!(format, OutputFormat::Csv) {
        bail!("csv output is not implemented yet for agent-spreadsheet")
    }

    let mut value = value.clone();
    prune_non_structural_empties(&mut value);
    apply_shape(&mut value, shape, projection_target);

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    let _ = (compact, quiet);
    serde_json::to_writer(&mut handle, &value)?;
    use std::io::Write;
    handle.write_all(b"\n")?;
    Ok(())
}

fn apply_shape(value: &mut Value, shape: OutputShape, projection_target: CompactProjectionTarget) {
    if !matches!(shape, OutputShape::Compact) {
        return;
    }

    match projection_target {
        CompactProjectionTarget::None => {}
        CompactProjectionTarget::RangeValues => project_range_values_compact(value),
        CompactProjectionTarget::ReadTable => project_read_table_compact(value),
        CompactProjectionTarget::SheetPage => project_sheet_page_compact(value),
        CompactProjectionTarget::FormulaTrace => project_formula_trace_compact(value),
    }
}

fn project_range_values_compact(value: &mut Value) {
    // Keep range-values contract stable across canonical/compact. Global pruning
    // already removes empty wrappers before projection.
    let _ = value;
}

fn project_read_table_compact(value: &mut Value) {
    // Ticket 3109 contract: keep root object and active payload branch unchanged;
    // only drop empty wrappers when present.
    let Value::Object(obj) = value else {
        return;
    };
    drop_empty_wrappers(obj, &["headers", "types"]);
}

fn project_sheet_page_compact(value: &mut Value) {
    // Ticket 3109 contract: preserve active payload branch (`rows` | `compact` |
    // `values_only`) and continuation fields without flattening or collapsing.
    // Global pruning already removed empty wrappers before projection.
    let _ = value;
}

fn project_formula_trace_compact(value: &mut Value) {
    // Ticket 3109 contract: compact projection omits per-layer highlights while
    // preserving depth/summary/edges/has_more and top-level continuation fields.
    let Some(layers) = value.get_mut("layers").and_then(Value::as_array_mut) else {
        return;
    };

    for layer in layers {
        if let Value::Object(layer_obj) = layer {
            layer_obj.remove("highlights");
        }
    }
}

fn drop_empty_wrappers(target: &mut Map<String, Value>, keys: &[&str]) {
    for key in keys {
        let should_remove = target
            .get(*key)
            .map(|value| {
                value.is_null()
                    || value.as_array().is_some_and(Vec::is_empty)
                    || value.as_object().is_some_and(Map::is_empty)
            })
            .unwrap_or(false);
        if should_remove {
            target.remove(*key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compact_shape_3109_formula_trace_omits_highlights_only_for_trace_target() {
        let mut compact_payload = json!({
            "workbook_id": "wb",
            "sheet_name": "Sheet1",
            "origin": "A1",
            "direction": "dependents",
            "next_cursor": { "depth": 1, "offset": 5 },
            "notes": ["note"],
            "layers": [{
                "depth": 1,
                "summary": { "total_nodes": 1 },
                "highlights": { "top_ranges": [], "top_formula_groups": [], "notable_cells": [] },
                "edges": [{ "from": "A1", "to": "B1" }],
                "has_more": true
            }]
        });

        apply_shape(
            &mut compact_payload,
            OutputShape::Compact,
            CompactProjectionTarget::FormulaTrace,
        );

        let layer = compact_payload["layers"]
            .as_array()
            .expect("layers array")
            .first()
            .cloned()
            .expect("first layer");
        assert!(layer.get("highlights").is_none());
        assert_eq!(layer["depth"], json!(1));
        assert!(layer.get("summary").is_some());
        assert!(layer.get("edges").is_some());
        assert_eq!(layer["has_more"], json!(true));
        assert_eq!(compact_payload["next_cursor"]["depth"], json!(1));
        assert_eq!(compact_payload["next_cursor"]["offset"], json!(5));
    }

    #[test]
    fn compact_shape_3109_none_target_does_not_apply_trace_projection() {
        let mut payload = json!({
            "layers": [{
                "depth": 1,
                "summary": { "total_nodes": 1 },
                "highlights": { "top_ranges": [] },
                "edges": [],
                "has_more": false
            }]
        });

        apply_shape(
            &mut payload,
            OutputShape::Compact,
            CompactProjectionTarget::None,
        );

        let layer = payload["layers"]
            .as_array()
            .expect("layers array")
            .first()
            .expect("first layer");
        assert!(layer.get("highlights").is_some());
    }

    #[test]
    fn compact_shape_3109_range_values_keeps_stable_shape() {
        let base_payload = json!({
            "workbook_id": "wb",
            "sheet_name": "Sheet1",
            "values": [{
                "range": "A1:B2",
                "rows": [[{"kind":"text","value":"x"}]],
                "next_start_row": 3
            }]
        });

        let mut compact_range_values = base_payload.clone();
        apply_shape(
            &mut compact_range_values,
            OutputShape::Compact,
            CompactProjectionTarget::RangeValues,
        );
        assert!(compact_range_values.get("values").is_some());
        assert!(compact_range_values.get("range").is_none());

        let mut compact_none_target = base_payload;
        apply_shape(
            &mut compact_none_target,
            OutputShape::Compact,
            CompactProjectionTarget::None,
        );
        assert!(compact_none_target.get("values").is_some());
        assert!(compact_none_target.get("range").is_none());
    }
}
