use crate::model::{RegionKind, SheetClassification, SheetRegion};
use crate::utils::column_number_to_name;
use crate::workbook::{SheetMetrics, StyleUsage};
use std::collections::HashMap;

pub fn classify(
    non_empty: u32,
    formulas: u32,
    rows: u32,
    columns: u32,
    comments: u32,
    _styles: &HashMap<String, StyleUsage>,
) -> SheetClassification {
    if non_empty == 0 {
        return SheetClassification::Empty;
    }
    let formula_ratio = if non_empty == 0 {
        0.0
    } else {
        formulas as f32 / non_empty as f32
    };
    if formula_ratio > 0.7 {
        SheetClassification::Calculator
    } else if formula_ratio > 0.2 {
        SheetClassification::Mixed
    } else if rows < 5 || columns < 3 || comments > 10 {
        SheetClassification::Metadata
    } else {
        SheetClassification::Data
    }
}

pub fn narrative(metrics: &SheetMetrics) -> String {
    let formula_ratio = if metrics.non_empty_cells == 0 {
        0.0
    } else {
        metrics.formula_cells as f32 / metrics.non_empty_cells as f32
    };

    format!(
        "{} sheet with {} rows, {} columns, {:.0}% formulas, {} style clusters",
        match metrics.classification {
            SheetClassification::Data => "Data-centric",
            SheetClassification::Calculator => "Calculator",
            SheetClassification::Mixed => "Mixed-use",
            SheetClassification::Metadata => "Metadata",
            SheetClassification::Empty => "Empty",
        },
        metrics.row_count,
        metrics.column_count,
        formula_ratio * 100.0,
        metrics.style_map.len()
    )
}

pub fn regions(metrics: &SheetMetrics) -> Vec<SheetRegion> {
    if metrics.non_empty_cells == 0 {
        return vec![];
    }
    let mut regions = Vec::new();
    let end_col = column_number_to_name(metrics.column_count.max(1));
    let end_cell = format!("{}{}", end_col, metrics.row_count.max(1));

    let kind = match metrics.classification {
        SheetClassification::Calculator => RegionKind::Calculator,
        SheetClassification::Metadata => RegionKind::Metadata,
        _ => RegionKind::Data,
    };

    regions.push(SheetRegion {
        kind,
        address: format!("A1:{}", end_cell),
        description: format!(
            "Primary region covering {:.0}% of sheet cells",
            density(metrics) * 100.0
        ),
    });
    regions
}

pub fn key_ranges(metrics: &SheetMetrics) -> Vec<String> {
    if metrics.non_empty_cells == 0 {
        return vec![];
    }

    let mut ranges = Vec::new();
    ranges.push("Header band likely in row 1".to_string());
    if matches!(metrics.classification, SheetClassification::Calculator) {
        ranges.push("Check final output cells near bottom rows".to_string());
    }
    ranges
}

fn density(metrics: &SheetMetrics) -> f32 {
    let total = (metrics.row_count.max(1) * metrics.column_count.max(1)) as f32;
    if total == 0.0 {
        0.0
    } else {
        metrics.non_empty_cells as f32 / total
    }
}
