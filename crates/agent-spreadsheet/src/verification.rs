use crate::model::{CellValue, NamedItemKind, NamedRangeDescriptor, NamedRangeScope};
use crate::workbook::{WorkbookContext, cell_to_value};
use anyhow::{Result, anyhow, bail};
use schemars::JsonSchema;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, JsonSchema, Default)]
pub struct TargetClassificationCounts {
    pub unchanged: u32,
    pub direct_edit: u32,
    pub recalc_result: u32,
    pub formula_shift: u32,
    pub new_error: u32,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct VerifySummary {
    pub target_count: u32,
    pub changed_targets: u32,
    pub new_error_count: u32,
    pub resolved_error_count: u32,
    pub preexisting_error_count: u32,
    pub named_range_delta_count: u32,
    pub target_classification_counts: TargetClassificationCounts,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TargetDelta {
    pub address: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<CellValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after: Option<CellValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_formula: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_formula: Option<String>,
    pub classification: String,
    pub changed: bool,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ErrorDelta {
    pub address: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_formula: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_formula: Option<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct NamedRangeDelta {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope_kind: Option<NamedRangeScope>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope_sheet_name: Option<String>,
    pub change: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_refers_to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_refers_to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_kind: Option<NamedItemKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_kind: Option<NamedItemKind>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct VerifyResponse {
    pub baseline: String,
    pub current: String,
    pub target_deltas: Vec<TargetDelta>,
    pub new_errors: Vec<ErrorDelta>,
    pub resolved_errors: Vec<ErrorDelta>,
    pub preexisting_errors: Vec<ErrorDelta>,
    pub named_range_deltas: Vec<NamedRangeDelta>,
    pub summary: VerifySummary,
}

#[derive(Debug, Clone, Default)]
pub struct VerifyOptions {
    pub targets: Vec<String>,
    pub sheet_filter: Option<String>,
    pub include_named_range_deltas: bool,
    pub errors_only: bool,
    pub targets_only: bool,
}

#[derive(Debug, Clone)]
struct TargetCellSnapshot {
    value: Option<CellValue>,
    formula: Option<String>,
}

#[derive(Debug, Clone)]
struct ErrorCellSnapshot {
    error: String,
    formula: Option<String>,
}

impl VerifyOptions {
    pub fn validate(&self) -> Result<()> {
        if self.errors_only && self.targets_only {
            bail!("invalid argument: --errors-only and --targets-only cannot be combined");
        }
        if self.errors_only && !self.targets.is_empty() {
            bail!(
                "invalid argument: --errors-only cannot be combined with explicit --targets; drop --targets or use default verify mode"
            );
        }
        if self.errors_only && self.include_named_range_deltas {
            bail!(
                "invalid argument: --errors-only cannot be combined with --named-ranges; use default verify mode to include named-range deltas"
            );
        }
        if self.targets_only && self.targets.is_empty() {
            bail!(
                "invalid argument: --targets-only requires --targets Sheet!A1,... to define the proof scope"
            );
        }
        if self.targets_only && self.include_named_range_deltas {
            bail!(
                "invalid argument: --targets-only cannot be combined with --named-ranges; use default verify mode to include named-range deltas"
            );
        }
        Ok(())
    }
}

pub fn compare_workbooks(
    baseline_label: impl Into<String>,
    current_label: impl Into<String>,
    baseline: &WorkbookContext,
    current: &WorkbookContext,
    options: &VerifyOptions,
    baseline_named_ranges: Option<&[NamedRangeDescriptor]>,
    current_named_ranges: Option<&[NamedRangeDescriptor]>,
) -> Result<VerifyResponse> {
    options.validate()?;

    let target_deltas = if options.errors_only {
        Vec::new()
    } else {
        collect_target_deltas(baseline, current, options.targets.clone())?
    };

    let (new_errors, resolved_errors, preexisting_errors) = if options.targets_only {
        (Vec::new(), Vec::new(), Vec::new())
    } else {
        let baseline_errors = collect_error_cells(baseline, options.sheet_filter.as_deref())?;
        let current_errors = collect_error_cells(current, options.sheet_filter.as_deref())?;
        compare_error_maps(&baseline_errors, &current_errors)
    };

    let named_range_deltas = if options.include_named_range_deltas {
        let baseline_named_ranges = baseline_named_ranges.ok_or_else(|| {
            anyhow!("internal error: baseline named ranges were not loaded for verification")
        })?;
        let current_named_ranges = current_named_ranges.ok_or_else(|| {
            anyhow!("internal error: current named ranges were not loaded for verification")
        })?;
        compare_named_ranges(baseline_named_ranges, current_named_ranges)
    } else {
        Vec::new()
    };

    let target_classification_counts = count_target_classifications(&target_deltas);
    Ok(VerifyResponse {
        baseline: baseline_label.into(),
        current: current_label.into(),
        summary: VerifySummary {
            target_count: target_deltas.len() as u32,
            changed_targets: target_deltas.iter().filter(|d| d.changed).count() as u32,
            new_error_count: new_errors.len() as u32,
            resolved_error_count: resolved_errors.len() as u32,
            preexisting_error_count: preexisting_errors.len() as u32,
            named_range_delta_count: named_range_deltas.len() as u32,
            target_classification_counts,
        },
        target_deltas,
        new_errors,
        resolved_errors,
        preexisting_errors,
        named_range_deltas,
    })
}

fn count_target_classifications(target_deltas: &[TargetDelta]) -> TargetClassificationCounts {
    let mut counts = TargetClassificationCounts::default();
    for delta in target_deltas {
        match delta.classification.as_str() {
            "unchanged" => counts.unchanged += 1,
            "direct_edit" => counts.direct_edit += 1,
            "recalc_result" => counts.recalc_result += 1,
            "formula_shift" => counts.formula_shift += 1,
            "new_error" => counts.new_error += 1,
            _ => {}
        }
    }
    counts
}

fn collect_target_deltas(
    baseline: &WorkbookContext,
    current: &WorkbookContext,
    targets: Vec<String>,
) -> Result<Vec<TargetDelta>> {
    let mut deltas = Vec::new();
    for target in targets {
        let (sheet_name, cell_ref) = parse_sheet_cell_ref(&target)?;
        let before = read_target_cell(baseline, &sheet_name, &cell_ref)?;
        let after = read_target_cell(current, &sheet_name, &cell_ref)?;
        let changed = !cell_values_equal(before.value.as_ref(), after.value.as_ref())
            || before.formula != after.formula;
        let classification = classify_target_delta(&before, &after, changed).to_string();
        deltas.push(TargetDelta {
            address: target,
            before: before.value,
            after: after.value,
            before_formula: before.formula,
            after_formula: after.formula,
            classification,
            changed,
        });
    }
    Ok(deltas)
}

fn parse_sheet_cell_ref(raw: &str) -> Result<(String, String)> {
    let (sheet_name, cell_ref) = raw.rsplit_once('!').ok_or_else(|| {
        anyhow!(
            "invalid argument: target '{}' must use Sheet!A1 notation",
            raw
        )
    })?;
    if sheet_name.trim().is_empty() || cell_ref.trim().is_empty() {
        bail!(
            "invalid argument: target '{}' must use Sheet!A1 notation",
            raw
        );
    }

    let sheet_name = extract_sheet_name(sheet_name);
    let cell_ref = parse_target_cell_ref(raw, cell_ref)?;
    Ok((sheet_name, cell_ref))
}

fn extract_sheet_name(raw: &str) -> String {
    let trimmed = raw.trim();
    if let Some(stripped) = trimmed.strip_prefix('\'')
        && let Some(inner) = stripped.strip_suffix('\'')
    {
        return inner.replace("''", "'");
    }
    trimmed.to_string()
}

fn parse_target_cell_ref(target: &str, raw_cell_ref: &str) -> Result<String> {
    let cell_ref = raw_cell_ref.trim();
    let (col, row, _, _) = umya_spreadsheet::helper::coordinate::index_from_coordinate(cell_ref);
    match (col, row) {
        (Some(c), Some(r)) if c > 0 && r > 0 => Ok(cell_ref.to_string()),
        _ => bail!(
            "invalid argument: target '{}' must use Sheet!A1 notation with a single A1 cell reference",
            target
        ),
    }
}

fn read_target_cell(
    workbook: &WorkbookContext,
    sheet_name: &str,
    cell_ref: &str,
) -> Result<TargetCellSnapshot> {
    workbook.with_sheet(sheet_name, |sheet| {
        if let Some(cell) = sheet.get_cell(cell_ref) {
            let formula = non_empty_formula(cell.get_formula());
            TargetCellSnapshot {
                value: cell_to_value(cell),
                formula,
            }
        } else {
            TargetCellSnapshot {
                value: None,
                formula: None,
            }
        }
    })
}

fn collect_error_cells(
    workbook: &WorkbookContext,
    sheet_filter: Option<&str>,
) -> Result<BTreeMap<String, ErrorCellSnapshot>> {
    let mut out = BTreeMap::new();
    let sheet_names = if let Some(sheet_name) = sheet_filter {
        vec![sheet_name.to_string()]
    } else {
        workbook.sheet_names()
    };

    for sheet_name in sheet_names {
        let sheet_errors = workbook.with_sheet(&sheet_name, |sheet| {
            let mut items = Vec::new();
            for cell in sheet.get_cell_collection() {
                let raw = cell.get_value();
                if !is_error_text(&raw) {
                    continue;
                }
                let address = format!("{}!{}", sheet_name, cell.get_coordinate().get_coordinate());
                items.push((
                    address,
                    ErrorCellSnapshot {
                        error: raw.to_string(),
                        formula: non_empty_formula(cell.get_formula()),
                    },
                ));
            }
            items
        })?;
        for (address, snapshot) in sheet_errors {
            out.insert(address, snapshot);
        }
    }
    Ok(out)
}

fn compare_error_maps(
    baseline: &BTreeMap<String, ErrorCellSnapshot>,
    current: &BTreeMap<String, ErrorCellSnapshot>,
) -> (Vec<ErrorDelta>, Vec<ErrorDelta>, Vec<ErrorDelta>) {
    let mut new_errors = Vec::new();
    let mut resolved_errors = Vec::new();
    let mut preexisting_errors = Vec::new();

    for (address, after) in current {
        if let Some(before) = baseline.get(address) {
            preexisting_errors.push(ErrorDelta {
                address: address.clone(),
                before_error: Some(before.error.clone()),
                after_error: Some(after.error.clone()),
                before_formula: before.formula.clone(),
                after_formula: after.formula.clone(),
            });
        } else {
            new_errors.push(ErrorDelta {
                address: address.clone(),
                before_error: None,
                after_error: Some(after.error.clone()),
                before_formula: None,
                after_formula: after.formula.clone(),
            });
        }
    }

    for (address, before) in baseline {
        if !current.contains_key(address) {
            resolved_errors.push(ErrorDelta {
                address: address.clone(),
                before_error: Some(before.error.clone()),
                after_error: None,
                before_formula: before.formula.clone(),
                after_formula: None,
            });
        }
    }

    (new_errors, resolved_errors, preexisting_errors)
}

fn compare_named_ranges(
    baseline: &[NamedRangeDescriptor],
    current: &[NamedRangeDescriptor],
) -> Vec<NamedRangeDelta> {
    let base_map: BTreeMap<String, &NamedRangeDescriptor> = baseline
        .iter()
        .map(|item| (named_range_key(item), item))
        .collect();
    let current_map: BTreeMap<String, &NamedRangeDescriptor> = current
        .iter()
        .map(|item| (named_range_key(item), item))
        .collect();

    let keys: BTreeSet<String> = base_map
        .keys()
        .cloned()
        .chain(current_map.keys().cloned())
        .collect();

    let mut deltas = Vec::new();
    for key in keys {
        let before = base_map.get(&key).copied();
        let after = current_map.get(&key).copied();
        match (before, after) {
            (Some(b), Some(a)) if b.refers_to != a.refers_to || b.kind != a.kind => {
                deltas.push(NamedRangeDelta {
                    name: a.name.clone(),
                    scope_kind: a.scope_kind,
                    scope_sheet_name: a.scope_sheet_name.clone(),
                    change: "changed".to_string(),
                    before_refers_to: Some(b.refers_to.clone()),
                    after_refers_to: Some(a.refers_to.clone()),
                    before_kind: Some(b.kind.clone()),
                    after_kind: Some(a.kind.clone()),
                });
            }
            (Some(b), None) => {
                deltas.push(NamedRangeDelta {
                    name: b.name.clone(),
                    scope_kind: b.scope_kind,
                    scope_sheet_name: b.scope_sheet_name.clone(),
                    change: "removed".to_string(),
                    before_refers_to: Some(b.refers_to.clone()),
                    after_refers_to: None,
                    before_kind: Some(b.kind.clone()),
                    after_kind: None,
                });
            }
            (None, Some(a)) => {
                deltas.push(NamedRangeDelta {
                    name: a.name.clone(),
                    scope_kind: a.scope_kind,
                    scope_sheet_name: a.scope_sheet_name.clone(),
                    change: "added".to_string(),
                    before_refers_to: None,
                    after_refers_to: Some(a.refers_to.clone()),
                    before_kind: None,
                    after_kind: Some(a.kind.clone()),
                });
            }
            _ => {}
        }
    }

    deltas
}

fn named_range_key(item: &NamedRangeDescriptor) -> String {
    format!(
        "{}|{:?}|{}|{:?}",
        item.name,
        item.scope_kind,
        item.scope_sheet_name.as_deref().unwrap_or(""),
        item.kind
    )
}

fn classify_target_delta(
    before: &TargetCellSnapshot,
    after: &TargetCellSnapshot,
    changed: bool,
) -> &'static str {
    if !changed {
        return "unchanged";
    }
    if is_error_value(after.value.as_ref()) && !is_error_value(before.value.as_ref()) {
        return "new_error";
    }
    if before.formula != after.formula {
        return "formula_shift";
    }
    if before.formula.is_none() && after.formula.is_none() {
        return "direct_edit";
    }
    "recalc_result"
}

fn is_error_value(value: Option<&CellValue>) -> bool {
    match value {
        Some(CellValue::Error(_)) => true,
        Some(other) => serde_json::to_value(other)
            .ok()
            .and_then(|json| {
                json.get("value")
                    .and_then(|v| v.as_str())
                    .map(is_error_text)
            })
            .unwrap_or(false),
        None => false,
    }
}

fn cell_values_equal(left: Option<&CellValue>, right: Option<&CellValue>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(l), Some(r)) => serde_json::to_value(l).ok() == serde_json::to_value(r).ok(),
        _ => false,
    }
}

fn non_empty_formula(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn is_error_text(raw: &str) -> bool {
    let upper = raw.trim().to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "#DIV/0!"
            | "#VALUE!"
            | "#NAME?"
            | "#REF!"
            | "#N/A"
            | "#NULL!"
            | "#NUM!"
            | "#SPILL!"
            | "#CALC!"
            | "#BUSY!"
            | "#FIELD!"
            | "#UNKNOWN!"
    )
}
