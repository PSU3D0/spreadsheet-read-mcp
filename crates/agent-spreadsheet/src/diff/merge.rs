use super::cells::RawCell;
use anyhow::Result;
use schemars::JsonSchema;
use serde::Serialize;
use std::cmp::Ordering;

#[derive(Debug, Serialize, Clone, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CellDiff {
    Added {
        address: String,
        value: Option<String>,
        formula: Option<String>,
    },
    Deleted {
        address: String,
        old_value: Option<String>,
    },
    Modified {
        address: String,
        subtype: ModificationType,
        old_value: Option<String>,
        new_value: Option<String>,
        old_formula: Option<String>,
        new_formula: Option<String>,
        old_style_id: Option<u32>,
        new_style_id: Option<u32>,
    },
}

#[derive(Debug, Serialize, Clone, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModificationType {
    FormulaEdit,
    RecalcResult,
    ValueEdit,
    StyleEdit,
}

pub fn diff_streams(
    base: impl Iterator<Item = Result<RawCell>>,
    fork: impl Iterator<Item = Result<RawCell>>,
) -> Result<Vec<CellDiff>> {
    let mut diffs = Vec::new();
    let mut base_iter = base.peekable();
    let mut fork_iter = fork.peekable();

    loop {
        // Handle errors in stream
        if let Some(Err(_)) = base_iter.peek() {
            return Err(base_iter.next().unwrap().unwrap_err());
        }
        if let Some(Err(_)) = fork_iter.peek() {
            return Err(fork_iter.next().unwrap().unwrap_err());
        }

        // Get references to Ok items
        let b_opt = base_iter.peek().map(|r| r.as_ref().unwrap());
        let f_opt = fork_iter.peek().map(|r| r.as_ref().unwrap());

        match (b_opt, f_opt) {
            (None, None) => break,
            (Some(b), None) => {
                diffs.push(CellDiff::Deleted {
                    address: b.address.original.clone(),
                    old_value: b.value.clone(),
                });
                base_iter.next();
            }
            (None, Some(f)) => {
                diffs.push(CellDiff::Added {
                    address: f.address.original.clone(),
                    value: f.value.clone(),
                    formula: f.formula.clone(),
                });
                fork_iter.next();
            }
            (Some(b), Some(f)) => {
                match b.address.cmp(&f.address) {
                    Ordering::Less => {
                        // Base is behind -> Deleted
                        diffs.push(CellDiff::Deleted {
                            address: b.address.original.clone(),
                            old_value: b.value.clone(),
                        });
                        base_iter.next();
                    }
                    Ordering::Greater => {
                        // Fork is behind -> Added
                        diffs.push(CellDiff::Added {
                            address: f.address.original.clone(),
                            value: f.value.clone(),
                            formula: f.formula.clone(),
                        });
                        fork_iter.next();
                    }
                    Ordering::Equal => {
                        // Same address -> Compare
                        if let Some(diff) = compare_cells(b, f) {
                            diffs.push(diff);
                        }
                        base_iter.next();
                        fork_iter.next();
                    }
                }
            }
        }
    }

    Ok(diffs)
}

fn compare_cells(base: &RawCell, fork: &RawCell) -> Option<CellDiff> {
    let formula_changed = base.formula != fork.formula;
    let value_changed = !values_equal(&base.value, &fork.value);
    let style_changed = base.style_id != fork.style_id;

    if !formula_changed && !value_changed && !style_changed {
        return None;
    }

    let subtype = if style_changed && !formula_changed && !value_changed {
        ModificationType::StyleEdit
    } else {
        match (formula_changed, value_changed, fork.formula.is_some()) {
            (true, _, _) => ModificationType::FormulaEdit,
            (false, true, true) => ModificationType::RecalcResult,
            (false, true, false) => ModificationType::ValueEdit,
            _ => return None, // Should be covered above
        }
    };

    Some(CellDiff::Modified {
        address: fork.address.original.clone(),
        subtype,
        old_value: base.value.clone(),
        new_value: fork.value.clone(),
        old_formula: base.formula.clone(),
        new_formula: fork.formula.clone(),
        old_style_id: if style_changed { base.style_id } else { None },
        new_style_id: if style_changed { fork.style_id } else { None },
    })
}

fn values_equal(a: &Option<String>, b: &Option<String>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            // Try numeric comparison with epsilon
            if let (Ok(fa), Ok(fb)) = (a.parse::<f64>(), b.parse::<f64>()) {
                (fa - fb).abs() < 1e-9
            } else {
                a == b
            }
        }
        _ => false,
    }
}
