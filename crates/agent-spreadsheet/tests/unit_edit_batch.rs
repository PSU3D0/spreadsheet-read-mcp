#![cfg(feature = "recalc")]

use serde_json::json;
use agent_spreadsheet::tools::write_normalize::{EditBatchParamsInput, normalize_edit_batch};

#[test]
fn edit_batch_accepts_shorthand_and_formula() {
    let input = json!({
        "fork_id": "f1",
        "sheet_name": "Inputs",
        "edits": [
            "A1=Hello",
            "B2==SUM(A3:A4)",
            { "address": "C3", "formula": "SUM(A1:A2)" },
            { "address": "D4", "value": "=NOT_A_FORMULA" },
            { "address": "E5", "value": "=ALWAYS_FORMULA", "is_formula": false },
            { "address": "F6", "formula": "=SUM(B1:B2)" }
        ]
    });

    let params: EditBatchParamsInput = serde_json::from_value(input).unwrap();
    let (normalized, warnings) = normalize_edit_batch(params).unwrap();

    assert_eq!(normalized.edits[0].address, "A1");
    assert_eq!(normalized.edits[0].value, "Hello");
    assert!(!normalized.edits[0].is_formula);

    assert_eq!(normalized.edits[1].address, "B2");
    assert_eq!(normalized.edits[1].value, "SUM(A3:A4)");
    assert!(normalized.edits[1].is_formula);

    assert_eq!(normalized.edits[2].value, "SUM(A1:A2)");
    assert!(normalized.edits[2].is_formula);

    assert_eq!(normalized.edits[4].value, "ALWAYS_FORMULA");
    assert!(normalized.edits[4].is_formula);

    assert_eq!(normalized.edits[5].value, "SUM(B1:B2)");
    assert!(normalized.edits[5].is_formula);

    // Noise warnings on documented syntax were removed; edits parse silently.
    assert!(warnings.is_empty());
}

#[test]
fn edit_batch_rejects_shorthand_without_equals() {
    let input = json!({
        "fork_id": "f1",
        "sheet_name": "Inputs",
        "edits": ["A1"]
    });

    let params: EditBatchParamsInput = serde_json::from_value(input).unwrap();
    let err = normalize_edit_batch(params).unwrap_err();

    assert!(err.to_string().contains("invalid shorthand edit"));
}

#[test]
fn edit_batch_shorthand_allows_space_before_formula() {
    let input = json!({
        "fork_id": "f1",
        "sheet_name": "Inputs",
        "edits": ["A1 = =SUM(A1:A2)"]
    });

    let params: EditBatchParamsInput = serde_json::from_value(input).unwrap();
    let (normalized, warnings) = normalize_edit_batch(params).unwrap();

    assert_eq!(normalized.edits[0].address, "A1");
    assert_eq!(normalized.edits[0].value, "SUM(A1:A2)");
    assert!(normalized.edits[0].is_formula);
    assert!(warnings.is_empty());
}
