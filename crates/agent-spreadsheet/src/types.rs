use serde::Serialize;

#[derive(Debug, Clone)]
pub struct CellEdit {
    pub address: String,
    pub value: String,
    pub is_formula: bool,
}

#[derive(Debug, Clone)]
pub struct CoreWarning {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct BasicDiffChange {
    pub sheet: String,
    pub address: String,
    pub change_type: String,
    pub original_value: Option<String>,
    pub original_formula: Option<String>,
    pub modified_value: Option<String>,
    pub modified_formula: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BasicDiffResponse {
    pub original: String,
    pub modified: String,
    pub change_count: usize,
    pub changes: Vec<BasicDiffChange>,
}

#[derive(Debug, Clone)]
pub struct RecalculateOutcome {
    pub backend: String,
    pub duration_ms: u64,
    pub cells_evaluated: Option<u64>,
    pub eval_errors: Option<Vec<String>>,
}
