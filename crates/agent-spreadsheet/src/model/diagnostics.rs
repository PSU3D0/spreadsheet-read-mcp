use formualizer_parse::parser::ParserError;
use formualizer_parse::tokenizer::{
    RecoveryAction, TokenDiagnostic, TokenStream, TokenSubType, TokenType,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const MAX_GROUPS: usize = 50;
const MAX_SAMPLE_ADDRESSES: usize = 5;
const FORMULA_PREVIEW_MAX_BYTES: usize = 80;

pub const FORMULA_PARSE_FAILED: &str = "FORMULA_PARSE_FAILED";
pub const FORMULA_PARSE_FAILED_PREFIX: &str = "formula parse failed: ";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, clap::ValueEnum, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum FormulaParsePolicy {
    /// Abort on any formula parse failure.
    Fail,
    /// Continue but collect diagnostics.
    #[default]
    Warn,
    /// Skip silently.
    Off,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CommandClass {
    SingleWrite,
    BatchWrite,
    ReadAnalysis,
}

impl FormulaParsePolicy {
    pub fn default_for_command_class(class: CommandClass) -> Self {
        match class {
            CommandClass::SingleWrite => FormulaParsePolicy::Fail,
            CommandClass::BatchWrite | CommandClass::ReadAnalysis => FormulaParsePolicy::Warn,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FormulaParseErrorGroup {
    pub error_code: String,
    pub error_message: String,
    pub sheet_name: String,
    pub formula_preview: String,
    pub count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sample_addresses: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FormulaParseDiagnostics {
    pub policy: FormulaParsePolicy,
    pub total_errors: usize,
    pub groups_truncated: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<FormulaParseErrorGroup>,
}

pub struct FormulaParseDiagnosticsBuilder {
    policy: FormulaParsePolicy,
    groups: BTreeMap<(String, String, String), GroupAccumulator>,
    total_errors: usize,
}

struct GroupAccumulator {
    error_code: String,
    error_message: String,
    formula_preview: String,
    count: usize,
    sample_addresses: Vec<String>,
}

impl FormulaParseDiagnosticsBuilder {
    pub fn new(policy: FormulaParsePolicy) -> Self {
        Self {
            policy,
            groups: BTreeMap::new(),
            total_errors: 0,
        }
    }

    pub fn record_error(&mut self, sheet: &str, address: &str, formula: &str, error: &str) {
        let formula_preview = truncate_formula_preview(formula);
        let normalized_formula = normalize_formula_for_grouping(formula);
        let normalized_error = normalize_error_for_grouping(error);
        let key = (sheet.to_string(), normalized_error, normalized_formula);

        self.total_errors += 1;

        let group = self.groups.entry(key).or_insert_with(|| GroupAccumulator {
            error_code: FORMULA_PARSE_FAILED.to_string(),
            error_message: error.to_string(),
            formula_preview,
            count: 0,
            sample_addresses: Vec::new(),
        });

        group.count += 1;
        if group.sample_addresses.len() < MAX_SAMPLE_ADDRESSES {
            group.sample_addresses.push(address.to_string());
        }
    }

    pub fn build(self) -> FormulaParseDiagnostics {
        let groups_truncated = self.groups.len() > MAX_GROUPS;
        let groups = self
            .groups
            .into_iter()
            .take(MAX_GROUPS)
            .map(
                |((sheet_name, _error_message, _normalized_key), group)| FormulaParseErrorGroup {
                    error_code: group.error_code,
                    error_message: group.error_message,
                    sheet_name,
                    formula_preview: group.formula_preview,
                    count: group.count,
                    sample_addresses: group.sample_addresses,
                },
            )
            .collect();

        FormulaParseDiagnostics {
            policy: self.policy,
            total_errors: self.total_errors,
            groups_truncated,
            groups,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.total_errors == 0
    }

    pub fn has_errors(&self) -> bool {
        self.total_errors > 0
    }
}

fn truncate_formula_preview(formula: &str) -> String {
    if formula.len() <= FORMULA_PREVIEW_MAX_BYTES {
        return formula.to_string();
    }

    let mut end = FORMULA_PREVIEW_MAX_BYTES;
    while end > 0 && !formula.is_char_boundary(end) {
        end -= 1;
    }

    let mut result = formula[..end].to_string();
    result.push('…');
    result
}

/// Normalize a formula for grouping by replacing cell/range references with
/// `$REF`. This collapses formulas that differ only in cell addresses (e.g.
/// `=IF(C4="",...)` and `=IF(C5="",...)`) into the same group key.
fn normalize_formula_for_grouping(formula: &str) -> String {
    if let Ok(stream) = TokenStream::new(formula) {
        let mut out = String::with_capacity(formula.len());
        if formula.starts_with('=') {
            out.push('=');
        }
        for span in &stream.spans {
            if span.token_type == TokenType::Operand && span.subtype == TokenSubType::Range {
                out.push_str("$REF");
            } else if let Some(val) = stream.source().get(span.start..span.end) {
                out.push_str(val);
            }
        }
        return truncate_formula_preview(&out);
    }

    // Fallback for unparsable formulas: use a simple regex-style substitution
    // to replace cell-like references (e.g. A1, $C$10, Sheet1!B2:C5).
    normalize_refs_regex(formula)
}

/// Regex-free cell reference normalization for malformed formulas.
/// Replaces patterns like A1, $B$2, C10, AA100 with $REF.
fn normalize_refs_regex(formula: &str) -> String {
    let bytes = formula.as_bytes();
    let mut out = String::with_capacity(formula.len());
    let mut i = 0;

    while i < bytes.len() {
        // Skip dollar signs that prefix column/row references
        let start = i;
        if bytes[i] == b'$' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_alphabetic() {
            // possible absolute ref like $A$1
        }

        // Try to match a cell reference: optional $, 1-3 alpha, optional $, 1+ digit
        let mut j = i;
        // skip leading $
        if j < bytes.len() && bytes[j] == b'$' {
            j += 1;
        }
        // require 1-3 alpha chars (column)
        let col_start = j;
        while j < bytes.len() && bytes[j].is_ascii_alphabetic() && j - col_start < 4 {
            j += 1;
        }
        let col_len = j - col_start;
        if (1..=3).contains(&col_len) {
            // skip optional $ before row
            if j < bytes.len() && bytes[j] == b'$' {
                j += 1;
            }
            // require 1+ digits (row)
            let row_start = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            let row_len = j - row_start;
            if row_len >= 1 {
                // Ensure this isn't part of a larger identifier (e.g. function name)
                let preceded_by_alpha = start > 0
                    && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_');
                let followed_by_alpha =
                    j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_');
                if !preceded_by_alpha && !followed_by_alpha {
                    out.push_str("$REF");
                    i = j;
                    continue;
                }
            }
        }

        out.push(bytes[i] as char);
        i += 1;
    }

    truncate_formula_preview(&out)
}

/// Normalize an error message for grouping by replacing numeric position/range
/// fields with placeholders.
fn normalize_error_for_grouping(error: &str) -> String {
    let mut out = String::with_capacity(error.len());
    let bytes = error.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Normalize "position <digits>" -> "position N"
        if i + 9 <= bytes.len() && &bytes[i..i + 9] == b"position " {
            out.push_str("position ");
            i += 9;
            if i < bytes.len() && bytes[i].is_ascii_digit() {
                out.push('N');
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                continue;
            }
        }

        // Normalize "bytes <digits>..<digits>" -> "bytes N..N"
        if i + 6 <= bytes.len() && &bytes[i..i + 6] == b"bytes " {
            let mut cursor = i + 6;
            if cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                out.push_str("bytes N");
                while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                    cursor += 1;
                }
                if cursor + 2 <= bytes.len() && &bytes[cursor..cursor + 2] == b".." {
                    cursor += 2;
                    out.push_str("..N");
                    while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                        cursor += 1;
                    }
                }
                i = cursor;
                continue;
            }
        }

        out.push(bytes[i] as char);
        i += 1;
    }

    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormulaParseFailure {
    pub parser_message: String,
    pub parser_position: Option<usize>,
    pub tokenizer: Option<TokenizerRecovery>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerRecovery {
    pub message: String,
    pub recovery: RecoveryAction,
    pub span_start: usize,
    pub span_end: usize,
}

impl FormulaParseFailure {
    fn from_parser_error(formula: &str, err: &ParserError) -> Self {
        let tokenizer = first_tokenizer_recovery(formula);
        Self {
            parser_message: err.message.clone(),
            parser_position: err.position,
            tokenizer,
        }
    }

    fn render_message(&self) -> String {
        let parser_message = match self.parser_position {
            Some(pos) => format!("parse error at position {pos}: {}", self.parser_message),
            None => format!("parse error: {}", self.parser_message),
        };

        if let Some(tokenizer) = &self.tokenizer {
            format!(
                "{parser_message} (tokenizer recovery {:?} at bytes {}..{}: {})",
                tokenizer.recovery, tokenizer.span_start, tokenizer.span_end, tokenizer.message
            )
        } else {
            parser_message
        }
    }
}

impl std::fmt::Display for FormulaParseFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render_message())
    }
}

impl std::error::Error for FormulaParseFailure {}

fn first_tokenizer_recovery(formula: &str) -> Option<TokenizerRecovery> {
    let stream = TokenStream::new_best_effort(formula);
    let diagnostic = stream.diagnostics_ref().first()?;
    Some(tokenizer_recovery_from_diagnostic(diagnostic))
}

fn tokenizer_recovery_from_diagnostic(diagnostic: &TokenDiagnostic) -> TokenizerRecovery {
    TokenizerRecovery {
        message: diagnostic.message.clone(),
        recovery: diagnostic.recovery,
        span_start: diagnostic.span.start,
        span_end: diagnostic.span.end,
    }
}

fn normalize_formula_input(formula: &str) -> String {
    let trimmed = formula.trim();
    if trimmed.starts_with('=') {
        trimmed.to_string()
    } else {
        format!("={trimmed}")
    }
}

pub fn validate_formula_detailed(formula: &str) -> Result<(), FormulaParseFailure> {
    let formula_in = normalize_formula_input(formula);
    formualizer_parse::parse(&formula_in)
        .map(|_| ())
        .map_err(|err| FormulaParseFailure::from_parser_error(&formula_in, &err))
}

pub fn format_formula_parse_failure(formula: &str, err: &ParserError) -> String {
    let formula_in = normalize_formula_input(formula);
    FormulaParseFailure::from_parser_error(&formula_in, err).to_string()
}

/// Validate a single formula string using the project's formula parser.
/// Returns Ok(()) if valid, Err(error_message) if invalid.
pub fn validate_formula(formula: &str) -> Result<(), String> {
    validate_formula_detailed(formula).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_default_is_warn() {
        assert_eq!(FormulaParsePolicy::default(), FormulaParsePolicy::Warn);
    }

    #[test]
    fn test_policy_default_for_single_write() {
        assert_eq!(
            FormulaParsePolicy::default_for_command_class(CommandClass::SingleWrite),
            FormulaParsePolicy::Fail
        );
    }

    #[test]
    fn test_policy_default_for_batch_write() {
        assert_eq!(
            FormulaParsePolicy::default_for_command_class(CommandClass::BatchWrite),
            FormulaParsePolicy::Warn
        );
    }

    #[test]
    fn test_policy_default_for_read_analysis() {
        assert_eq!(
            FormulaParsePolicy::default_for_command_class(CommandClass::ReadAnalysis),
            FormulaParsePolicy::Warn
        );
    }

    #[test]
    fn test_policy_serde_roundtrip() {
        let cases = [
            (FormulaParsePolicy::Fail, "fail"),
            (FormulaParsePolicy::Warn, "warn"),
            (FormulaParsePolicy::Off, "off"),
        ];

        for (policy, expected) in cases {
            let serialized = serde_json::to_string(&policy).expect("serialize policy");
            assert_eq!(serialized, format!("\"{expected}\""));

            let deserialized: FormulaParsePolicy =
                serde_json::from_str(&serialized).expect("deserialize policy");
            assert_eq!(deserialized, policy);
        }
    }

    #[test]
    fn test_empty_builder() {
        let builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);
        assert!(builder.is_empty());
        assert!(!builder.has_errors());

        let diagnostics = builder.build();
        assert_eq!(diagnostics.policy, FormulaParsePolicy::Warn);
        assert_eq!(diagnostics.total_errors, 0);
        assert!(!diagnostics.groups_truncated);
        assert!(diagnostics.groups.is_empty());
    }

    #[test]
    fn test_single_error_group() {
        let mut builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);
        builder.record_error("Sheet1", "A1", "=SUM(A:A)", "unexpected token");

        let diagnostics = builder.build();
        assert_eq!(diagnostics.total_errors, 1);
        assert_eq!(diagnostics.groups.len(), 1);
        let group = &diagnostics.groups[0];
        assert_eq!(group.count, 1);
        assert_eq!(group.error_code, FORMULA_PARSE_FAILED);
    }

    #[test]
    fn test_grouping_same_key() {
        let mut builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);
        builder.record_error("Sheet1", "A1", "=SUM(A:A)", "unexpected token");
        builder.record_error("Sheet1", "A2", "=SUM(A:A)", "unexpected token");
        builder.record_error("Sheet1", "A3", "=SUM(A:A)", "unexpected token");

        let diagnostics = builder.build();
        assert_eq!(diagnostics.groups.len(), 1);
        let group = &diagnostics.groups[0];
        assert_eq!(group.count, 3);
        assert_eq!(group.sample_addresses, vec!["A1", "A2", "A3"]);
    }

    #[test]
    fn test_grouping_different_sheets() {
        let mut builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);
        builder.record_error("A", "A1", "=SUM(A:A)", "unexpected token");
        builder.record_error("B", "A1", "=SUM(A:A)", "unexpected token");

        let diagnostics = builder.build();
        assert_eq!(diagnostics.groups.len(), 2);
        assert_eq!(diagnostics.groups[0].sheet_name, "A");
        assert_eq!(diagnostics.groups[1].sheet_name, "B");
    }

    #[test]
    fn test_grouping_different_messages() {
        let mut builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);
        builder.record_error("Sheet1", "A1", "=SUM(A:A)", "unexpected token");
        builder.record_error("Sheet1", "B1", "=SUM(A:A)", "unknown function");

        let diagnostics = builder.build();
        assert_eq!(diagnostics.groups.len(), 2);
    }

    #[test]
    fn test_sample_address_cap_at_5() {
        let mut builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);

        for i in 1..=8 {
            builder.record_error("Sheet1", &format!("A{i}"), "=SUM(A:A)", "unexpected token");
        }

        let diagnostics = builder.build();
        let group = &diagnostics.groups[0];
        assert_eq!(group.count, 8);
        assert_eq!(group.sample_addresses.len(), 5);
        assert_eq!(group.sample_addresses, vec!["A1", "A2", "A3", "A4", "A5"]);
    }

    #[test]
    fn test_deterministic_ordering() {
        let mut builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);
        builder.record_error("C", "A1", "=1", "err");
        builder.record_error("A", "A1", "=1", "err");
        builder.record_error("B", "A1", "=1", "err");

        let diagnostics = builder.build();
        let sheets: Vec<&str> = diagnostics
            .groups
            .iter()
            .map(|group| group.sheet_name.as_str())
            .collect();
        assert_eq!(sheets, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_groups_truncated_at_50() {
        let mut builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);

        // Use structurally distinct formulas (different function names) so they
        // don't collapse under reference normalization.
        for i in 0..60 {
            builder.record_error(
                "Sheet1",
                "A1",
                &format!("=FUNC{i}(A1)"),
                &format!("error variant {i}"),
            );
        }

        let diagnostics = builder.build();
        assert_eq!(diagnostics.total_errors, 60);
        assert_eq!(diagnostics.groups.len(), 50);
        assert!(diagnostics.groups_truncated);
    }

    #[test]
    fn test_formula_preview_truncation() {
        let mut builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);
        let formula = format!("={}", "A".repeat(119));
        assert_eq!(formula.len(), 120);

        builder.record_error("Sheet1", "A1", &formula, "unexpected token");
        let diagnostics = builder.build();

        let preview = &diagnostics.groups[0].formula_preview;
        assert!(preview.ends_with('…'));
        assert_ne!(preview, &formula);
        assert!(preview.len() <= FORMULA_PREVIEW_MAX_BYTES + '…'.len_utf8());
    }

    #[test]
    fn test_diagnostics_json_structure() {
        let mut builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);
        builder.record_error("Sheet1", "A1", "=SUM(A:A)", "unexpected token");
        let diagnostics = builder.build();

        let value = serde_json::to_value(diagnostics).expect("serialize diagnostics");
        assert_eq!(value["policy"], serde_json::json!("warn"));
        assert_eq!(value["total_errors"], serde_json::json!(1));
        assert_eq!(value["groups_truncated"], serde_json::json!(false));
        assert!(value["groups"].is_array());

        let group = &value["groups"][0];
        assert_eq!(group["error_code"], serde_json::json!(FORMULA_PARSE_FAILED));
        assert_eq!(
            group["error_message"],
            serde_json::json!("unexpected token")
        );
        assert_eq!(group["sheet_name"], serde_json::json!("Sheet1"));
        assert_eq!(group["formula_preview"], serde_json::json!("=SUM(A:A)"));
        assert_eq!(group["count"], serde_json::json!(1));
        assert!(group["sample_addresses"].is_array());
    }

    #[test]
    fn test_has_errors_after_record() {
        let mut builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);
        builder.record_error("Sheet1", "A1", "=SUM(A:A)", "unexpected token");

        assert!(builder.has_errors());
        assert!(!builder.is_empty());
    }

    #[test]
    fn test_validate_formula_valid() {
        assert!(validate_formula("SUM(A1:A10)").is_ok());
        assert!(validate_formula("=SUM(A1:A10)").is_ok());
        assert!(validate_formula("A1+B1").is_ok());
        assert!(validate_formula("IF(A1>0,1,0)").is_ok());
    }

    #[test]
    fn test_validate_formula_invalid() {
        assert!(validate_formula("SUM(A1:A10").is_err()); // unclosed paren
        assert!(validate_formula("SUM(A1:A10))").is_err()); // extra closing paren
    }

    #[test]
    fn test_validate_formula_detailed_includes_recovery_context() {
        let err = validate_formula_detailed("SUM(A1:A10")
            .expect_err("unterminated formula should return parse diagnostics");
        let rendered = err.to_string();
        assert!(rendered.contains("parse error"));
        assert!(rendered.contains("tokenizer recovery"));
        assert!(rendered.contains("bytes "));
    }

    #[test]
    fn test_normalize_error_for_grouping_normalizes_bytes_ranges() {
        let n1 = normalize_error_for_grouping(
            "parse error at position 14 (tokenizer recovery UnmatchedOpener at bytes 1..9: x)",
        );
        let n2 = normalize_error_for_grouping(
            "parse error at position 29 (tokenizer recovery UnmatchedOpener at bytes 7..15: x)",
        );
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_grouping_normalizes_cell_references() {
        // Formulas that differ only in cell references should group together.
        // This is the exact scenario from the Production_Readiness workbook.
        let mut builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);
        builder.record_error(
            "Assessments",
            "D4",
            "=IF(C4=\"\",\"\",IF(C4=\"N/A\",\"\",0))",
            "parse error at position 42",
        );
        builder.record_error(
            "Assessments",
            "D5",
            "=IF(C5=\"\",\"\",IF(C5=\"N/A\",\"\",0))",
            "parse error at position 42",
        );
        builder.record_error(
            "Assessments",
            "D10",
            "=IF(C10=\"\",\"\",IF(C10=\"N/A\",\"\",0))",
            "parse error at position 42",
        );

        let diagnostics = builder.build();
        // All three should collapse into ONE group (same structure, same error)
        assert_eq!(diagnostics.total_errors, 3);
        assert_eq!(diagnostics.groups.len(), 1);

        let group = &diagnostics.groups[0];
        assert_eq!(group.count, 3);
        assert_eq!(group.sample_addresses, vec!["D4", "D5", "D10"]);
        // formula_preview should show the first formula encountered (human-readable, not normalized)
        assert!(group.formula_preview.contains("C4"));
    }

    #[test]
    fn test_grouping_different_structure_not_collapsed() {
        // Formulas with genuinely different structure should NOT collapse.
        let mut builder = FormulaParseDiagnosticsBuilder::new(FormulaParsePolicy::Warn);
        builder.record_error("Sheet1", "A1", "=SUM(A1:A10)", "unexpected token");
        builder.record_error("Sheet1", "A2", "=AVERAGE(B1:B10)", "unexpected token");

        let diagnostics = builder.build();
        assert_eq!(diagnostics.groups.len(), 2);
    }

    #[test]
    fn test_normalize_formula_for_grouping() {
        // Direct unit test for the normalization function
        let n1 = normalize_formula_for_grouping("=IF(C4=\"\",\"\",0)");
        let n2 = normalize_formula_for_grouping("=IF(C5=\"\",\"\",0)");
        let n3 = normalize_formula_for_grouping("=IF(C100=\"\",\"\",0)");
        assert_eq!(n1, n2);
        assert_eq!(n2, n3);

        // Different structure should differ
        let n4 = normalize_formula_for_grouping("=SUM(A1:A10)");
        let n5 = normalize_formula_for_grouping("=AVERAGE(A1:A10)");
        assert_ne!(n4, n5);

        // Unparsable formula falls back to truncated preview
        let bad = normalize_formula_for_grouping("=SUM(A1:A10))");
        assert!(!bad.is_empty());
    }
}
