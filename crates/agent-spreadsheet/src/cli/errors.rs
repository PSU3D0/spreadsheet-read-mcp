use crate::cli::OutputFormat;
use crate::model::{FORMULA_PARSE_FAILED, FORMULA_PARSE_FAILED_PREFIX};
use anyhow::{Result, bail};
use serde::Serialize;

pub fn ensure_output_supported(format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => Ok(()),
        OutputFormat::Csv => {
            bail!("csv output is not implemented yet for this CLI; use --output-format json")
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ErrorEnvelope {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub did_you_mean: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub try_this: Option<String>,
}

pub fn envelope_for(error: &anyhow::Error) -> ErrorEnvelope {
    let message = error.to_string();

    if let Some((requested, suggested)) = parse_sheet_suggestion(&message) {
        return ErrorEnvelope {
            code: "SHEET_NOT_FOUND".to_string(),
            message: format!("sheet '{}' was not found", requested),
            did_you_mean: Some(suggested),
            try_this: Some("run `asp read sheets <file>` to inspect valid names".to_string()),
        };
    }

    if let Some(requested) = parse_sheet_not_found(&message) {
        return ErrorEnvelope {
            code: "SHEET_NOT_FOUND".to_string(),
            message: format!("sheet '{}' was not found", requested),
            did_you_mean: None,
            try_this: Some("run `asp read sheets <file>` to inspect valid names".to_string()),
        };
    }

    if let Some(detail) = message.strip_prefix("invalid argument: ") {
        let try_this = if detail.contains("session payload kind") {
            "run `asp example session op transform.write_matrix` or `asp schema session op transform.write_matrix` to inspect supported kinds and canonical payloads".to_string()
        } else {
            "run the command with --help to inspect valid arguments".to_string()
        };
        return ErrorEnvelope {
            code: "INVALID_ARGUMENT".to_string(),
            message: detail.to_string(),
            did_you_mean: None,
            try_this: Some(try_this),
        };
    }

    if let Some(detail) = message.strip_prefix("invalid ops payload: ") {
        return ErrorEnvelope {
            code: "INVALID_OPS_PAYLOAD".to_string(),
            message: detail.to_string(),
            did_you_mean: None,
            try_this: Some("pass --ops @<path-to-json> with payload {\"ops\":[...]}".to_string()),
        };
    }

    if message.starts_with("invalid session ops payload") {
        return ErrorEnvelope {
            code: "INVALID_OPS_PAYLOAD".to_string(),
            message,
            did_you_mean: None,
            try_this: Some(
                "run `asp example session op transform.write_matrix` or `asp schema session op transform.write_matrix` to inspect the canonical payload contract".to_string(),
            ),
        };
    }

    if let Some(detail) = message.strip_prefix("output exists: ") {
        return ErrorEnvelope {
            code: "OUTPUT_EXISTS".to_string(),
            message: detail.to_string(),
            did_you_mean: None,
            try_this: Some("choose a new --output path or re-run with --force".to_string()),
        };
    }

    if let Some(detail) = message.strip_prefix("unsafe clone template: ") {
        return ErrorEnvelope {
            code: "UNSAFE_CLONE_TEMPLATE".to_string(),
            message: detail.to_string(),
            did_you_mean: None,
            try_this: Some(
                "re-run with --merge-policy safe or choose a different template row".to_string(),
            ),
        };
    }

    if let Some(detail) = message.strip_prefix("write failed: ") {
        return ErrorEnvelope {
            code: "WRITE_FAILED".to_string(),
            message: detail.to_string(),
            did_you_mean: None,
            try_this: Some("check destination permissions and available disk space".to_string()),
        };
    }

    if message.contains("does not exist") {
        return ErrorEnvelope {
            code: "FILE_NOT_FOUND".to_string(),
            message,
            did_you_mean: None,
            try_this: Some("check the workbook path and permissions".to_string()),
        };
    }

    if message.contains("at least one range") {
        return ErrorEnvelope {
            code: "INVALID_ARGUMENT".to_string(),
            message,
            did_you_mean: None,
            try_this: Some("pass one or more A1 ranges, for example: `A1:C10`".to_string()),
        };
    }

    if message.contains("at least one edit") {
        return ErrorEnvelope {
            code: "INVALID_ARGUMENT".to_string(),
            message,
            did_you_mean: None,
            try_this: Some("add one or more edits like `A1=42` or `B2==SUM(A1:A1)`".to_string()),
        };
    }

    if message.contains("invalid shorthand edit") {
        return ErrorEnvelope {
            code: "INVALID_EDIT_SYNTAX".to_string(),
            message,
            did_you_mean: None,
            try_this: Some(
                "use `<cell>=<value>` for values or `<cell>==<formula>` for formulas".to_string(),
            ),
        };
    }

    if message.contains("csv output is not implemented") {
        return ErrorEnvelope {
            code: "OUTPUT_FORMAT_UNSUPPORTED".to_string(),
            message,
            did_you_mean: Some("json".to_string()),
            try_this: Some("re-run with `--output-format json`".to_string()),
        };
    }

    if message.starts_with(FORMULA_PARSE_FAILED_PREFIX) {
        return ErrorEnvelope {
            code: FORMULA_PARSE_FAILED.to_string(),
            message,
            did_you_mean: None,
            try_this: Some(
                "re-run with --formula-parse-policy warn to collect diagnostics instead of aborting"
                    .to_string(),
            ),
        };
    }

    ErrorEnvelope {
        code: "COMMAND_FAILED".to_string(),
        message,
        did_you_mean: None,
        try_this: None,
    }
}

fn parse_sheet_suggestion(message: &str) -> Option<(String, String)> {
    let prefix = "sheet '";
    let not_found = "' not found; did you mean '";
    let suffix = "' ?";

    let start = message.find(prefix)? + prefix.len();
    let rest = &message[start..];
    let mid = rest.find(not_found)?;
    let requested = &rest[..mid];
    let suggestion_start = start + mid + not_found.len();
    let suggestion_rest = &message[suggestion_start..];
    let suggestion_end = suggestion_rest.find(suffix)?;
    let suggested = &suggestion_rest[..suggestion_end];
    Some((requested.to_string(), suggested.to_string()))
}

fn parse_sheet_not_found(message: &str) -> Option<String> {
    let rest = message.strip_prefix("sheet ")?;
    if rest.contains(" not found; did you mean ") {
        return None;
    }
    if let Some(stripped) = rest.strip_prefix('\'')
        && let Some(requested) = stripped.strip_suffix("' not found")
    {
        return Some(requested.to_string());
    }
    rest.strip_suffix(" not found").map(str::to_string)
}
