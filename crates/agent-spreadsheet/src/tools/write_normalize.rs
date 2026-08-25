use crate::core::write::{normalize_object_edit, normalize_shorthand_edit};
use crate::errors::InvalidParamsError;
use crate::model::{FormulaParsePolicy, Warning};
use crate::tools::fork::{CellEdit, EditBatchParams};
use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct EditBatchParamsInput {
    pub fork_id: String,
    pub sheet_name: String,
    pub edits: Vec<CellEditInput>,
    #[serde(default)]
    pub formula_parse_policy: Option<FormulaParsePolicy>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum CellEditInput {
    Shorthand(String),
    Object(CellEditV2),
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct CellEditV2 {
    pub address: String,
    #[serde(default)]
    pub value: Option<String>,
    #[serde(default)]
    pub formula: Option<String>,
    #[serde(default)]
    pub is_formula: Option<bool>,
}

pub fn normalize_edit_batch(
    params: EditBatchParamsInput,
) -> Result<(EditBatchParams, Vec<Warning>)> {
    let mut warnings = Vec::new();
    let mut edits = Vec::with_capacity(params.edits.len());

    for (idx, edit) in params.edits.into_iter().enumerate() {
        match edit {
            CellEditInput::Shorthand(entry) => {
                let (normalized, core_warnings) =
                    normalize_shorthand_edit(&entry).map_err(|err| {
                        InvalidParamsError::new("edit_batch", err.to_string())
                            .with_path(format!("edits[{idx}]"))
                    })?;
                edits.push(CellEdit {
                    address: normalized.address,
                    value: normalized.value,
                    is_formula: normalized.is_formula,
                });
                warnings.extend(core_warnings.into_iter().map(|warning| Warning {
                    code: warning.code,
                    message: warning.message,
                }));
            }
            CellEditInput::Object(obj) => {
                let normalized =
                    normalize_object_edit(&obj.address, obj.value, obj.formula, obj.is_formula)
                        .map_err(|err| {
                            let path = if err.to_string().contains("address") {
                                format!("edits[{idx}].address")
                            } else {
                                format!("edits[{idx}]")
                            };
                            InvalidParamsError::new("edit_batch", err.to_string()).with_path(path)
                        })?;

                edits.push(CellEdit {
                    address: normalized.0.address,
                    value: normalized.0.value,
                    is_formula: normalized.0.is_formula,
                });
                warnings.extend(normalized.1.into_iter().map(|warning| Warning {
                    code: warning.code,
                    message: warning.message,
                }));
            }
        }
    }

    Ok((
        EditBatchParams {
            fork_id: params.fork_id,
            sheet_name: params.sheet_name,
            edits,
        },
        warnings,
    ))
}
