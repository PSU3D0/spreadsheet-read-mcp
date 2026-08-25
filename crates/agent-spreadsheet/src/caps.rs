use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BackendCaps {
    pub backend: BackendKind,
    pub supports_styles: bool,
    pub supports_tables: bool,
    pub supports_comments: bool,
    pub supports_defined_names: bool,
    pub supports_conditional_formatting: bool,
    pub supports_formula_graph: bool,
}

impl BackendCaps {
    pub fn xlsx() -> Self {
        Self {
            backend: BackendKind::XlsxUmya,
            supports_styles: true,
            supports_tables: true,
            supports_comments: true,
            supports_defined_names: true,
            supports_conditional_formatting: true,
            supports_formula_graph: true,
        }
    }

    pub fn degraded_for_ods() -> Self {
        Self {
            backend: BackendKind::OdsFuture,
            supports_styles: false,
            supports_tables: false,
            supports_comments: false,
            supports_defined_names: true,
            supports_conditional_formatting: false,
            supports_formula_graph: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    XlsxUmya,
    OdsFuture,
}
