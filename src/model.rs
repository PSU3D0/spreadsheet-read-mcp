use crate::caps::BackendCaps;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct WorkbookId(pub String);

impl WorkbookId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WorkbookDescriptor {
    pub workbook_id: WorkbookId,
    pub short_id: String,
    pub slug: String,
    pub folder: Option<String>,
    pub path: String,
    pub bytes: u64,
    pub last_modified: Option<String>,
    pub caps: BackendCaps,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WorkbookListResponse {
    pub workbooks: Vec<WorkbookDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WorkbookDescription {
    pub workbook_id: WorkbookId,
    pub short_id: String,
    pub slug: String,
    pub path: String,
    pub bytes: u64,
    pub sheet_count: usize,
    pub defined_names: usize,
    pub tables: usize,
    pub macros_present: bool,
    pub last_modified: Option<String>,
    pub caps: BackendCaps,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SheetSummary {
    pub name: String,
    pub visible: bool,
    pub row_count: u32,
    pub column_count: u32,
    pub non_empty_cells: u32,
    pub formula_cells: u32,
    pub cached_values: u32,
    pub classification: SheetClassification,
    pub style_tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SheetClassification {
    Data,
    Calculator,
    Mixed,
    Metadata,
    Empty,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SheetListResponse {
    pub workbook_id: WorkbookId,
    pub workbook_short_id: String,
    pub sheets: Vec<SheetSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SheetOverviewResponse {
    pub workbook_id: WorkbookId,
    pub workbook_short_id: String,
    pub sheet_name: String,
    pub narrative: String,
    pub regions: Vec<SheetRegion>,
    pub key_ranges: Vec<String>,
    pub formula_ratio: f32,
    pub notable_features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SheetRegion {
    pub kind: RegionKind,
    pub address: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RegionKind {
    Table,
    Calculator,
    Metadata,
    Styles,
    Comments,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SheetPageResponse {
    pub workbook_id: WorkbookId,
    pub workbook_short_id: String,
    pub sheet_name: String,
    pub rows: Vec<RowSnapshot>,
    pub has_more: bool,
    pub next_start_row: Option<u32>,
    pub header_row: Option<RowSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RowSnapshot {
    pub row_index: u32,
    pub cells: Vec<CellSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CellSnapshot {
    pub address: String,
    pub value: Option<CellValue>,
    pub formula: Option<String>,
    pub cached_value: Option<CellValue>,
    pub number_format: Option<String>,
    pub style_tags: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "value")]
pub enum CellValue {
    Text(String),
    Number(f64),
    Bool(bool),
    Error(String),
    Date(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SheetStatisticsResponse {
    pub workbook_id: WorkbookId,
    pub workbook_short_id: String,
    pub sheet_name: String,
    pub row_count: u32,
    pub column_count: u32,
    pub density: f32,
    pub numeric_columns: Vec<ColumnSummary>,
    pub text_columns: Vec<ColumnSummary>,
    pub null_counts: BTreeMap<String, u32>,
    pub duplicate_warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ColumnSummary {
    pub header: Option<String>,
    pub column: String,
    pub samples: Vec<CellValue>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SheetFormulaMapResponse {
    pub workbook_id: WorkbookId,
    pub workbook_short_id: String,
    pub sheet_name: String,
    pub groups: Vec<FormulaGroup>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FormulaGroup {
    pub fingerprint: String,
    pub addresses: Vec<String>,
    pub formula: String,
    pub is_array: bool,
    pub is_shared: bool,
    pub is_volatile: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FormulaTraceResponse {
    pub workbook_id: WorkbookId,
    pub workbook_short_id: String,
    pub sheet_name: String,
    pub origin: String,
    pub direction: TraceDirection,
    pub layers: Vec<TraceLayer>,
    pub next_cursor: Option<TraceCursor>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FormulaTraceEdge {
    pub from: String,
    pub to: String,
    pub formula: Option<String>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TraceLayer {
    pub depth: u32,
    pub summary: TraceLayerSummary,
    pub highlights: TraceLayerHighlights,
    pub edges: Vec<FormulaTraceEdge>,
    pub has_more: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TraceLayerSummary {
    pub total_nodes: usize,
    pub formula_nodes: usize,
    pub value_nodes: usize,
    pub blank_nodes: usize,
    pub external_nodes: usize,
    pub unique_formula_groups: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TraceLayerHighlights {
    pub top_ranges: Vec<TraceRangeHighlight>,
    pub top_formula_groups: Vec<TraceFormulaGroupHighlight>,
    pub notable_cells: Vec<TraceCellHighlight>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TraceRangeHighlight {
    pub start: String,
    pub end: String,
    pub count: usize,
    pub literals: usize,
    pub formulas: usize,
    pub blanks: usize,
    pub sample_values: Vec<CellValue>,
    pub sample_formulas: Vec<String>,
    pub sample_addresses: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TraceFormulaGroupHighlight {
    pub fingerprint: String,
    pub formula: String,
    pub count: usize,
    pub sample_addresses: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TraceCellHighlight {
    pub address: String,
    pub kind: TraceCellKind,
    pub value: Option<CellValue>,
    pub formula: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TraceCellKind {
    Formula,
    Literal,
    Blank,
    External,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TraceCursor {
    pub depth: u32,
    pub offset: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TraceDirection {
    Precedents,
    Dependents,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct NamedRangeDescriptor {
    pub name: String,
    pub scope: Option<String>,
    pub refers_to: String,
    pub kind: NamedItemKind,
    pub sheet_name: Option<String>,
    pub comment: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NamedItemKind {
    NamedRange,
    Table,
    Formula,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct NamedRangesResponse {
    pub workbook_id: WorkbookId,
    pub workbook_short_id: String,
    pub items: Vec<NamedRangeDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FindFormulaMatch {
    pub address: String,
    pub sheet_name: String,
    pub formula: String,
    pub cached_value: Option<CellValue>,
    pub context: Vec<RowSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FindFormulaResponse {
    pub workbook_id: WorkbookId,
    pub workbook_short_id: String,
    pub matches: Vec<FindFormulaMatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VolatileScanEntry {
    pub address: String,
    pub sheet_name: String,
    pub function: String,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VolatileScanResponse {
    pub workbook_id: WorkbookId,
    pub workbook_short_id: String,
    pub items: Vec<VolatileScanEntry>,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SheetStylesResponse {
    pub workbook_id: WorkbookId,
    pub workbook_short_id: String,
    pub sheet_name: String,
    pub styles: Vec<StyleSummary>,
    pub conditional_rules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StyleSummary {
    pub style_id: String,
    pub occurrences: u32,
    pub tags: Vec<String>,
    pub example_cells: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ManifestStubResponse {
    pub workbook_id: WorkbookId,
    pub workbook_short_id: String,
    pub slug: String,
    pub sheets: Vec<ManifestSheetStub>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ManifestSheetStub {
    pub sheet_name: String,
    pub classification: SheetClassification,
    pub candidate_expectations: Vec<String>,
    pub notes: Vec<String>,
}
