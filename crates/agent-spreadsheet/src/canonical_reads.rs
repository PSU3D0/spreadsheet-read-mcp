use crate::model::*;
use crate::operations::{CanonicalErrorCode, CanonicalErrorEnvelope, ResourceId};
use crate::state::AppState;
use crate::tools::{self, FilterOp, FormulaSortBy, MatchMode, SampleMode, StyleGranularity};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListWorkbooksRequest {
    #[serde(default)]
    pub slug_prefix: Option<String>,
    #[serde(default)]
    pub folder: Option<String>,
    #[serde(default)]
    pub path_glob: Option<String>,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub offset: Option<u32>,
    #[serde(default)]
    pub include_paths: Option<bool>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanonicalWorkbookDescriptor {
    pub resource_id: ResourceId,
    pub metadata: WorkbookDiscoveryMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paths: Option<WorkbookPaths>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_capabilities: Option<crate::caps::BackendCaps>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkbookDiscoveryMetadata {
    pub short_id: String,
    pub slug: String,
    pub bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_modified: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkbookPaths {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub internal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client: Option<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListWorkbooksData {
    pub workbooks: Vec<CanonicalWorkbookDescriptor>,
    pub next_offset: Option<u32>,
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DescribeInclude {
    Summary,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DescribeWorkbookRequest {
    pub resource_id: ResourceId,
    #[serde(default)]
    pub include: Vec<DescribeInclude>,
    #[serde(default)]
    pub summary: Option<DescribeSummaryOptions>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DescribeSummaryOptions {
    #[serde(default)]
    pub include_entry_points: Option<bool>,
    #[serde(default)]
    pub include_named_ranges: Option<bool>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DescribeWorkbookData {
    pub metadata: WorkbookExactMetadata,
    pub paths: WorkbookPaths,
    pub capabilities: WorkbookCapabilities,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<WorkbookDerivedSummary>,
    pub warnings: Vec<Warning>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkbookExactMetadata {
    pub short_id: String,
    pub slug: String,
    pub bytes: u64,
    pub last_modified: Option<String>,
    pub sheet_count: usize,
    pub defined_name_count: usize,
    pub table_count: usize,
    pub macros_present: bool,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkbookCapabilities {
    pub backend: crate::caps::BackendCaps,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkbookDerivedSummary {
    pub status: CoverageStatus,
    pub total_cells: u64,
    pub total_formulas: u64,
    pub breakdown: WorkbookBreakdown,
    pub region_counts: RegionCountSummary,
    pub key_named_ranges: Vec<NamedRangeDescriptor>,
    pub suggested_entry_points: Vec<EntryPoint>,
    pub notes: Vec<String>,
    pub coverage: WorkbookSummaryCoverage,
}

#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CoverageStatus {
    Complete,
    Bounded,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkbookSummaryCoverage {
    pub sheets_scanned: usize,
    pub sheets_total: usize,
    pub bounded: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReadCellsFormat {
    Dense,
    Values,
    Csv,
    Json,
    Rows,
    Full,
    Compact,
    ValuesOnly,
}

impl ReadCellsFormat {
    fn range_format(self) -> TableOutputFormat {
        match self {
            Self::Dense => TableOutputFormat::Dense,
            Self::Values | Self::Compact | Self::ValuesOnly => TableOutputFormat::Values,
            Self::Csv => TableOutputFormat::Csv,
            Self::Json | Self::Full => TableOutputFormat::Json,
            Self::Rows => TableOutputFormat::Rows,
        }
    }

    fn row_format(self) -> SheetPageFormat {
        match self {
            Self::Full | Self::Json => SheetPageFormat::Full,
            Self::Compact | Self::Dense | Self::Csv | Self::Rows => SheetPageFormat::Compact,
            Self::Values | Self::ValuesOnly => SheetPageFormat::ValuesOnly,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReadCellField {
    Value,
    Formula,
    CachedValue,
    StoredKind,
    NumberFormat,
    StyleTags,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReadCellsSelection {
    Range {
        ranges: Vec<String>,
        #[serde(default)]
        include_headers: Option<bool>,
    },
    Rows {
        #[serde(default = "one")]
        start_row: u32,
        row_count: u32,
        #[serde(default)]
        columns: Option<RowColumnSelection>,
        #[serde(default)]
        include_header: Option<bool>,
    },
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RowColumnSelection {
    All,
    Letters {
        values: Vec<String>,
    },
    Headers {
        values: Vec<String>,
        #[serde(default)]
        header_row: Option<u32>,
    },
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanonicalReadPage {
    #[serde(default)]
    pub limit_rows: Option<u32>,
    #[serde(default)]
    pub cursor: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadCellsRequest {
    pub resource_id: ResourceId,
    pub sheet_name: String,
    pub selection: ReadCellsSelection,
    #[serde(default)]
    pub fields: Vec<ReadCellField>,
    #[serde(default)]
    pub include_formulas: Option<bool>,
    #[serde(default)]
    pub include_styles: Option<bool>,
    #[serde(default)]
    pub format: Option<ReadCellsFormat>,
    #[serde(default)]
    pub encoding: Option<ReadCellsFormat>,
    #[serde(default)]
    pub page_size: Option<u32>,
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub page: Option<CanonicalReadPage>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadCellsData {
    pub sheet_name: String,
    pub selection_kind: ReadCellsSelectionKind,
    pub encoding: ReadCellsFormat,
    pub blocks: Vec<ReadCellsBlock>,
    pub header: Option<RowSnapshot>,
    pub calculation: CalculationMetadata,
    pub page: ReadCellsPage,
    pub warnings: Vec<Warning>,
}

#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReadCellsSelectionKind {
    Range,
    Rows,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadCellsBlock {
    pub selection_index: usize,
    pub requested_range: String,
    pub returned_range: String,
    pub row_count: usize,
    pub column_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub row_indices: Option<Vec<u32>>,
    pub payload: ReadCellsPayload,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadCellsPayload {
    pub encoding: ReadCellsFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows: Option<Vec<Vec<Option<CellValue>>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formulas: Option<Vec<Vec<Option<String>>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<Vec<Option<CellValuePrimitive>>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense: Option<RangeValuesDensePayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub csv: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows_keyed: Option<Vec<RangeValuesRowEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshots: Option<Vec<RowSnapshot>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact: Option<SheetPageCompact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values_only: Option<SheetPageValues>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadCellsPage {
    pub rows_returned: usize,
    pub cells_returned: usize,
    pub complete: bool,
    pub next_cursor: Option<String>,
    pub limits: ReadCellsLimits,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadCellsLimits {
    pub requested_rows: u32,
    pub max_cells: Option<usize>,
    pub max_payload_bytes: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadCellsCursor {
    revision_id: String,
    fingerprint: String,
    selection_index: usize,
    next_row: u32,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InspectCellsRequest {
    pub resource_id: ResourceId,
    pub sheet_name: String,
    pub targets: Vec<String>,
    #[serde(default)]
    pub include_empty: Option<bool>,
    #[serde(default)]
    pub budget: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NamedRangesRequest {
    pub resource_id: ResourceId,
    #[serde(default)]
    pub sheet_name: Option<String>,
    #[serde(default)]
    pub name_prefix: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchValuesRequest {
    pub resource_id: ResourceId,
    pub query: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub mode: Option<FindMode>,
    #[serde(default)]
    pub match_mode: Option<MatchMode>,
    #[serde(default)]
    pub case_sensitive: bool,
    #[serde(default)]
    pub sheet_name: Option<String>,
    #[serde(default)]
    pub region_id: Option<u32>,
    #[serde(default)]
    pub table_name: Option<String>,
    #[serde(default)]
    pub value_types: Option<Vec<tools::ValueTypeFilter>>,
    #[serde(default)]
    pub search_headers_only: bool,
    #[serde(default)]
    pub direction: Option<LabelDirection>,
    #[serde(default = "fifty")]
    pub limit: u32,
    #[serde(default)]
    pub offset: Option<u32>,
    #[serde(default)]
    pub context: Option<tools::FindContext>,
    #[serde(default)]
    pub context_width: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FormulaQuery {
    pub text: String,
    #[serde(default)]
    pub match_mode: Option<MatchMode>,
    #[serde(default)]
    pub case_sensitive: bool,
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct FormulaFilter {
    #[serde(default)]
    pub volatile: Option<bool>,
    #[serde(default)]
    pub function_names: Vec<String>,
    #[serde(default)]
    pub has_external_references: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FormulaSearchScope {
    Workbook,
    Sheet {
        sheet_name: String,
        #[serde(default)]
        range: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FormulaResultMode {
    Cells,
    Groups,
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FormulaGroupBy {
    Function,
    NormalizedFormula,
    Fingerprint,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FormulaContextRequest {
    pub rows: u32,
    pub columns: u32,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchFormulasRequest {
    pub resource_id: ResourceId,
    #[serde(default)]
    pub scope: Option<FormulaSearchScope>,
    #[serde(default)]
    pub query: Option<FormulaQuery>,
    #[serde(default)]
    pub filter: Option<FormulaFilter>,
    pub result_mode: FormulaResultMode,
    #[serde(default)]
    pub group_by: Option<FormulaGroupBy>,
    #[serde(default)]
    pub include_addresses: Option<bool>,
    #[serde(default)]
    pub addresses_per_group: Option<u32>,
    #[serde(default)]
    pub include_context: Option<FormulaContextRequest>,
    #[serde(default)]
    pub formula_parse_policy: Option<FormulaParsePolicy>,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub offset: Option<u32>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SearchFormulasData {
    pub result_mode: FormulaResultMode,
    pub matches: Vec<FormulaCellMatch>,
    pub groups: Vec<FormulaSearchGroup>,
    pub summary: FormulaSearchSummary,
    pub formula_parse_diagnostics: Option<FormulaParseDiagnostics>,
    pub next_offset: Option<u32>,
    pub warnings: Vec<Warning>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FormulaCellMatch {
    pub sheet_name: String,
    pub address: String,
    pub formula: String,
    pub cached_value: Option<CellValue>,
    pub classifications: FormulaClassifications,
    pub context: Vec<RowSnapshot>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FormulaClassifications {
    pub volatile: bool,
    pub functions: Vec<String>,
    pub external_references: bool,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FormulaSearchGroup {
    pub group_key: String,
    pub formula: Option<String>,
    pub fingerprint: Option<String>,
    pub volatile: bool,
    pub functions: Vec<String>,
    pub cell_count: usize,
    pub addresses: Vec<String>,
    pub addresses_complete: bool,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FormulaSearchSummary {
    pub formula_cells_scanned: usize,
    pub matched_cells: usize,
    pub matched_groups: usize,
    pub scan_complete: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FormulaTraceRequest {
    pub resource_id: ResourceId,
    pub sheet_name: String,
    pub cell_address: String,
    pub direction: TraceDirection,
    #[serde(default)]
    pub depth: Option<u32>,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub page_size: Option<usize>,
    #[serde(default)]
    pub cursor: Option<TraceCursor>,
    #[serde(default)]
    pub formula_parse_policy: Option<FormulaParsePolicy>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FormulaMapRequest {
    pub resource_id: ResourceId,
    pub sheet_name: String,
    #[serde(default)]
    pub range: Option<String>,
    #[serde(default)]
    pub expand: bool,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub sort_by: Option<FormulaSortBy>,
    #[serde(default)]
    pub summary_only: Option<bool>,
    #[serde(default)]
    pub include_addresses: Option<bool>,
    #[serde(default)]
    pub addresses_limit: Option<u32>,
    #[serde(default)]
    pub formula_parse_policy: Option<FormulaParsePolicy>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProfileTableRequest {
    pub resource_id: ResourceId,
    #[serde(default)]
    pub sheet_name: Option<String>,
    #[serde(default)]
    pub region_id: Option<u32>,
    #[serde(default)]
    pub table_name: Option<String>,
    #[serde(default)]
    pub sample_mode: Option<SampleMode>,
    #[serde(default)]
    pub sample_size: Option<u32>,
    #[serde(default)]
    pub summary_only: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SheetStatisticsRequest {
    pub resource_id: ResourceId,
    pub sheet_name: String,
    #[serde(default)]
    pub sample_rows: Option<usize>,
    #[serde(default)]
    pub summary_only: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadLayoutRequest {
    pub resource_id: ResourceId,
    pub sheet_name: String,
    #[serde(default)]
    pub range: Option<String>,
    #[serde(default)]
    pub mode: Option<LayoutMode>,
    #[serde(default)]
    pub max_col_width: Option<u32>,
    #[serde(default)]
    pub fit_columns: Option<bool>,
    #[serde(default)]
    pub trim_empty_columns: Option<bool>,
    #[serde(default)]
    pub render: Option<LayoutRender>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadLayoutData {
    pub lossiness: LayoutLossiness,
    pub layout: LayoutPageResponse,
}

#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LayoutLossiness {
    Lossy,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportGridRequest {
    pub resource_id: ResourceId,
    pub sheet_name: String,
    pub range: String,
    #[serde(default)]
    pub page_size: Option<u32>,
    #[serde(default)]
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportGridData {
    pub losslessness: GridLosslessness,
    pub requested_range: String,
    pub returned_range: String,
    pub grid: GridPayload,
    pub complete: bool,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GridLosslessness {
    Lossless,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AnalyzeStylesScope {
    Workbook,
    Sheet {
        sheet_name: String,
        #[serde(default)]
        selection: Option<StyleSelection>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StyleSelection {
    All,
    Range { range: String },
    Region { region_id: u32 },
}

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StyleInclude {
    Descriptors,
    Ranges,
    ExampleCells,
    Theme,
    ConditionalFormats,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnalyzeStylesLimits {
    #[serde(default)]
    pub cells_scanned: Option<u32>,
    #[serde(default)]
    pub examples_per_style: Option<u32>,
    #[serde(default)]
    pub ranges_per_style: Option<u32>,
    #[serde(default)]
    pub styles: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnalyzeStylesRequest {
    pub resource_id: ResourceId,
    pub scope: AnalyzeStylesScope,
    #[serde(default)]
    pub group_by: Option<StyleGranularity>,
    #[serde(default)]
    pub include: Vec<StyleInclude>,
    #[serde(default)]
    pub limits: Option<AnalyzeStylesLimits>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnalyzeStylesData {
    pub scope: AnalyzeStylesScope,
    pub styles: Vec<CanonicalStyleUsage>,
    pub theme: Option<ThemeSummary>,
    pub conditional_formats: Vec<ConditionalFormatSummary>,
    pub coverage: StyleCoverage,
    pub warnings: Vec<Warning>,
}

#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AnalyzeStylesScopeKind {
    Workbook,
    Sheet,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanonicalStyleUsage {
    pub style_id: String,
    pub occurrences: u32,
    pub tags: Vec<String>,
    pub descriptor: Option<StyleDescriptor>,
    pub ranges: Vec<String>,
    pub ranges_complete: bool,
    pub example_cells: Vec<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StyleCoverage {
    pub status: CoverageStatus,
    pub cells_scanned: u64,
    pub cells_in_scope: Option<u64>,
    pub counts_exact: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanonicalTableFilter {
    pub column: String,
    pub op: FilterOp,
    pub value: Value,
}

impl From<CanonicalTableFilter> for tools::TableFilter {
    fn from(value: CanonicalTableFilter) -> Self {
        Self {
            column: value.column,
            op: value.op,
            value: value.value,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadTableRequest {
    pub resource_id: ResourceId,
    #[serde(default)]
    pub sheet_name: Option<String>,
    #[serde(default)]
    pub table_name: Option<String>,
    #[serde(default)]
    pub region_id: Option<u32>,
    #[serde(default)]
    pub range: Option<String>,
    #[serde(default)]
    pub header_row: Option<u32>,
    #[serde(default)]
    pub header_rows: Option<u32>,
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    #[serde(default)]
    pub filters: Option<Vec<CanonicalTableFilter>>,
    #[serde(default)]
    pub sample_mode: Option<SampleMode>,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub offset: Option<u32>,
    #[serde(default)]
    pub format: Option<TableOutputFormat>,
    #[serde(default)]
    pub include_headers: Option<bool>,
    #[serde(default)]
    pub include_types: Option<bool>,
}

fn one() -> u32 {
    1
}
fn fifty() -> u32 {
    50
}

pub fn fingerprint_read_cells(request: &ReadCellsRequest) -> String {
    let invariant = format!(
        "{}|{:?}|{:?}|{:?}|{:?}|{:?}",
        request.sheet_name,
        request.selection,
        request.fields,
        request.include_formulas,
        request.include_styles,
        request.format.or(request.encoding)
    );
    format!("{:x}", Sha256::digest(invariant.as_bytes()))
}

fn encode_cursor(cursor: &ReadCellsCursor) -> Result<String, serde_json::Error> {
    serde_json::to_vec(cursor).map(|bytes| {
        let mut encoded = String::from("rc1_");
        for byte in bytes {
            encoded.push_str(&format!("{byte:02x}"));
        }
        encoded
    })
}

fn decode_cursor(value: &str) -> Result<ReadCellsCursor, String> {
    let hex = value
        .strip_prefix("rc1_")
        .ok_or_else(|| "cursor is not a read_cells v1 cursor".to_string())?;
    if hex.len() % 2 != 0 {
        return Err("cursor has invalid length".to_string());
    }
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&hex[index..index + 2], 16)
                .map_err(|_| "cursor is not valid hexadecimal".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    serde_json::from_slice(&bytes).map_err(|_| "cursor payload is invalid".to_string())
}

fn parse_range(range: &str) -> Result<(u32, u32, u32, u32), String> {
    let bare = range.rsplit_once('!').map_or(range, |(_, value)| value);
    let mut parts = bare.split(':');
    let first = parts.next().unwrap_or_default();
    let second = parts.next().unwrap_or(first);
    if parts.next().is_some() {
        return Err(format!("invalid range: {range}"));
    }
    let parse = |cell: &str| -> Result<(u32, u32), String> {
        let cell = cell.replace('$', "");
        let split = cell
            .find(|ch: char| ch.is_ascii_digit())
            .ok_or_else(|| format!("invalid range: {range}"))?;
        let (letters, digits) = cell.split_at(split);
        if letters.is_empty()
            || digits.is_empty()
            || !letters.chars().all(|c| c.is_ascii_alphabetic())
        {
            return Err(format!("invalid range: {range}"));
        }
        let col = letters.bytes().fold(0_u32, |acc, byte| {
            acc * 26 + u32::from(byte.to_ascii_uppercase() - b'A' + 1)
        });
        let row = digits
            .parse::<u32>()
            .map_err(|_| format!("invalid range: {range}"))?;
        if row == 0 {
            return Err(format!("invalid range: {range}"));
        }
        Ok((col, row))
    };
    let (c1, r1) = parse(first)?;
    let (c2, r2) = parse(second)?;
    if c1 > c2 || r1 > r2 {
        return Err(format!("invalid range: {range}"));
    }
    Ok((c1, r1, c2, r2))
}

fn col_name(mut col: u32) -> String {
    let mut result = Vec::new();
    while col > 0 {
        col -= 1;
        result.push((b'A' + (col % 26) as u8) as char);
        col /= 26;
    }
    result.iter().rev().collect()
}

fn range_string(c1: u32, r1: u32, c2: u32, r2: u32) -> String {
    format!("{}{r1}:{}{r2}", col_name(c1), col_name(c2))
}

fn requested_cursor(request: &ReadCellsRequest) -> Option<&str> {
    request.cursor.as_deref().or_else(|| {
        request
            .page
            .as_ref()
            .and_then(|page| page.cursor.as_deref())
    })
}

fn requested_page_size(request: &ReadCellsRequest) -> u32 {
    request
        .page_size
        .or_else(|| request.page.as_ref().and_then(|page| page.limit_rows))
        .unwrap_or(50)
        .clamp(1, 500)
}

fn requested_encoding(request: &ReadCellsRequest) -> ReadCellsFormat {
    request
        .format
        .or(request.encoding)
        .unwrap_or(ReadCellsFormat::Dense)
}

fn canonical_error(
    code: CanonicalErrorCode,
    operation: &str,
    message: impl Into<String>,
    path: Option<&str>,
) -> CanonicalErrorEnvelope {
    CanonicalErrorEnvelope::new(code, message, Some(operation), path.map(str::to_string))
}

pub async fn execute_read_cells(
    state: Arc<AppState>,
    request: &ReadCellsRequest,
    revision_id: &str,
) -> Result<ReadCellsData, CanonicalErrorEnvelope> {
    let operation = "read_cells";
    let fingerprint = fingerprint_read_cells(request);
    let cursor = if let Some(value) = requested_cursor(request) {
        let cursor = decode_cursor(value).map_err(|message| {
            canonical_error(
                CanonicalErrorCode::InvalidRequest,
                operation,
                message,
                Some("$.cursor"),
            )
        })?;
        if cursor.revision_id != revision_id {
            return Err(canonical_error(
                CanonicalErrorCode::StaleCursor,
                operation,
                "cursor is bound to a different resource revision",
                Some("$.cursor"),
            ));
        }
        if cursor.fingerprint != fingerprint {
            return Err(canonical_error(
                CanonicalErrorCode::CursorMismatch,
                operation,
                "cursor does not match the invariant read_cells request",
                Some("$.cursor"),
            ));
        }
        Some(cursor)
    } else {
        None
    };

    let workbook_id = request.resource_id.to_workbook_id();
    let encoding = requested_encoding(request);
    let requested_rows = requested_page_size(request);
    let include_formulas = request
        .include_formulas
        .unwrap_or_else(|| request.fields.contains(&ReadCellField::Formula));
    let include_styles = request.include_styles.unwrap_or_else(|| {
        request.fields.iter().any(|field| {
            matches!(
                field,
                ReadCellField::NumberFormat | ReadCellField::StyleTags
            )
        })
    });
    let calculation = state
        .open_workbook(&workbook_id)
        .await
        .map_err(|error| {
            canonical_error(
                CanonicalErrorCode::ResourceNotFound,
                operation,
                error.to_string(),
                Some("$.resource_id"),
            )
        })?
        .calculation_metadata();
    let mut blocks = Vec::new();
    let mut rows_returned = 0_usize;
    let mut cells_returned = 0_usize;
    let mut next_cursor = None;
    let mut header = None;

    match &request.selection {
        ReadCellsSelection::Range {
            ranges,
            include_headers,
        } => {
            if ranges.is_empty() {
                return Err(canonical_error(
                    CanonicalErrorCode::InvalidRequest,
                    operation,
                    "selection.ranges must not be empty",
                    Some("$.selection.ranges"),
                ));
            }
            let start_index = cursor.as_ref().map_or(0, |value| value.selection_index);
            for (index, requested_range) in ranges.iter().enumerate().skip(start_index) {
                if rows_returned >= requested_rows as usize {
                    break;
                }
                let (c1, original_r1, c2, r2) =
                    parse_range(requested_range).map_err(|message| {
                        canonical_error(
                            CanonicalErrorCode::InvalidRequest,
                            operation,
                            message,
                            Some("$.selection.ranges"),
                        )
                    })?;
                let r1 = if index == start_index {
                    cursor.as_ref().map_or(original_r1, |value| value.next_row)
                } else {
                    original_r1
                };
                if r1 > r2 {
                    continue;
                }
                let remaining = requested_rows as usize - rows_returned;
                let range = range_string(c1, r1, c2, r2);
                let response = tools::range_values(
                    state.clone(),
                    tools::RangeValuesParams {
                        workbook_or_fork_id: workbook_id.clone(),
                        sheet_name: request.sheet_name.clone(),
                        ranges: vec![range],
                        include_headers: *include_headers,
                        include_formulas: Some(include_formulas),
                        format: Some(encoding.range_format()),
                        page_size: Some(remaining as u32),
                    },
                )
                .await
                .map_err(|error| {
                    canonical_error(
                        CanonicalErrorCode::OperationFailed,
                        operation,
                        error.to_string(),
                        None,
                    )
                })?;
                let entry = response.values.into_iter().next().ok_or_else(|| {
                    canonical_error(
                        CanonicalErrorCode::OperationFailed,
                        operation,
                        "range read returned no correlated entry",
                        None,
                    )
                })?;
                let row_count = entry
                    .rows
                    .as_ref()
                    .map(Vec::len)
                    .or_else(|| entry.values.as_ref().map(Vec::len))
                    .or_else(|| entry.dense.as_ref().map(|dense| dense.row_runs.len()))
                    .or_else(|| entry.rows_keyed.as_ref().map(Vec::len))
                    .or_else(|| entry.csv.as_ref().map(|csv| csv.lines().count()))
                    .unwrap_or(0);
                let returned_end = if row_count == 0 {
                    r1.saturating_sub(1)
                } else {
                    r1 + row_count as u32 - 1
                };
                let returned_range = if row_count == 0 {
                    range_string(c1, r1, c2, r1)
                } else {
                    range_string(c1, r1, c2, returned_end)
                };
                let continuation_row = entry
                    .next_start_row
                    .or_else(|| (returned_end < r2).then_some(returned_end + 1));
                let payload = ReadCellsPayload {
                    encoding,
                    rows: entry.rows,
                    formulas: entry.formulas,
                    values: entry.values,
                    dense: entry.dense,
                    csv: entry.csv,
                    rows_keyed: entry.rows_keyed,
                    snapshots: None,
                    compact: None,
                    values_only: None,
                };
                let column_count = (c2 - c1 + 1) as usize;
                rows_returned += row_count;
                cells_returned += row_count * column_count;
                blocks.push(ReadCellsBlock {
                    selection_index: index,
                    requested_range: requested_range.clone(),
                    returned_range,
                    row_count,
                    column_count,
                    row_indices: None,
                    payload,
                });
                if let Some(next_row) = continuation_row {
                    next_cursor = Some(
                        encode_cursor(&ReadCellsCursor {
                            revision_id: revision_id.to_string(),
                            fingerprint: fingerprint.clone(),
                            selection_index: index,
                            next_row,
                        })
                        .map_err(|error| {
                            canonical_error(
                                CanonicalErrorCode::OperationFailed,
                                operation,
                                error.to_string(),
                                None,
                            )
                        })?,
                    );
                    break;
                }
                if index + 1 < ranges.len() && rows_returned >= requested_rows as usize {
                    let (_, next_row, _, _) =
                        parse_range(&ranges[index + 1]).map_err(|message| {
                            canonical_error(
                                CanonicalErrorCode::InvalidRequest,
                                operation,
                                message,
                                Some("$.selection.ranges"),
                            )
                        })?;
                    next_cursor = Some(
                        encode_cursor(&ReadCellsCursor {
                            revision_id: revision_id.to_string(),
                            fingerprint: fingerprint.clone(),
                            selection_index: index + 1,
                            next_row,
                        })
                        .map_err(|error| {
                            canonical_error(
                                CanonicalErrorCode::OperationFailed,
                                operation,
                                error.to_string(),
                                None,
                            )
                        })?,
                    );
                    break;
                }
            }
        }
        ReadCellsSelection::Rows {
            start_row,
            row_count,
            columns,
            include_header,
        } => {
            if *row_count == 0 {
                return Err(canonical_error(
                    CanonicalErrorCode::InvalidRequest,
                    operation,
                    "selection.row_count must be greater than zero",
                    Some("$.selection.row_count"),
                ));
            }
            let current_row = cursor
                .as_ref()
                .map_or((*start_row).max(1), |value| value.next_row);
            let final_row = start_row
                .max(&1)
                .saturating_add(*row_count)
                .saturating_sub(1);
            let remaining = final_row
                .saturating_sub(current_row)
                .saturating_add(1)
                .min(requested_rows);
            let (letters, headers, header_row) = match columns {
                Some(RowColumnSelection::Letters { values }) => (Some(values.clone()), None, 1),
                Some(RowColumnSelection::Headers { values, header_row }) => {
                    (None, Some(values.clone()), header_row.unwrap_or(1).max(1))
                }
                _ => (None, None, 1),
            };
            let response = tools::sheet_page_with_header_row(
                state.clone(),
                tools::SheetPageParams {
                    workbook_or_fork_id: workbook_id,
                    sheet_name: request.sheet_name.clone(),
                    start_row: current_row,
                    page_size: remaining,
                    columns: letters,
                    columns_by_header: headers,
                    include_formulas,
                    include_styles,
                    include_header: include_header.unwrap_or(true),
                    format: Some(encoding.row_format()),
                },
                header_row,
            )
            .await
            .map_err(|error| {
                canonical_error(
                    CanonicalErrorCode::OperationFailed,
                    operation,
                    error.to_string(),
                    None,
                )
            })?;
            header = response.header_row.clone();
            let snapshots = response.rows.clone();
            let encoded_row_count = response
                .compact
                .as_ref()
                .map(|compact| compact.rows.len())
                .or_else(|| {
                    response
                        .values_only
                        .as_ref()
                        .map(|values| values.rows.len())
                })
                .unwrap_or(snapshots.len());
            let row_indices = if snapshots.is_empty() {
                (current_row..current_row.saturating_add(encoded_row_count as u32))
                    .collect::<Vec<_>>()
            } else {
                snapshots
                    .iter()
                    .map(|row| row.row_index)
                    .collect::<Vec<_>>()
            };
            let returned_row_count = row_indices.len();
            let column_count = snapshots.first().map_or_else(
                || {
                    response.compact.as_ref().map_or_else(
                        || {
                            response
                                .values_only
                                .as_ref()
                                .and_then(|values| values.rows.first())
                                .map_or(0, Vec::len)
                        },
                        |compact| compact.headers.len(),
                    )
                },
                |row| row.cells.len(),
            );
            let returned_end = row_indices.last().copied().unwrap_or(current_row);
            let requested_range = format!("rows:{start_row}+{row_count}");
            let returned_range = format!("rows:{current_row}-{returned_end}");
            let next_row = response
                .next_start_row
                .filter(|next| *next <= final_row)
                .or_else(|| (returned_end < final_row).then_some(returned_end + 1));
            if let Some(next_row) = next_row {
                next_cursor = Some(
                    encode_cursor(&ReadCellsCursor {
                        revision_id: revision_id.to_string(),
                        fingerprint,
                        selection_index: 0,
                        next_row,
                    })
                    .map_err(|error| {
                        canonical_error(
                            CanonicalErrorCode::OperationFailed,
                            operation,
                            error.to_string(),
                            None,
                        )
                    })?,
                );
            }
            blocks.push(ReadCellsBlock {
                selection_index: 0,
                requested_range,
                returned_range,
                row_count: returned_row_count,
                column_count,
                row_indices: Some(row_indices),
                payload: ReadCellsPayload {
                    encoding,
                    rows: None,
                    formulas: None,
                    values: None,
                    dense: None,
                    csv: None,
                    rows_keyed: None,
                    snapshots: (!snapshots.is_empty()).then_some(snapshots),
                    compact: response.compact,
                    values_only: response.values_only,
                },
            });
            rows_returned = returned_row_count;
            cells_returned = returned_row_count * column_count;
        }
    }
    let config = state.config();
    Ok(ReadCellsData {
        sheet_name: request.sheet_name.clone(),
        selection_kind: match request.selection {
            ReadCellsSelection::Range { .. } => ReadCellsSelectionKind::Range,
            ReadCellsSelection::Rows { .. } => ReadCellsSelectionKind::Rows,
        },
        encoding,
        blocks,
        header,
        calculation,
        page: ReadCellsPage {
            rows_returned,
            cells_returned,
            complete: next_cursor.is_none(),
            next_cursor,
            limits: ReadCellsLimits {
                requested_rows,
                max_cells: config.max_cells(),
                max_payload_bytes: config.max_payload_bytes(),
            },
        },
        warnings: Vec::new(),
    })
}

pub async fn execute_list_workbooks(
    state: Arc<AppState>,
    request: ListWorkbooksRequest,
) -> anyhow::Result<ListWorkbooksData> {
    let response = tools::list_workbooks(
        state,
        tools::ListWorkbooksParams {
            slug_prefix: request.slug_prefix,
            folder: request.folder,
            path_glob: request.path_glob,
            limit: request.limit,
            offset: request.offset,
            include_paths: request.include_paths,
        },
    )
    .await?;
    let workbooks = response
        .workbooks
        .into_iter()
        .map(|workbook| {
            Ok(CanonicalWorkbookDescriptor {
                resource_id: ResourceId::bind_workbook(&workbook.workbook_id)
                    .map_err(anyhow::Error::msg)?,
                metadata: WorkbookDiscoveryMetadata {
                    short_id: workbook.short_id,
                    slug: workbook.slug,
                    bytes: workbook.bytes,
                    last_modified: workbook.last_modified,
                    revision_id: workbook.revision_id,
                },
                paths: (workbook.path.is_some() || workbook.client_path.is_some()).then_some(
                    WorkbookPaths {
                        internal: workbook.path,
                        client: workbook.client_path,
                    },
                ),
                backend_capabilities: workbook.caps,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(ListWorkbooksData {
        workbooks,
        next_offset: response.next_offset,
    })
}

pub async fn execute_describe(
    state: Arc<AppState>,
    request: DescribeWorkbookRequest,
) -> anyhow::Result<DescribeWorkbookData> {
    let workbook_id = request.resource_id.to_workbook_id();
    let description = tools::describe_workbook(
        state.clone(),
        tools::DescribeWorkbookParams {
            workbook_or_fork_id: workbook_id.clone(),
        },
    )
    .await?;
    let summary = if request.include.contains(&DescribeInclude::Summary) {
        let options = request.summary.unwrap_or(DescribeSummaryOptions {
            include_entry_points: None,
            include_named_ranges: None,
        });
        let value = tools::workbook_summary(
            state,
            tools::WorkbookSummaryParams {
                workbook_or_fork_id: workbook_id,
                summary_only: Some(false),
                include_entry_points: options.include_entry_points,
                include_named_ranges: options.include_named_ranges,
            },
        )
        .await?;
        Some(WorkbookDerivedSummary {
            status: CoverageStatus::Complete,
            total_cells: value.total_cells,
            total_formulas: value.total_formulas,
            breakdown: value.breakdown,
            region_counts: value.region_counts,
            key_named_ranges: value.key_named_ranges,
            suggested_entry_points: value.suggested_entry_points,
            notes: value.notes,
            coverage: WorkbookSummaryCoverage {
                sheets_scanned: description.sheet_count,
                sheets_total: description.sheet_count,
                bounded: false,
            },
        })
    } else {
        None
    };
    Ok(DescribeWorkbookData {
        metadata: WorkbookExactMetadata {
            short_id: description.short_id,
            slug: description.slug,
            bytes: description.bytes,
            last_modified: description.last_modified,
            sheet_count: description.sheet_count,
            defined_name_count: description.defined_names,
            table_count: description.tables,
            macros_present: description.macros_present,
        },
        paths: WorkbookPaths {
            internal: Some(description.path),
            client: description.client_path,
        },
        capabilities: WorkbookCapabilities {
            backend: description.caps,
        },
        summary,
        warnings: Vec::new(),
    })
}

pub async fn execute_search_formulas(
    state: Arc<AppState>,
    request: SearchFormulasRequest,
) -> anyhow::Result<SearchFormulasData> {
    if request.query.is_none()
        && request.filter.as_ref().is_none_or(|filter| {
            filter.volatile.is_none()
                && filter.function_names.is_empty()
                && filter.has_external_references.is_none()
        })
    {
        anyhow::bail!("at least one of query or a non-empty filter is required");
    }
    let workbook_id = request.resource_id.to_workbook_id();
    let sheet_name = match request.scope.as_ref() {
        Some(FormulaSearchScope::Sheet { sheet_name, .. }) => Some(sheet_name.clone()),
        _ => None,
    };
    let find = tools::find_formula(
        state,
        tools::FindFormulaParams {
            workbook_or_fork_id: workbook_id,
            query: String::new(),
            sheet_name,
            case_sensitive: true,
            include_context: request.include_context.is_some(),
            limit: 500,
            offset: 0,
            context_rows: request.include_context.as_ref().map(|value| value.rows),
            context_cols: request.include_context.as_ref().map(|value| value.columns),
        },
    )
    .await?;
    let scanned = find.matches.len();
    let mut cells = Vec::new();
    for matched in find.matches {
        let functions = formula_functions(&matched.formula);
        let volatile = functions.iter().any(|name| is_volatile_function(name));
        let external = matched.formula.contains('[') && matched.formula.contains(']');
        let query_matches = request
            .query
            .as_ref()
            .is_none_or(|query| formula_query_matches(&matched.formula, query));
        let filter_matches = request.filter.as_ref().is_none_or(|filter| {
            filter.volatile.is_none_or(|wanted| wanted == volatile)
                && (filter.function_names.is_empty()
                    || filter.function_names.iter().all(|wanted| {
                        functions
                            .iter()
                            .any(|actual| actual.eq_ignore_ascii_case(wanted))
                    }))
                && filter
                    .has_external_references
                    .is_none_or(|wanted| wanted == external)
        });
        if query_matches && filter_matches {
            cells.push(FormulaCellMatch {
                sheet_name: matched.sheet_name,
                address: matched.address,
                formula: matched.formula,
                cached_value: matched.cached_value,
                classifications: FormulaClassifications {
                    volatile,
                    functions,
                    external_references: external,
                },
                context: matched.context,
            });
        }
    }
    let total_matches = cells.len();
    let offset = request.offset.unwrap_or(0) as usize;
    let limit = request.limit.unwrap_or(50).clamp(1, 500) as usize;
    let mut groups = Vec::new();
    let mut total_groups = 0_usize;
    if request.result_mode == FormulaResultMode::Groups {
        let mut grouped: BTreeMap<String, Vec<&FormulaCellMatch>> = BTreeMap::new();
        for cell in &cells {
            let keys = match request.group_by.unwrap_or(FormulaGroupBy::Function) {
                FormulaGroupBy::Function if !cell.classifications.functions.is_empty() => {
                    cell.classifications.functions.clone()
                }
                FormulaGroupBy::NormalizedFormula => vec![cell.formula.to_ascii_uppercase()],
                FormulaGroupBy::Fingerprint => {
                    vec![format!("{:x}", Sha256::digest(cell.formula.as_bytes()))]
                }
                FormulaGroupBy::Function => vec!["OTHER".to_string()],
            };
            for key in keys {
                grouped.entry(key).or_default().push(cell);
            }
        }
        let address_limit = request.addresses_per_group.unwrap_or(15) as usize;
        groups = grouped
            .into_iter()
            .map(|(key, values)| {
                let addresses = if request.include_addresses.unwrap_or(true) {
                    values
                        .iter()
                        .take(address_limit)
                        .map(|cell| format!("{}!{}", cell.sheet_name, cell.address))
                        .collect()
                } else {
                    Vec::new()
                };
                FormulaSearchGroup {
                    group_key: key.clone(),
                    formula: matches!(request.group_by, Some(FormulaGroupBy::NormalizedFormula))
                        .then(|| values[0].formula.clone()),
                    fingerprint: matches!(request.group_by, Some(FormulaGroupBy::Fingerprint))
                        .then_some(key),
                    volatile: values.iter().any(|cell| cell.classifications.volatile),
                    functions: values
                        .iter()
                        .flat_map(|cell| cell.classifications.functions.clone())
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter()
                        .collect(),
                    cell_count: values.len(),
                    addresses_complete: addresses.len() == values.len(),
                    addresses,
                }
            })
            .collect();
        total_groups = groups.len();
        groups = groups.into_iter().skip(offset).take(limit).collect();
        cells.clear();
    } else {
        cells = cells.into_iter().skip(offset).take(limit).collect();
    }
    let returned = if request.result_mode == FormulaResultMode::Groups {
        groups.len()
    } else {
        cells.len()
    };
    let next_offset = (offset + returned
        < if request.result_mode == FormulaResultMode::Groups {
            total_groups
        } else {
            total_matches
        })
    .then_some((offset + returned) as u32);
    Ok(SearchFormulasData {
        result_mode: request.result_mode,
        matches: cells,
        groups,
        summary: FormulaSearchSummary {
            formula_cells_scanned: scanned,
            matched_cells: total_matches,
            matched_groups: total_groups,
            scan_complete: find.next_offset.is_none(),
        },
        formula_parse_diagnostics: None,
        next_offset,
        warnings: Vec::new(),
    })
}

fn formula_query_matches(formula: &str, query: &FormulaQuery) -> bool {
    let (formula, needle) = if query.case_sensitive {
        (formula.to_string(), query.text.clone())
    } else {
        (
            formula.to_ascii_lowercase(),
            query.text.to_ascii_lowercase(),
        )
    };
    match query.match_mode.unwrap_or(MatchMode::Contains) {
        MatchMode::Contains => formula.contains(&needle),
        MatchMode::Exact => formula == needle,
        MatchMode::Prefix => formula.starts_with(&needle),
        MatchMode::Regex => {
            regex::Regex::new(&query.text).is_ok_and(|regex| regex.is_match(formula.as_str()))
        }
    }
}

fn formula_functions(formula: &str) -> Vec<String> {
    let regex =
        regex::Regex::new(r"(?i)([A-Z][A-Z0-9_.]*)\s*\(").expect("static formula function regex");
    regex
        .captures_iter(formula)
        .filter_map(|capture| capture.get(1))
        .map(|name| name.as_str().to_ascii_uppercase())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn is_volatile_function(name: &str) -> bool {
    matches!(
        name,
        "NOW" | "TODAY" | "RAND" | "RANDBETWEEN" | "OFFSET" | "INDIRECT" | "CELL" | "INFO"
    )
}

pub async fn execute_analyze_styles(
    state: Arc<AppState>,
    request: AnalyzeStylesRequest,
) -> anyhow::Result<AnalyzeStylesData> {
    let workbook_id = request.resource_id.to_workbook_id();
    let include_descriptor = request.include.contains(&StyleInclude::Descriptors);
    let include_examples = request.include.contains(&StyleInclude::ExampleCells);
    let include_ranges = request.include.contains(&StyleInclude::Ranges);
    match request.scope {
        AnalyzeStylesScope::Workbook => {
            if request.include.contains(&StyleInclude::Ranges) {
                anyhow::bail!("ranges are only available for sheet scope");
            }
            let max_scan = request
                .limits
                .as_ref()
                .and_then(|limits| limits.cells_scanned);
            let response = tools::workbook_style_summary(
                state,
                tools::WorkbookStyleSummaryParams {
                    workbook_or_fork_id: workbook_id,
                    max_styles: request.limits.as_ref().and_then(|limits| limits.styles),
                    max_conditional_formats: None,
                    max_cells_scan: max_scan,
                    summary_only: Some(false),
                    include_descriptor: Some(include_descriptor),
                    include_example_cells: Some(include_examples),
                    include_theme: Some(request.include.contains(&StyleInclude::Theme)),
                    include_conditional_formats: Some(
                        request.include.contains(&StyleInclude::ConditionalFormats),
                    ),
                },
            )
            .await?;
            let bounded = response.scan_truncated;
            let cells_scanned = response
                .styles
                .iter()
                .map(|style| u64::from(style.occurrences))
                .sum();
            Ok(AnalyzeStylesData {
                scope: AnalyzeStylesScope::Workbook,
                styles: response
                    .styles
                    .into_iter()
                    .map(|style| {
                        let mut examples = style.example_cells;
                        examples.sort();
                        CanonicalStyleUsage {
                            style_id: style.style_id,
                            occurrences: style.occurrences,
                            tags: style.tags,
                            descriptor: style.descriptor,
                            ranges: Vec::new(),
                            ranges_complete: true,
                            example_cells: examples,
                        }
                    })
                    .collect(),
                theme: response.theme,
                conditional_formats: response.conditional_formats,
                coverage: StyleCoverage {
                    status: if bounded {
                        CoverageStatus::Bounded
                    } else {
                        CoverageStatus::Complete
                    },
                    cells_scanned,
                    cells_in_scope: (!bounded).then_some(cells_scanned),
                    counts_exact: !bounded,
                },
                warnings: response
                    .notes
                    .into_iter()
                    .map(|message| Warning {
                        code: "STYLE_ANALYSIS_NOTE".to_string(),
                        message,
                    })
                    .collect(),
            })
        }
        AnalyzeStylesScope::Sheet {
            sheet_name,
            selection,
        } => {
            if request.include.contains(&StyleInclude::Theme) {
                anyhow::bail!("theme is only available for workbook scope");
            }
            let response_scope = AnalyzeStylesScope::Sheet {
                sheet_name: sheet_name.clone(),
                selection: selection.clone(),
            };
            let scope = match selection {
                Some(StyleSelection::Range { range }) => {
                    Some(tools::SheetStylesScope::Range { range })
                }
                Some(StyleSelection::Region { region_id }) => {
                    Some(tools::SheetStylesScope::Region { region_id })
                }
                _ => None,
            };
            let response = tools::sheet_styles(
                state.clone(),
                tools::SheetStylesParams {
                    workbook_or_fork_id: workbook_id.clone(),
                    sheet_name: sheet_name.clone(),
                    scope,
                    granularity: request.group_by,
                    max_items: request
                        .limits
                        .as_ref()
                        .and_then(|limits| limits.styles)
                        .map(|value| value as usize),
                    summary_only: Some(false),
                    include_descriptor: Some(include_descriptor),
                    include_ranges: Some(include_ranges),
                    include_example_cells: Some(include_examples),
                },
            )
            .await?;
            let conditional_formats = if request.include.contains(&StyleInclude::ConditionalFormats)
            {
                tools::workbook_style_summary(
                    state,
                    tools::WorkbookStyleSummaryParams {
                        workbook_or_fork_id: workbook_id,
                        max_styles: Some(0),
                        max_conditional_formats: None,
                        max_cells_scan: Some(0),
                        summary_only: Some(false),
                        include_descriptor: Some(false),
                        include_example_cells: Some(false),
                        include_theme: Some(false),
                        include_conditional_formats: Some(true),
                    },
                )
                .await?
                .conditional_formats
                .into_iter()
                .filter(|format| format.sheet_name == sheet_name)
                .collect()
            } else {
                Vec::new()
            };
            let bounded = response.styles_truncated;
            let cells_scanned = response
                .styles
                .iter()
                .map(|style| u64::from(style.occurrences))
                .sum();
            Ok(AnalyzeStylesData {
                scope: response_scope,
                styles: response
                    .styles
                    .into_iter()
                    .map(|style| {
                        let mut examples = style.example_cells;
                        examples.sort();
                        let mut ranges = style.cell_ranges;
                        ranges.sort();
                        CanonicalStyleUsage {
                            style_id: style.style_id,
                            occurrences: style.occurrences,
                            tags: style.tags,
                            descriptor: style.descriptor,
                            ranges,
                            ranges_complete: !style.ranges_truncated,
                            example_cells: examples,
                        }
                    })
                    .collect(),
                theme: None,
                conditional_formats,
                coverage: StyleCoverage {
                    status: if bounded {
                        CoverageStatus::Bounded
                    } else {
                        CoverageStatus::Complete
                    },
                    cells_scanned,
                    cells_in_scope: (!bounded).then_some(cells_scanned),
                    counts_exact: !bounded,
                },
                warnings: Vec::new(),
            })
        }
    }
}

pub async fn execute_export_grid(
    state: Arc<AppState>,
    request: ExportGridRequest,
    revision_id: &str,
) -> Result<ExportGridData, CanonicalErrorEnvelope> {
    let operation = "export_grid";
    let (c1, r1, c2, r2) = parse_range(&request.range).map_err(|message| {
        canonical_error(
            CanonicalErrorCode::InvalidRequest,
            operation,
            message,
            Some("$.range"),
        )
    })?;
    let fingerprint = format!(
        "{:x}",
        Sha256::digest(format!("{}|{}", request.sheet_name, request.range).as_bytes())
    );
    let start_row = if let Some(value) = &request.cursor {
        let cursor = decode_cursor(value).map_err(|message| {
            canonical_error(
                CanonicalErrorCode::InvalidRequest,
                operation,
                message,
                Some("$.cursor"),
            )
        })?;
        if cursor.revision_id != revision_id {
            return Err(canonical_error(
                CanonicalErrorCode::StaleCursor,
                operation,
                "cursor is bound to a different resource revision",
                Some("$.cursor"),
            ));
        }
        if cursor.fingerprint != fingerprint {
            return Err(canonical_error(
                CanonicalErrorCode::CursorMismatch,
                operation,
                "cursor does not match the export_grid request",
                Some("$.cursor"),
            ));
        }
        cursor.next_row
    } else {
        r1
    };
    let end_row = r2.min(
        start_row
            .saturating_add(request.page_size.unwrap_or(500).clamp(1, 5000))
            .saturating_sub(1),
    );
    let returned_range = range_string(c1, start_row, c2, end_row);
    let mut grid = tools::grid_export(
        state,
        tools::GridExportParams {
            workbook_or_fork_id: request.resource_id.to_workbook_id(),
            sheet_name: request.sheet_name,
            range: returned_range.clone(),
        },
    )
    .await
    .map_err(|error| {
        canonical_error(
            CanonicalErrorCode::OperationFailed,
            operation,
            error.to_string(),
            None,
        )
    })?;
    let mut populated = BTreeMap::new();
    for row in std::mem::take(&mut grid.rows) {
        for cell in row.cells {
            populated.insert(cell.offset, cell);
        }
    }
    grid.rows = (0..=end_row - start_row)
        .map(|row_offset| GridRow {
            cells: (0..=c2 - c1)
                .map(|column_offset| {
                    populated
                        .remove(&[row_offset, column_offset])
                        .unwrap_or(GridCell {
                            offset: [row_offset, column_offset],
                            v: None,
                            f: None,
                            fmt: None,
                            style: None,
                        })
                })
                .collect(),
        })
        .collect();
    let next_cursor = if end_row < r2 {
        Some(
            encode_cursor(&ReadCellsCursor {
                revision_id: revision_id.to_string(),
                fingerprint,
                selection_index: 0,
                next_row: end_row + 1,
            })
            .map_err(|error| {
                canonical_error(
                    CanonicalErrorCode::OperationFailed,
                    operation,
                    error.to_string(),
                    None,
                )
            })?,
        )
    } else {
        None
    };
    Ok(ExportGridData {
        losslessness: GridLosslessness::Lossless,
        requested_range: request.range,
        returned_range,
        grid,
        complete: next_cursor.is_none(),
        next_cursor,
    })
}
