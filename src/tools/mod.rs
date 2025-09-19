pub mod filters;

use crate::analysis::{formula::FormulaGraph, stats};
use crate::model::*;
use crate::state::AppState;
use crate::workbook::{WorkbookContext, cell_to_value};
use anyhow::{Result, anyhow};
use schemars::JsonSchema;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const DEFAULT_TRACE_PAGE_SIZE: usize = 20;
const TRACE_PAGE_MIN: usize = 5;
const TRACE_PAGE_MAX: usize = 200;
const TRACE_RANGE_THRESHOLD: usize = 4;
const TRACE_RANGE_HIGHLIGHT_LIMIT: usize = 3;
const TRACE_GROUP_HIGHLIGHT_LIMIT: usize = 3;
const TRACE_CELL_HIGHLIGHT_LIMIT: usize = 5;
const TRACE_RANGE_VALUE_SAMPLES: usize = 3;
const TRACE_RANGE_FORMULA_SAMPLES: usize = 2;
const TRACE_GROUP_SAMPLE_LIMIT: usize = 5;

pub async fn list_workbooks(
    state: Arc<AppState>,
    params: ListWorkbooksParams,
) -> Result<WorkbookListResponse> {
    let filter = params.into_filter()?;
    state.list_workbooks(filter)
}

pub async fn describe_workbook(
    state: Arc<AppState>,
    params: DescribeWorkbookParams,
) -> Result<WorkbookDescription> {
    let workbook = state.open_workbook(&params.workbook_id).await?;
    let desc = workbook.describe();
    Ok(desc)
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListWorkbooksParams {
    pub slug_prefix: Option<String>,
    pub folder: Option<String>,
    pub path_glob: Option<String>,
}

impl ListWorkbooksParams {
    fn into_filter(self) -> Result<filters::WorkbookFilter> {
        filters::WorkbookFilter::new(self.slug_prefix, self.folder, self.path_glob)
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DescribeWorkbookParams {
    pub workbook_id: WorkbookId,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListSheetsParams {
    pub workbook_id: WorkbookId,
}

pub async fn list_sheets(
    state: Arc<AppState>,
    params: ListSheetsParams,
) -> Result<SheetListResponse> {
    let workbook = state.open_workbook(&params.workbook_id).await?;
    let summaries = workbook.list_summaries()?;
    let response = SheetListResponse {
        workbook_id: workbook.id.clone(),
        workbook_short_id: workbook.short_id.clone(),
        sheets: summaries,
    };
    Ok(response)
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SheetOverviewParams {
    pub workbook_id: WorkbookId,
    pub sheet_name: String,
}

pub async fn sheet_overview(
    state: Arc<AppState>,
    params: SheetOverviewParams,
) -> Result<SheetOverviewResponse> {
    let workbook = state.open_workbook(&params.workbook_id).await?;
    let overview = workbook.sheet_overview(&params.sheet_name)?;
    Ok(overview)
}

fn default_start_row() -> u32 {
    1
}

fn default_page_size() -> u32 {
    50
}

fn default_include_formulas() -> bool {
    true
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SheetPageParams {
    pub workbook_id: WorkbookId,
    pub sheet_name: String,
    #[serde(default = "default_start_row")]
    pub start_row: u32,
    #[serde(default = "default_page_size")]
    pub page_size: u32,
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    #[serde(default = "default_include_formulas")]
    pub include_formulas: bool,
    #[serde(default)]
    pub include_styles: bool,
}

pub async fn sheet_page(
    state: Arc<AppState>,
    params: SheetPageParams,
) -> Result<SheetPageResponse> {
    if params.page_size == 0 {
        return Err(anyhow!("page_size must be greater than zero"));
    }

    let workbook = state.open_workbook(&params.workbook_id).await?;
    let metrics = workbook.get_sheet_metrics(&params.sheet_name)?;

    let start_row = params.start_row.max(1);
    let page_size = params.page_size.min(500);
    let include_formulas = params.include_formulas;
    let include_styles = params.include_styles;
    let columns = params.columns.clone();

    let page = workbook.with_sheet(&params.sheet_name, |sheet| {
        build_page(
            sheet,
            start_row,
            page_size,
            columns.clone(),
            include_formulas,
            include_styles,
        )
    })?;

    let has_more = page.end_row < metrics.metrics.row_count;
    let next_start_row = if has_more {
        Some(page.end_row + 1)
    } else {
        None
    };

    let response = SheetPageResponse {
        workbook_id: workbook.id.clone(),
        workbook_short_id: workbook.short_id.clone(),
        sheet_name: params.sheet_name,
        rows: page.rows,
        has_more,
        next_start_row,
        header_row: page.header,
    };
    Ok(response)
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SheetFormulaMapParams {
    pub workbook_id: WorkbookId,
    pub sheet_name: String,
    pub range: Option<String>,
    #[serde(default)]
    pub expand: bool,
}

pub async fn sheet_formula_map(
    state: Arc<AppState>,
    params: SheetFormulaMapParams,
) -> Result<SheetFormulaMapResponse> {
    let workbook = state.open_workbook(&params.workbook_id).await?;
    let graph = workbook.formula_graph(&params.sheet_name)?;
    let mut groups = Vec::new();
    let mut truncated = false;

    for mut group in graph.groups() {
        if let Some(range) = &params.range {
            group.addresses = group
                .addresses
                .into_iter()
                .filter(|addr| address_in_range(addr, range))
                .collect();
            if group.addresses.is_empty() {
                continue;
            }
        }
        if !params.expand && group.addresses.len() > 15 {
            group.addresses.truncate(15);
            truncated = true;
        }
        groups.push(group);
    }

    let response = SheetFormulaMapResponse {
        workbook_id: workbook.id.clone(),
        workbook_short_id: workbook.short_id.clone(),
        sheet_name: params.sheet_name.clone(),
        groups,
        truncated,
    };
    Ok(response)
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FormulaTraceParams {
    pub workbook_id: WorkbookId,
    pub sheet_name: String,
    pub cell_address: String,
    pub direction: TraceDirection,
    pub depth: Option<u32>,
    pub limit: Option<u32>,
    #[serde(default)]
    pub page_size: Option<usize>,
    #[serde(default)]
    pub cursor: Option<TraceCursor>,
}

pub async fn formula_trace(
    state: Arc<AppState>,
    params: FormulaTraceParams,
) -> Result<FormulaTraceResponse> {
    let workbook = state.open_workbook(&params.workbook_id).await?;
    let graph = workbook.formula_graph(&params.sheet_name)?;
    let formula_lookup = build_formula_lookup(&graph);
    let depth = params.depth.unwrap_or(3).max(1).min(5);
    let page_size = params
        .page_size
        .or_else(|| params.limit.map(|v| v as usize))
        .unwrap_or(DEFAULT_TRACE_PAGE_SIZE)
        .clamp(TRACE_PAGE_MIN, TRACE_PAGE_MAX);

    let origin = params.cell_address.to_uppercase();
    let (layers, next_cursor, notes) = build_trace_layers(
        &workbook,
        &graph,
        &formula_lookup,
        &params.direction,
        &origin,
        &params.sheet_name,
        depth,
        page_size,
        params.cursor.clone(),
    )?;

    let response = FormulaTraceResponse {
        workbook_id: workbook.id.clone(),
        workbook_short_id: workbook.short_id.clone(),
        sheet_name: params.sheet_name.clone(),
        origin,
        direction: params.direction.clone(),
        layers,
        next_cursor,
        notes,
    };
    Ok(response)
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct NamedRangesParams {
    pub workbook_id: WorkbookId,
    pub sheet_name: Option<String>,
    pub name_prefix: Option<String>,
}

pub async fn named_ranges(
    state: Arc<AppState>,
    params: NamedRangesParams,
) -> Result<NamedRangesResponse> {
    let workbook = state.open_workbook(&params.workbook_id).await?;
    let mut items = workbook.named_items()?;

    if let Some(sheet_filter) = &params.sheet_name {
        items.retain(|item| {
            item.sheet_name
                .as_ref()
                .map(|name| name.eq_ignore_ascii_case(sheet_filter))
                .unwrap_or(false)
        });
    }
    if let Some(prefix) = &params.name_prefix {
        let prefix_lower = prefix.to_ascii_lowercase();
        items.retain(|item| item.name.to_ascii_lowercase().starts_with(&prefix_lower));
    }

    let response = NamedRangesResponse {
        workbook_id: workbook.id.clone(),
        workbook_short_id: workbook.short_id.clone(),
        items,
    };
    Ok(response)
}

struct PageBuildResult {
    rows: Vec<RowSnapshot>,
    header: Option<RowSnapshot>,
    end_row: u32,
}

fn build_page(
    sheet: &umya_spreadsheet::Worksheet,
    start_row: u32,
    page_size: u32,
    columns: Option<Vec<String>>,
    include_formulas: bool,
    include_styles: bool,
) -> PageBuildResult {
    let max_col = sheet.get_highest_column();
    let end_row = (start_row + page_size - 1).min(sheet.get_highest_row().max(start_row));
    let column_indices = resolve_columns(columns.as_ref(), max_col);

    let header = build_row_snapshot(sheet, 1, &column_indices, include_formulas, include_styles);

    let mut rows = Vec::new();
    for row_idx in start_row..=end_row {
        rows.push(build_row_snapshot(
            sheet,
            row_idx,
            &column_indices,
            include_formulas,
            include_styles,
        ));
    }

    PageBuildResult {
        rows,
        header: Some(header),
        end_row,
    }
}

fn build_row_snapshot(
    sheet: &umya_spreadsheet::Worksheet,
    row_index: u32,
    columns: &[u32],
    include_formulas: bool,
    include_styles: bool,
) -> RowSnapshot {
    let mut cells = Vec::new();
    for &col in columns {
        if let Some(cell) = sheet.get_cell((row_index, col)) {
            cells.push(build_cell_snapshot(cell, include_formulas, include_styles));
        } else {
            let address = crate::utils::cell_address(col, row_index);
            cells.push(CellSnapshot {
                address,
                value: None,
                formula: None,
                cached_value: None,
                number_format: None,
                style_tags: Vec::new(),
                notes: Vec::new(),
            });
        }
    }

    RowSnapshot { row_index, cells }
}

fn build_cell_snapshot(
    cell: &umya_spreadsheet::Cell,
    include_formulas: bool,
    include_styles: bool,
) -> CellSnapshot {
    let address = cell.get_coordinate().get_coordinate();
    let value = crate::workbook::cell_to_value(cell);
    let formula = if include_formulas && cell.is_formula() {
        Some(cell.get_formula().to_string())
    } else {
        None
    };
    let cached_value = if cell.is_formula() {
        value.clone()
    } else {
        None
    };
    let number_format = if include_styles {
        cell.get_style()
            .get_number_format()
            .map(|fmt| fmt.get_format_code().to_string())
    } else {
        None
    };
    let style_tags = if include_styles {
        crate::analysis::style::tag_cell(cell)
            .map(|(_, tagging)| tagging.tags)
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    CellSnapshot {
        address,
        value,
        formula,
        cached_value,
        number_format,
        style_tags,
        notes: Vec::new(),
    }
}

fn resolve_columns(columns: Option<&Vec<String>>, max_column: u32) -> Vec<u32> {
    use std::collections::BTreeSet;
    use umya_spreadsheet::helper::coordinate::column_index_from_string;

    let mut indices = BTreeSet::new();
    if let Some(specs) = columns {
        for spec in specs {
            if let Some((start, end)) = spec.split_once(':') {
                let start_idx = column_index_from_string(start);
                let end_idx = column_index_from_string(end);
                let (min_idx, max_idx) = if start_idx <= end_idx {
                    (start_idx, end_idx)
                } else {
                    (end_idx, start_idx)
                };
                for idx in min_idx..=max_idx {
                    indices.insert(idx);
                }
            } else {
                indices.insert(column_index_from_string(spec));
            }
        }
    } else {
        for idx in 1..=max_column.max(1) {
            indices.insert(idx);
        }
    }

    indices.into_iter().collect()
}
fn default_stats_sample() -> usize {
    500
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SheetStatisticsParams {
    pub workbook_id: WorkbookId,
    pub sheet_name: String,
    #[serde(default)]
    pub sample_rows: Option<usize>,
}

pub async fn sheet_statistics(
    state: Arc<AppState>,
    params: SheetStatisticsParams,
) -> Result<SheetStatisticsResponse> {
    let workbook = state.open_workbook(&params.workbook_id).await?;
    let sheet_metrics = workbook.get_sheet_metrics(&params.sheet_name)?;
    let sample_rows = params.sample_rows.unwrap_or_else(default_stats_sample);
    let stats = workbook.with_sheet(&params.sheet_name, |sheet| {
        stats::compute_sheet_statistics(sheet, sample_rows)
    })?;
    let response = SheetStatisticsResponse {
        workbook_id: workbook.id.clone(),
        workbook_short_id: workbook.short_id.clone(),
        sheet_name: params.sheet_name,
        row_count: sheet_metrics.metrics.row_count,
        column_count: sheet_metrics.metrics.column_count,
        density: stats.density,
        numeric_columns: stats.numeric_columns,
        text_columns: stats.text_columns,
        null_counts: stats.null_counts,
        duplicate_warnings: stats.duplicate_warnings,
    };
    Ok(response)
}

fn address_in_range(address: &str, range: &str) -> bool {
    parse_range(range).map_or(true, |((start_col, start_row), (end_col, end_row))| {
        if let Some((col, row)) = parse_address(address) {
            col >= start_col && col <= end_col && row >= start_row && row <= end_row
        } else {
            false
        }
    })
}

fn parse_range(range: &str) -> Option<((u32, u32), (u32, u32))> {
    let mut parts = range.split(':');
    let start = parts.next()?;
    let end = parts.next().unwrap_or(start);
    let start_idx = parse_address(start)?;
    let end_idx = parse_address(end)?;
    Some((
        (start_idx.0.min(end_idx.0), start_idx.1.min(end_idx.1)),
        (start_idx.0.max(end_idx.0), start_idx.1.max(end_idx.1)),
    ))
}

fn parse_address(address: &str) -> Option<(u32, u32)> {
    use umya_spreadsheet::helper::coordinate::index_from_coordinate;
    let (col, row, _, _) = index_from_coordinate(address);
    match (col, row) {
        (Some(c), Some(r)) => Some((c, r)),
        _ => None,
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindFormulaParams {
    pub workbook_id: WorkbookId,
    pub query: String,
    pub sheet_name: Option<String>,
    #[serde(default)]
    pub case_sensitive: bool,
}

pub async fn find_formula(
    state: Arc<AppState>,
    params: FindFormulaParams,
) -> Result<FindFormulaResponse> {
    let workbook = state.open_workbook(&params.workbook_id).await?;
    let query = if params.case_sensitive {
        params.query.clone()
    } else {
        params.query.to_ascii_lowercase()
    };
    let mut matches = Vec::new();

    let sheet_names: Vec<String> = if let Some(sheet) = &params.sheet_name {
        vec![sheet.clone()]
    } else {
        workbook.sheet_names()
    };

    for sheet_name in sheet_names {
        let sheet_matches = workbook.with_sheet(&sheet_name, |sheet| {
            collect_formula_matches(sheet, &sheet_name, &query, params.case_sensitive)
        })?;
        matches.extend(sheet_matches);
    }

    let response = FindFormulaResponse {
        workbook_id: workbook.id.clone(),
        workbook_short_id: workbook.short_id.clone(),
        matches,
    };
    Ok(response)
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScanVolatilesParams {
    pub workbook_id: WorkbookId,
    pub sheet_name: Option<String>,
}

pub async fn scan_volatiles(
    state: Arc<AppState>,
    params: ScanVolatilesParams,
) -> Result<VolatileScanResponse> {
    let workbook = state.open_workbook(&params.workbook_id).await?;
    let target_sheets: Vec<String> = if let Some(sheet) = &params.sheet_name {
        vec![sheet.clone()]
    } else {
        workbook.sheet_names()
    };

    let mut items = Vec::new();
    let mut truncated = false;

    for sheet_name in target_sheets {
        let graph = workbook.formula_graph(&sheet_name)?;
        for group in graph.groups() {
            if !group.is_volatile {
                continue;
            }
            for address in group.addresses.iter().take(50) {
                items.push(VolatileScanEntry {
                    address: address.clone(),
                    sheet_name: sheet_name.clone(),
                    function: "volatile".to_string(),
                    note: Some(group.formula.clone()),
                });
            }
            if group.addresses.len() > 50 {
                truncated = true;
            }
        }
    }

    let response = VolatileScanResponse {
        workbook_id: workbook.id.clone(),
        workbook_short_id: workbook.short_id.clone(),
        items,
        truncated,
    };
    Ok(response)
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SheetStylesParams {
    pub workbook_id: WorkbookId,
    pub sheet_name: String,
}

pub async fn sheet_styles(
    state: Arc<AppState>,
    params: SheetStylesParams,
) -> Result<SheetStylesResponse> {
    let workbook = state.open_workbook(&params.workbook_id).await?;
    let entry = workbook.get_sheet_metrics(&params.sheet_name)?;

    let styles = entry
        .metrics
        .style_map
        .iter()
        .map(|(style_id, usage)| StyleSummary {
            style_id: style_id.clone(),
            occurrences: usage.occurrences,
            tags: usage.tags.clone(),
            example_cells: usage.example_cells.clone(),
        })
        .collect();

    let response = SheetStylesResponse {
        workbook_id: workbook.id.clone(),
        workbook_short_id: workbook.short_id.clone(),
        sheet_name: params.sheet_name.clone(),
        styles,
        conditional_rules: Vec::new(),
    };
    Ok(response)
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ManifestStubParams {
    pub workbook_id: WorkbookId,
    pub sheet_filter: Option<String>,
}

pub async fn get_manifest_stub(
    state: Arc<AppState>,
    params: ManifestStubParams,
) -> Result<ManifestStubResponse> {
    let workbook = state.open_workbook(&params.workbook_id).await?;
    let mut summaries = workbook.list_summaries()?;

    if let Some(filter) = &params.sheet_filter {
        summaries.retain(|summary| summary.name.eq_ignore_ascii_case(filter));
    }

    let sheets = summaries
        .into_iter()
        .map(|summary| ManifestSheetStub {
            sheet_name: summary.name.clone(),
            classification: summary.classification.clone(),
            candidate_expectations: vec![format!(
                "Review {} sheet for expectation candidates",
                format!("{:?}", summary.classification).to_ascii_lowercase()
            )],
            notes: summary.style_tags,
        })
        .collect();

    let response = ManifestStubResponse {
        workbook_id: workbook.id.clone(),
        workbook_short_id: workbook.short_id.clone(),
        slug: workbook.slug.clone(),
        sheets,
    };
    Ok(response)
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CloseWorkbookParams {
    pub workbook_id: WorkbookId,
}

pub async fn close_workbook(state: Arc<AppState>, params: CloseWorkbookParams) -> Result<String> {
    state.close_workbook(&params.workbook_id)?;
    let data = format!("workbook {} evicted", params.workbook_id.as_str());
    Ok(data)
}
fn collect_formula_matches(
    sheet: &umya_spreadsheet::Worksheet,
    sheet_name: &str,
    query: &str,
    case_sensitive: bool,
) -> Vec<FindFormulaMatch> {
    use crate::workbook::cell_to_value;

    let mut results = Vec::new();
    for cell in sheet.get_cell_collection() {
        if !cell.is_formula() {
            continue;
        }
        let formula = cell.get_formula();
        let haystack = if case_sensitive {
            formula.to_string()
        } else {
            formula.to_ascii_lowercase()
        };
        if !haystack.contains(query) {
            continue;
        }
        let coord = cell.get_coordinate();
        let column = *coord.get_col_num();
        let row = *coord.get_row_num();
        let columns = vec![column];
        let context_row = build_row_snapshot(sheet, row, &columns, true, false);
        let header_row = build_row_snapshot(sheet, 1, &columns, false, false);

        results.push(FindFormulaMatch {
            address: coord.get_coordinate(),
            sheet_name: sheet_name.to_string(),
            formula: formula.to_string(),
            cached_value: if cell.is_formula() {
                cell_to_value(cell)
            } else {
                None
            },
            context: vec![header_row, context_row],
        });
    }
    results
}

#[derive(Clone)]
struct TraceFormulaInfo {
    fingerprint: String,
    formula: String,
}

#[derive(Clone)]
struct TraceEdgeRaw {
    from: String,
    to: String,
    neighbor: String,
}

#[derive(Clone)]
struct LayerLinks {
    depth: u32,
    edges: Vec<TraceEdgeRaw>,
}

#[derive(Clone)]
struct NeighborDetail {
    address: String,
    column: Option<u32>,
    row: Option<u32>,
    kind: TraceCellKind,
    value: Option<CellValue>,
    formula: Option<String>,
    fingerprint: Option<String>,
    external: bool,
}

fn build_formula_lookup(graph: &FormulaGraph) -> HashMap<String, TraceFormulaInfo> {
    let mut map = HashMap::new();
    for group in graph.groups() {
        for address in group.addresses.clone() {
            map.insert(
                address.to_ascii_uppercase(),
                TraceFormulaInfo {
                    fingerprint: group.fingerprint.clone(),
                    formula: group.formula.clone(),
                },
            );
        }
    }
    map
}

fn build_trace_layers(
    workbook: &WorkbookContext,
    graph: &FormulaGraph,
    formula_lookup: &HashMap<String, TraceFormulaInfo>,
    direction: &TraceDirection,
    origin: &str,
    sheet_name: &str,
    depth_limit: u32,
    page_size: usize,
    cursor: Option<TraceCursor>,
) -> Result<(Vec<TraceLayer>, Option<TraceCursor>, Vec<String>)> {
    let layer_links = collect_layer_links(graph, direction, origin, depth_limit);
    let mut layers = Vec::new();
    let mut next_cursor = None;
    let mut notes = Vec::new();
    let focus_depth = cursor.as_ref().map(|c| c.depth);

    for layer in layer_links {
        let produce_edges = focus_depth.map_or(true, |depth| depth == layer.depth);
        let offset = cursor
            .as_ref()
            .filter(|c| c.depth == layer.depth)
            .map(|c| c.offset)
            .unwrap_or(0);

        let mut node_set: HashSet<String> = HashSet::new();
        for edge in &layer.edges {
            node_set.insert(edge.neighbor.clone());
        }
        let mut nodes: Vec<String> = node_set.into_iter().collect();
        nodes.sort_by(|a, b| compare_addresses(a, b));

        let details = workbook.with_sheet(sheet_name, |sheet| {
            collect_neighbor_details(sheet, sheet_name, &nodes, formula_lookup)
        })?;
        let total_nodes = details.len();
        let start = offset.min(total_nodes);
        let end = if produce_edges {
            (start + page_size).min(total_nodes)
        } else {
            start
        };
        let selected_slice = if produce_edges {
            &details[start..end]
        } else {
            &details[0..0]
        };
        let selected_addresses: HashSet<String> = selected_slice
            .iter()
            .map(|detail| detail.address.clone())
            .collect();

        let summary = build_layer_summary(&details);
        let range_highlights = build_range_highlights(&details);
        let group_highlights = build_formula_group_highlights(&details);
        let notable_cells = build_notable_cells(&details, &range_highlights, &group_highlights);

        let highlights = TraceLayerHighlights {
            top_ranges: range_highlights.clone(),
            top_formula_groups: group_highlights.clone(),
            notable_cells,
        };

        let edges = if produce_edges {
            build_edges_for_layer(&layer.edges, &selected_addresses, formula_lookup)
        } else {
            Vec::new()
        };

        let has_more = produce_edges && end < total_nodes;
        if has_more && next_cursor.is_none() {
            next_cursor = Some(TraceCursor {
                depth: layer.depth,
                offset: end,
            });
        }
        if has_more {
            notes.push(format!(
                "Layer {} truncated at {} of {} nodes; supply cursor.depth={} and cursor.offset={} to continue",
                layer.depth, end, total_nodes, layer.depth, end
            ));
        }

        layers.push(TraceLayer {
            depth: layer.depth,
            summary,
            highlights,
            edges,
            has_more,
        });
    }

    Ok((layers, next_cursor, notes))
}

fn collect_layer_links(
    graph: &FormulaGraph,
    direction: &TraceDirection,
    origin: &str,
    depth_limit: u32,
) -> Vec<LayerLinks> {
    let mut visited: HashSet<String> = HashSet::new();
    visited.insert(origin.to_string());
    let mut frontier = vec![origin.to_string()];
    let mut layers = Vec::new();

    for depth in 1..=depth_limit {
        let mut next_frontier_set: HashSet<String> = HashSet::new();
        let mut edges = Vec::new();

        for cell in &frontier {
            let neighbors = match direction {
                TraceDirection::Precedents => graph.precedents(cell),
                TraceDirection::Dependents => graph.dependents(cell),
            };

            for neighbor in neighbors {
                let neighbor_upper = neighbor.to_ascii_uppercase();
                let edge = match direction {
                    TraceDirection::Precedents => TraceEdgeRaw {
                        from: cell.clone(),
                        to: neighbor_upper.clone(),
                        neighbor: neighbor_upper.clone(),
                    },
                    TraceDirection::Dependents => TraceEdgeRaw {
                        from: neighbor_upper.clone(),
                        to: cell.clone(),
                        neighbor: neighbor_upper.clone(),
                    },
                };
                edges.push(edge);
                if visited.insert(neighbor_upper.clone()) {
                    next_frontier_set.insert(neighbor_upper);
                }
            }
        }

        if edges.is_empty() {
            break;
        }

        layers.push(LayerLinks { depth, edges });
        if next_frontier_set.is_empty() {
            break;
        }
        let mut next_frontier: Vec<String> = next_frontier_set.into_iter().collect();
        next_frontier.sort();
        frontier = next_frontier;
    }

    layers
}

fn collect_neighbor_details(
    sheet: &umya_spreadsheet::Worksheet,
    current_sheet: &str,
    addresses: &[String],
    formula_lookup: &HashMap<String, TraceFormulaInfo>,
) -> Vec<NeighborDetail> {
    let mut details = Vec::new();
    for address in addresses {
        let (sheet_part, cell_part) = split_sheet_and_cell(address);
        let normalized_sheet = sheet_part
            .as_ref()
            .map(|s| clean_sheet_name(s).to_ascii_lowercase());
        let is_external = normalized_sheet
            .as_ref()
            .map(|s| !s.eq_ignore_ascii_case(current_sheet))
            .unwrap_or(false);

        let Some(cell_ref) = cell_part else {
            details.push(NeighborDetail {
                address: address.clone(),
                column: None,
                row: None,
                kind: TraceCellKind::External,
                value: None,
                formula: None,
                fingerprint: None,
                external: true,
            });
            continue;
        };

        let cell_ref_upper = cell_ref.to_ascii_uppercase();

        if is_external {
            let formula_info = lookup_formula_info(formula_lookup, &cell_ref_upper, address);
            details.push(NeighborDetail {
                address: address.clone(),
                column: None,
                row: None,
                kind: TraceCellKind::External,
                value: None,
                formula: formula_info.map(|info| info.formula.clone()),
                fingerprint: formula_info.map(|info| info.fingerprint.clone()),
                external: true,
            });
            continue;
        }

        let Some((col, row)) = parse_address(&cell_ref_upper) else {
            details.push(NeighborDetail {
                address: address.clone(),
                column: None,
                row: None,
                kind: TraceCellKind::External,
                value: None,
                formula: None,
                fingerprint: None,
                external: true,
            });
            continue;
        };

        let cell_opt = sheet.get_cell((&col, &row));
        let formula_info = lookup_formula_info(formula_lookup, &cell_ref_upper, address);
        if let Some(cell) = cell_opt {
            let value = cell_to_value(cell);
            let kind = if cell.is_formula() {
                TraceCellKind::Formula
            } else if value.is_some() {
                TraceCellKind::Literal
            } else {
                TraceCellKind::Blank
            };
            details.push(NeighborDetail {
                address: address.clone(),
                column: Some(col),
                row: Some(row),
                kind,
                value,
                formula: formula_info.map(|info| info.formula.clone()),
                fingerprint: formula_info.map(|info| info.fingerprint.clone()),
                external: false,
            });
        } else {
            details.push(NeighborDetail {
                address: address.clone(),
                column: Some(col),
                row: Some(row),
                kind: TraceCellKind::Blank,
                value: None,
                formula: formula_info.map(|info| info.formula.clone()),
                fingerprint: formula_info.map(|info| info.fingerprint.clone()),
                external: false,
            });
        }
    }
    details
}

fn build_layer_summary(details: &[NeighborDetail]) -> TraceLayerSummary {
    let mut summary = TraceLayerSummary {
        total_nodes: details.len(),
        formula_nodes: 0,
        value_nodes: 0,
        blank_nodes: 0,
        external_nodes: 0,
        unique_formula_groups: 0,
    };

    let mut fingerprints: HashSet<String> = HashSet::new();

    for detail in details {
        match detail.kind {
            TraceCellKind::Formula => {
                summary.formula_nodes += 1;
                if let Some(fp) = &detail.fingerprint {
                    fingerprints.insert(fp.clone());
                }
            }
            TraceCellKind::Literal => summary.value_nodes += 1,
            TraceCellKind::Blank => summary.blank_nodes += 1,
            TraceCellKind::External => summary.external_nodes += 1,
        }
    }

    summary.unique_formula_groups = fingerprints.len();
    summary
}

fn build_formula_group_highlights(details: &[NeighborDetail]) -> Vec<TraceFormulaGroupHighlight> {
    let mut aggregates: HashMap<String, (String, usize, Vec<String>)> = HashMap::new();
    for detail in details {
        if let (Some(fingerprint), Some(formula)) = (&detail.fingerprint, &detail.formula) {
            let entry = aggregates
                .entry(fingerprint.clone())
                .or_insert_with(|| (formula.clone(), 0, Vec::new()));
            entry.1 += 1;
            if entry.2.len() < TRACE_GROUP_SAMPLE_LIMIT {
                entry.2.push(detail.address.clone());
            }
        }
    }

    let mut highlights: Vec<TraceFormulaGroupHighlight> = aggregates
        .into_iter()
        .map(
            |(fingerprint, (formula, count, sample_addresses))| TraceFormulaGroupHighlight {
                fingerprint,
                formula,
                count,
                sample_addresses,
            },
        )
        .collect();

    highlights.sort_by(|a, b| b.count.cmp(&a.count));
    highlights.truncate(TRACE_GROUP_HIGHLIGHT_LIMIT);
    highlights
}

fn build_range_highlights(details: &[NeighborDetail]) -> Vec<TraceRangeHighlight> {
    let mut by_column: HashMap<u32, Vec<&NeighborDetail>> = HashMap::new();
    for detail in details {
        if let (Some(col), Some(_row)) = (detail.column, detail.row) {
            if !detail.external {
                by_column.entry(col).or_default().push(detail);
            }
        }
    }

    for column_entries in by_column.values_mut() {
        column_entries.sort_by(|a, b| a.row.cmp(&b.row));
    }

    let mut ranges = Vec::new();
    for entries in by_column.values() {
        let mut current: Vec<&NeighborDetail> = Vec::new();
        for detail in entries {
            if current.is_empty() {
                current.push(detail);
                continue;
            }
            let prev_row = current.last().and_then(|d| d.row).unwrap_or(0);
            if detail.row.unwrap_or(0) == prev_row + 1 {
                current.push(detail);
            } else {
                if current.len() >= TRACE_RANGE_THRESHOLD {
                    ranges.push(make_range_highlight(&current));
                }
                current.clear();
                current.push(detail);
            }
        }
        if current.len() >= TRACE_RANGE_THRESHOLD {
            ranges.push(make_range_highlight(&current));
        }
    }

    ranges.sort_by(|a, b| b.count.cmp(&a.count));
    ranges.truncate(TRACE_RANGE_HIGHLIGHT_LIMIT);
    ranges
}

fn make_range_highlight(details: &[&NeighborDetail]) -> TraceRangeHighlight {
    let mut literals = 0usize;
    let mut formulas = 0usize;
    let mut blanks = 0usize;
    let mut sample_values = Vec::new();
    let mut sample_formulas = Vec::new();
    let mut sample_addresses = Vec::new();

    for detail in details {
        match detail.kind {
            TraceCellKind::Formula => {
                formulas += 1;
                if let Some(formula) = &detail.formula {
                    if sample_formulas.len() < TRACE_RANGE_FORMULA_SAMPLES
                        && !sample_formulas.contains(formula)
                    {
                        sample_formulas.push(formula.clone());
                    }
                }
            }
            TraceCellKind::Literal => {
                literals += 1;
                if let Some(value) = &detail.value {
                    if sample_values.len() < TRACE_RANGE_VALUE_SAMPLES {
                        sample_values.push(value.clone());
                    }
                }
            }
            TraceCellKind::Blank => blanks += 1,
            TraceCellKind::External => {}
        }
        if sample_addresses.len() < TRACE_RANGE_VALUE_SAMPLES {
            sample_addresses.push(detail.address.clone());
        }
    }

    TraceRangeHighlight {
        start: details
            .first()
            .map(|d| d.address.clone())
            .unwrap_or_default(),
        end: details
            .last()
            .map(|d| d.address.clone())
            .unwrap_or_default(),
        count: details.len(),
        literals,
        formulas,
        blanks,
        sample_values,
        sample_formulas,
        sample_addresses,
    }
}

fn build_notable_cells(
    details: &[NeighborDetail],
    ranges: &[TraceRangeHighlight],
    groups: &[TraceFormulaGroupHighlight],
) -> Vec<TraceCellHighlight> {
    let mut exclude: HashSet<String> = HashSet::new();
    for range in ranges {
        exclude.insert(range.start.clone());
        exclude.insert(range.end.clone());
        for addr in &range.sample_addresses {
            exclude.insert(addr.clone());
        }
    }
    for group in groups {
        for addr in &group.sample_addresses {
            exclude.insert(addr.clone());
        }
    }

    let mut highlights = Vec::new();
    let mut kind_counts: HashMap<TraceCellKind, usize> = HashMap::new();

    for detail in details {
        if highlights.len() >= TRACE_CELL_HIGHLIGHT_LIMIT {
            break;
        }
        if exclude.contains(&detail.address) {
            continue;
        }
        let counter = kind_counts.entry(detail.kind.clone()).or_insert(0);
        if *counter >= 2 && detail.kind != TraceCellKind::External {
            continue;
        }
        highlights.push(TraceCellHighlight {
            address: detail.address.clone(),
            kind: detail.kind.clone(),
            value: detail.value.clone(),
            formula: detail.formula.clone(),
        });
        *counter += 1;
    }

    highlights
}

fn build_edges_for_layer(
    raw_edges: &[TraceEdgeRaw],
    selected: &HashSet<String>,
    formula_lookup: &HashMap<String, TraceFormulaInfo>,
) -> Vec<FormulaTraceEdge> {
    let mut edges = Vec::new();
    for edge in raw_edges {
        if selected.contains(&edge.neighbor) {
            let formula = lookup_formula_info(formula_lookup, &edge.neighbor, &edge.neighbor)
                .map(|info| info.formula.clone());
            edges.push(FormulaTraceEdge {
                from: edge.from.clone(),
                to: edge.to.clone(),
                formula,
                note: None,
            });
        }
    }
    edges.sort_by(|a, b| compare_addresses(&a.to, &b.to));
    edges
}

fn lookup_formula_info<'a>(
    lookup: &'a HashMap<String, TraceFormulaInfo>,
    cell_ref: &str,
    original: &str,
) -> Option<&'a TraceFormulaInfo> {
    if let Some(info) = lookup.get(cell_ref) {
        return Some(info);
    }
    if let (Some(_sheet), Some(cell)) = split_sheet_and_cell(original) {
        let upper = cell.to_ascii_uppercase();
        return lookup.get(&upper);
    }
    None
}

fn compare_addresses(left: &str, right: &str) -> Ordering {
    let (sheet_left, cell_left) = split_sheet_and_cell(left);
    let (sheet_right, cell_right) = split_sheet_and_cell(right);

    let sheet_left_key = sheet_left
        .as_ref()
        .map(|s| clean_sheet_name(s).to_ascii_uppercase())
        .unwrap_or_default();
    let sheet_right_key = sheet_right
        .as_ref()
        .map(|s| clean_sheet_name(s).to_ascii_uppercase())
        .unwrap_or_default();

    match sheet_left_key.cmp(&sheet_right_key) {
        Ordering::Equal => {
            let left_core = cell_left.unwrap_or_else(|| left.to_string());
            let right_core = cell_right.unwrap_or_else(|| right.to_string());
            let left_coords = parse_address(&left_core.to_ascii_uppercase());
            let right_coords = parse_address(&right_core.to_ascii_uppercase());
            match (left_coords, right_coords) {
                (Some((lc, lr)), Some((rc, rr))) => lc
                    .cmp(&rc)
                    .then_with(|| lr.cmp(&rr))
                    .then_with(|| left_core.cmp(&right_core)),
                _ => left_core.cmp(&right_core),
            }
        }
        other => other,
    }
}

fn split_sheet_and_cell(address: &str) -> (Option<String>, Option<String>) {
    if let Some(idx) = address.rfind('!') {
        let sheet = address[..idx].to_string();
        let cell = address[idx + 1..].to_string();
        (Some(sheet), Some(cell))
    } else {
        (None, Some(address.to_string()))
    }
}

fn clean_sheet_name(sheet: &str) -> String {
    let trimmed = sheet.trim_matches(|c| c == '\'' || c == '"');
    let after_bracket = trimmed.rsplit(']').next().unwrap_or(trimmed);
    after_bracket
        .trim_matches(|c| c == '\'' || c == '"')
        .to_string()
}
