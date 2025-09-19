use crate::analysis::{
    classification,
    formula::{FormulaAtlas, FormulaGraph},
    style,
};
use crate::caps::BackendCaps;
use crate::config::ServerConfig;
use crate::model::{
    NamedItemKind, NamedRangeDescriptor, SheetClassification, SheetOverviewResponse, SheetSummary,
    WorkbookDescription, WorkbookId, WorkbookListResponse,
};
use crate::tools::filters::WorkbookFilter;
use crate::utils::{hash_path_metadata, make_short_workbook_id, system_time_to_rfc3339};
use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use umya_spreadsheet::reader::xlsx;
use umya_spreadsheet::{DefinedName, Spreadsheet, Worksheet};

pub struct WorkbookContext {
    pub id: WorkbookId,
    pub short_id: String,
    pub slug: String,
    pub path: PathBuf,
    pub caps: BackendCaps,
    pub bytes: u64,
    pub last_modified: Option<DateTime<Utc>>,
    spreadsheet: Arc<RwLock<Spreadsheet>>,
    sheet_cache: RwLock<HashMap<String, Arc<SheetCacheEntry>>>,
    formula_atlas: Arc<FormulaAtlas>,
}

pub struct SheetCacheEntry {
    pub metrics: SheetMetrics,
    pub style_tags: Vec<String>,
    pub named_ranges: Vec<NamedRangeDescriptor>,
}

#[derive(Debug, Clone)]
pub struct SheetMetrics {
    pub row_count: u32,
    pub column_count: u32,
    pub non_empty_cells: u32,
    pub formula_cells: u32,
    pub cached_values: u32,
    pub comments: u32,
    pub style_map: HashMap<String, StyleUsage>,
    pub classification: SheetClassification,
}

#[derive(Debug, Clone)]
pub struct StyleUsage {
    pub occurrences: u32,
    pub tags: Vec<String>,
    pub example_cells: Vec<String>,
}

impl WorkbookContext {
    pub fn load(_config: &Arc<ServerConfig>, path: &Path) -> Result<Self> {
        let metadata = fs::metadata(path)
            .with_context(|| format!("unable to read metadata for {:?}", path))?;
        let slug = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "workbook".to_string());
        let bytes = metadata.len();
        let last_modified = metadata.modified().ok().and_then(system_time_to_rfc3339);
        let id = WorkbookId(hash_path_metadata(path, &metadata));
        let spreadsheet =
            xlsx::read(path).with_context(|| format!("failed to parse workbook {:?}", path))?;
        let short_id = make_short_workbook_id(&slug, id.as_str());

        Ok(Self {
            id,
            short_id,
            slug,
            path: path.to_path_buf(),
            caps: BackendCaps::xlsx(),
            bytes,
            last_modified,
            spreadsheet: Arc::new(RwLock::new(spreadsheet)),
            sheet_cache: RwLock::new(HashMap::new()),
            formula_atlas: Arc::new(FormulaAtlas::default()),
        })
    }

    pub fn sheet_names(&self) -> Vec<String> {
        let book = self.spreadsheet.read();
        book.get_sheet_collection()
            .iter()
            .map(|sheet| sheet.get_name().to_string())
            .collect()
    }

    pub fn describe(&self) -> WorkbookDescription {
        let book = self.spreadsheet.read();
        let defined_names_count = book.get_defined_names().len();
        let table_count: usize = book
            .get_sheet_collection()
            .iter()
            .map(|sheet| sheet.get_tables().len())
            .sum();
        let macros_present = false;

        WorkbookDescription {
            workbook_id: self.id.clone(),
            short_id: self.short_id.clone(),
            slug: self.slug.clone(),
            path: self.path.to_string_lossy().to_string(),
            bytes: self.bytes,
            sheet_count: book.get_sheet_collection().len(),
            defined_names: defined_names_count,
            tables: table_count,
            macros_present,
            last_modified: self
                .last_modified
                .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
            caps: self.caps.clone(),
        }
    }

    pub fn get_sheet_metrics(&self, sheet_name: &str) -> Result<Arc<SheetCacheEntry>> {
        if let Some(entry) = self.sheet_cache.read().get(sheet_name) {
            return Ok(entry.clone());
        }

        let mut writer = self.sheet_cache.write();
        if let Some(entry) = writer.get(sheet_name) {
            return Ok(entry.clone());
        }

        let book = self.spreadsheet.read();
        let sheet = book
            .get_sheet_by_name(sheet_name)
            .ok_or_else(|| anyhow!("sheet {} not found", sheet_name))?;
        let (metrics, style_tags) = compute_sheet_metrics(sheet);
        let named_ranges = gather_named_ranges(sheet, book.get_defined_names());

        let entry = Arc::new(SheetCacheEntry {
            metrics,
            style_tags,
            named_ranges,
        });

        writer.insert(sheet_name.to_string(), entry.clone());
        Ok(entry)
    }

    pub fn list_summaries(&self) -> Result<Vec<SheetSummary>> {
        let book = self.spreadsheet.read();
        let mut summaries = Vec::new();
        for sheet in book.get_sheet_collection() {
            let name = sheet.get_name().to_string();
            let entry = self.get_sheet_metrics(&name)?;
            summaries.push(SheetSummary {
                name: name.clone(),
                visible: sheet.get_sheet_state() != "hidden",
                row_count: entry.metrics.row_count,
                column_count: entry.metrics.column_count,
                non_empty_cells: entry.metrics.non_empty_cells,
                formula_cells: entry.metrics.formula_cells,
                cached_values: entry.metrics.cached_values,
                classification: entry.metrics.classification.clone(),
                style_tags: entry.style_tags.clone(),
            });
        }
        Ok(summaries)
    }

    pub fn with_sheet<T, F>(&self, sheet_name: &str, func: F) -> Result<T>
    where
        F: FnOnce(&Worksheet) -> T,
    {
        let book = self.spreadsheet.read();
        let sheet = book
            .get_sheet_by_name(sheet_name)
            .ok_or_else(|| anyhow!("sheet {} not found", sheet_name))?;
        Ok(func(sheet))
    }

    pub fn formula_graph(&self, sheet_name: &str) -> Result<FormulaGraph> {
        self.with_sheet(sheet_name, |sheet| {
            FormulaGraph::build(sheet, &self.formula_atlas)
        })?
    }

    pub fn named_items(&self) -> Result<Vec<NamedRangeDescriptor>> {
        let book = self.spreadsheet.read();
        let sheet_names: Vec<String> = book
            .get_sheet_collection()
            .iter()
            .map(|sheet| sheet.get_name().to_string())
            .collect();
        let mut items = Vec::new();

        for defined in book.get_defined_names() {
            let refers_to = defined.get_address();
            let scope = if defined.has_local_sheet_id() {
                let idx = *defined.get_local_sheet_id() as usize;
                sheet_names.get(idx).cloned()
            } else {
                None
            };
            let kind = if refers_to.starts_with('=') {
                NamedItemKind::Formula
            } else {
                NamedItemKind::NamedRange
            };

            items.push(NamedRangeDescriptor {
                name: defined.get_name().to_string(),
                scope: scope.clone(),
                refers_to: refers_to.clone(),
                kind,
                sheet_name: scope,
                comment: None,
            });
        }

        for sheet in book.get_sheet_collection() {
            for table in sheet.get_tables() {
                let start = table.get_area().0.get_coordinate();
                let end = table.get_area().1.get_coordinate();
                items.push(NamedRangeDescriptor {
                    name: table.get_name().to_string(),
                    scope: Some(sheet.get_name().to_string()),
                    refers_to: format!("{}:{}", start, end),
                    kind: NamedItemKind::Table,
                    sheet_name: Some(sheet.get_name().to_string()),
                    comment: None,
                });
            }
        }

        Ok(items)
    }

    pub fn sheet_overview(&self, sheet_name: &str) -> Result<SheetOverviewResponse> {
        let entry = self.get_sheet_metrics(sheet_name)?;
        let narrative = classification::narrative(&entry.metrics);
        let regions = classification::regions(&entry.metrics);
        let key_ranges = classification::key_ranges(&entry.metrics);

        Ok(SheetOverviewResponse {
            workbook_id: self.id.clone(),
            workbook_short_id: self.short_id.clone(),
            sheet_name: sheet_name.to_string(),
            narrative,
            regions,
            key_ranges,
            formula_ratio: if entry.metrics.non_empty_cells == 0 {
                0.0
            } else {
                entry.metrics.formula_cells as f32 / entry.metrics.non_empty_cells as f32
            },
            notable_features: entry.style_tags.clone(),
        })
    }
}

pub fn cell_to_value(cell: &umya_spreadsheet::Cell) -> Option<crate::model::CellValue> {
    let raw = cell.get_value();
    if raw.is_empty() {
        return None;
    }
    if let Ok(number) = raw.parse::<f64>() {
        return Some(crate::model::CellValue::Number(number));
    }

    let lower = raw.to_ascii_lowercase();
    if lower == "true" {
        return Some(crate::model::CellValue::Bool(true));
    }
    if lower == "false" {
        return Some(crate::model::CellValue::Bool(false));
    }

    Some(crate::model::CellValue::Text(raw.to_string()))
}

pub fn compute_sheet_metrics(sheet: &Worksheet) -> (SheetMetrics, Vec<String>) {
    use std::collections::HashMap as StdHashMap;
    let mut non_empty = 0u32;
    let mut formulas = 0u32;
    let mut cached = 0u32;
    let comments = sheet.get_comments().len() as u32;
    let mut style_usage: StdHashMap<String, StyleUsage> = StdHashMap::new();

    for cell in sheet.get_cell_collection() {
        let value = cell.get_value();
        if !value.is_empty() {
            non_empty += 1;
        }
        if cell.is_formula() {
            formulas += 1;
            if !cell.get_value().is_empty() {
                cached += 1;
            }
        }

        if let Some((style_key, usage)) = style::tag_cell(cell) {
            let entry = style_usage.entry(style_key).or_insert_with(|| StyleUsage {
                occurrences: 0,
                tags: usage.tags.clone(),
                example_cells: Vec::new(),
            });
            entry.occurrences += 1;
            if entry.example_cells.len() < 5 {
                entry.example_cells.push(usage.example_cell.clone());
            }
        }
    }

    let (max_col, max_row) = sheet.get_highest_column_and_row();

    let classification = classification::classify(
        non_empty,
        formulas,
        max_row,
        max_col,
        comments,
        &style_usage,
    );

    let style_tags: Vec<String> = style_usage
        .values()
        .flat_map(|usage| usage.tags.clone())
        .collect();

    let metrics = SheetMetrics {
        row_count: max_row,
        column_count: max_col,
        non_empty_cells: non_empty,
        formula_cells: formulas,
        cached_values: cached,
        comments,
        style_map: style_usage,
        classification,
    };
    (metrics, style_tags)
}

fn gather_named_ranges(
    sheet: &Worksheet,
    defined_names: &[DefinedName],
) -> Vec<NamedRangeDescriptor> {
    let name_str = sheet.get_name();
    defined_names
        .iter()
        .filter(|name| name.get_address().contains(name_str))
        .map(|name| NamedRangeDescriptor {
            name: name.get_name().to_string(),
            scope: if name.has_local_sheet_id() {
                Some(name_str.to_string())
            } else {
                None
            },
            refers_to: name.get_address(),
            kind: NamedItemKind::NamedRange,
            sheet_name: Some(name_str.to_string()),
            comment: None,
        })
        .collect()
}

pub fn build_workbook_list(
    config: &Arc<ServerConfig>,
    filter: &WorkbookFilter,
) -> Result<WorkbookListResponse> {
    let mut descriptors = Vec::new();

    if let Some(single) = config.single_workbook() {
        let metadata = fs::metadata(single)
            .with_context(|| format!("unable to read metadata for {:?}", single))?;
        let id = WorkbookId(hash_path_metadata(single, &metadata));
        let slug = single
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "workbook".to_string());
        let folder = derive_folder(config, single);
        let short_id = make_short_workbook_id(&slug, id.as_str());
        let caps = BackendCaps::xlsx();

        if filter.matches(&slug, folder.as_deref(), single) {
            let descriptor = crate::model::WorkbookDescriptor {
                workbook_id: id,
                short_id,
                slug,
                folder,
                path: single
                    .strip_prefix(&config.workspace_root)
                    .unwrap_or(single)
                    .to_string_lossy()
                    .to_string(),
                bytes: metadata.len(),
                last_modified: metadata
                    .modified()
                    .ok()
                    .and_then(system_time_to_rfc3339)
                    .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
                caps,
            };
            descriptors.push(descriptor);
        }

        return Ok(WorkbookListResponse {
            workbooks: descriptors,
        });
    }

    use walkdir::WalkDir;

    for entry in WalkDir::new(&config.workspace_root) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if !has_supported_extension(&config.supported_extensions, path) {
            continue;
        }
        let metadata = entry.metadata()?;
        let id = WorkbookId(hash_path_metadata(path, &metadata));
        let slug = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "workbook".to_string());
        let folder = derive_folder(config, path);
        let short_id = make_short_workbook_id(&slug, id.as_str());
        let caps = BackendCaps::xlsx();

        if !filter.matches(&slug, folder.as_deref(), path) {
            continue;
        }

        let descriptor = crate::model::WorkbookDescriptor {
            workbook_id: id,
            short_id,
            slug,
            folder,
            path: path
                .strip_prefix(&config.workspace_root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string(),
            bytes: metadata.len(),
            last_modified: metadata
                .modified()
                .ok()
                .and_then(system_time_to_rfc3339)
                .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
            caps,
        };
        descriptors.push(descriptor);
    }

    descriptors.sort_by(|a, b| a.slug.cmp(&b.slug));

    Ok(WorkbookListResponse {
        workbooks: descriptors,
    })
}

fn derive_folder(config: &Arc<ServerConfig>, path: &Path) -> Option<String> {
    path.strip_prefix(&config.workspace_root)
        .ok()
        .and_then(|relative| relative.parent())
        .and_then(|parent| parent.file_name())
        .map(|os| os.to_string_lossy().to_string())
}

fn has_supported_extension(allowed: &[String], path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let lower = ext.to_ascii_lowercase();
            allowed.iter().any(|candidate| candidate == &lower)
        })
        .unwrap_or(false)
}
