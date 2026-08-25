pub mod address;
pub mod cells;
pub mod hash;
pub mod merge;
pub mod names;
pub mod sst;
pub mod tables;

use anyhow::Result;
use cells::CellIterator;
use merge::{CellDiff, diff_streams};
use names::{DefinedName, NameDiff, NameKey, diff_names, parse_defined_names};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use schemars::JsonSchema;
use serde::Serialize;
use sst::Sst;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tables::{TableDiff, TableInfo, diff_tables, parse_table_xml};
use zip::ZipArchive;

#[derive(Debug, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum Change {
    Cell(CellChange),
    Table(TableDiff),
    Name(NameDiff),
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct CellChange {
    pub sheet: String,
    #[serde(flatten)]
    pub diff: CellDiff,
}

// Legacy alias for tests until updated
pub type DiffResult = CellChange;

pub fn calculate_changeset(
    base_path: &Path,
    fork_path: &Path,
    sheet_filter: Option<&str>,
) -> Result<Vec<Change>> {
    let mut base_zip = ZipArchive::new(File::open(base_path)?)?;
    let mut fork_zip = ZipArchive::new(File::open(fork_path)?)?;

    // Load SSTs
    let base_sst = load_sst(&mut base_zip).ok();
    let fork_sst = load_sst(&mut fork_zip).ok();
    let base_sst_hash = base_zip
        .by_name("xl/sharedStrings.xml")
        .ok()
        .and_then(|f| hash::compute_hash(f).ok())
        .unwrap_or(0);
    let fork_sst_hash = fork_zip
        .by_name("xl/sharedStrings.xml")
        .ok()
        .and_then(|f| hash::compute_hash(f).ok())
        .unwrap_or(0);

    // Load Workbook Meta (Sheets + Names)
    let base_meta = load_workbook_meta(&mut base_zip)?;
    let fork_meta = load_workbook_meta(&mut fork_zip)?;

    let mut all_changes = Vec::new();

    // 1. Diff Names
    // Names are global (or scoped), not filtered by sheet_filter usually,
    // unless scope matches? For now return all name changes.
    // Ideally we filter scoped names by sheet_filter.
    let name_diffs = diff_names(&base_meta.names, &fork_meta.names, &base_meta.sheet_id_map);
    for d in name_diffs {
        if let Some(filter) = sheet_filter {
            match &d {
                NameDiff::NameAdded {
                    scope_sheet: Some(s),
                    ..
                } if s != filter => continue,
                NameDiff::NameDeleted {
                    scope_sheet: Some(s),
                    ..
                } if s != filter => continue,
                NameDiff::NameModified {
                    scope_sheet: Some(s),
                    ..
                } if s != filter => continue,
                _ => {}
            }
        }
        all_changes.push(Change::Name(d));
    }

    // 2. Diff Tables
    let base_tables = load_tables(&mut base_zip, &base_meta.sheet_map)?;
    let fork_tables = load_tables(&mut fork_zip, &fork_meta.sheet_map)?;
    let table_diffs = diff_tables(&base_tables, &fork_tables);
    for d in table_diffs {
        if let Some(filter) = sheet_filter {
            match &d {
                TableDiff::TableAdded { sheet, .. } if sheet != filter => continue,
                TableDiff::TableDeleted { sheet, .. } if sheet != filter => continue,
                TableDiff::TableModified { sheet, .. } if sheet != filter => continue,
                _ => {}
            }
        }
        all_changes.push(Change::Table(d));
    }

    // 3. Diff Cells (per sheet)
    // We iterate the UNION of sheets
    let mut all_sheets: Vec<_> = base_meta
        .sheet_map
        .keys()
        .chain(fork_meta.sheet_map.keys())
        .collect();
    all_sheets.sort();
    all_sheets.dedup();

    for name in all_sheets {
        if let Some(filter) = sheet_filter
            && name != filter
        {
            continue;
        }

        let base_path_str = base_meta.sheet_map.get(name);
        let fork_path_str = fork_meta.sheet_map.get(name);

        // Hash Check (optimization)
        let base_hash = if let Some(p) = base_path_str {
            if let Ok(f) = base_zip.by_name(p) {
                hash::compute_hash(f)?
            } else {
                0
            }
        } else {
            0
        };

        let fork_hash = if let Some(p) = fork_path_str {
            if let Ok(f) = fork_zip.by_name(p) {
                hash::compute_hash(f)?
            } else {
                0
            }
        } else {
            0
        };

        if base_hash != 0 && base_hash == fork_hash && base_sst_hash == fork_sst_hash {
            continue;
        }

        // Diff Streams
        let base_iter = if let Some(p) = base_path_str {
            if let Ok(f) = base_zip.by_name(p) {
                Some(CellIterator::new(BufReader::new(f), base_sst.as_ref()))
            } else {
                None
            }
        } else {
            None
        };

        let fork_iter = if let Some(p) = fork_path_str {
            if let Ok(f) = fork_zip.by_name(p) {
                Some(CellIterator::new(BufReader::new(f), fork_sst.as_ref()))
            } else {
                None
            }
        } else {
            None
        };

        let diffs = match (base_iter, fork_iter) {
            (Some(b), Some(f)) => diff_streams(b, f)?,
            (Some(b), None) => diff_streams(b, std::iter::empty())?,
            (None, Some(f)) => diff_streams(std::iter::empty(), f)?,
            (None, None) => Vec::new(),
        };

        for d in diffs {
            all_changes.push(Change::Cell(CellChange {
                sheet: name.clone(),
                diff: d,
            }));
        }
    }

    Ok(all_changes)
}

fn load_sst(zip: &mut ZipArchive<File>) -> Result<Sst> {
    let f = zip.by_name("xl/sharedStrings.xml")?;
    Sst::from_reader(BufReader::new(f))
}

struct WorkbookMeta {
    sheet_map: HashMap<String, String>, // name -> path
    sheet_id_map: HashMap<u32, String>, // index (0-based from sheetId or array?) -> name
    // Spec says localSheetId is 0-based index of sheet in workbook
    names: HashMap<NameKey, DefinedName>,
}

fn load_workbook_meta(zip: &mut ZipArchive<File>) -> Result<WorkbookMeta> {
    // 1. Parse workbook.xml for name -> rId, sheetId, and definedNames
    let mut name_to_rid = HashMap::new();
    let mut sheet_id_map = HashMap::new();
    let mut defined_names = HashMap::new();

    // We need to know the *order* of sheets for localSheetId (0, 1, 2...)
    // Iterate sheets in order of appearance?
    let mut sheet_order = Vec::new();

    {
        let workbook_xml = zip.by_name("xl/workbook.xml")?;
        let mut reader = Reader::from_reader(BufReader::new(workbook_xml));
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                    if e.name().as_ref() == b"sheet" {
                        let mut name = String::new();
                        let mut rid = String::new();
                        for attr in e.attributes() {
                            let attr = attr?;
                            if attr.key.as_ref() == b"name" {
                                name = String::from_utf8_lossy(&attr.value).to_string();
                            } else if attr.key.as_ref() == b"r:id" {
                                rid = String::from_utf8_lossy(&attr.value).to_string();
                            }
                        }
                        if !name.is_empty() && !rid.is_empty() {
                            name_to_rid.insert(rid, name.clone());
                            sheet_order.push(name);
                        }
                    } else if e.name().as_ref() == b"definedNames" {
                        // Switch to parsing defined names
                        // We need to pass the reader to the names module?
                        // But we are inside a loop borrowing the reader.
                        // Since `parse_defined_names` takes `&mut Reader`, we can call it.
                        // It will consume until </definedNames>.
                        defined_names = parse_defined_names(&mut reader)?;
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(e.into()),
                _ => {}
            }
            buf.clear();
        }
    }

    for (idx, name) in sheet_order.into_iter().enumerate() {
        sheet_id_map.insert(idx as u32, name);
    }

    // 2. Parse _rels/workbook.xml.rels for rId -> Target
    let mut rid_to_target = HashMap::new();
    if let Ok(rels_xml) = zip.by_name("xl/_rels/workbook.xml.rels") {
        let mut reader = Reader::from_reader(BufReader::new(rels_xml));
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                    if e.name().as_ref() == b"Relationship" {
                        let mut id = String::new();
                        let mut target = String::new();
                        for attr in e.attributes() {
                            let attr = attr?;
                            if attr.key.as_ref() == b"Id" {
                                id = String::from_utf8_lossy(&attr.value).to_string();
                            } else if attr.key.as_ref() == b"Target" {
                                target = String::from_utf8_lossy(&attr.value).to_string();
                            }
                        }
                        rid_to_target.insert(id, target);
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(e.into()),
                _ => {}
            }
            buf.clear();
        }
    }

    // 3. Join
    let mut sheet_map = HashMap::new();
    for (rid, name) in name_to_rid {
        if let Some(target) = rid_to_target.get(&rid) {
            let path = if target.starts_with('/') {
                target.trim_start_matches('/').to_string()
            } else {
                format!("xl/{}", target)
            };
            sheet_map.insert(name, path);
        }
    }

    Ok(WorkbookMeta {
        sheet_map,
        sheet_id_map,
        names: defined_names,
    })
}

fn load_tables(
    zip: &mut ZipArchive<File>,
    sheet_map: &HashMap<String, String>,
) -> Result<HashMap<String, TableInfo>> {
    let mut tables = HashMap::new();

    for (sheet_name, sheet_path) in sheet_map {
        let path = Path::new(sheet_path);
        let parent = path.parent().unwrap();
        let filename = path.file_name().unwrap().to_str().unwrap();
        let rels_path = parent.join("_rels").join(format!("{}.rels", filename));
        let rels_path_str = rels_path.to_str().unwrap();

        // 1. Find table files in rels
        let mut table_files = Vec::new();
        if let Ok(f) = zip.by_name(rels_path_str) {
            let mut reader = Reader::from_reader(BufReader::new(f));
            let mut buf = Vec::new();
            loop {
                match reader.read_event_into(&mut buf) {
                    Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                        if e.name().as_ref() == b"Relationship" {
                            let mut target = String::new();
                            let mut type_attr = String::new();
                            for attr in e.attributes() {
                                let attr = attr?;
                                if attr.key.as_ref() == b"Target" {
                                    target = String::from_utf8_lossy(&attr.value).to_string();
                                } else if attr.key.as_ref() == b"Type" {
                                    type_attr = String::from_utf8_lossy(&attr.value).to_string();
                                }
                            }
                            if type_attr.ends_with("/table") {
                                table_files.push(target);
                            }
                        }
                    }
                    Ok(Event::Eof) => break,
                    _ => {}
                }
                buf.clear();
            }
        }

        // 2. Parse table files
        for target in table_files {
            // Target is relative to sheet path parent (e.g. "../tables/table1.xml")
            // parent is "xl/worksheets"
            // We need to resolve this path.
            // Simple normalization: if starts with "../", strip last component of parent.
            // parent: "xl/worksheets" -> parent of that is "xl"
            // target: "tables/table1.xml" -> "xl/tables/table1.xml"

            // Or use PathBuf logic
            let mut full_path = parent.to_path_buf();
            for component in Path::new(&target).components() {
                match component {
                    std::path::Component::ParentDir => {
                        full_path.pop();
                    }
                    std::path::Component::Normal(c) => {
                        full_path.push(c);
                    }
                    _ => {}
                }
            }

            let full_path_str = full_path.to_str().unwrap();

            if let Ok(f) = zip.by_name(full_path_str) {
                let mut reader = Reader::from_reader(BufReader::new(f));
                if let Ok(info) = parse_table_xml(&mut reader, sheet_name.clone()) {
                    tables.insert(info.display_name.clone(), info);
                }
            }
        }
    }

    Ok(tables)
}
