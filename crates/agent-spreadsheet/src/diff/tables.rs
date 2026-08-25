use anyhow::Result;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use schemars::JsonSchema;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

#[derive(Debug, Clone, PartialEq)]
pub struct TableInfo {
    pub display_name: String,
    pub range: String, // "A1:D5"
    pub sheet: String, // "Sheet1"
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TableDiff {
    TableAdded {
        display_name: String,
        sheet: String,
        range: String,
    },
    TableDeleted {
        display_name: String,
        sheet: String,
    },
    TableModified {
        display_name: String,
        sheet: String,
        old_range: String,
        new_range: String,
    },
}

pub fn parse_table_xml<R: BufRead>(
    reader: &mut Reader<R>,
    sheet_name: String,
) -> Result<TableInfo> {
    let mut buf = Vec::new();
    let mut display_name = String::new();
    let mut range = String::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"table" => {
                for attr in e.attributes() {
                    let attr = attr?;
                    match attr.key.as_ref() {
                        b"displayName" => {
                            display_name = String::from_utf8_lossy(&attr.value).to_string()
                        }
                        b"ref" => range = String::from_utf8_lossy(&attr.value).to_string(),
                        _ => {}
                    }
                }
                // We only need the top-level attributes
                break;
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(e.into()),
            _ => {}
        }
        buf.clear();
    }

    if display_name.is_empty() || range.is_empty() {
        // Fallback or error? Some tables might lack displayName?
        // Spec says displayName is required.
        // If we didn't find <table ...>, return error
        if display_name.is_empty() {
            return Err(anyhow::anyhow!("Missing displayName in table definition"));
        }
    }

    Ok(TableInfo {
        display_name,
        range,
        sheet: sheet_name,
    })
}

pub fn diff_tables(
    base_tables: &HashMap<String, TableInfo>, // Keyed by displayName
    fork_tables: &HashMap<String, TableInfo>,
) -> Vec<TableDiff> {
    let mut diffs = Vec::new();
    let all_keys: HashSet<_> = base_tables.keys().chain(fork_tables.keys()).collect();

    for key in all_keys {
        let base = base_tables.get(key);
        let fork = fork_tables.get(key);

        match (base, fork) {
            (None, Some(f)) => {
                diffs.push(TableDiff::TableAdded {
                    display_name: f.display_name.clone(),
                    sheet: f.sheet.clone(),
                    range: f.range.clone(),
                });
            }
            (Some(b), None) => {
                diffs.push(TableDiff::TableDeleted {
                    display_name: b.display_name.clone(),
                    sheet: b.sheet.clone(),
                });
            }
            (Some(b), Some(f)) => {
                // If semantic identity (displayName) matches, check for changes
                if b.range != f.range {
                    diffs.push(TableDiff::TableModified {
                        display_name: b.display_name.clone(),
                        sheet: b.sheet.clone(),
                        old_range: b.range.clone(),
                        new_range: f.range.clone(),
                    });
                }
                // Note: If sheet changed (e.g. table moved to another sheet),
                // it would look like a modification here if displayName is preserved.
                // In practice, moving a table across sheets is rare/complex.
                // If it happens, we might want to report it.
                // For V1, we ignore sheet changes in TableModified, or we could add it.
            }
            (None, None) => unreachable!(),
        }
    }

    // Sort
    diffs.sort_by(|a, b| {
        let name_a = match a {
            TableDiff::TableAdded { display_name, .. } => display_name,
            TableDiff::TableDeleted { display_name, .. } => display_name,
            TableDiff::TableModified { display_name, .. } => display_name,
        };
        let name_b = match b {
            TableDiff::TableAdded { display_name, .. } => display_name,
            TableDiff::TableDeleted { display_name, .. } => display_name,
            TableDiff::TableModified { display_name, .. } => display_name,
        };
        name_a.cmp(name_b)
    });

    diffs
}
