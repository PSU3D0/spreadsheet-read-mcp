use anyhow::Result;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use schemars::JsonSchema;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NameKey {
    pub name: String,
    pub scope: Option<u32>, // localSheetId
}

#[derive(Debug, Clone, PartialEq)]
pub struct DefinedName {
    pub key: NameKey,
    pub formula: String,
    pub hidden: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NameDiff {
    NameAdded {
        name: String,
        formula: String,
        scope_sheet: Option<String>,
    },
    NameDeleted {
        name: String,
        scope_sheet: Option<String>,
    },
    NameModified {
        name: String,
        scope_sheet: Option<String>,
        old_formula: String,
        new_formula: String,
    },
}

pub fn parse_defined_names<R: BufRead>(
    reader: &mut Reader<R>,
) -> Result<HashMap<NameKey, DefinedName>> {
    let mut names = HashMap::new();
    let mut buf = Vec::new();
    let mut inner_buf = Vec::new();

    // The reader should be positioned at the start of <definedNames> or we assume the caller
    // is iterating through workbook.xml.
    // For simplicity in integration, we'll scan for definedNames block if not inside one,
    // or just assume we are scanning a stream that might contain it.

    // However, to keep it stateless relative to the outer loop, let's assume the caller
    // passes a reader that is iterating the *entire* workbook.xml, and we just hook into the relevant events.
    // OR better: The caller iterates workbook.xml, finds <definedNames>, and calls us to consume it.

    // Let's implement assuming we consume the *content* of <definedNames>.

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"definedName" => {
                let mut name_attr = String::new();
                let mut scope_attr = None;
                let mut hidden = false;

                for attr in e.attributes() {
                    let attr = attr?;
                    match attr.key.as_ref() {
                        b"name" => name_attr = String::from_utf8_lossy(&attr.value).to_string(),
                        b"localSheetId" => {
                            if let Ok(val) = String::from_utf8_lossy(&attr.value).parse::<u32>() {
                                scope_attr = Some(val);
                            }
                        }
                        b"hidden" => {
                            let val = attr.value.as_ref();
                            hidden = val == b"1" || val == b"true";
                        }
                        _ => {}
                    }
                }

                // Formula is the text content
                let mut formula = String::new();
                loop {
                    match reader.read_event_into(&mut inner_buf) {
                        Ok(Event::Text(e)) => formula.push_str(&e.unescape()?),
                        Ok(Event::End(ref end)) if end.name() == e.name() => break,
                        Ok(Event::Eof) => break,
                        Err(e) => return Err(e.into()),
                        _ => {}
                    }
                    inner_buf.clear();
                }

                let key = NameKey {
                    name: name_attr,
                    scope: scope_attr,
                };

                names.insert(
                    key.clone(),
                    DefinedName {
                        key,
                        formula,
                        hidden,
                    },
                );
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"definedNames" => break,
            Ok(Event::Eof) => break,
            Err(e) => return Err(e.into()),
            _ => {}
        }
        buf.clear();
    }

    Ok(names)
}

pub fn diff_names(
    base_names: &HashMap<NameKey, DefinedName>,
    fork_names: &HashMap<NameKey, DefinedName>,
    sheet_id_map: &HashMap<u32, String>, // index -> sheet name
) -> Vec<NameDiff> {
    let mut diffs = Vec::new();
    let all_keys: HashSet<_> = base_names.keys().chain(fork_names.keys()).collect();

    for key in all_keys {
        let base = base_names.get(key);
        let fork = fork_names.get(key);

        if let Some(b) = base
            && b.hidden
        {
            continue;
        }
        if let Some(f) = fork
            && f.hidden
        {
            continue;
        }

        let sheet_name = key.scope.and_then(|id| sheet_id_map.get(&id).cloned());

        match (base, fork) {
            (None, Some(f)) => {
                diffs.push(NameDiff::NameAdded {
                    name: key.name.clone(),
                    formula: f.formula.clone(),
                    scope_sheet: sheet_name,
                });
            }
            (Some(_), None) => {
                diffs.push(NameDiff::NameDeleted {
                    name: key.name.clone(),
                    scope_sheet: sheet_name,
                });
            }
            (Some(b), Some(f)) => {
                if b.formula != f.formula {
                    diffs.push(NameDiff::NameModified {
                        name: key.name.clone(),
                        scope_sheet: sheet_name,
                        old_formula: b.formula.clone(),
                        new_formula: f.formula.clone(),
                    });
                }
            }
            (None, None) => unreachable!(),
        }
    }

    // Sort for stability
    diffs.sort_by(|a, b| {
        let name_a = match a {
            NameDiff::NameAdded { name, .. } => name,
            NameDiff::NameDeleted { name, .. } => name,
            NameDiff::NameModified { name, .. } => name,
        };
        let name_b = match b {
            NameDiff::NameAdded { name, .. } => name,
            NameDiff::NameDeleted { name, .. } => name,
            NameDiff::NameModified { name, .. } => name,
        };
        name_a.cmp(name_b)
    });

    diffs
}
