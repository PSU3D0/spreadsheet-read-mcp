use crate::runtime::stateless::StatelessRuntime;
use anyhow::{Result, anyhow, bail};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const DIFF_LIMIT_MAX: u32 = 2_000;
const GROUP_PREVIEW_LIMIT: usize = 25;

#[derive(Debug, Clone, Copy)]
struct A1Bounds {
    start_col: u32,
    end_col: u32,
    start_row: u32,
    end_row: u32,
}

#[derive(Debug, Clone, Serialize)]
struct DiffGroup {
    group_id: String,
    kind: String,
    group_type: String,
    review_priority: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    sheet: Option<String>,
    change_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    range: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    sample_addresses: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    sample_items: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SheetDiffSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    sheet: Option<String>,
    total_changes: u32,
    direct_change_count: u32,
    recalc_result_change_count: u32,
    counts_by_subtype: BTreeMap<String, u32>,
    group_count: u32,
    counts_by_group_type: BTreeMap<String, u32>,
    group_preview: Vec<DiffGroup>,
    group_preview_truncated: bool,
}

#[derive(Debug, Clone)]
struct DiffGroupBuilder {
    kind: String,
    group_type: String,
    sheet: Option<String>,
    change_count: u32,
    min_col: Option<u32>,
    max_col: Option<u32>,
    min_row: Option<u32>,
    max_row: Option<u32>,
    last_address: Option<String>,
    sample_addresses: Vec<String>,
    sample_items: Vec<String>,
}

pub struct DiffCommandArgs {
    pub original: PathBuf,
    pub modified: PathBuf,
    pub sheet: Option<String>,
    pub sheets: Option<Vec<String>>,
    pub range: Option<String>,
    pub details: bool,
    pub limit: u32,
    pub offset: u32,
    pub exclude_recalc_result: bool,
}

pub async fn diff(args: DiffCommandArgs) -> Result<Value> {
    let DiffCommandArgs {
        original,
        modified,
        sheet,
        sheets,
        range,
        details,
        limit,
        offset,
        exclude_recalc_result,
    } = args;
    if sheet.is_some() && sheets.is_some() {
        bail!("invalid argument: --sheet and --sheets are mutually exclusive");
    }

    let runtime = StatelessRuntime;
    let original = runtime.normalize_existing_file(&original)?;
    let modified = runtime.normalize_existing_file(&modified)?;

    if details && (limit == 0 || limit > DIFF_LIMIT_MAX) {
        bail!(
            "invalid argument: --limit must be between 1 and {}",
            DIFF_LIMIT_MAX
        );
    }

    let sheet_filters: Vec<String> = if let Some(s) = sheet {
        vec![s]
    } else {
        sheets.unwrap_or_default()
    };

    let range_bounds = if let Some(range) = range.as_ref() {
        Some(
            parse_a1_range(range)
                .ok_or_else(|| anyhow!("invalid argument: --range must be A1 notation"))?,
        )
    } else {
        None
    };

    let mut payload = runtime.diff_json(&original, &modified)?;
    let changes = payload
        .get_mut("changes")
        .and_then(Value::as_array_mut)
        .map(std::mem::take)
        .unwrap_or_default();

    let mut counts_by_kind: BTreeMap<String, u32> = BTreeMap::new();
    let mut counts_by_type: BTreeMap<String, u32> = BTreeMap::new();
    let mut counts_by_subtype: BTreeMap<String, u32> = BTreeMap::new();
    let mut affected_sheets: BTreeSet<String> = BTreeSet::new();

    let mut filtered = Vec::new();
    let mut recalc_result_change_count = 0u32;
    for change in changes {
        if !change_matches_filters(&change, &sheet_filters, range_bounds) {
            continue;
        }

        let subtype = change_subtype_key(&change).map(str::to_string);
        if exclude_recalc_result && subtype.as_deref() == Some("recalc_result") {
            continue;
        }

        let kind = change_kind(&change).to_string();
        *counts_by_kind.entry(kind).or_default() += 1;

        let type_key = change_type_key(&change).to_string();
        *counts_by_type.entry(type_key).or_default() += 1;

        if let Some(subtype_key) = subtype {
            if subtype_key == "recalc_result" {
                recalc_result_change_count += 1;
            }
            *counts_by_subtype.entry(subtype_key).or_default() += 1;
        }

        if let Some(sheet_name) = change_sheet_name(&change) {
            affected_sheets.insert(sheet_name.to_string());
        }

        filtered.push(change);
    }

    let total_changes = filtered.len() as u32;
    let direct_change_count = total_changes.saturating_sub(recalc_result_change_count);
    let groups = build_groups(&filtered);
    let sheet_summaries = build_sheet_summaries(&filtered, &groups);
    let mut counts_by_group_type: BTreeMap<String, u32> = BTreeMap::new();
    for group in &groups {
        *counts_by_group_type
            .entry(group.group_type.clone())
            .or_default() += 1;
    }
    let group_preview: Vec<Value> = groups
        .iter()
        .take(GROUP_PREVIEW_LIMIT)
        .map(|group| serde_json::to_value(group).expect("group to value"))
        .collect();
    let group_preview_truncated = groups.len() > GROUP_PREVIEW_LIMIT;

    let (returned_changes, paged_changes, truncated, next_offset) = if details {
        let offset = offset as usize;
        let limit = limit as usize;
        let total = filtered.len();
        let page: Vec<Value> = filtered.into_iter().skip(offset).take(limit).collect();
        let returned = page.len() as u32;
        let consumed = offset.saturating_add(returned as usize);
        let truncated = consumed < total;
        let next_offset = truncated.then_some(consumed as u32);
        (returned, page, truncated, next_offset)
    } else {
        (0, Vec::new(), false, None)
    };

    let summary = json!({
        "total_changes": total_changes,
        "returned_changes": returned_changes,
        "truncated": truncated,
        "next_offset": next_offset,
        "counts_by_kind": counts_by_kind,
        "counts_by_type": counts_by_type,
        "counts_by_subtype": counts_by_subtype,
        "affected_sheets": affected_sheets.into_iter().collect::<Vec<_>>(),
        "recalc_result_change_count": recalc_result_change_count,
        "direct_change_count": direct_change_count,
        "group_count": groups.len(),
        "counts_by_group_type": counts_by_group_type,
        "group_preview": group_preview,
        "group_preview_truncated": group_preview_truncated,
        "sheet_summaries": sheet_summaries,
        "filters": {
            "exclude_recalc_result": exclude_recalc_result,
        }
    });

    let mut response = Map::new();
    response.insert(
        "original".to_string(),
        Value::String(original.display().to_string()),
    );
    response.insert(
        "modified".to_string(),
        Value::String(modified.display().to_string()),
    );
    response.insert("change_count".to_string(), Value::from(total_changes));
    response.insert("summary".to_string(), summary);

    if details {
        response.insert("changes".to_string(), Value::Array(paged_changes));
        response.insert(
            "groups".to_string(),
            Value::Array(
                groups
                    .into_iter()
                    .map(|group| serde_json::to_value(group).expect("group to value"))
                    .collect(),
            ),
        );
    }

    Ok(Value::Object(response))
}

fn build_groups(changes: &[Value]) -> Vec<DiffGroup> {
    let mut ordered = changes.to_vec();
    ordered.sort_by_key(group_sort_key);

    let mut out = Vec::new();
    let mut current: Option<DiffGroupBuilder> = None;

    for change in &ordered {
        let next = group_builder_for_change(change);
        match current.take() {
            Some(mut active) if can_merge_group(&active, &next, change) => {
                merge_group(&mut active, change);
                current = Some(active);
            }
            Some(active) => {
                out.push(finalize_group(active, 0));
                current = Some(next);
            }
            None => current = Some(next),
        }
    }

    if let Some(active) = current {
        out.push(finalize_group(active, 0));
    }

    out.sort_by_key(diff_group_sort_key);
    for (idx, group) in out.iter_mut().enumerate() {
        group.group_id = format!("grp_{:04}", idx + 1);
    }
    out
}

fn group_builder_for_change(change: &Value) -> DiffGroupBuilder {
    let kind = change_kind(change).to_string();
    let group_type = change_group_type(change).to_string();
    let sheet = change_sheet_name(change).map(str::to_string);
    let mut builder = DiffGroupBuilder {
        kind,
        group_type,
        sheet,
        change_count: 0,
        min_col: None,
        max_col: None,
        min_row: None,
        max_row: None,
        last_address: None,
        sample_addresses: Vec::new(),
        sample_items: Vec::new(),
    };
    merge_group(&mut builder, change);
    builder
}

fn can_merge_group(current: &DiffGroupBuilder, next: &DiffGroupBuilder, change: &Value) -> bool {
    if current.kind != "cell" || next.kind != "cell" {
        return false;
    }
    if current.group_type != next.group_type || current.sheet != next.sheet {
        return false;
    }
    let Some(last_address) = current.last_address.as_deref() else {
        return false;
    };
    let Some(next_address) = change_address(change) else {
        return false;
    };
    addresses_are_adjacent(last_address, next_address)
}

fn merge_group(group: &mut DiffGroupBuilder, change: &Value) {
    group.change_count += 1;

    if let Some(address) = change_address(change) {
        group.last_address = Some(address.to_string());
        if group.sample_addresses.len() < 5 {
            group.sample_addresses.push(address.to_string());
        }
        if let Some((col, row)) = parse_a1_coord(address) {
            group.min_col = Some(group.min_col.map_or(col, |v| v.min(col)));
            group.max_col = Some(group.max_col.map_or(col, |v| v.max(col)));
            group.min_row = Some(group.min_row.map_or(row, |v| v.min(row)));
            group.max_row = Some(group.max_row.map_or(row, |v| v.max(row)));
        }
        return;
    }

    if let Some(item_name) = change_item_name(change)
        && group.sample_items.len() < 5
    {
        group.sample_items.push(item_name.to_string());
    }
}

fn finalize_group(group: DiffGroupBuilder, index: usize) -> DiffGroup {
    let group_type = group.group_type;
    DiffGroup {
        group_id: format!("grp_{:04}", index + 1),
        kind: group.kind,
        review_priority: review_priority_label(&group_type).to_string(),
        group_type,
        sheet: group.sheet,
        change_count: group.change_count,
        range: match (group.min_col, group.max_col, group.min_row, group.max_row) {
            (Some(start_col), Some(end_col), Some(start_row), Some(end_row)) => {
                Some(format_a1_range(start_col, end_col, start_row, end_row))
            }
            _ => None,
        },
        sample_addresses: group.sample_addresses,
        sample_items: group.sample_items,
    }
}

fn change_group_type(change: &Value) -> &str {
    if let Some(subtype) = change_subtype_key(change) {
        return subtype;
    }
    change_type_key(change)
}

fn change_kind(change: &Value) -> &'static str {
    if change.get("address").is_some() {
        "cell"
    } else if change.get("display_name").is_some() {
        "table"
    } else if change.get("name").is_some() {
        "name"
    } else {
        "unknown"
    }
}

fn change_type_key(change: &Value) -> &str {
    match change_kind(change) {
        "cell" => change
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown"),
        "table" => match change.get("type").and_then(Value::as_str) {
            Some("table_added") => "table_added",
            Some("table_deleted") => "table_deleted",
            Some("table_modified") => "table_modified",
            _ => "table_unknown",
        },
        "name" => match change.get("type").and_then(Value::as_str) {
            Some("name_added") => "name_added",
            Some("name_deleted") => "name_deleted",
            Some("name_modified") => "name_modified",
            _ => "name_unknown",
        },
        _ => "unknown",
    }
}

fn change_subtype_key(change: &Value) -> Option<&str> {
    change.get("subtype").and_then(Value::as_str)
}

fn change_sheet_name(change: &Value) -> Option<&str> {
    change
        .get("sheet")
        .and_then(Value::as_str)
        .or_else(|| change.get("scope_sheet").and_then(Value::as_str))
}

fn change_address(change: &Value) -> Option<&str> {
    change.get("address").and_then(Value::as_str)
}

fn change_item_name(change: &Value) -> Option<&str> {
    change
        .get("display_name")
        .and_then(Value::as_str)
        .or_else(|| change.get("name").and_then(Value::as_str))
}

fn group_sort_key(change: &Value) -> (String, String, u32, u32, u32, u32, String) {
    let (start_row, start_col, end_row, end_col, label) =
        change_position_key(change_address(change), change_item_name(change));
    (
        change_sheet_name(change).unwrap_or("").to_string(),
        change_group_type(change).to_string(),
        start_row,
        start_col,
        end_row,
        end_col,
        label,
    )
}

fn diff_group_sort_key(group: &DiffGroup) -> (u8, String, u32, u32, u32, u32, String, String) {
    let (start_row, start_col, end_row, end_col, label) = change_position_key(
        group
            .range
            .as_deref()
            .or_else(|| group.sample_addresses.first().map(String::as_str)),
        group.sample_items.first().map(String::as_str),
    );
    (
        review_priority_rank(&group.group_type),
        group.sheet.clone().unwrap_or_default(),
        start_row,
        start_col,
        end_row,
        end_col,
        group.group_type.clone(),
        label,
    )
}

fn review_priority_rank(group_type: &str) -> u8 {
    match group_type {
        "formula_edit" | "value_edit" | "style_edit" | "added" | "deleted" => 0,
        "table_modified" | "name_modified" | "table_added" | "table_deleted" | "name_added"
        | "name_deleted" => 1,
        "recalc_result" => 2,
        _ => 3,
    }
}

fn review_priority_label(group_type: &str) -> &'static str {
    match review_priority_rank(group_type) {
        0 => "direct",
        1 => "structural",
        2 => "derived",
        _ => "other",
    }
}

fn change_position_key(
    address_or_range: Option<&str>,
    item_name: Option<&str>,
) -> (u32, u32, u32, u32, String) {
    if let Some(text) = address_or_range {
        if let Some(bounds) = parse_a1_range(text) {
            return (
                bounds.start_row,
                bounds.start_col,
                bounds.end_row,
                bounds.end_col,
                text.to_string(),
            );
        }
        if let Some((col, row)) = parse_a1_coord(text) {
            return (row, col, row, col, text.to_string());
        }
    }

    (
        u32::MAX,
        u32::MAX,
        u32::MAX,
        u32::MAX,
        item_name.unwrap_or("").to_string(),
    )
}

fn build_sheet_summaries(changes: &[Value], groups: &[DiffGroup]) -> Vec<SheetDiffSummary> {
    let mut counts_by_sheet: BTreeMap<Option<String>, SheetDiffSummary> = BTreeMap::new();

    for change in changes {
        let sheet_key = change_sheet_name(change).map(str::to_string);
        let entry = counts_by_sheet
            .entry(sheet_key.clone())
            .or_insert_with(|| SheetDiffSummary {
                sheet: sheet_key.clone(),
                total_changes: 0,
                direct_change_count: 0,
                recalc_result_change_count: 0,
                counts_by_subtype: BTreeMap::new(),
                group_count: 0,
                counts_by_group_type: BTreeMap::new(),
                group_preview: Vec::new(),
                group_preview_truncated: false,
            });
        entry.total_changes += 1;
        match change_subtype_key(change) {
            Some("recalc_result") => {
                entry.recalc_result_change_count += 1;
                *entry
                    .counts_by_subtype
                    .entry("recalc_result".to_string())
                    .or_default() += 1;
            }
            Some(subtype) => {
                entry.direct_change_count += 1;
                *entry
                    .counts_by_subtype
                    .entry(subtype.to_string())
                    .or_default() += 1;
            }
            None => {
                entry.direct_change_count += 1;
            }
        }
    }

    for group in groups {
        let entry = counts_by_sheet
            .entry(group.sheet.clone())
            .or_insert_with(|| SheetDiffSummary {
                sheet: group.sheet.clone(),
                total_changes: 0,
                direct_change_count: 0,
                recalc_result_change_count: 0,
                counts_by_subtype: BTreeMap::new(),
                group_count: 0,
                counts_by_group_type: BTreeMap::new(),
                group_preview: Vec::new(),
                group_preview_truncated: false,
            });
        entry.group_count += 1;
        *entry
            .counts_by_group_type
            .entry(group.group_type.clone())
            .or_default() += 1;
        if entry.group_preview.len() < 10 {
            entry.group_preview.push(group.clone());
        } else {
            entry.group_preview_truncated = true;
        }
    }

    counts_by_sheet.into_values().collect()
}

fn change_matches_filters(
    change: &Value,
    sheet_filters: &[String],
    range: Option<A1Bounds>,
) -> bool {
    if !sheet_filters.is_empty() {
        let Some(sheet_name) = change_sheet_name(change) else {
            return false;
        };
        if !sheet_filters
            .iter()
            .any(|f| sheet_name.eq_ignore_ascii_case(f))
        {
            return false;
        }
    }

    let Some(bounds) = range else {
        return true;
    };

    if let Some(address) = change.get("address").and_then(Value::as_str) {
        return address_in_bounds(address, bounds);
    }

    ["range", "old_range", "new_range"]
        .iter()
        .filter_map(|key| change.get(*key).and_then(Value::as_str))
        .any(|candidate| range_intersects(candidate, bounds))
}

fn address_in_bounds(address: &str, bounds: A1Bounds) -> bool {
    let Some((col, row)) = parse_a1_coord(address) else {
        return false;
    };
    col >= bounds.start_col
        && col <= bounds.end_col
        && row >= bounds.start_row
        && row <= bounds.end_row
}

fn range_intersects(range: &str, bounds: A1Bounds) -> bool {
    let Some(candidate) = parse_a1_range(range) else {
        return false;
    };

    !(candidate.end_col < bounds.start_col
        || candidate.start_col > bounds.end_col
        || candidate.end_row < bounds.start_row
        || candidate.start_row > bounds.end_row)
}

fn addresses_are_adjacent(left: &str, right: &str) -> bool {
    let (left_col, left_row) = match parse_a1_coord(left) {
        Some(v) => v,
        None => return false,
    };
    let (right_col, right_row) = match parse_a1_coord(right) {
        Some(v) => v,
        None => return false,
    };

    (left_row == right_row && left_col.abs_diff(right_col) == 1)
        || (left_col == right_col && left_row.abs_diff(right_row) == 1)
}

fn format_a1_range(start_col: u32, end_col: u32, start_row: u32, end_row: u32) -> String {
    let start = format!("{}{}", column_number_to_name(start_col), start_row);
    let end = format!("{}{}", column_number_to_name(end_col), end_row);
    if start == end {
        start
    } else {
        format!("{}:{}", start, end)
    }
}

fn column_number_to_name(mut col: u32) -> String {
    let mut chars = Vec::new();
    while col > 0 {
        let rem = ((col - 1) % 26) as u8;
        chars.push((b'A' + rem) as char);
        col = (col - 1) / 26;
    }
    chars.iter().rev().collect()
}

fn parse_a1_range(raw: &str) -> Option<A1Bounds> {
    let mut text = raw.trim();
    if text.is_empty() {
        return None;
    }

    if let Some((_, tail)) = text.rsplit_once('!') {
        text = tail;
    }

    let (left, right) = text.split_once(':').map_or((text, text), |(a, b)| (a, b));
    let (c1, r1) = parse_a1_coord(left)?;
    let (c2, r2) = parse_a1_coord(right)?;

    Some(A1Bounds {
        start_col: c1.min(c2),
        end_col: c1.max(c2),
        start_row: r1.min(r2),
        end_row: r1.max(r2),
    })
}

fn parse_a1_coord(raw: &str) -> Option<(u32, u32)> {
    let coord = raw.trim().trim_start_matches('$');
    if coord.is_empty() {
        return None;
    }

    let mut letters = String::new();
    let mut digits = String::new();
    for ch in coord.chars() {
        if ch == '$' {
            continue;
        }
        if ch.is_ascii_alphabetic() {
            if !digits.is_empty() {
                return None;
            }
            letters.push(ch.to_ascii_uppercase());
        } else if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            return None;
        }
    }

    if letters.is_empty() || digits.is_empty() {
        return None;
    }

    let mut col = 0u32;
    for ch in letters.bytes() {
        col = col
            .saturating_mul(26)
            .saturating_add((ch - b'A' + 1) as u32);
    }

    let row: u32 = digits.parse().ok()?;
    if col == 0 || row == 0 {
        return None;
    }

    Some((col, row))
}
