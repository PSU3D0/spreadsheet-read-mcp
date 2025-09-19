use chrono::{DateTime, SecondsFormat, Utc};
use sha2::{Digest, Sha256};
use std::fs::Metadata;
use std::path::Path;
use std::time::SystemTime;

pub fn system_time_to_datetime(time: SystemTime) -> Option<DateTime<Utc>> {
    Some(DateTime::<Utc>::from(time))
}

pub fn system_time_to_rfc3339(time: SystemTime) -> Option<DateTime<Utc>> {
    system_time_to_datetime(time)
}

pub fn hash_path_metadata(path: &Path, metadata: &Metadata) -> String {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(metadata.len().to_le_bytes());
    if let Ok(modified) = metadata.modified() {
        if let Some(dt) = system_time_to_datetime(modified) {
            hasher.update(dt.to_rfc3339_opts(SecondsFormat::Micros, true).as_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

pub fn column_number_to_name(column: u32) -> String {
    let mut column = column;
    let mut name = String::new();
    while column > 0 {
        let rem = ((column - 1) % 26) as u8;
        name.insert(0, (b'A' + rem) as char);
        column = (column - 1) / 26;
    }
    name
}

pub fn cell_address(column: u32, row: u32) -> String {
    format!("{}{}", column_number_to_name(column), row)
}

pub fn make_short_workbook_id(slug: &str, canonical_id: &str) -> String {
    let mut slug_part: String = slug
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    if slug_part.is_empty() {
        slug_part = "wb".to_string();
    }
    if slug_part.len() > 12 {
        slug_part.truncate(12);
    }
    let short_hash: String = canonical_id.chars().take(8).collect();
    format!("{}-{}", slug_part, short_hash)
}
