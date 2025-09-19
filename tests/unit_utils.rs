use std::fs;
use std::time::{Duration, SystemTime};

use spreadsheet_read_mcp::utils::{
    cell_address, column_number_to_name, hash_path_metadata, make_short_workbook_id,
    system_time_to_rfc3339,
};

#[test]
fn column_name_and_cell_address_round_trip() {
    assert_eq!(column_number_to_name(1), "A");
    assert_eq!(column_number_to_name(26), "Z");
    assert_eq!(column_number_to_name(27), "AA");
    assert_eq!(column_number_to_name(702), "ZZ");
    assert_eq!(cell_address(1, 1), "A1");
    assert_eq!(cell_address(28, 42), "AB42");
}

#[test]
fn make_short_workbook_id_sanitizes_slug() {
    let short = make_short_workbook_id("Quarterly P&L 🚀", "abcdef0123456789");
    assert_eq!(short, "quarterlypl-abcdef01");

    let fallback = make_short_workbook_id("!!!", "1234567890");
    assert_eq!(fallback, "wb-12345678");
}

#[test]
fn system_time_to_rfc3339_returns_instant() {
    let now = SystemTime::now();
    let dt = system_time_to_rfc3339(now).expect("timestamp");
    let formatted = dt.to_rfc3339();
    assert!(formatted.contains('T'));

    let earlier = now - Duration::from_secs(60);
    let earlier_dt = system_time_to_rfc3339(earlier).expect("earlier");
    assert!(dt > earlier_dt);
}

#[test]
fn hash_path_metadata_changes_with_file_contents() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let path = tempdir.path().join("sample.xlsx");
    fs::write(&path, b"a").expect("write file");
    let meta1 = fs::metadata(&path).expect("metadata");
    let hash1 = hash_path_metadata(&path, &meta1);

    fs::write(&path, b"ab").expect("rewrite file");
    let meta2 = fs::metadata(&path).expect("metadata");
    let hash2 = hash_path_metadata(&path, &meta2);

    assert_ne!(hash1, hash2);
    assert_eq!(hash1.len(), 64);
    assert_eq!(hash2.len(), 64);
}
