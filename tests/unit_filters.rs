use std::path::Path;

use spreadsheet_read_mcp::tools::filters::WorkbookFilter;

#[test]
fn matches_slug_folder_and_glob_case_insensitive() {
    let filter = WorkbookFilter::new(
        Some("Fin".to_string()),
        Some("reports".to_string()),
        Some("**/2024-*.xlsx".to_string()),
    )
    .expect("filter");

    let path = Path::new("/workspace/reports/2024-q1.xlsx");
    assert!(filter.matches("FinancialSummary", Some("reports"), path));

    let wrong_slug = filter.matches("Budget", Some("reports"), path);
    assert!(!wrong_slug);

    let wrong_folder = filter.matches("FinancialSummary", Some("ops"), path);
    assert!(!wrong_folder);

    let wrong_path = filter.matches(
        "FinancialSummary",
        Some("reports"),
        Path::new("/workspace/reports/quarterly.xlsx"),
    );
    assert!(!wrong_path);
}

#[test]
fn invalid_glob_is_error() {
    let result = WorkbookFilter::new(None, None, Some("[".to_string()));
    match result {
        Ok(_) => panic!("expected glob construction to fail"),
        Err(err) => assert!(err.to_string().contains("invalid glob pattern")),
    }
}
