use std::sync::Arc;

use spreadsheet_read_mcp::tools::filters::WorkbookFilter;
use spreadsheet_read_mcp::workbook::{WorkbookContext, build_workbook_list};

mod support;

#[test]
fn build_workbook_list_respects_filters() {
    let workspace = support::TestWorkspace::new();
    workspace.create_workbook("reports/summary.xlsx", |_| {});
    workspace.create_workbook("ops/dashboard.xlsb", |_| {});
    support::touch_file(&workspace.path("other/notes.txt"));

    let config = Arc::new(workspace.config());
    let filter = WorkbookFilter::new(Some("sum".to_string()), Some("reports".to_string()), None)
        .expect("filter");

    let response = build_workbook_list(&config, &filter).expect("list workbooks");
    assert_eq!(response.workbooks.len(), 1);
    let descriptor = &response.workbooks[0];
    assert_eq!(descriptor.slug, "summary");
    assert_eq!(descriptor.folder.as_deref(), Some("reports"));
    assert_eq!(descriptor.path, "reports/summary.xlsx");
    assert!(descriptor.bytes > 0);
    assert!(descriptor.last_modified.is_some());
}

#[test]
fn workbook_context_caches_sheet_metrics() {
    let workspace = support::TestWorkspace::new();
    let path = workspace.create_workbook("metrics.xlsx", |book| {
        let sheet = book.get_sheet_by_name_mut("Sheet1").unwrap();
        for row in 1..=3 {
            for col in 1..=3 {
                sheet
                    .get_cell_mut((col, row))
                    .set_value_number((row * 10 + col) as i32);
            }
        }
        sheet.get_cell_mut("A4").set_formula("SUM(A1:A3)");
    });

    let config = Arc::new(workspace.config());
    let context = WorkbookContext::load(&config, &path).expect("load workbook");

    let metrics_first = context.get_sheet_metrics("Sheet1").expect("metrics");
    let metrics_second = context.get_sheet_metrics("Sheet1").expect("metrics");
    assert!(Arc::ptr_eq(&metrics_first, &metrics_second));

    assert_eq!(metrics_first.metrics.non_empty_cells, 9);
    assert_eq!(metrics_first.metrics.formula_cells, 1);
    assert_eq!(metrics_first.metrics.cached_values, 0);

    let summary = context.describe();
    assert_eq!(summary.sheet_count, 1);
    assert_eq!(summary.slug, "metrics");
    assert!(summary.caps.supports_styles);
    assert!(summary.caps.supports_formula_graph);
    assert!(summary.bytes > 0);
}

#[test]
fn build_workbook_list_single_mode_filters_properly() {
    let workspace = support::TestWorkspace::new();
    let focus_path = workspace.create_workbook("focus/only.xlsx", |_| {});
    workspace.create_workbook("other/ignored.xlsx", |_| {});

    let config = Arc::new(workspace.config_with(|cfg| {
        cfg.single_workbook = Some(focus_path.clone());
    }));
    let filter = WorkbookFilter::default();

    let response = build_workbook_list(&config, &filter).expect("list workbooks");
    assert_eq!(response.workbooks.len(), 1);
    let descriptor = &response.workbooks[0];
    assert_eq!(descriptor.slug, "only");
    assert_eq!(descriptor.folder.as_deref(), Some("focus"));
    assert_eq!(descriptor.path, "focus/only.xlsx");
}
