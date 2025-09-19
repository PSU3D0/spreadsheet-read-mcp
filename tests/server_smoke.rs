use anyhow::Result;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::ErrorCode;
use std::collections::HashSet;
use std::sync::Arc;

use spreadsheet_read_mcp::SpreadsheetServer;
use spreadsheet_read_mcp::tools::{ListWorkbooksParams, SheetPageParams};

mod support;

#[tokio::test(flavor = "current_thread")]
async fn server_tool_handlers_return_json() -> Result<()> {
    let workspace = support::TestWorkspace::new();
    workspace.create_workbook("simple.xlsx", |book| {
        let sheet = book.get_sheet_by_name_mut("Sheet1").unwrap();
        sheet.get_cell_mut((1, 1)).set_value("Name".to_string());
        sheet.get_cell_mut((2, 1)).set_value("Value".to_string());
        sheet.get_cell_mut((1, 2)).set_value("Alpha".to_string());
        sheet.get_cell_mut((2, 2)).set_value_number(10_f64);
        sheet
            .get_cell_mut((2, 3))
            .set_formula("B2*2")
            .set_formula_result_default("20");
    });

    let server = workspace.server().await?;

    let list = server
        .list_workbooks(Parameters(ListWorkbooksParams {
            slug_prefix: None,
            folder: None,
            path_glob: None,
        }))
        .await
        .expect("list workbooks")
        .0;
    assert_eq!(list.workbooks.len(), 1);
    let workbook_id = list.workbooks[0].workbook_id.clone();

    let error = match server
        .sheet_page(Parameters(SheetPageParams {
            workbook_id: workbook_id.clone(),
            sheet_name: "Missing".to_string(),
            start_row: 1,
            page_size: 10,
            columns: None,
            include_formulas: true,
            include_styles: false,
        }))
        .await
    {
        Ok(_) => panic!("missing sheet should error"),
        Err(err) => err,
    };
    assert!(error.message.contains("sheet Missing"));

    let page = server
        .sheet_page(Parameters(SheetPageParams {
            workbook_id,
            sheet_name: "Sheet1".to_string(),
            start_row: 1,
            page_size: 10,
            columns: None,
            include_formulas: true,
            include_styles: false,
        }))
        .await
        .expect("page fetch")
        .0;
    assert_eq!(page.rows.len(), 3);
    let contains_b3 = page
        .rows
        .iter()
        .flat_map(|row| row.cells.iter())
        .any(|cell| cell.address == "B3");
    assert!(contains_b3);

    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn disabled_tools_return_invalid_request() -> Result<()> {
    let workspace = support::TestWorkspace::new();
    let workbook_path = workspace.create_workbook("locked.xlsx", |_| {});

    let mut enabled = HashSet::new();
    enabled.insert("list_workbooks".to_string());

    let config = workspace.config_with(|cfg| {
        cfg.enabled_tools = Some(enabled);
        cfg.single_workbook = Some(workbook_path.clone());
    });
    let server = SpreadsheetServer::new(Arc::new(config)).await?;

    let list = server
        .list_workbooks(Parameters(ListWorkbooksParams {
            slug_prefix: None,
            folder: None,
            path_glob: None,
        }))
        .await?
        .0;
    assert_eq!(list.workbooks.len(), 1);
    let workbook_id = list.workbooks[0].workbook_id.clone();

    let error = match server
        .sheet_page(Parameters(SheetPageParams {
            workbook_id,
            sheet_name: "Sheet1".to_string(),
            start_row: 1,
            page_size: 5,
            columns: None,
            include_formulas: true,
            include_styles: false,
        }))
        .await
    {
        Ok(_) => panic!("sheet_page should be disabled"),
        Err(err) => err,
    };

    assert_eq!(error.code, ErrorCode::INVALID_REQUEST);
    assert!(
        error
            .message
            .contains("tool 'sheet_page' is disabled by server configuration")
    );

    Ok(())
}
