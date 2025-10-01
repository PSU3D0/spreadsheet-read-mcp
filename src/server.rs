use crate::config::ServerConfig;
use crate::model::{
    FindFormulaResponse, FormulaTraceResponse, ManifestStubResponse, NamedRangesResponse,
    SheetFormulaMapResponse, SheetListResponse, SheetOverviewResponse, SheetPageResponse,
    SheetStatisticsResponse, SheetStylesResponse, VolatileScanResponse, WorkbookDescription,
    WorkbookListResponse,
};
use crate::state::AppState;
use crate::tools;
use anyhow::Result;
use rmcp::{
    ErrorData as McpError, Json, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::stdio,
};
use std::sync::Arc;
use thiserror::Error;

#[derive(Clone)]
pub struct SpreadsheetServer {
    state: Arc<AppState>,
    tool_router: ToolRouter<SpreadsheetServer>,
}

impl SpreadsheetServer {
    pub async fn new(config: Arc<ServerConfig>) -> Result<Self> {
        config.ensure_workspace_root()?;
        let state = Arc::new(AppState::new(config));
        Ok(Self::from_state(state))
    }

    pub fn from_state(state: Arc<AppState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    pub async fn run_stdio(self) -> Result<()> {
        let service = self
            .serve(stdio())
            .await
            .inspect_err(|error| tracing::error!("serving error: {:?}", error))?;
        service.waiting().await?;
        Ok(())
    }

    pub async fn run(self) -> Result<()> {
        self.run_stdio().await
    }

    fn ensure_tool_enabled(&self, tool: &str) -> Result<()> {
        tracing::info!(tool = tool, "tool invocation requested");
        if self.state.config().is_tool_enabled(tool) {
            Ok(())
        } else {
            Err(ToolDisabledError::new(tool).into())
        }
    }
}

#[tool_router]
impl SpreadsheetServer {
    #[tool(
        name = "list_workbooks",
        description = "List spreadsheet files in the workspace"
    )]
    pub async fn list_workbooks(
        &self,
        Parameters(params): Parameters<tools::ListWorkbooksParams>,
    ) -> Result<Json<WorkbookListResponse>, McpError> {
        self.ensure_tool_enabled("list_workbooks")
            .map_err(to_mcp_error)?;
        tools::list_workbooks(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(name = "describe_workbook", description = "Describe workbook metadata")]
    pub async fn describe_workbook(
        &self,
        Parameters(params): Parameters<tools::DescribeWorkbookParams>,
    ) -> Result<Json<WorkbookDescription>, McpError> {
        self.ensure_tool_enabled("describe_workbook")
            .map_err(to_mcp_error)?;
        tools::describe_workbook(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(name = "list_sheets", description = "List sheets with summaries")]
    pub async fn list_sheets(
        &self,
        Parameters(params): Parameters<tools::ListSheetsParams>,
    ) -> Result<Json<SheetListResponse>, McpError> {
        self.ensure_tool_enabled("list_sheets")
            .map_err(to_mcp_error)?;
        tools::list_sheets(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(
        name = "sheet_overview",
        description = "Get narrative overview for a sheet"
    )]
    pub async fn sheet_overview(
        &self,
        Parameters(params): Parameters<tools::SheetOverviewParams>,
    ) -> Result<Json<SheetOverviewResponse>, McpError> {
        self.ensure_tool_enabled("sheet_overview")
            .map_err(to_mcp_error)?;
        tools::sheet_overview(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(name = "sheet_page", description = "Page through sheet cells")]
    pub async fn sheet_page(
        &self,
        Parameters(params): Parameters<tools::SheetPageParams>,
    ) -> Result<Json<SheetPageResponse>, McpError> {
        self.ensure_tool_enabled("sheet_page")
            .map_err(to_mcp_error)?;
        tools::sheet_page(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(
        name = "sheet_statistics",
        description = "Get aggregated sheet statistics"
    )]
    pub async fn sheet_statistics(
        &self,
        Parameters(params): Parameters<tools::SheetStatisticsParams>,
    ) -> Result<Json<SheetStatisticsResponse>, McpError> {
        self.ensure_tool_enabled("sheet_statistics")
            .map_err(to_mcp_error)?;
        tools::sheet_statistics(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(
        name = "sheet_formula_map",
        description = "Summarize formula groups across a sheet"
    )]
    pub async fn sheet_formula_map(
        &self,
        Parameters(params): Parameters<tools::SheetFormulaMapParams>,
    ) -> Result<Json<SheetFormulaMapResponse>, McpError> {
        self.ensure_tool_enabled("sheet_formula_map")
            .map_err(to_mcp_error)?;
        tools::sheet_formula_map(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(
        name = "formula_trace",
        description = "Trace formula precedents or dependents"
    )]
    pub async fn formula_trace(
        &self,
        Parameters(params): Parameters<tools::FormulaTraceParams>,
    ) -> Result<Json<FormulaTraceResponse>, McpError> {
        self.ensure_tool_enabled("formula_trace")
            .map_err(to_mcp_error)?;
        tools::formula_trace(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(name = "named_ranges", description = "List named ranges and tables")]
    pub async fn named_ranges(
        &self,
        Parameters(params): Parameters<tools::NamedRangesParams>,
    ) -> Result<Json<NamedRangesResponse>, McpError> {
        self.ensure_tool_enabled("named_ranges")
            .map_err(to_mcp_error)?;
        tools::named_ranges(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(name = "find_formula", description = "Search formulas containing text")]
    pub async fn find_formula(
        &self,
        Parameters(params): Parameters<tools::FindFormulaParams>,
    ) -> Result<Json<FindFormulaResponse>, McpError> {
        self.ensure_tool_enabled("find_formula")
            .map_err(to_mcp_error)?;
        tools::find_formula(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(name = "scan_volatiles", description = "Scan for volatile formulas")]
    pub async fn scan_volatiles(
        &self,
        Parameters(params): Parameters<tools::ScanVolatilesParams>,
    ) -> Result<Json<VolatileScanResponse>, McpError> {
        self.ensure_tool_enabled("scan_volatiles")
            .map_err(to_mcp_error)?;
        tools::scan_volatiles(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(
        name = "sheet_styles",
        description = "Summarise style usage for a sheet"
    )]
    pub async fn sheet_styles(
        &self,
        Parameters(params): Parameters<tools::SheetStylesParams>,
    ) -> Result<Json<SheetStylesResponse>, McpError> {
        self.ensure_tool_enabled("sheet_styles")
            .map_err(to_mcp_error)?;
        tools::sheet_styles(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(
        name = "get_manifest_stub",
        description = "Generate manifest scaffold for workbook"
    )]
    pub async fn get_manifest_stub(
        &self,
        Parameters(params): Parameters<tools::ManifestStubParams>,
    ) -> Result<Json<ManifestStubResponse>, McpError> {
        self.ensure_tool_enabled("get_manifest_stub")
            .map_err(to_mcp_error)?;
        tools::get_manifest_stub(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }

    #[tool(name = "close_workbook", description = "Evict a workbook from cache")]
    pub async fn close_workbook(
        &self,
        Parameters(params): Parameters<tools::CloseWorkbookParams>,
    ) -> Result<Json<String>, McpError> {
        self.ensure_tool_enabled("close_workbook")
            .map_err(to_mcp_error)?;
        tools::close_workbook(self.state.clone(), params)
            .await
            .map(Json)
            .map_err(to_mcp_error)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SpreadsheetServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(
                "Spreadsheet MCP server. Use tools to explore workbooks in the configured workspace.".to_string(),
            ),
            ..ServerInfo::default()
        }
    }
}

fn to_mcp_error(error: anyhow::Error) -> McpError {
    if error.downcast_ref::<ToolDisabledError>().is_some() {
        McpError::invalid_request(error.to_string(), None)
    } else {
        McpError::internal_error(error.to_string(), None)
    }
}

#[derive(Debug, Error)]
#[error("tool '{tool_name}' is disabled by server configuration")]
struct ToolDisabledError {
    tool_name: String,
}

impl ToolDisabledError {
    fn new(tool_name: &str) -> Self {
        Self {
            tool_name: tool_name.to_ascii_lowercase(),
        }
    }
}
