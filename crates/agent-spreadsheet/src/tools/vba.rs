use crate::model::{
    VbaModuleDescriptor, VbaModuleSourceResponse, VbaProjectSummaryResponse,
    VbaReferenceDescriptor, WorkbookId,
};
use crate::state::AppState;
use anyhow::{Result, anyhow, bail};
use schemars::JsonSchema;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use zip::result::ZipError;

const MAX_VBA_PROJECT_BYTES: u64 = 20 * 1024 * 1024;
const DEFAULT_MAX_MODULES: u32 = 200;
const DEFAULT_INCLUDE_REFERENCES: bool = true;

const DEFAULT_OFFSET_LINES: u32 = 0;
const DEFAULT_LIMIT_LINES: u32 = 200;
const MAX_LIMIT_LINES: u32 = 5_000;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct VbaProjectSummaryParams {
    #[serde(alias = "workbook_id")]
    pub workbook_or_fork_id: WorkbookId,
    #[serde(default)]
    pub max_modules: Option<u32>,
    #[serde(default)]
    pub include_references: Option<bool>,
}

pub async fn vba_project_summary(
    state: Arc<AppState>,
    params: VbaProjectSummaryParams,
) -> Result<VbaProjectSummaryResponse> {
    let workbook = state.open_workbook(&params.workbook_or_fork_id).await?;
    let raw = extract_vba_project_bin(&workbook.path)?;

    if raw.is_none() {
        return Ok(VbaProjectSummaryResponse {
            workbook_id: workbook.id.clone(),
            has_vba: false,
            code_page: None,
            sys_kind: None,
            modules: Vec::new(),
            modules_truncated: false,
            references: Vec::new(),
            references_truncated: false,
            notes: vec!["No xl/vbaProject.bin found in workbook".to_string()],
        });
    }

    let project = ovba::open_project(raw.unwrap())?;

    let max_modules = params.max_modules.unwrap_or(DEFAULT_MAX_MODULES).max(1);
    let include_references = params
        .include_references
        .unwrap_or(DEFAULT_INCLUDE_REFERENCES);

    let mut modules: Vec<VbaModuleDescriptor> = Vec::new();
    for module in project.modules.iter().take(max_modules as usize) {
        let module_type = match module.module_type {
            ovba::ModuleType::Procedural => "procedural",
            ovba::ModuleType::DocClsDesigner => "doc_cls_designer",
        }
        .to_string();

        modules.push(VbaModuleDescriptor {
            name: module.name.clone(),
            stream_name: module.stream_name.clone(),
            doc_string: module.doc_string.clone(),
            text_offset: module.text_offset as u64,
            help_context: module.help_context,
            module_type,
            read_only: module.read_only,
            private: module.private,
        });
    }

    let modules_truncated = project.modules.len() > max_modules as usize;

    let mut references: Vec<VbaReferenceDescriptor> = Vec::new();
    let mut references_truncated = false;
    if include_references {
        for reference in project.references.iter() {
            let (kind, debug) = summarize_reference(reference);
            references.push(VbaReferenceDescriptor { kind, debug });
            if references.len() >= 200 {
                references_truncated = project.references.len() > references.len();
                break;
            }
        }
    }

    let sys_kind = Some(
        match project.information.sys_kind {
            ovba::SysKind::Win16 => "win16",
            ovba::SysKind::Win32 => "win32",
            ovba::SysKind::MacOs => "macos",
            ovba::SysKind::Win64 => "win64",
        }
        .to_string(),
    );

    Ok(VbaProjectSummaryResponse {
        workbook_id: workbook.id.clone(),
        has_vba: true,
        code_page: Some(project.information.code_page),
        sys_kind,
        modules,
        modules_truncated,
        references,
        references_truncated,
        notes: Vec::new(),
    })
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct VbaModuleSourceParams {
    #[serde(alias = "workbook_id")]
    pub workbook_or_fork_id: WorkbookId,
    pub module_name: String,
    #[serde(default = "default_offset_lines")]
    pub offset_lines: u32,
    #[serde(default = "default_limit_lines")]
    pub limit_lines: u32,
}

fn default_offset_lines() -> u32 {
    DEFAULT_OFFSET_LINES
}

fn default_limit_lines() -> u32 {
    DEFAULT_LIMIT_LINES
}

pub async fn vba_module_source(
    state: Arc<AppState>,
    params: VbaModuleSourceParams,
) -> Result<VbaModuleSourceResponse> {
    let workbook = state.open_workbook(&params.workbook_or_fork_id).await?;
    let raw = extract_vba_project_bin(&workbook.path)?
        .ok_or_else(|| anyhow!("No xl/vbaProject.bin found in workbook"))?;

    let project = ovba::open_project(raw)?;
    let source = project.module_source(&params.module_name)?;

    let offset = params.offset_lines;
    let limit = params.limit_lines.clamp(1, MAX_LIMIT_LINES);

    let mut total_lines: u32 = 0;
    let mut selected: Vec<&str> = Vec::new();

    for (idx, line) in source.lines().enumerate() {
        let idx = idx as u32;
        total_lines = total_lines.saturating_add(1);
        if idx < offset {
            continue;
        }
        if selected.len() >= limit as usize {
            continue;
        }
        selected.push(line);
    }

    if total_lines == 0 && !source.is_empty() {
        total_lines = 1;
    }

    let truncated = total_lines.saturating_sub(offset) > limit;

    let mut page = selected.join("\n");
    if !page.is_empty() {
        page.push('\n');
    }

    Ok(VbaModuleSourceResponse {
        workbook_id: workbook.id.clone(),
        module_name: params.module_name,
        offset_lines: offset,
        limit_lines: limit,
        total_lines,
        truncated,
        source: page,
    })
}

fn extract_vba_project_bin(path: &Path) -> Result<Option<Vec<u8>>> {
    let file = File::open(path)
        .map_err(|e| anyhow!("failed to open workbook {}: {}", path.display(), e))?;

    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| anyhow!("failed to open workbook zip {}: {}", path.display(), e))?;

    let mut entry = match archive.by_name("xl/vbaProject.bin") {
        Ok(f) => f,
        Err(ZipError::FileNotFound) => return Ok(None),
        Err(e) => return Err(anyhow!("failed to locate xl/vbaProject.bin: {}", e)),
    };

    let declared_size = entry.size();
    if declared_size > MAX_VBA_PROJECT_BYTES {
        bail!(
            "xl/vbaProject.bin too large ({} bytes; max {} bytes)",
            declared_size,
            MAX_VBA_PROJECT_BYTES
        );
    }

    let mut buf: Vec<u8> = Vec::with_capacity(declared_size.min(1024 * 1024) as usize);
    entry
        .read_to_end(&mut buf)
        .map_err(|e| anyhow!("failed to read xl/vbaProject.bin: {}", e))?;

    if buf.len() as u64 > MAX_VBA_PROJECT_BYTES {
        bail!(
            "xl/vbaProject.bin too large after read ({} bytes; max {} bytes)",
            buf.len(),
            MAX_VBA_PROJECT_BYTES
        );
    }

    Ok(Some(buf))
}

fn summarize_reference(reference: &ovba::Reference) -> (String, String) {
    let kind = match reference {
        ovba::Reference::Control(_) => "control",
        ovba::Reference::Original(_) => "original",
        ovba::Reference::Registered(_) => "registered",
        ovba::Reference::Project(_) => "project",
    }
    .to_string();

    let mut debug = format!("{:?}", reference);
    const MAX_DEBUG_BYTES: usize = 4096;
    if debug.len() > MAX_DEBUG_BYTES {
        debug.truncate(MAX_DEBUG_BYTES);
        debug.push_str("...[truncated]");
    }

    (kind, debug)
}
