use crate::runtime::stateless::StatelessRuntime;
use crate::tools::{self, NamedRangesParams};
use crate::verification::{VerifyOptions, compare_workbooks};
use anyhow::Result;
use serde_json::Value;
use std::path::PathBuf;

pub async fn verify(
    baseline: PathBuf,
    current: PathBuf,
    targets: Option<Vec<String>>,
    sheet_name: Option<String>,
    named_ranges: bool,
    errors_only: bool,
    targets_only: bool,
) -> Result<Value> {
    let options = VerifyOptions {
        targets: targets.unwrap_or_default(),
        sheet_filter: sheet_name.clone(),
        include_named_range_deltas: named_ranges,
        errors_only,
        targets_only,
    };
    options.validate()?;

    let runtime = StatelessRuntime;
    let baseline = runtime.normalize_existing_file(&baseline)?;
    let current = runtime.normalize_existing_file(&current)?;

    let (baseline_state, baseline_id) = runtime.open_state_for_file(&baseline).await?;
    let (current_state, current_id) = runtime.open_state_for_file(&current).await?;

    let baseline_workbook = baseline_state.open_workbook(&baseline_id).await?;
    let current_workbook = current_state.open_workbook(&current_id).await?;

    let baseline_named = if named_ranges {
        Some(
            tools::named_ranges(
                baseline_state.clone(),
                NamedRangesParams {
                    workbook_or_fork_id: baseline_id.clone(),
                    sheet_name: sheet_name.clone(),
                    name_prefix: None,
                },
            )
            .await?,
        )
    } else {
        None
    };
    let current_named = if named_ranges {
        Some(
            tools::named_ranges(
                current_state.clone(),
                NamedRangesParams {
                    workbook_or_fork_id: current_id.clone(),
                    sheet_name,
                    name_prefix: None,
                },
            )
            .await?,
        )
    } else {
        None
    };

    let response = compare_workbooks(
        baseline.display().to_string(),
        current.display().to_string(),
        &baseline_workbook,
        &current_workbook,
        &options,
        baseline_named.as_ref().map(|r| r.items.as_slice()),
        current_named.as_ref().map(|r| r.items.as_slice()),
    )?;

    Ok(serde_json::to_value(response)?)
}
