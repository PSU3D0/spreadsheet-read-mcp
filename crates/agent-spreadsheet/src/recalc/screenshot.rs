use super::RecalcConfig;
use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::{fs, task, time};

use super::macro_uri::export_screenshot_uri;

pub struct ScreenshotResult {
    pub output_path: PathBuf,
    pub size_bytes: u64,
    pub duration_ms: u64,
}

pub struct ScreenshotExecutor {
    soffice_path: PathBuf,
    timeout: Duration,
}

impl ScreenshotExecutor {
    pub fn new(config: &RecalcConfig) -> Self {
        Self {
            soffice_path: config
                .soffice_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("/usr/bin/soffice")),
            timeout: Duration::from_millis(config.timeout_ms.unwrap_or(30_000)),
        }
    }

    pub async fn screenshot(
        &self,
        workbook_path: &Path,
        output_path: &Path,
        sheet_name: &str,
        range: Option<&str>,
    ) -> Result<ScreenshotResult> {
        let start = Instant::now();

        let abs_path = workbook_path
            .canonicalize()
            .map_err(|e| anyhow!("failed to canonicalize workbook path: {}", e))?;

        let file_path = abs_path.display().to_string();
        let range_arg = range.unwrap_or("A1:M40");

        // LibreOffice will try to write cache/config under HOME and XDG dirs.
        // In containerized environments we want a known-writable location.
        let _ = std::fs::create_dir_all("/tmp/.cache");
        let _ = std::fs::create_dir_all("/tmp/.config");

        fn truncate_for_log(s: &str, max_bytes: usize) -> String {
            if s.len() <= max_bytes {
                return s.to_string();
            }

            // Keep it cheap; best-effort to not cut mid-char.
            let mut end = max_bytes;
            while end > 0 && !s.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}...[truncated]", &s[..end])
        }

        struct SofficeLogs {
            stdout: String,
            stderr: String,
        }

        let run_macro = |macro_uri: String| async move {
            let macro_result = time::timeout(self.timeout, {
                let mut cmd = Command::new(&self.soffice_path);
                if let Ok(root) = std::env::var("SPREADSHEET_MCP_LIBREOFFICE_USER_INSTALLATION")
                    && !root.trim().is_empty()
                {
                    let root = root.trim();
                    let uri = if root.starts_with("file://") {
                        root.to_string()
                    } else {
                        format!("file:///{}", root.trim_start_matches('/'))
                    };
                    cmd.arg(format!("-env:UserInstallation={}", uri));
                }
                cmd.args([
                    "--headless",
                    "--norestore",
                    "--nodefault",
                    "--nofirststartwizard",
                    "--nolockcheck",
                    "--calc",
                    &macro_uri,
                ])
                .env("HOME", "/tmp")
                .env("XDG_CACHE_HOME", "/tmp/.cache")
                .env("XDG_CONFIG_HOME", "/tmp/.config")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
            })
            .await
            .map_err(|_| anyhow!("soffice timed out after {:?}", self.timeout))
            .and_then(|res| res.map_err(|e| anyhow!("failed to spawn soffice: {}", e)))?;

            // LibreOffice Basic errors often land in stderr, but the process can still exit 0.
            // Capture stdout/stderr even on success so we can surface it on downstream failures.
            let stdout_raw = String::from_utf8_lossy(&macro_result.stdout);
            let stderr_raw = String::from_utf8_lossy(&macro_result.stderr);
            let stdout = truncate_for_log(stdout_raw.trim(), 16 * 1024);
            let stderr = truncate_for_log(stderr_raw.trim(), 16 * 1024);

            if !stdout.is_empty() || !stderr.is_empty() {
                tracing::debug!(
                    soffice_stdout = %stdout,
                    soffice_stderr = %stderr,
                    "soffice screenshot macro output"
                );
            }

            if !macro_result.status.success() {
                return Err(anyhow!(
                    "soffice screenshot macro failed (exit {}): stderr={}, stdout={}",
                    macro_result.status.code().unwrap_or(-1),
                    stderr,
                    stdout
                ));
            }

            Ok(SofficeLogs { stdout, stderr })
        };

        let pdf_output_path = output_path.with_extension("pdf");
        let macro_uri_pdf = export_screenshot_uri(
            &file_path,
            &pdf_output_path.display().to_string(),
            sheet_name,
            range_arg,
        )?;

        let macro_logs = run_macro(macro_uri_pdf).await?;

        fs::metadata(&pdf_output_path).await.map_err(|_| {
            anyhow!(
                "screenshot PDF output file not created at {} (soffice stderr={}, stdout={})",
                pdf_output_path.display(),
                macro_logs.stderr,
                macro_logs.stdout
            )
        })?;

        let out_dir = output_path
            .parent()
            .ok_or_else(|| anyhow!("output path has no parent directory"))?;

        let out_dir_str = out_dir
            .to_str()
            .ok_or_else(|| anyhow!("output directory is not valid UTF-8"))?;

        let pdf_str = pdf_output_path
            .to_str()
            .ok_or_else(|| anyhow!("pdf output path is not valid UTF-8"))?;

        let convert_result = time::timeout(self.timeout, {
            let mut cmd = Command::new(&self.soffice_path);
            if let Ok(root) = std::env::var("SPREADSHEET_MCP_LIBREOFFICE_USER_INSTALLATION")
                && !root.trim().is_empty()
            {
                let root = root.trim();
                let uri = if root.starts_with("file://") {
                    root.to_string()
                } else {
                    format!("file:///{}", root.trim_start_matches('/'))
                };
                cmd.arg(format!("-env:UserInstallation={}", uri));
            }
            cmd.args([
                "--headless",
                "--norestore",
                "--nodefault",
                "--nofirststartwizard",
                "--nolockcheck",
                "--convert-to",
                "png",
                "--outdir",
                out_dir_str,
                pdf_str,
            ])
            .env("HOME", "/tmp")
            .env("XDG_CACHE_HOME", "/tmp/.cache")
            .env("XDG_CONFIG_HOME", "/tmp/.config")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
        })
        .await
        .map_err(|_| {
            anyhow!(
                "soffice PDF->PNG conversion timed out after {:?}",
                self.timeout
            )
        })
        .and_then(|res| {
            res.map_err(|e| anyhow!("failed to spawn soffice for conversion: {}", e))
        })?;

        if !convert_result.status.success() {
            let stderr = String::from_utf8_lossy(&convert_result.stderr);
            let stdout = String::from_utf8_lossy(&convert_result.stdout);
            return Err(anyhow!(
                "soffice PDF->PNG conversion failed (exit {}): stderr={}, stdout={}",
                convert_result.status.code().unwrap_or(-1),
                stderr,
                stdout
            ));
        }

        let mut png_path = if fs::metadata(output_path).await.is_ok() {
            Some(output_path.to_path_buf())
        } else {
            let stem = pdf_output_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let mut dir = fs::read_dir(out_dir).await?;
            let mut found: Option<PathBuf> = None;
            while let Some(entry) = dir.next_entry().await? {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("png")
                    && let Some(file_stem) = path.file_stem().and_then(|s| s.to_str())
                    && file_stem.starts_with(stem)
                {
                    found = Some(path);
                    break;
                }
            }
            found
        };

        if png_path.is_none() {
            let prefix = output_path.with_extension("");
            let prefix_str = prefix
                .to_str()
                .ok_or_else(|| anyhow!("PNG prefix path is not valid UTF-8"))?;

            let pdftoppm_result = time::timeout(
                self.timeout,
                Command::new("pdftoppm")
                    .args(["-png", "-singlefile", pdf_str, prefix_str])
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .output(),
            )
            .await
            .map_err(|_| anyhow!("pdftoppm conversion timed out after {:?}", self.timeout))
            .and_then(|res| res.map_err(|e| anyhow!("failed to spawn pdftoppm: {}", e)))?;

            if !pdftoppm_result.status.success() {
                let stderr = String::from_utf8_lossy(&pdftoppm_result.stderr);
                let stdout = String::from_utf8_lossy(&pdftoppm_result.stdout);
                return Err(anyhow!(
                    "pdftoppm PDF->PNG conversion failed (exit {}): stderr={}, stdout={}",
                    pdftoppm_result.status.code().unwrap_or(-1),
                    stderr,
                    stdout
                ));
            }

            if fs::metadata(output_path).await.is_ok() {
                png_path = Some(output_path.to_path_buf());
            }
        }

        let png_path = png_path
            .ok_or_else(|| anyhow!("PNG output file not created in {}", out_dir.display()))?;

        let _ = fs::remove_file(&pdf_output_path).await;

        crop_png_best_effort(&png_path).await;

        let metadata = fs::metadata(&png_path).await.map_err(|_| {
            anyhow!(
                "screenshot PNG output file not created at {}",
                png_path.display()
            )
        })?;

        Ok(ScreenshotResult {
            output_path: png_path,
            size_bytes: metadata.len(),
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    pub fn is_available(&self) -> bool {
        self.soffice_path.exists()
    }
}

async fn crop_png_best_effort(path: &Path) {
    let path = path.to_path_buf();
    let _ = task::spawn_blocking(move || crop_png_in_place(&path)).await;
}

fn crop_png_in_place(path: &Path) -> Result<()> {
    use image::ImageFormat;

    let img = image::ImageReader::open(path)
        .and_then(|r| r.with_guessed_format())
        .map_err(|e| anyhow!("failed to read png {}: {}", path.display(), e))?
        .decode()
        .map_err(|e| anyhow!("failed to decode png {}: {}", path.display(), e))?;

    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    if width == 0 || height == 0 {
        return Ok(());
    }

    let background = estimate_background_color(&rgba, width, height);
    // First pass uses a conservative threshold to avoid cropping to noise.
    // If that fails (e.g., very faint gridlines), fall back to a more sensitive pass.
    let bbox = find_foreground_bbox(&rgba, background, 20, 100)
        .or_else(|| find_foreground_bbox(&rgba, background, 8, 20));

    let Some((min_x, min_y, max_x, max_y)) = bbox else {
        return Ok(());
    };

    let padding = 8u32;
    let min_x = min_x.saturating_sub(padding);
    let min_y = min_y.saturating_sub(padding);
    let max_x = (max_x + padding).min(width - 1);
    let max_y = (max_y + padding).min(height - 1);
    let crop_w = max_x - min_x + 1;
    let crop_h = max_y - min_y + 1;

    if crop_w == width && crop_h == height {
        return Ok(());
    }

    let cropped = image::imageops::crop_imm(&rgba, min_x, min_y, crop_w, crop_h).to_image();
    let tmp_path = path.with_extension("tmp.png");
    cropped.save_with_format(&tmp_path, ImageFormat::Png)?;
    std::fs::rename(&tmp_path, path)?;

    Ok(())
}

fn find_foreground_bbox(
    rgba: &image::RgbaImage,
    background: [u8; 3],
    epsilon: i32,
    min_foreground: u64,
) -> Option<(u32, u32, u32, u32)> {
    let (width, height) = rgba.dimensions();
    let epsilon_sq: i32 = epsilon * epsilon;

    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut foreground_count: u64 = 0;

    for y in 0..height {
        for x in 0..width {
            let p = rgba.get_pixel(x, y);
            let alpha = p[3] as i32;
            let dr = p[0] as i32 - background[0] as i32;
            let dg = p[1] as i32 - background[1] as i32;
            let db = p[2] as i32 - background[2] as i32;
            let dist_sq = dr * dr + dg * dg + db * db;

            if alpha < 250 || dist_sq > epsilon_sq {
                foreground_count += 1;
                if x < min_x {
                    min_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if x > max_x {
                    max_x = x;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
    }

    if foreground_count < min_foreground || min_x > max_x || min_y > max_y {
        None
    } else {
        Some((min_x, min_y, max_x, max_y))
    }
}

fn estimate_background_color(img: &image::RgbaImage, width: u32, height: u32) -> [u8; 3] {
    let sample = 8u32.min(width).min(height).max(1);

    let corner_mean = |x0: u32, y0: u32| -> [f32; 3] {
        let mut sum = [0f32; 3];
        let mut count = 0f32;
        for y in y0..(y0 + sample) {
            for x in x0..(x0 + sample) {
                let p = img.get_pixel(x, y);
                sum[0] += p[0] as f32;
                sum[1] += p[1] as f32;
                sum[2] += p[2] as f32;
                count += 1.0;
            }
        }
        [sum[0] / count, sum[1] / count, sum[2] / count]
    };

    let tl = corner_mean(0, 0);
    let tr = corner_mean(width - sample, 0);
    let bl = corner_mean(0, height - sample);
    let br = corner_mean(width - sample, height - sample);
    let corners = [tl, tr, bl, br];

    let luminance = |c: [f32; 3]| (c[0] + c[1] + c[2]) / 3.0;
    let lums = [luminance(tl), luminance(tr), luminance(bl), luminance(br)];
    let (min_lum, max_lum) = lums.iter().fold((f32::MAX, f32::MIN), |acc, v| {
        (acc.0.min(*v), acc.1.max(*v))
    });

    // If one corner is substantially brighter than others, treat it as paper/background.
    // This avoids selecting Calc's gray row/column header strip as background.
    if max_lum - min_lum >= 12.0 {
        let mut best = 0usize;
        for i in 1..lums.len() {
            if lums[i] > lums[best] {
                best = i;
            }
        }
        let c = corners[best];
        return [c[0] as u8, c[1] as u8, c[2] as u8];
    }

    let dist2 = |a: [f32; 3], b: [f32; 3]| -> f32 {
        let dr = a[0] - b[0];
        let dg = a[1] - b[1];
        let db = a[2] - b[2];
        dr * dr + dg * dg + db * db
    };

    let mut best_idx = 0usize;
    let mut best_score = f32::MAX;
    for i in 0..corners.len() {
        let mut score = 0f32;
        for j in 0..corners.len() {
            if i != j {
                score += dist2(corners[i], corners[j]);
            }
        }
        if score < best_score {
            best_score = score;
            best_idx = i;
        }
    }

    let bg = corners[best_idx];
    [
        bg[0].round() as u8,
        bg[1].round() as u8,
        bg[2].round() as u8,
    ]
}
