//! Scene -> RGBA pixmap -> PNG. tiny-skia and png only.

use tiny_skia::{
    FillRule, LineCap, Mask, Paint, PathBuilder, Pixmap, Rect as SkRect, Stroke, StrokeDash,
    Transform,
};

use crate::scene::{Cmd, Dash, HAlign, Rect, Rgba, Scene, VAlign};
use crate::text::Fonts;

/// Hard bounds on the pixmap. The canonical surface already caps the range at
/// 100 rows by 30 columns; these guard a hand-built scene.
pub const MAX_DIM: u32 = 8192;
pub const MAX_PIXELS: u64 = 16_777_216;

/// Line box height as a multiple of the font size. Matches the auto-fit model
/// in extraction, so a wrapped cell's computed row height and its painted
/// lines agree.
pub const LINE_HEIGHT_FACTOR: f32 = 1.2;

/// PNG zlib compression level.
///
/// The 5003 bake-off measured encode at 35.6 ms of a 51.4 ms median render:
/// it is the dominant cost, not rasterisation. `Fastest` is 8.3x faster and
/// 5.4x larger in the aggregate. For MCP image content the bytes are
/// base64-inflated into a model context, so the default stays `Balanced` and
/// the level is an explicit knob rather than whatever `Pixmap::encode_png`
/// picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PngLevel {
    Fast,
    #[default]
    Balanced,
    Best,
}

impl PngLevel {
    fn to_png(self) -> png::Compression {
        match self {
            PngLevel::Fast => png::Compression::Fast,
            PngLevel::Balanced => png::Compression::Balanced,
            PngLevel::Best => png::Compression::High,
        }
    }

    /// Stable lowercase name, for CLI and report round-tripping.
    pub const fn as_str(self) -> &'static str {
        match self {
            PngLevel::Fast => "fast",
            PngLevel::Balanced => "balanced",
            PngLevel::Best => "best",
        }
    }
}

/// Paint the scene. The command list is already in paint order.
pub fn render(scene: &Scene, fonts: &Fonts) -> Result<Pixmap, crate::RenderError> {
    let width = scene.width.ceil().max(1.0) as u32;
    let height = scene.height.ceil().max(1.0) as u32;
    if u64::from(width) * u64::from(height) > MAX_PIXELS || width > MAX_DIM || height > MAX_DIM {
        return Err(crate::RenderError::TooLarge { width, height });
    }
    let mut pixmap =
        Pixmap::new(width, height).ok_or(crate::RenderError::TooLarge { width, height })?;
    pixmap.fill(tiny_skia::Color::WHITE);

    for cmd in &scene.cmds {
        match cmd {
            Cmd::FillRect { rect, color } => fill_rect(&mut pixmap, rect, *color),
            Cmd::Line {
                x1,
                y1,
                x2,
                y2,
                width,
                color,
                dash,
            } => draw_line(&mut pixmap, *x1, *y1, *x2, *y2, *width, *color, *dash),
            Cmd::Text { .. } => draw_text(&mut pixmap, cmd, fonts),
        }
    }
    Ok(pixmap)
}

fn fill_rect(pixmap: &mut Pixmap, rect: &Rect, color: Rgba) {
    let Some(sk) = SkRect::from_xywh(rect.x, rect.y, rect.w.max(0.01), rect.h.max(0.01)) else {
        return;
    };
    let mut paint = Paint::default();
    paint.set_color_rgba8(color.0, color.1, color.2, color.3);
    paint.anti_alias = false;
    pixmap.fill_rect(sk, &paint, Transform::identity(), None);
}

#[allow(clippy::too_many_arguments)]
fn draw_line(
    pixmap: &mut Pixmap,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    width: f32,
    color: Rgba,
    dash: Dash,
) {
    let horizontal = (y1 - y2).abs() < 0.01;
    let vertical = (x1 - x2).abs() < 0.01;
    let mut paint = Paint::default();
    paint.set_color_rgba8(color.0, color.1, color.2, color.3);
    paint.anti_alias = false;

    // A solid axis-aligned rule is a filled band, not a stroke: snapping it to
    // whole pixels keeps hairlines crisp and, more importantly, keeps the
    // output byte-identical across targets.
    if matches!(dash, Dash::Solid) && (horizontal || vertical) {
        let band = if horizontal {
            SkRect::from_xywh(
                x1.min(x2),
                (y1 - width / 2.0).round(),
                (x2 - x1).abs().max(0.01),
                width.round().max(1.0),
            )
        } else {
            SkRect::from_xywh(
                (x1 - width / 2.0).round(),
                y1.min(y2),
                width.round().max(1.0),
                (y2 - y1).abs().max(0.01),
            )
        };
        if let Some(band) = band {
            pixmap.fill_rect(band, &paint, Transform::identity(), None);
        }
        return;
    }

    let mut stroke = Stroke {
        width,
        line_cap: LineCap::Butt,
        ..Stroke::default()
    };
    stroke.dash = dash_pattern(dash, width);
    let mut paint_at = |offset: f32| {
        let mut builder = PathBuilder::new();
        let snap = |v: f32| v.floor() + 0.5;
        if horizontal {
            builder.move_to(x1, snap(y1) + offset);
            builder.line_to(x2, snap(y2) + offset);
        } else if vertical {
            builder.move_to(snap(x1) + offset, y1);
            builder.line_to(snap(x2) + offset, y2);
        } else {
            builder.move_to(x1, y1);
            builder.line_to(x2, y2);
        }
        if let Some(path) = builder.finish() {
            pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    };
    paint_at(0.0);
    // A double rule is two bands one gap apart.
    if matches!(dash, Dash::Double) {
        paint_at(2.0 * width);
    }
}

fn dash_pattern(dash: Dash, width: f32) -> Option<StrokeDash> {
    let build = |rhythm: &[f32]| StrokeDash::new(rhythm.iter().map(|k| k * width).collect(), 0.0);
    match dash {
        Dash::Solid | Dash::Double => None,
        Dash::Dotted => build(&[1.0, 3.0]),
        Dash::Dashed => build(&[4.0, 3.0]),
        Dash::DashDot => build(&[4.0, 3.0, 1.0, 3.0]),
        Dash::DashDotDot => build(&[4.0, 3.0, 1.0, 3.0, 1.0, 3.0]),
    }
}

fn draw_text(pixmap: &mut Pixmap, cmd: &Cmd, fonts: &Fonts) {
    let Cmd::Text {
        rect,
        text,
        size_px,
        color,
        bold,
        italic,
        underline,
        strike,
        halign,
        valign,
        wrap,
        clip,
    } = cmd
    else {
        return;
    };
    let lines: Vec<String> = if *wrap {
        wrap_lines(text, rect.w, *size_px, *bold, fonts)
    } else {
        vec![text.replace('\n', " ")]
    };
    let line_height = size_px * LINE_HEIGHT_FACTOR;
    let block_height = line_height * lines.len() as f32;
    let mut y = match valign {
        VAlign::Top => rect.y,
        VAlign::Center => rect.y + (rect.h - block_height) / 2.0,
        VAlign::Bottom => rect.y + rect.h - block_height,
    };

    // A full-pixmap mask per run is the dominant raster cost, so only pay for
    // it when the run can actually escape its clip rect.
    let widest = lines
        .iter()
        .map(|line| fonts.measure(line, *size_px, *bold))
        .fold(0.0f32, f32::max);
    let needs_clip = widest > clip.w + 0.5 || block_height > clip.h + 0.5 || y < clip.y - 0.5;
    let mask = if needs_clip {
        clip_mask(pixmap, clip)
    } else {
        None
    };

    for line in &lines {
        let advance = fonts.measure(line, *size_px, *bold);
        let x = match halign {
            HAlign::Left => rect.x,
            HAlign::Center => rect.x + (rect.w - advance) / 2.0,
            HAlign::Right => rect.x + rect.w - advance,
        };
        let baseline = y + fonts.ascent_px(*size_px, *bold);
        fonts.draw(
            pixmap,
            line,
            x,
            baseline,
            *size_px,
            *bold,
            *italic,
            *color,
            mask.as_ref(),
        );
        let mut rule = |offset: f32| {
            let mut paint = Paint::default();
            paint.set_color_rgba8(color.0, color.1, color.2, color.3);
            paint.anti_alias = false;
            if let Some(sk) = SkRect::from_xywh(
                x,
                baseline + offset,
                advance.max(0.01),
                (size_px * 0.07).max(1.0),
            ) {
                let path = PathBuilder::from_rect(sk);
                pixmap.fill_path(
                    &path,
                    &paint,
                    FillRule::Winding,
                    Transform::identity(),
                    mask.as_ref(),
                );
            }
        };
        if *underline {
            rule(size_px * 0.12);
        }
        if *strike {
            rule(-size_px * 0.28);
        }
        y += line_height;
    }
}

fn clip_mask(pixmap: &Pixmap, clip: &Rect) -> Option<Mask> {
    let mut mask = Mask::new(pixmap.width(), pixmap.height())?;
    let rect = SkRect::from_xywh(clip.x, clip.y, clip.w.max(0.01), clip.h.max(0.01))?;
    mask.fill_path(
        &PathBuilder::from_rect(rect),
        FillRule::Winding,
        false,
        Transform::identity(),
    );
    Some(mask)
}

/// Greedy word wrap on the measured advance. Shared with extraction's row
/// auto-fit so the computed height and the painted lines agree.
pub fn wrap_lines(
    text: &str,
    max_width: f32,
    size_px: f32,
    bold: bool,
    fonts: &Fonts,
) -> Vec<String> {
    let mut out = Vec::new();
    for paragraph in text.split('\n') {
        let mut current = String::new();
        for word in paragraph.split_whitespace() {
            let candidate = if current.is_empty() {
                word.to_string()
            } else {
                format!("{current} {word}")
            };
            if fonts.measure(&candidate, size_px, bold) <= max_width || current.is_empty() {
                current = candidate;
            } else {
                out.push(std::mem::take(&mut current));
                current = word.to_string();
            }
        }
        out.push(current);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Encode at an explicit zlib level. Deliberately not `Pixmap::encode_png`,
/// which fixes the level.
pub fn encode_png(pixmap: &Pixmap, level: PngLevel) -> Result<Vec<u8>, crate::RenderError> {
    let mut buffer = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buffer, pixmap.width(), pixmap.height());
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(level.to_png());
        let mut writer = encoder
            .write_header()
            .map_err(|e| crate::RenderError::Encode(e.to_string()))?;
        writer
            .write_image_data(pixmap.data())
            .map_err(|e| crate::RenderError::Encode(e.to_string()))?;
    }
    Ok(buffer)
}
