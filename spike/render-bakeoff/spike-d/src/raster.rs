//! Scene -> RGBA pixmap -> PNG, tiny-skia only.

use tiny_skia::{
    FillRule, LineCap, Paint, PathBuilder, Pixmap, Rect as SkRect, Stroke, StrokeDash, Transform,
};

use crate::scene::{Cmd, Dash, HAlign, Rgba, Scene, VAlign};
use crate::text::Fonts;

pub const MAX_DIM: u32 = 8192;
pub const MAX_PIXELS: u64 = 16_777_216;

#[derive(Debug)]
pub enum RenderError {
    Empty,
    TooLarge { w: u32, h: u32 },
    Encode(String),
}

fn dash_pattern(d: Dash, w: f32) -> Option<StrokeDash> {
    let m = |v: &[f32]| StrokeDash::new(v.iter().map(|k| k * w).collect(), 0.0);
    match d {
        Dash::Solid | Dash::Double => None,
        Dash::Dotted => m(&[1.0, 3.0]),
        Dash::Dashed => m(&[4.0, 3.0]),
        Dash::DashDot => m(&[4.0, 3.0, 1.0, 3.0]),
        Dash::DashDotDot => m(&[4.0, 3.0, 1.0, 3.0, 1.0, 3.0]),
    }
}

pub fn render(scene: &Scene, fonts: &Fonts) -> Result<Pixmap, RenderError> {
    let w = scene.width.ceil().max(1.0) as u32;
    let h = scene.height.ceil().max(1.0) as u32;
    if w == 0 || h == 0 {
        return Err(RenderError::Empty);
    }
    if w > MAX_DIM || h > MAX_DIM || u64::from(w) * u64::from(h) > MAX_PIXELS {
        return Err(RenderError::TooLarge { w, h });
    }
    let mut pix = Pixmap::new(w, h).ok_or(RenderError::TooLarge { w, h })?;
    pix.fill(tiny_skia::Color::WHITE);

    for cmd in &scene.cmds {
        match cmd {
            Cmd::FillRect { rect, color } => {
                let Some(r) = SkRect::from_xywh(rect.x, rect.y, rect.w.max(0.01), rect.h.max(0.01))
                else {
                    continue;
                };
                let mut paint = Paint::default();
                paint.set_color_rgba8(color.0, color.1, color.2, color.3);
                paint.anti_alias = false;
                pix.fill_rect(r, &paint, Transform::identity(), None);
            }
            Cmd::Line {
                x1,
                y1,
                x2,
                y2,
                width,
                color,
                dash,
            } => {
                let mut paint = Paint::default();
                paint.set_color_rgba8(color.0, color.1, color.2, color.3);
                paint.anti_alias = false;
                let mut stroke = Stroke {
                    width: *width,
                    line_cap: LineCap::Butt,
                    ..Stroke::default()
                };
                stroke.dash = dash_pattern(*dash, *width);
                let draw = |pix: &mut Pixmap, off: f32| {
                    let mut pb = PathBuilder::new();
                    // Nudge onto a half-pixel so 1px rules stay crisp.
                    let snap = |v: f32| v.floor() + 0.5;
                    if (y1 - y2).abs() < 0.01 {
                        pb.move_to(*x1, snap(*y1) + off);
                        pb.line_to(*x2, snap(*y2) + off);
                    } else if (x1 - x2).abs() < 0.01 {
                        pb.move_to(snap(*x1) + off, *y1);
                        pb.line_to(snap(*x2) + off, *y2);
                    } else {
                        pb.move_to(*x1, *y1);
                        pb.line_to(*x2, *y2);
                    }
                    if let Some(p) = pb.finish() {
                        pix.stroke_path(&p, &paint, &stroke, Transform::identity(), None);
                    }
                };
                draw(&mut pix, 0.0);
                if matches!(dash, Dash::Double) {
                    draw(&mut pix, 2.0 * *width);
                }
            }
            Cmd::Text {
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
            } => {
                let lines: Vec<String> = if *wrap {
                    wrap_lines(text, rect.w, *size_px, *bold, fonts)
                } else {
                    vec![text.replace('\n', " ")]
                };
                let line_h = size_px * 1.2;
                let block_h = line_h * lines.len() as f32;
                let mut y = match valign {
                    VAlign::Top => rect.y,
                    VAlign::Center => rect.y + (rect.h - block_h) / 2.0,
                    VAlign::Bottom => rect.y + rect.h - block_h,
                };
                // A full-pixmap Mask per text run is the dominant raster cost,
                // so only pay for it when the run can actually escape its clip.
                let widest = lines
                    .iter()
                    .map(|l| fonts.measure(l, *size_px, *bold))
                    .fold(0.0f32, f32::max);
                let needs_clip = widest > clip.w + 0.5
                    || block_h > clip.h + 0.5
                    || y < clip.y - 0.5;
                let mask = if needs_clip { clip_mask(&pix, clip) } else { None };
                for line in &lines {
                    let adv = fonts.measure(line, *size_px, *bold);
                    let x = match halign {
                        HAlign::Left => rect.x,
                        HAlign::Center => rect.x + (rect.w - adv) / 2.0,
                        HAlign::Right => rect.x + rect.w - adv,
                    };
                    let baseline = y + fonts.ascent_px(*size_px, *bold);
                    fonts.draw(
                        &mut pix,
                        line,
                        x,
                        baseline,
                        *size_px,
                        *bold,
                        *italic,
                        *color,
                        mask.as_ref(),
                    );
                    if *underline || *strike {
                        let mut paint = Paint::default();
                        paint.set_color_rgba8(color.0, color.1, color.2, color.3);
                        let yy = if *underline {
                            baseline + size_px * 0.12
                        } else {
                            baseline - size_px * 0.28
                        };
                        if let Some(r) =
                            SkRect::from_xywh(x, yy, adv.max(0.01), (size_px * 0.07).max(1.0))
                        {
                            pix.fill_rect(r, &paint, Transform::identity(), None);
                        }
                        if *underline && *strike {
                            let yy2 = baseline - size_px * 0.28;
                            if let Some(r) =
                                SkRect::from_xywh(x, yy2, adv.max(0.01), (size_px * 0.07).max(1.0))
                            {
                                pix.fill_rect(r, &paint, Transform::identity(), None);
                            }
                        }
                    }
                    y += line_h;
                }
            }
        }
    }
    Ok(pix)
}

fn clip_mask(pix: &Pixmap, clip: &crate::scene::Rect) -> Option<tiny_skia::Mask> {
    let mut mask = tiny_skia::Mask::new(pix.width(), pix.height())?;
    let r = SkRect::from_xywh(clip.x, clip.y, clip.w.max(0.01), clip.h.max(0.01))?;
    let path = PathBuilder::from_rect(r);
    mask.fill_path(&path, FillRule::Winding, false, Transform::identity());
    Some(mask)
}

fn wrap_lines(text: &str, max_w: f32, size_px: f32, bold: bool, fonts: &Fonts) -> Vec<String> {
    let mut out = Vec::new();
    for para in text.split('\n') {
        let mut cur = String::new();
        for word in para.split_whitespace() {
            let cand = if cur.is_empty() {
                word.to_string()
            } else {
                format!("{cur} {word}")
            };
            if fonts.measure(&cand, size_px, bold) <= max_w || cur.is_empty() {
                cur = cand;
            } else {
                out.push(std::mem::take(&mut cur));
                cur = word.to_string();
            }
        }
        out.push(cur);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// PNG encode at a chosen zlib level. tiny-skia's own `encode_png` uses the
/// png crate's default compression, which turned out to dominate spike-D's
/// total time, so 5004 needs an explicit knob here.
pub fn to_png_level(pix: &Pixmap, level: png::Compression) -> Result<Vec<u8>, RenderError> {
    let mut buf = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut buf, pix.width(), pix.height());
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_compression(level);
        let mut w = enc
            .write_header()
            .map_err(|e| RenderError::Encode(e.to_string()))?;
        w.write_image_data(pix.data())
            .map_err(|e| RenderError::Encode(e.to_string()))?;
    }
    Ok(buf)
}

pub fn to_png(pix: &Pixmap) -> Result<Vec<u8>, RenderError> {
    pix.encode_png().map_err(|e| RenderError::Encode(e.to_string()))
}

/// Best-effort crop of uniform white margin, mirroring
/// `crop_png_in_place` in crates/agent-spreadsheet/src/recalc/screenshot.rs.
pub fn crop_white(pix: &Pixmap) -> (u32, u32, u32, u32) {
    let (w, h) = (pix.width(), pix.height());
    let data = pix.data();
    let is_content = |x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        let (r, g, b) = (data[i], data[i + 1], data[i + 2]);
        r < 250 || g < 250 || b < 250
    };
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
    for y in 0..h {
        for x in 0..w {
            if is_content(x, y) {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if x0 > x1 || y0 > y1 {
        (0, 0, w, h)
    } else {
        (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    }
}

pub const _RGBA_HINT: Rgba = Rgba::BLACK;
