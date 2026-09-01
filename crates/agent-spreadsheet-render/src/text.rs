//! The text stack, behind a `Fonts` facade.
//!
//! ab_glyph, pinned by the 5003 bake-off: 434,186 fewer wasm bytes than
//! rustybuzz for no visible quality difference on Latin text. The facade
//! (`measure`, `ascent_px`, `draw`, `has_glyph`) is the seam a future shaping
//! stack swaps behind.
//!
//! Two things the bake-off found the hard way and this module must not lose:
//!
//! * `Font::outline` hands back a flat segment list, not contours. They have
//!   to be stitched by continuity before the path goes to tiny-skia, or the
//!   winding fill collapses to hairline fragments. See `stitch_contours`.
//! * The subset faces drop glyphs outside their coverage. They must never be
//!   dropped silently: `has_glyph` is false for them, extraction raises
//!   `font_substituted`, and `draw` paints a visible `.notdef` box.

use ab_glyph::{Font, FontRef, OutlineCurve, ScaleFont};
use tiny_skia::{FillRule, Mask, Paint, PathBuilder, Pixmap, Transform};

use crate::scene::Rgba;

/// Subset Carlito, built with the `pyftsubset` command recorded in
/// `assets/README.md`. Carlito is metrically compatible with Calibri, which is
/// what the column metric model assumes.
pub static REGULAR: &[u8] = include_bytes!("../assets/Carlito-Regular-subset.ttf");
pub static BOLD: &[u8] = include_bytes!("../assets/Carlito-Bold-subset.ttf");

/// Shear applied for synthetic italic. There is no real italic face; both
/// reference renderers synthesise it the same way.
pub const ITALIC_SHEAR: f32 = 0.21;

/// `.notdef` box geometry, in em, for a codepoint the subset does not carry.
const NOTDEF_ADVANCE_EM: f32 = 0.55;
const NOTDEF_HEIGHT_EM: f32 = 0.66;
const NOTDEF_INSET_EM: f32 = 0.06;
const NOTDEF_STROKE_EM: f32 = 0.05;

pub struct Fonts {
    regular: FontRef<'static>,
    bold: FontRef<'static>,
}

impl Default for Fonts {
    fn default() -> Self {
        Self::new()
    }
}

impl Fonts {
    pub fn new() -> Self {
        Self {
            // The faces are compiled in and are checked by a unit test, so a
            // parse failure here is a build-time defect, not a runtime one.
            regular: FontRef::try_from_slice(REGULAR).expect("embedded Carlito Regular is valid"),
            bold: FontRef::try_from_slice(BOLD).expect("embedded Carlito Bold is valid"),
        }
    }

    fn face(&self, bold: bool) -> &FontRef<'static> {
        if bold { &self.bold } else { &self.regular }
    }

    /// Whether the embedded subset carries this codepoint. Glyph id 0 is
    /// `.notdef` in every TrueType face.
    pub fn has_glyph(&self, c: char) -> bool {
        self.face(false).glyph_id(c).0 != 0 || c == '\u{0}'
    }

    /// True when any codepoint of `text` falls outside the subset.
    pub fn has_missing_glyphs(&self, text: &str) -> bool {
        text.chars()
            .any(|c| !c.is_control() && !c.is_whitespace() && !self.has_glyph(c))
    }

    pub fn ascent_px(&self, size_px: f32, bold: bool) -> f32 {
        self.face(bold).as_scaled(size_px).ascent()
    }

    /// Distance from ascent top to descent bottom, in px. Used by row
    /// auto-fit.
    pub fn line_extent_px(&self, size_px: f32, bold: bool) -> f32 {
        let scaled = self.face(bold).as_scaled(size_px);
        scaled.ascent() - scaled.descent()
    }

    /// Advance width in px, kerning included. A codepoint outside the subset
    /// measures as the `.notdef` box it will be drawn as.
    pub fn measure(&self, text: &str, size_px: f32, bold: bool) -> f32 {
        let scaled = self.face(bold).as_scaled(size_px);
        let mut width = 0.0f32;
        let mut previous = None;
        for c in text.chars() {
            let gid = scaled.glyph_id(c);
            if gid.0 == 0 && c != '\u{0}' {
                width += NOTDEF_ADVANCE_EM * size_px;
                previous = None;
                continue;
            }
            if let Some(p) = previous {
                width += scaled.kern(p, gid);
            }
            width += scaled.h_advance(gid);
            previous = Some(gid);
        }
        width
    }

    /// Paint `text` with its left edge at `x` and its baseline at `baseline`.
    /// Returns the advance actually drawn.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        pixmap: &mut Pixmap,
        text: &str,
        x: f32,
        baseline: f32,
        size_px: f32,
        bold: bool,
        italic: bool,
        color: Rgba,
        clip: Option<&Mask>,
    ) -> f32 {
        let font = self.face(bold);
        let scaled = font.as_scaled(size_px);
        let units_per_em = font.units_per_em().unwrap_or(1000.0);
        let scale = size_px / units_per_em;

        let mut paint = Paint::default();
        paint.set_color_rgba8(color.0, color.1, color.2, color.3);
        paint.anti_alias = true;

        let mut pen = 0.0f32;
        let mut previous = None;
        for c in text.chars() {
            let gid = scaled.glyph_id(c);
            if gid.0 == 0 && c != '\u{0}' {
                draw_notdef(pixmap, &paint, x + pen, baseline, size_px, clip);
                pen += NOTDEF_ADVANCE_EM * size_px;
                previous = None;
                continue;
            }
            if let Some(p) = previous {
                pen += scaled.kern(p, gid);
            }
            if let Some(curves) = font.outline(gid)
                && let Some(path) = stitch_contours(&curves.curves)
            {
                // ab_glyph outlines are in font units and y-up; the pixmap is
                // y-down, hence the negative y scale.
                // Glyph coordinates are y-up, so a positive x-skew leans the
                // top of the glyph to the right, which is the direction a
                // real italic leans.
                let shear = if italic { ITALIC_SHEAR } else { 0.0 };
                let transform =
                    Transform::from_row(scale, 0.0, shear * scale, -scale, x + pen, baseline);
                pixmap.fill_path(&path, &paint, FillRule::Winding, transform, clip);
            }
            pen += scaled.h_advance(gid);
            previous = Some(gid);
        }
        pen
    }
}

/// Stitch ab_glyph's flat segment list back into closed contours.
///
/// ab_glyph returns every segment of every contour in one list with no contour
/// boundaries. Feeding that straight to a `PathBuilder` produces one open
/// subpath and the winding fill collapses to hairline fragments — the defect
/// that invalidated the bake-off's first ab_glyph measurement run. A new
/// contour starts wherever a segment's start point is not the previous
/// segment's end point.
pub fn stitch_contours(curves: &[OutlineCurve]) -> Option<tiny_skia::Path> {
    let mut builder = PathBuilder::new();
    let mut cursor: Option<(f32, f32)> = None;
    let mut open = false;
    for segment in curves {
        let start = match segment {
            OutlineCurve::Line(a, _)
            | OutlineCurve::Quad(a, _, _)
            | OutlineCurve::Cubic(a, _, _, _) => (a.x, a.y),
        };
        let continues =
            cursor.is_some_and(|(x, y)| (x - start.0).abs() < 1e-4 && (y - start.1).abs() < 1e-4);
        if !continues {
            if open {
                builder.close();
            }
            builder.move_to(start.0, start.1);
            open = true;
        }
        cursor = Some(match segment {
            OutlineCurve::Line(_, b) => {
                builder.line_to(b.x, b.y);
                (b.x, b.y)
            }
            OutlineCurve::Quad(_, b, c) => {
                builder.quad_to(b.x, b.y, c.x, c.y);
                (c.x, c.y)
            }
            OutlineCurve::Cubic(_, b, c, d) => {
                builder.cubic_to(b.x, b.y, c.x, c.y, d.x, d.y);
                (d.x, d.y)
            }
        });
    }
    if open {
        builder.close();
    }
    builder.finish()
}

/// Count the closed contours a glyph outline stitches into. Used by the
/// two-contour golden test, which is the guard against the stitching
/// regression described above.
pub fn contour_count(curves: &[OutlineCurve]) -> usize {
    let mut count = 0usize;
    let mut cursor: Option<(f32, f32)> = None;
    for segment in curves {
        let (start, end) = match segment {
            OutlineCurve::Line(a, b) => ((a.x, a.y), (b.x, b.y)),
            OutlineCurve::Quad(a, _, c) => ((a.x, a.y), (c.x, c.y)),
            OutlineCurve::Cubic(a, _, _, d) => ((a.x, a.y), (d.x, d.y)),
        };
        let continues =
            cursor.is_some_and(|(x, y)| (x - start.0).abs() < 1e-4 && (y - start.1).abs() < 1e-4);
        if !continues {
            count += 1;
        }
        cursor = Some(end);
    }
    count
}

/// A hollow box for a codepoint the subset does not carry. Mandatory: the
/// bake-off measured the subset silently dropping the Greek row of the unicode
/// fixture, which is the part that must not ship.
fn draw_notdef(
    pixmap: &mut Pixmap,
    paint: &Paint<'_>,
    x: f32,
    baseline: f32,
    size_px: f32,
    clip: Option<&Mask>,
) {
    let inset = NOTDEF_INSET_EM * size_px;
    let stroke = (NOTDEF_STROKE_EM * size_px).max(1.0);
    let width = NOTDEF_ADVANCE_EM * size_px - 2.0 * inset;
    let height = NOTDEF_HEIGHT_EM * size_px;
    if width <= 2.0 * stroke || height <= 2.0 * stroke {
        return;
    }
    let left = x + inset;
    let top = baseline - height;
    let bars = [
        (left, top, width, stroke),
        (left, top + height - stroke, width, stroke),
        (left, top, stroke, height),
        (left + width - stroke, top, stroke, height),
    ];
    for (bx, by, bw, bh) in bars {
        if let Some(rect) = tiny_skia::Rect::from_xywh(bx, by, bw, bh) {
            let path = PathBuilder::from_rect(rect);
            pixmap.fill_path(&path, paint, FillRule::Winding, Transform::identity(), clip);
        }
    }
}

/// Total embedded font bytes. Reported by the crate README's size claim and
/// asserted by a unit test so the assets cannot silently grow.
pub fn embedded_font_bytes() -> usize {
    REGULAR.len() + BOLD.len()
}
