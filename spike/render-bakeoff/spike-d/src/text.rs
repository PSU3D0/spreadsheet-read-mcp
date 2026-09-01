//! Text stack behind a feature switch: rustybuzz (full shaping) vs ab_glyph
//! (advance + kerning only). Both use the same embedded Carlito faces, which
//! are themselves behind a `font-full` / `font-subset` switch.
//!
//! The public surface is identical for both so the raster path is shared:
//! `measure` returns an advance in px, `outline` yields filled paths.

use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Transform};

use crate::scene::Rgba;

#[cfg(all(feature = "font-full", not(feature = "font-subset")))]
pub static REGULAR: &[u8] = include_bytes!("../assets/Carlito-Regular.ttf");
#[cfg(all(feature = "font-full", not(feature = "font-subset")))]
pub static BOLD: &[u8] = include_bytes!("../assets/Carlito-Bold.ttf");

#[cfg(feature = "font-subset")]
pub static REGULAR: &[u8] = include_bytes!("../assets/Carlito-Regular-subset.ttf");
#[cfg(feature = "font-subset")]
pub static BOLD: &[u8] = include_bytes!("../assets/Carlito-Bold-subset.ttf");

/// Skew applied for synthetic italic (BetterOffice uses 0.21).
pub const ITALIC_SHEAR: f32 = 0.21;

// ---------------------------------------------------------------------------
// rustybuzz
// ---------------------------------------------------------------------------
#[cfg(feature = "text-rustybuzz")]
mod imp {
    use super::*;
    use rustybuzz::{Face, UnicodeBuffer, ttf_parser};

    pub struct Fonts {
        regular: Face<'static>,
        bold: Face<'static>,
    }

    impl Fonts {
        pub fn new() -> Self {
            Self {
                regular: Face::from_slice(REGULAR, 0).expect("carlito regular"),
                bold: Face::from_slice(BOLD, 0).expect("carlito bold"),
            }
        }
        fn face(&self, bold: bool) -> &Face<'static> {
            if bold { &self.bold } else { &self.regular }
        }
        pub fn units_per_em(&self, bold: bool) -> f32 {
            self.face(bold).units_per_em() as f32
        }
        pub fn ascent_px(&self, size_px: f32, bold: bool) -> f32 {
            let f = self.face(bold);
            f32::from(f.ascender()) / f.units_per_em() as f32 * size_px
        }

        pub fn measure(&self, text: &str, size_px: f32, bold: bool) -> f32 {
            let face = self.face(bold);
            let mut buf = UnicodeBuffer::new();
            buf.push_str(text);
            let g = rustybuzz::shape(face, &[], buf);
            let upem = face.units_per_em() as f32;
            let adv: i32 = g.glyph_positions().iter().map(|p| p.x_advance).sum();
            adv as f32 / upem * size_px
        }

        /// Shape and paint. Returns the advance actually drawn.
        pub fn draw(
            &self,
            pix: &mut Pixmap,
            text: &str,
            x: f32,
            baseline: f32,
            size_px: f32,
            bold: bool,
            italic: bool,
            color: Rgba,
            clip: Option<&tiny_skia::Mask>,
        ) -> f32 {
            let face = self.face(bold);
            let upem = face.units_per_em() as f32;
            let scale = size_px / upem;
            let mut buf = UnicodeBuffer::new();
            buf.push_str(text);
            let g = rustybuzz::shape(face, &[], buf);

            let mut paint = Paint::default();
            paint.set_color_rgba8(color.0, color.1, color.2, color.3);
            paint.anti_alias = true;

            let mut pen = 0.0f32;
            for (info, pos) in g.glyph_infos().iter().zip(g.glyph_positions()) {
                let gid = ttf_parser::GlyphId(info.glyph_id as u16);
                let mut b = OutlineSink {
                    pb: PathBuilder::new(),
                };
                if face.outline_glyph(gid, &mut b).is_some()
                    && let Some(path) = b.pb.finish()
                {
                    let gx = x + (pen + pos.x_offset as f32) * scale;
                    let gy = baseline - (pos.y_offset as f32) * scale;
                    // y flip: font units are y-up, pixmap is y-down.
                    let shear = if italic { -ITALIC_SHEAR } else { 0.0 };
                    let ts = Transform::from_row(scale, 0.0, shear * scale, -scale, gx, gy);
                    pix.fill_path(&path, &paint, FillRule::Winding, ts, clip);
                }
                pen += pos.x_advance as f32;
            }
            pen * scale
        }
    }

    struct OutlineSink {
        pb: PathBuilder,
    }
    impl ttf_parser::OutlineBuilder for OutlineSink {
        fn move_to(&mut self, x: f32, y: f32) {
            self.pb.move_to(x, y);
        }
        fn line_to(&mut self, x: f32, y: f32) {
            self.pb.line_to(x, y);
        }
        fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
            self.pb.quad_to(x1, y1, x, y);
        }
        fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
            self.pb.cubic_to(x1, y1, x2, y2, x, y);
        }
        fn close(&mut self) {
            self.pb.close();
        }
    }
}

// ---------------------------------------------------------------------------
// ab_glyph (advance + kerning, no complex shaping)
// ---------------------------------------------------------------------------
#[cfg(all(feature = "text-abglyph", not(feature = "text-rustybuzz")))]
mod imp {
    use super::*;
    use ab_glyph::{Font, FontRef, OutlineCurve, ScaleFont};

    pub struct Fonts {
        regular: FontRef<'static>,
        bold: FontRef<'static>,
    }

    impl Fonts {
        pub fn new() -> Self {
            Self {
                regular: FontRef::try_from_slice(REGULAR).expect("carlito regular"),
                bold: FontRef::try_from_slice(BOLD).expect("carlito bold"),
            }
        }
        fn face(&self, bold: bool) -> &FontRef<'static> {
            if bold { &self.bold } else { &self.regular }
        }
        pub fn units_per_em(&self, bold: bool) -> f32 {
            self.face(bold).units_per_em().unwrap_or(1000.0)
        }
        pub fn ascent_px(&self, size_px: f32, bold: bool) -> f32 {
            self.face(bold).as_scaled(size_px).ascent()
        }

        pub fn measure(&self, text: &str, size_px: f32, bold: bool) -> f32 {
            let f = self.face(bold).as_scaled(size_px);
            let mut w = 0.0f32;
            let mut prev = None;
            for c in text.chars() {
                let g = f.glyph_id(c);
                if let Some(p) = prev {
                    w += f.kern(p, g);
                }
                w += f.h_advance(g);
                prev = Some(g);
            }
            w
        }

        pub fn draw(
            &self,
            pix: &mut Pixmap,
            text: &str,
            x: f32,
            baseline: f32,
            size_px: f32,
            bold: bool,
            italic: bool,
            color: Rgba,
            clip: Option<&tiny_skia::Mask>,
        ) -> f32 {
            let font = self.face(bold);
            let f = font.as_scaled(size_px);
            let upem = font.units_per_em().unwrap_or(1000.0);
            let scale = size_px / upem;

            let mut paint = Paint::default();
            paint.set_color_rgba8(color.0, color.1, color.2, color.3);
            paint.anti_alias = true;

            let mut pen = 0.0f32;
            let mut prev = None;
            for c in text.chars() {
                let gid = f.glyph_id(c);
                if let Some(p) = prev {
                    pen += f.kern(p, gid);
                }
                if let Some(curves) = font.outline(gid) {
                    let mut pb = PathBuilder::new();
                    for seg in &curves.curves {
                        match seg {
                            OutlineCurve::Line(a, b) => {
                                pb.move_to(a.x, a.y);
                                pb.line_to(b.x, b.y);
                            }
                            OutlineCurve::Quad(a, b, c) => {
                                pb.move_to(a.x, a.y);
                                pb.quad_to(b.x, b.y, c.x, c.y);
                            }
                            OutlineCurve::Cubic(a, b, c, d) => {
                                pb.move_to(a.x, a.y);
                                pb.cubic_to(b.x, b.y, c.x, c.y, d.x, d.y);
                            }
                        }
                    }
                    if let Some(path) = pb.finish() {
                        // ab_glyph outlines are already in font units (y-up).
                        let shear = if italic { -ITALIC_SHEAR } else { 0.0 };
                        let ts = Transform::from_row(
                            scale,
                            0.0,
                            shear * scale,
                            -scale,
                            x + pen,
                            baseline,
                        );
                        pix.fill_path(&path, &paint, FillRule::Winding, ts, clip);
                    }
                }
                pen += f.h_advance(gid);
                prev = Some(gid);
            }
            let _ = upem;
            pen
        }
    }
}

pub use imp::Fonts;

/// Which stack was compiled in. Recorded in the harness output.
pub const STACK: &str = if cfg!(feature = "text-rustybuzz") {
    "rustybuzz"
} else {
    "ab_glyph"
};

pub const FONT_STRATEGY: &str = if cfg!(feature = "font-subset") {
    "carlito-subset"
} else {
    "carlito-full"
};

pub fn embedded_font_bytes() -> usize {
    REGULAR.len() + BOLD.len()
}
