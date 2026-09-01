//! Spike D scene model.
//!
//! Deliberately a flat, ordered command list with absolute pixel coordinates,
//! matching the shape both BetterOffice (`DrawCmd`) and readany-render (`Item`)
//! converged on. Painter's algorithm: fills, then borders/gridlines, then text.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgba(pub u8, pub u8, pub u8, pub u8);

impl Rgba {
    pub const BLACK: Rgba = Rgba(0, 0, 0, 255);
    pub const WHITE: Rgba = Rgba(255, 255, 255, 255);
    pub const GRIDLINE: Rgba = Rgba(0xd4, 0xd4, 0xd4, 255);
    pub const HEADING_BG: Rgba = Rgba(0xf0, 0xf0, 0xf0, 255);
    pub const HEADING_FG: Rgba = Rgba(0x44, 0x44, 0x44, 255);

    pub fn from_argb_hex(s: &str) -> Option<Rgba> {
        let s = s.trim().trim_start_matches('#');
        match s.len() {
            6 => Some(Rgba(hx(s, 0)?, hx(s, 2)?, hx(s, 4)?, 255)),
            8 => Some(Rgba(hx(s, 2)?, hx(s, 4)?, hx(s, 6)?, 255)),
            _ => None,
        }
    }
}

fn hx(s: &str, i: usize) -> Option<u8> {
    u8::from_str_radix(s.get(i..i + 2)?, 16).ok()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HAlign {
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VAlign {
    Top,
    Center,
    Bottom,
}

/// Dash rhythm in multiples of the stroke width. `None` is solid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dash {
    Solid,
    Dotted,
    Dashed,
    DashDot,
    DashDotDot,
    Double,
}

#[derive(Debug, Clone)]
pub enum Cmd {
    FillRect {
        rect: Rect,
        color: Rgba,
    },
    /// Axis-aligned rule. Borders and gridlines both land here.
    Line {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        width: f32,
        color: Rgba,
        dash: Dash,
    },
    Text {
        /// Left edge of the text box (not the glyph origin).
        rect: Rect,
        text: String,
        size_px: f32,
        color: Rgba,
        bold: bool,
        italic: bool,
        underline: bool,
        strike: bool,
        halign: HAlign,
        valign: VAlign,
        wrap: bool,
        /// Clip rect; for spilling text this is wider than `rect`.
        clip: Rect,
    },
}

#[derive(Debug, Clone, Default)]
pub struct Scene {
    pub width: f32,
    pub height: f32,
    pub cmds: Vec<Cmd>,
    pub warnings: Vec<Warning>,
}

/// Closed warning enum, mirroring the one ticket 5004 pins for the report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Warning {
    ConditionalFormatOmitted,
    ChartOmitted,
    ImageOmitted,
    FontSubstituted,
    RichTextFlattened,
    NumberFormatApproximated,
    FormulasUnevaluated,
    TextRotationOmitted,
    PatternFillApproximated,
}

impl Scene {
    pub fn warn(&mut self, w: Warning) {
        if !self.warnings.contains(&w) {
            self.warnings.push(w);
        }
    }
}
