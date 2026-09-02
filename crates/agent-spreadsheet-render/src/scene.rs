//! The scene model: a flat, ordered command list in absolute device pixels.
//!
//! Pinned by the 5003 bake-off. Three variants only — `FillRect`, `Line`
//! (axis-aligned) and `Text`. No `Path`, no `Group`: nothing in the 5004 scope
//! needs them, and `Path` is where chart rendering would start.
//!
//! This module has no rendering imports. Warnings are collected here during
//! extraction so the report is a by-product of extraction, not a second pass.

use serde::{Deserialize, Serialize};

/// Paint order the rasterizer relies on. Extraction emits commands in exactly
/// this order: background, fills, gridlines, borders, text, headings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }
}

/// Packed RGBA. Deliberately not a `#rrggbb` string: BetterOffice re-parses
/// colour strings inside its raster loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rgba(pub u8, pub u8, pub u8, pub u8);

impl Rgba {
    pub const BLACK: Rgba = Rgba(0, 0, 0, 255);
    pub const WHITE: Rgba = Rgba(255, 255, 255, 255);
    pub const GRIDLINE: Rgba = Rgba(0xd4, 0xd4, 0xd4, 255);
    pub const HEADING_BG: Rgba = Rgba(0xf0, 0xf0, 0xf0, 255);
    pub const HEADING_FG: Rgba = Rgba(0x44, 0x44, 0x44, 255);
    pub const HEADING_RULE: Rgba = Rgba(0x9a, 0x9a, 0x9a, 255);
    /// `[Red]` in a number-format section.
    pub const FORMAT_RED: Rgba = Rgba(0xff, 0x00, 0x00, 255);

    /// Parse `RRGGBB` or `AARRGGBB`, with or without a leading `#`.
    pub fn from_argb_hex(s: &str) -> Option<Rgba> {
        let s = s.trim().trim_start_matches('#');
        match s.len() {
            6 => Some(Rgba(hex_at(s, 0)?, hex_at(s, 2)?, hex_at(s, 4)?, 255)),
            8 => Some(Rgba(hex_at(s, 2)?, hex_at(s, 4)?, hex_at(s, 6)?, 255)),
            _ => None,
        }
    }
}

fn hex_at(s: &str, i: usize) -> Option<u8> {
    u8::from_str_radix(s.get(i..i + 2)?, 16).ok()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HAlign {
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VAlign {
    Top,
    Center,
    Bottom,
}

/// Dash rhythm in multiples of the stroke width. `Solid` is undashed;
/// `Double` is two solid bands drawn one stroke-gap apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dash {
    Solid,
    Dotted,
    Dashed,
    DashDot,
    DashDotDot,
    Double,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Cmd {
    FillRect {
        rect: Rect,
        color: Rgba,
    },
    /// Axis-aligned rule. Gridlines and borders both land here. Borders are
    /// bands anchored to the grid boundary (office2pdf's model), not strokes
    /// centred on it, so `width` is the band width and the band grows inward.
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
        /// The text box (padded cell interior), not the glyph origin.
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
        /// Clip rect. Wider than `rect` for text spilling across empty
        /// neighbours to the right.
        clip: Rect,
    },
}

/// Closed set of things this renderer does not reproduce faithfully. Nothing
/// unsupported disappears silently: every omission raises one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

impl Warning {
    /// Stable snake_case name, matching the serde representation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Warning::ConditionalFormatOmitted => "conditional_format_omitted",
            Warning::ChartOmitted => "chart_omitted",
            Warning::ImageOmitted => "image_omitted",
            Warning::FontSubstituted => "font_substituted",
            Warning::RichTextFlattened => "rich_text_flattened",
            Warning::NumberFormatApproximated => "number_format_approximated",
            Warning::FormulasUnevaluated => "formulas_unevaluated",
            Warning::TextRotationOmitted => "text_rotation_omitted",
            Warning::PatternFillApproximated => "pattern_fill_approximated",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub width: f32,
    pub height: f32,
    pub cmds: Vec<Cmd>,
    /// Sorted and deduplicated, so goldens are order-independent.
    pub warnings: Vec<Warning>,
}

impl Scene {
    /// Raise a warning once per scene. Sorted insertion keeps the report
    /// deterministic regardless of the order cells were visited in.
    pub fn warn(&mut self, warning: Warning) {
        if let Err(index) = self.warnings.binary_search(&warning) {
            self.warnings.insert(index, warning);
        }
    }
}
