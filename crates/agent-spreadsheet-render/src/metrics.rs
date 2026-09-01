//! Geometry and style metrics ported verbatim from office2pdf 0.6.7
//! (`src/parser/xlsx_cells.rs`, `src/parser/xlsx_style.rs`).
//!
//! These encode 17 one-factor native-Excel probes we cannot re-derive, and the
//! hardcoded digit-advance table is exactly what makes the geometry identical
//! under wasm and on font-less machines.
//!
//! Two things were deliberately NOT ported, per the 5003 bake-off:
//!
//! * `native_excel_pdf_row_height` (the `* 0.92` factor). It compensates for
//!   Excel's compacted *print* grid; we raster for screen at 96/72.
//! * `resolve_style_color`. It delegates to umya's `get_argb_with_theme`,
//!   which short-circuits on indexed colours and silently loses the tint.
//!   Colour resolution here is deliberately explicit instead
//!   (see [`crate::color`]).

use crate::scene::{Dash, Rgba};

/// Device pixels per point at 96 dpi.
pub const PX_PER_PT: f32 = 96.0 / 72.0;

/// office2pdf `EXCEL_DEFAULT_ROW_HEIGHT_PT` — Excel's fallback row height when
/// the sheet declares none (Calibri 11).
pub const EXCEL_DEFAULT_ROW_HEIGHT_PT: f64 = 15.0;

/// office2pdf `XLSX_CELL_PADDING`, in points.
pub const PAD_TOP_PT: f64 = 1.0;
pub const PAD_RIGHT_PT: f64 = 3.0;
pub const PAD_BOTTOM_PT: f64 = 1.5;
pub const PAD_LEFT_PT: f64 = 3.0;

/// office2pdf `CALIBRI_DIGIT_ADVANCE_EM` — the max digit advance of Calibri
/// (and of metrically identical Carlito), Excel's default Normal font, and the
/// last-resort metric for a family the reference table does not know.
pub const CALIBRI_DIGIT_ADVANCE_EM: f64 = 0.506836;

/// office2pdf `reference_digit_advance_em`: maximum digit advances (em over
/// U+0030..=U+0039) of the faces Excel itself ships, read from their `hmtx`
/// tables by the issue #621 probe tooling.
///
/// These outrank live font resolution on purpose. A converting machine may
/// substitute a digit-incompatible face (Calibri -> Liberation Sans advances
/// 0.556em against Calibri's 0.5068), which would shift column geometry per
/// machine, while Excel's own print metric always comes from the face Excel
/// resolves.
pub fn reference_digit_advance_em(family: &str) -> Option<f64> {
    match family.to_ascii_lowercase().as_str() {
        "calibri" | "carlito" => Some(CALIBRI_DIGIT_ADVANCE_EM),
        "arial" | "helvetica" | "liberation sans" => Some(0.556152),
        "verdana" => Some(0.635742),
        "courier new" => Some(0.600098),
        "times new roman" => Some(0.500000),
        "malgun gothic" | "\u{b9d1}\u{c740} \u{ace0}\u{b515}" => Some(0.550781),
        _ => None,
    }
}

/// office2pdf `round_half_up_pt`. Excel's column metric rounds half UP, not
/// half-even: the Times New Roman 13 probe lands exactly on 6.500pt and prints
/// a 7pt unit. Inputs are non-negative.
pub fn round_half_up_pt(value: f64) -> f64 {
    (value + 0.5).floor()
}

/// office2pdf `column_unit_pt`: the points Excel allots to one column
/// character unit for a Normal font — `round_half_up(max digit advance x
/// size)`, an INTEGER point count.
///
/// office2pdf falls back to live font resolution before the Calibri constant;
/// we have no font enumeration (and want none, for wasm), so an unknown family
/// goes straight to Calibri's advance.
pub fn column_unit_pt(family: &str, size_pt: f64) -> f64 {
    let digit_advance_em = reference_digit_advance_em(family).unwrap_or(CALIBRI_DIGIT_ADVANCE_EM);
    round_half_up_pt(digit_advance_em * size_pt)
}

/// office2pdf `column_width_to_pt`. An OOXML width is in character units
/// relative to the Normal font's column unit and already carries Excel's cell
/// padding adjustment, so layout must not add padding again.
pub fn column_width_to_pt(char_width: f64, column_unit_pt: f64) -> f64 {
    round_half_up_pt(char_width * column_unit_pt)
}

/// office2pdf `default_column_width_pt`. With no declared `defaultColWidth`,
/// Excel prints `baseColWidth x unit + 5` points — not 8.43 character units —
/// where `baseColWidth` defaults to 8 when `sheetFormatPr` omits it too.
pub fn default_column_width_pt(
    declared_width_chars: Option<f64>,
    base_col_width_chars: Option<u32>,
    column_unit_pt: f64,
) -> f64 {
    match declared_width_chars {
        Some(width_chars) => round_half_up_pt(width_chars * column_unit_pt),
        None => f64::from(base_col_width_chars.unwrap_or(8)) * column_unit_pt + 5.0,
    }
}

/// office2pdf `pattern_ink_coverage`: the fraction of a cell a spreadsheetML
/// pattern's foreground covers.
///
/// Measured on a one-factor probe: one cell per `patternType`, `fgColor`
/// `FF000000` over `bgColor` `FFFFFFFF`, exported by LibreOffice 24.2 and read
/// back as the mean grey of the swatch interior at 300 DPI. Two values do not
/// follow from the names: `darkGrid` covers a half, not three quarters, and
/// `lightTrellis` covers three eighths rather than `lightGrid`'s seven
/// sixteenths.
pub fn pattern_ink_coverage(pattern: &umya_spreadsheet::PatternValues) -> f64 {
    use umya_spreadsheet::PatternValues::*;
    match pattern {
        Solid => 1.0,
        DarkGray | DarkTrellis => 0.75,
        MediumGray | DarkHorizontal | DarkVertical | DarkDown | DarkUp | DarkGrid => 0.5,
        LightGrid => 0.4375,
        LightTrellis => 0.375,
        LightGray | LightHorizontal | LightVertical | LightDown | LightUp => 0.25,
        Gray125 => 0.125,
        Gray0625 => 0.0625,
        None => 0.0,
    }
}

/// office2pdf `blend_color`: composite `foreground` over `background` at
/// `coverage`.
pub fn blend_color(background: Rgba, foreground: Rgba, coverage: f64) -> Rgba {
    let mix = |below: u8, above: u8| -> u8 {
        (f64::from(below) * (1.0 - coverage) + f64::from(above) * coverage).round() as u8
    };
    Rgba(
        mix(background.0, foreground.0),
        mix(background.1, foreground.1),
        mix(background.2, foreground.2),
        255,
    )
}

/// office2pdf `border_style_to_width`: the printed band width in points.
///
/// Measured on a native Excel 16.111 one-factor probe: every border prints as
/// a filled band anchored to the grid boundary — `thin` and `hair` 1pt,
/// `medium` 2pt, `thick` 3pt. A `double` rule is two 1pt bands with a 1pt gap,
/// so each band carries the `thin` weight rather than the combined one.
pub fn border_style_to_width(style: &str) -> Option<f32> {
    match style {
        "hair" => Some(1.0),
        "thin" | "dashed" | "dotted" | "dashDot" | "dashDotDot" => Some(1.0),
        "double" => Some(1.0),
        "medium" | "mediumDashed" | "mediumDashDot" | "mediumDashDotDot" | "slantDashDot" => {
            Some(2.0)
        }
        "thick" => Some(3.0),
        _ => None,
    }
}

/// office2pdf `border_style_to_line_style`. Excel prints `hair` as a
/// dot-textured 1pt band, not a solid line.
pub fn border_style_to_line_style(style: &str) -> Dash {
    match style {
        "dashed" | "mediumDashed" => Dash::Dashed,
        "dotted" | "hair" => Dash::Dotted,
        "dashDot" | "mediumDashDot" | "slantDashDot" => Dash::DashDot,
        "dashDotDot" | "mediumDashDotDot" => Dash::DashDotDot,
        "double" => Dash::Double,
        _ => Dash::Solid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_unit_rounds_half_up() {
        // Times New Roman 13 lands exactly on 6.500pt and must print a 7pt
        // unit; half-even would give 6.
        assert_eq!(column_unit_pt("times new roman", 13.0), 7.0);
        // Calibri 11 is the canonical 6pt unit.
        assert_eq!(column_unit_pt("Calibri", 11.0), 6.0);
        // Calibri 10 is 5pt, which kills every integer-96dpi-pixel model.
        assert_eq!(column_unit_pt("Calibri", 10.0), 5.0);
        // An unknown family falls back to Calibri's advance.
        assert_eq!(column_unit_pt("Wingdings", 11.0), 6.0);
    }

    #[test]
    fn default_column_width_uses_base_col_width_rule() {
        // No declared default, no baseColWidth: 8 * unit + 5.
        assert_eq!(default_column_width_pt(None, None, 6.0), 53.0);
        assert_eq!(default_column_width_pt(None, None, 5.0), 45.0);
        assert_eq!(default_column_width_pt(None, None, 7.0), 61.0);
        // Declared baseColWidth outranks the 8 default.
        assert_eq!(default_column_width_pt(None, Some(10), 6.0), 65.0);
        assert_eq!(default_column_width_pt(None, Some(12), 6.0), 77.0);
        // A declared defaultColWidth outranks baseColWidth and quantizes.
        assert_eq!(default_column_width_pt(Some(10.6), Some(12), 6.0), 64.0);
    }

    #[test]
    fn declared_width_quantizes_to_integer_points() {
        // Probe calibri11frac: width 10.6 at the 6pt Calibri-11 unit prints
        // 64pt, not 63.6pt.
        assert_eq!(column_width_to_pt(10.6, 6.0), 64.0);
    }

    #[test]
    fn pattern_coverage_matches_the_probe() {
        use umya_spreadsheet::PatternValues as P;
        assert_eq!(pattern_ink_coverage(&P::Solid), 1.0);
        assert_eq!(pattern_ink_coverage(&P::DarkGrid), 0.5);
        assert_eq!(pattern_ink_coverage(&P::LightTrellis), 0.375);
        assert_eq!(pattern_ink_coverage(&P::LightGrid), 0.4375);
        assert_eq!(pattern_ink_coverage(&P::None), 0.0);
    }

    #[test]
    fn blend_matches_the_gantt_reference() {
        // 735773 over an omitted (white) background at lightUp's quarter
        // prints DCD5DC to within one level.
        let blended = blend_color(Rgba::WHITE, Rgba(0x73, 0x57, 0x73, 255), 0.25);
        assert_eq!(blended, Rgba(0xdc, 0xd5, 0xdc, 255));
    }

    #[test]
    fn border_widths_and_styles() {
        assert_eq!(border_style_to_width("hair"), Some(1.0));
        assert_eq!(border_style_to_width("double"), Some(1.0));
        assert_eq!(border_style_to_width("medium"), Some(2.0));
        assert_eq!(border_style_to_width("slantDashDot"), Some(2.0));
        assert_eq!(border_style_to_width("thick"), Some(3.0));
        assert_eq!(border_style_to_width("none"), None);
        assert_eq!(border_style_to_line_style("hair"), Dash::Dotted);
        assert_eq!(border_style_to_line_style("thin"), Dash::Solid);
        assert_eq!(border_style_to_line_style("double"), Dash::Double);
    }
}
