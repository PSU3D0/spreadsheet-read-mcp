//! umya `Worksheet` -> `Scene` for a bounded range.
//!
//! Geometry model ported from office2pdf `src/parser/xlsx_cells.rs` (the
//! column-unit / half-up rounding model, calibrated against native Excel
//! probes), with the print-only 0.92 row compaction deliberately dropped
//! because we raster for screen, not for Excel's PDF grid.

use umya_spreadsheet::{Spreadsheet, Worksheet};

use crate::scene::{Cmd, Dash, HAlign, Rect, Rgba, Scene, VAlign, Warning};

pub const PX_PER_PT: f32 = 96.0 / 72.0;
/// office2pdf `CALIBRI_DIGIT_ADVANCE_EM`.
pub const CALIBRI_DIGIT_ADVANCE_EM: f64 = 0.506_836;
pub const EXCEL_DEFAULT_ROW_HEIGHT_PT: f64 = 15.0;
/// office2pdf `XLSX_CELL_PADDING`, in points.
pub const PAD_LEFT_PT: f32 = 3.0;
pub const PAD_RIGHT_PT: f32 = 3.0;
pub const PAD_TOP_PT: f32 = 1.0;
pub const PAD_BOTTOM_PT: f32 = 1.5;

pub const HEADING_COL_H_PX: f32 = 20.0;
pub const HEADING_ROW_W_PX: f32 = 44.0;

/// office2pdf `reference_digit_advance_em` — hardcoded on purpose so the
/// geometry is machine-independent and identical under wasm.
fn digit_advance_em(family: &str) -> f64 {
    match family.to_ascii_lowercase().as_str() {
        "calibri" | "carlito" => CALIBRI_DIGIT_ADVANCE_EM,
        "arial" | "helvetica" | "liberation sans" => 0.556_152,
        "verdana" => 0.635_742,
        "courier new" => 0.600_098,
        "times new roman" => 0.500_000,
        _ => CALIBRI_DIGIT_ADVANCE_EM,
    }
}

fn round_half_up(v: f64) -> f64 {
    (v + 0.5).floor()
}

fn column_unit_pt(family: &str, size_pt: f64) -> f64 {
    round_half_up(digit_advance_em(family) * size_pt)
}

#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// 1-based inclusive bounds.
    pub first_col: u32,
    pub last_col: u32,
    pub first_row: u32,
    pub last_row: u32,
    pub headings: bool,
    pub scale: f32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            first_col: 1,
            last_col: 13,
            first_row: 1,
            last_row: 40,
            headings: true,
            scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Track {
    /// index (1-based)
    idx: u32,
    /// device px offset of the leading edge
    offset: f32,
    size: f32,
    hidden: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct Merge {
    c0: u32,
    r0: u32,
    c1: u32,
    r1: u32,
}

fn col_name(mut n: u32) -> String {
    let mut s = String::new();
    while n > 0 {
        let r = ((n - 1) % 26) as u8;
        s.insert(0, (b'A' + r) as char);
        n = (n - 1) / 26;
    }
    s
}

/// Normal font (name, size) read out of the workbook's first style. umya gives
/// us the cell style directly, so unlike office2pdf we do not have to reopen
/// styles.xml — but a cell on `cellXfs[0]` can still report no font.
fn normal_font(book: &Spreadsheet, sheet: &Worksheet) -> (String, f64) {
    let _ = book;
    let mut counts: Vec<(String, f64, usize)> = Vec::new();
    for cell in sheet.get_cell_collection() {
        let f = cell.get_style().get_font();
        let (name, size) = match f {
            Some(f) => (f.get_name().to_string(), *f.get_size()),
            None => continue,
        };
        if name.is_empty() {
            continue;
        }
        if let Some(e) = counts
            .iter_mut()
            .find(|e| e.0 == name && (e.1 - size).abs() < 0.01)
        {
            e.2 += 1;
        } else {
            counts.push((name, size, 1));
        }
    }
    // umya's `get_cell_collection()` iteration order is NOT stable across runs,
    // so a bare frequency sort produces a different "normal font" (and hence a
    // different column unit and a different image width) between runs whenever
    // two families tie. Tie-break on name and size to make it deterministic.
    // Ticket 5004 needs PNG hash goldens, so this matters.
    counts.sort_by(|a, b| {
        b.2.cmp(&a.2)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.total_cmp(&b.1))
    });
    counts
        .first()
        .map(|e| (e.0.clone(), e.1))
        .unwrap_or_else(|| ("Calibri".to_string(), 11.0))
}

fn build_col_tracks(sheet: &Worksheet, o: &Options, unit_pt: f64) -> Vec<Track> {
    let props = sheet.get_sheet_format_properties();
    let declared_default = {
        let v = *props.get_default_column_width();
        if v > 0.0 { Some(v) } else { None }
    };
    let base = {
        let v = *props.get_base_column_width();
        if v > 0 { Some(v) } else { None }
    };
    let default_pt = match declared_default {
        Some(chars) => round_half_up(chars * unit_pt),
        None => f64::from(base.unwrap_or(8)) * unit_pt + 5.0,
    };

    let mut out = Vec::new();
    let mut x = 0.0f32;
    for c in o.first_col..=o.last_col {
        let dim = sheet.get_column_dimension_by_number(&c);
        let (w_pt, hidden) = match dim {
            Some(d) => {
                let hidden = *d.get_hidden();
                let w = *d.get_width();
                let pt = if w > 0.0 {
                    round_half_up(w * unit_pt)
                } else {
                    default_pt
                };
                (pt, hidden)
            }
            None => (default_pt, false),
        };
        let size = if hidden {
            0.0
        } else {
            (w_pt as f32) * PX_PER_PT * o.scale
        };
        out.push(Track {
            idx: c,
            offset: x,
            size,
            hidden,
        });
        x += size;
    }
    out
}

fn build_row_tracks(sheet: &Worksheet, o: &Options) -> Vec<Track> {
    let props = sheet.get_sheet_format_properties();
    let default_pt = {
        let v = *props.get_default_row_height();
        if v > 0.0 { v } else { EXCEL_DEFAULT_ROW_HEIGHT_PT }
    };
    let mut out = Vec::new();
    let mut y = 0.0f32;
    for r in o.first_row..=o.last_row {
        let dim = sheet.get_row_dimension(&r);
        let (h_pt, hidden) = match dim {
            Some(d) => {
                let hidden = *d.get_hidden();
                let h = *d.get_height();
                (if h > 0.0 { h } else { default_pt }, hidden)
            }
            None => (default_pt, false),
        };
        let size = if hidden {
            0.0
        } else {
            (h_pt as f32) * PX_PER_PT * o.scale
        };
        out.push(Track {
            idx: r,
            offset: y,
            size,
            hidden,
        });
        y += size;
    }
    out
}

fn merges(sheet: &Worksheet) -> Vec<Merge> {
    sheet
        .get_merge_cells()
        .iter()
        .map(|r| {
            let c0 = r
                .get_coordinate_start_col()
                .map(|c| *c.get_num())
                .unwrap_or(1);
            let r0 = r
                .get_coordinate_start_row()
                .map(|c| *c.get_num())
                .unwrap_or(1);
            Merge {
                c0,
                r0,
                c1: r.get_coordinate_end_col().map(|c| *c.get_num()).unwrap_or(c0),
                r1: r.get_coordinate_end_row().map(|c| *c.get_num()).unwrap_or(r0),
            }
        })
        .collect()
}

/// office2pdf `pattern_ink_coverage`.
fn pattern_coverage(p: &str) -> f64 {
    match p {
        "solid" => 1.0,
        "darkGray" | "darkTrellis" => 0.75,
        "mediumGray" | "darkHorizontal" | "darkVertical" | "darkDown" | "darkUp" | "darkGrid" => 0.5,
        "lightGrid" => 0.4375,
        "lightTrellis" => 0.375,
        "lightGray" | "lightHorizontal" | "lightVertical" | "lightDown" | "lightUp" => 0.25,
        "gray125" => 0.125,
        "gray0625" => 0.0625,
        _ => 0.0,
    }
}

fn blend(below: Rgba, above: Rgba, cov: f64) -> Rgba {
    let mix = |b: u8, a: u8| ((f64::from(b) * (1.0 - cov)) + f64::from(a) * cov).round() as u8;
    Rgba(
        mix(below.0, above.0),
        mix(below.1, above.1),
        mix(below.2, above.2),
        255,
    )
}

/// office2pdf `border_style_to_width` (measured against native Excel 16.111).
fn border_width_pt(style: &str) -> Option<f32> {
    match style {
        "hair" | "thin" | "dashed" | "dotted" | "dashDot" | "dashDotDot" | "double" => Some(1.0),
        "medium" | "mediumDashed" | "mediumDashDot" | "mediumDashDotDot" | "slantDashDot" => {
            Some(2.0)
        }
        "thick" => Some(3.0),
        _ => None,
    }
}

fn border_dash(style: &str) -> Dash {
    match style {
        "dashed" | "mediumDashed" => Dash::Dashed,
        "dotted" | "hair" => Dash::Dotted,
        "dashDot" | "mediumDashDot" | "slantDashDot" => Dash::DashDot,
        "dashDotDot" | "mediumDashDotDot" => Dash::DashDotDot,
        "double" => Dash::Double,
        _ => Dash::Solid,
    }
}

pub fn extract(book: &Spreadsheet, sheet: &Worksheet, o: &Options) -> Scene {
    let mut scene = Scene::default();
    let (nf_name, nf_size) = normal_font(book, sheet);
    let unit_pt = column_unit_pt(&nf_name, nf_size);

    let cols = build_col_tracks(sheet, o, unit_pt);
    let rows = build_row_tracks(sheet, o);
    let ms = merges(sheet);

    let ox = if o.headings {
        HEADING_ROW_W_PX * o.scale
    } else {
        0.0
    };
    let oy = if o.headings {
        HEADING_COL_H_PX * o.scale
    } else {
        0.0
    };
    let grid_w = cols.last().map(|t| t.offset + t.size).unwrap_or(0.0);
    let grid_h = rows.last().map(|t| t.offset + t.size).unwrap_or(0.0);
    scene.width = ox + grid_w;
    scene.height = oy + grid_h;

    let cx = |t: &Track| ox + t.offset;
    let cy = |t: &Track| oy + t.offset;

    // 1. background
    scene.cmds.push(Cmd::FillRect {
        rect: Rect {
            x: 0.0,
            y: 0.0,
            w: scene.width,
            h: scene.height,
        },
        color: Rgba::WHITE,
    });

    // 2. cell fills (merged cells paint from their top-left)
    let skip = |c: u32, r: u32| -> Option<Merge> {
        ms.iter()
            .find(|m| c >= m.c0 && c <= m.c1 && r >= m.r0 && r <= m.r1)
            .copied()
    };

    let span_rect = |m: &Merge| -> Option<Rect> {
        let c0 = cols.iter().find(|t| t.idx == m.c0)?;
        let r0 = rows.iter().find(|t| t.idx == m.r0)?;
        let c1 = cols.iter().find(|t| t.idx == m.c1.min(o.last_col))?;
        let r1 = rows.iter().find(|t| t.idx == m.r1.min(o.last_row))?;
        Some(Rect {
            x: cx(c0),
            y: cy(r0),
            w: (c1.offset + c1.size) - c0.offset,
            h: (r1.offset + r1.size) - r0.offset,
        })
    };

    let mut texts: Vec<Cmd> = Vec::new();
    let mut borders: Vec<Cmd> = Vec::new();

    for rt in &rows {
        if rt.hidden {
            continue;
        }
        for ct in &cols {
            if ct.hidden {
                continue;
            }
            let m = skip(ct.idx, rt.idx);
            if let Some(m) = m
                && (m.c0 != ct.idx || m.r0 != rt.idx)
            {
                continue; // interior of a merge
            }
            let rect = match m {
                Some(m) => match span_rect(&m) {
                    Some(r) => r,
                    None => continue,
                },
                None => Rect {
                    x: cx(ct),
                    y: cy(rt),
                    w: ct.size,
                    h: rt.size,
                },
            };

            let cell = sheet.get_cell((ct.idx, rt.idx));
            let Some(cell) = cell else { continue };
            let style = cell.get_style();

            // fill
            if let Some(pf) = style.get_fill().and_then(|f| f.get_pattern_fill()) {
                let ptype = format!("{:?}", pf.get_pattern_type());
                let ptype = pattern_key(&ptype);
                let cov = pattern_coverage(&ptype);
                if cov > 0.0 {
                    let fg = pf
                        .get_foreground_color()
                        .and_then(|c| Rgba::from_argb_hex(c.get_argb()))
                        .unwrap_or(Rgba::BLACK);
                    let color = if cov >= 1.0 {
                        fg
                    } else {
                        scene.warn(Warning::PatternFillApproximated);
                        let bg = pf
                            .get_background_color()
                            .and_then(|c| Rgba::from_argb_hex(c.get_argb()))
                            .unwrap_or(Rgba::WHITE);
                        blend(bg, fg, cov)
                    };
                    scene.cmds.push(Cmd::FillRect { rect, color });
                }
            }

            // borders
            if let Some(bs) = style.get_borders() {
                let mut edge = |style_str: &str, argb: Option<&str>, x1, y1, x2, y2| {
                    if let Some(wpt) = border_width_pt(style_str) {
                        borders.push(Cmd::Line {
                            x1,
                            y1,
                            x2,
                            y2,
                            width: wpt * PX_PER_PT * o.scale,
                            color: argb
                                .and_then(Rgba::from_argb_hex)
                                .unwrap_or(Rgba::BLACK),
                            dash: border_dash(style_str),
                        });
                    }
                };
                let t = bs.get_top();
                edge(
                    t.get_border_style(),
                    Some(t.get_color().get_argb()),
                    rect.x,
                    rect.y,
                    rect.x + rect.w,
                    rect.y,
                );
                let b = bs.get_bottom();
                edge(
                    b.get_border_style(),
                    Some(b.get_color().get_argb()),
                    rect.x,
                    rect.y + rect.h,
                    rect.x + rect.w,
                    rect.y + rect.h,
                );
                let l = bs.get_left();
                edge(
                    l.get_border_style(),
                    Some(l.get_color().get_argb()),
                    rect.x,
                    rect.y,
                    rect.x,
                    rect.y + rect.h,
                );
                let r = bs.get_right();
                edge(
                    r.get_border_style(),
                    Some(r.get_color().get_argb()),
                    rect.x + rect.w,
                    rect.y,
                    rect.x + rect.w,
                    rect.y + rect.h,
                );
            }

            // text
            let value = cell.get_formatted_value();
            if value.is_empty() {
                continue;
            }
            if cell.get_formula().is_empty() {
                // cached value present or literal
            } else if cell.get_cell_value().get_value().is_empty() {
                scene.warn(Warning::FormulasUnevaluated);
            }
            let font = style.get_font();
            let (bold, italic, underline, strike, size_pt, fcolor, fname) = match font {
                Some(f) => (
                    *f.get_bold(),
                    *f.get_italic(),
                    format!("{:?}", f.get_font_underline().get_val()) != "None",
                    *f.get_strikethrough(),
                    *f.get_size(),
                    Rgba::from_argb_hex(f.get_color().get_argb()).unwrap_or(Rgba::BLACK),
                    f.get_name().to_string(),
                ),
                None => (
                    false,
                    false,
                    false,
                    false,
                    nf_size,
                    Rgba::BLACK,
                    nf_name.clone(),
                ),
            };
            if !fname.is_empty() && !fname.eq_ignore_ascii_case("Calibri") {
                scene.warn(Warning::FontSubstituted);
            }
            let (halign, valign, wrap, rotated) = match style.get_alignment() {
                Some(a) => {
                    let h = match format!("{:?}", a.get_horizontal()).as_str() {
                        "Center" | "CenterContinuous" => Some(HAlign::Center),
                        "Right" => Some(HAlign::Right),
                        "Left" => Some(HAlign::Left),
                        _ => None,
                    };
                    let v = match format!("{:?}", a.get_vertical()).as_str() {
                        "Center" => VAlign::Center,
                        "Top" => VAlign::Top,
                        _ => VAlign::Bottom,
                    };
                    (h, v, *a.get_wrap_text(), *a.get_text_rotation() != 0)
                }
                None => (None, VAlign::Bottom, false, false),
            };
            if rotated {
                scene.warn(Warning::TextRotationOmitted);
            }
            // general alignment: numbers right, text left
            let numeric = cell.get_value_number().is_some();
            let halign = halign.unwrap_or(if numeric { HAlign::Right } else { HAlign::Left });
            if !style
                .get_number_format()
                .map(|n| n.get_format_code().is_empty())
                .unwrap_or(true)
            {
                scene.warn(Warning::NumberFormatApproximated);
            }

            let pad_l = PAD_LEFT_PT * PX_PER_PT * o.scale;
            let pad_r = PAD_RIGHT_PT * PX_PER_PT * o.scale;
            let pad_t = PAD_TOP_PT * PX_PER_PT * o.scale;
            let pad_b = PAD_BOTTOM_PT * PX_PER_PT * o.scale;
            let inner = Rect {
                x: rect.x + pad_l,
                y: rect.y + pad_t,
                w: (rect.w - pad_l - pad_r).max(0.0),
                h: (rect.h - pad_t - pad_b).max(0.0),
            };
            // Unwrapped left/general text spills across empty neighbours to the
            // right (office2pdf compute_spill_width, simplified).
            let mut clip = Rect {
                x: rect.x,
                y: rect.y,
                w: rect.w,
                h: rect.h,
            };
            if !wrap && m.is_none() && halign == HAlign::Left {
                let mut extra = 0.0;
                for nc in cols.iter().filter(|t| t.idx > ct.idx && !t.hidden) {
                    let occupied = sheet
                        .get_cell((nc.idx, rt.idx))
                        .map(|c| !c.get_formatted_value().is_empty())
                        .unwrap_or(false);
                    if occupied {
                        break;
                    }
                    extra += nc.size;
                }
                clip.w += extra;
            }

            texts.push(Cmd::Text {
                rect: inner,
                text: value,
                size_px: (size_pt as f32) * PX_PER_PT * o.scale,
                color: fcolor,
                bold,
                italic,
                underline,
                strike,
                halign,
                valign,
                wrap,
                clip,
            });
        }
    }

    // 3. gridlines (below borders)
    let show_grid = sheet
        .get_sheets_views()
        .get_sheet_view_list()
        .first()
        .map(|v| *v.get_show_grid_lines())
        .unwrap_or(true);
    if show_grid {
        for ct in cols.iter().filter(|t| !t.hidden) {
            let x = cx(ct);
            scene.cmds.push(Cmd::Line {
                x1: x,
                y1: oy,
                x2: x,
                y2: scene.height,
                width: 1.0,
                color: Rgba::GRIDLINE,
                dash: Dash::Solid,
            });
        }
        scene.cmds.push(Cmd::Line {
            x1: ox + grid_w,
            y1: oy,
            x2: ox + grid_w,
            y2: scene.height,
            width: 1.0,
            color: Rgba::GRIDLINE,
            dash: Dash::Solid,
        });
        for rt in rows.iter().filter(|t| !t.hidden) {
            let y = cy(rt);
            scene.cmds.push(Cmd::Line {
                x1: ox,
                y1: y,
                x2: scene.width,
                y2: y,
                width: 1.0,
                color: Rgba::GRIDLINE,
                dash: Dash::Solid,
            });
        }
        scene.cmds.push(Cmd::Line {
            x1: ox,
            y1: oy + grid_h,
            x2: scene.width,
            y2: oy + grid_h,
            width: 1.0,
            color: Rgba::GRIDLINE,
            dash: Dash::Solid,
        });
    }

    scene.cmds.extend(borders);
    scene.cmds.extend(texts);

    // 4. headings last so they stay legible
    if o.headings {
        scene.cmds.push(Cmd::FillRect {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: scene.width,
                h: oy,
            },
            color: Rgba::HEADING_BG,
        });
        scene.cmds.push(Cmd::FillRect {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                w: ox,
                h: scene.height,
            },
            color: Rgba::HEADING_BG,
        });
        for ct in cols.iter().filter(|t| !t.hidden && t.size > 6.0) {
            scene.cmds.push(Cmd::Text {
                rect: Rect {
                    x: cx(ct),
                    y: 2.0,
                    w: ct.size,
                    h: oy - 4.0,
                },
                text: col_name(ct.idx),
                size_px: 11.0 * o.scale,
                color: Rgba::HEADING_FG,
                bold: false,
                italic: false,
                underline: false,
                strike: false,
                halign: HAlign::Center,
                valign: VAlign::Center,
                wrap: false,
                clip: Rect {
                    x: cx(ct),
                    y: 0.0,
                    w: ct.size,
                    h: oy,
                },
            });
        }
        for rt in rows.iter().filter(|t| !t.hidden && t.size > 6.0) {
            scene.cmds.push(Cmd::Text {
                rect: Rect {
                    x: 2.0,
                    y: cy(rt),
                    w: ox - 6.0,
                    h: rt.size,
                },
                text: rt.idx.to_string(),
                size_px: 11.0 * o.scale,
                color: Rgba::HEADING_FG,
                bold: false,
                italic: false,
                underline: false,
                strike: false,
                halign: HAlign::Right,
                valign: VAlign::Center,
                wrap: false,
                clip: Rect {
                    x: 0.0,
                    y: cy(rt),
                    w: ox,
                    h: rt.size,
                },
            });
        }
        scene.cmds.push(Cmd::Line {
            x1: 0.0,
            y1: oy,
            x2: scene.width,
            y2: oy,
            width: 1.0,
            color: Rgba(0x9a, 0x9a, 0x9a, 255),
            dash: Dash::Solid,
        });
        scene.cmds.push(Cmd::Line {
            x1: ox,
            y1: 0.0,
            x2: ox,
            y2: scene.height,
            width: 1.0,
            color: Rgba(0x9a, 0x9a, 0x9a, 255),
            dash: Dash::Solid,
        });
    }

    if !sheet.get_conditional_formatting_collection().is_empty() {
        scene.warn(Warning::ConditionalFormatOmitted);
    }
    if !sheet.get_image_collection().is_empty() {
        scene.warn(Warning::ImageOmitted);
    }
    if !sheet.get_chart_collection().is_empty() {
        scene.warn(Warning::ChartOmitted);
    }

    scene
}

/// umya's `PatternValues` Debug names are CamelCase; the OOXML codes are
/// lowerCamelCase. Normalise.
fn pattern_key(debug_name: &str) -> String {
    let mut c = debug_name.chars();
    match c.next() {
        Some(f) => f.to_ascii_lowercase().to_string() + c.as_str(),
        None => String::new(),
    }
}
