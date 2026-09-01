//! Ticket 5003 bake-off harness.
//!
//! Usage: bakeoff <corpus-dir> <out-dir> [ref-a|ref-b|spike-d]...
//! Emits one JSON object per (backend, workbook) line to stdout and writes
//! PNGs under <out-dir>/<backend>/<group>__<name>.png.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn corpus(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    let mut groups: Vec<_> = fs::read_dir(dir)
        .expect("corpus dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    groups.sort();
    for g in groups {
        let gname = g.file_name().unwrap().to_string_lossy().to_string();
        let mut files: Vec<_> = fs::read_dir(&g)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("xlsx"))
            .collect();
        files.sort();
        for f in files {
            let n = f.file_stem().unwrap().to_string_lossy().to_string();
            out.push((format!("{gname}__{n}"), f));
        }
    }
    out
}

#[derive(Default)]
struct Row {
    backend: String,
    file: String,
    status: String,
    error: String,
    parse_ms: f64,
    scene_ms: f64,
    raster_ms: f64,
    encode_ms: f64,
    png_bytes: usize,
    width: u32,
    height: u32,
    warnings: Vec<String>,
}

impl Row {
    fn emit(&self) {
        let j = serde_json::json!({
            "backend": self.backend,
            "file": self.file,
            "status": self.status,
            "error": self.error,
            "parse_ms": (self.parse_ms * 1000.0).round() / 1000.0,
            "scene_ms": (self.scene_ms * 1000.0).round() / 1000.0,
            "raster_ms": (self.raster_ms * 1000.0).round() / 1000.0,
            "encode_ms": (self.encode_ms * 1000.0).round() / 1000.0,
            "png_bytes": self.png_bytes,
            "width": self.width,
            "height": self.height,
            "warnings": self.warnings,
        });
        println!("{j}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus_dir = PathBuf::from(args.get(1).cloned().unwrap_or("corpus".into()));
    let out_dir = PathBuf::from(args.get(2).cloned().unwrap_or("out".into()));
    let backends: Vec<String> = if args.len() > 3 {
        args[3..].to_vec()
    } else {
        vec!["ref-a".into(), "ref-b".into(), "spike-d".into()]
    };

    let files = corpus(&corpus_dir);
    eprintln!("corpus: {} workbooks", files.len());

    for backend in &backends {
        let bdir = out_dir.join(backend);
        let _ = fs::create_dir_all(&bdir);
        for (name, path) in &files {
            let row = match backend.as_str() {
                #[cfg(feature = "ref-a")]
                "ref-a" => run_ref_a(name, path, &bdir),
                #[cfg(feature = "ref-b")]
                "ref-b" => run_ref_b(name, path, &bdir),
                #[cfg(feature = "spike-d")]
                "spike-d" => run_spike_d(name, path, &bdir),
                other => Row {
                    backend: other.into(),
                    file: name.clone(),
                    status: "skipped".into(),
                    error: "backend not compiled in".into(),
                    ..Row::default()
                },
            };
            row.emit();
        }
    }
}

fn png_dims(b: &[u8]) -> (u32, u32) {
    if b.len() < 24 {
        return (0, 0);
    }
    let w = u32::from_be_bytes([b[16], b[17], b[18], b[19]]);
    let h = u32::from_be_bytes([b[20], b[21], b[22], b[23]]);
    (w, h)
}

// ---------------------------------------------------------------------------
// Reference A: BetterOffice
// ---------------------------------------------------------------------------
#[cfg(feature = "ref-a")]
fn run_ref_a(name: &str, path: &Path, out: &Path) -> Row {
    use betteroffice_xlsx::{SheetId, Workbook, viewport_for_range};
    use betteroffice_xlsx::{CellRange, CellRef};

    let mut row = Row {
        backend: "ref-a".into(),
        file: name.into(),
        ..Row::default()
    };
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            row.status = "read_error".into();
            row.error = e.to_string();
            return row;
        }
    };
    let t = Instant::now();
    let wb = match std::panic::catch_unwind(|| Workbook::open_for_read(&bytes)) {
        Ok(Ok(w)) => w,
        Ok(Err(e)) => {
            row.parse_ms = ms(t);
            row.status = "parse_error".into();
            row.error = format!("{e:?}");
            return row;
        }
        Err(_) => {
            row.parse_ms = ms(t);
            row.status = "parse_panic".into();
            return row;
        }
    };
    row.parse_ms = ms(t);

    // Bound to the same window spike D uses: A1:M40.
    let range = CellRange::new(CellRef::new(0, 0), CellRef::new(39, 12));
    let sheet = SheetId(0);
    let vp = match wb.sheet(sheet) {
        Ok(s) => viewport_for_range(s, range),
        Err(e) => {
            row.status = "sheet_error".into();
            row.error = format!("{e:?}");
            return row;
        }
    };
    let t = Instant::now();
    match wb.render_png_for(sheet, &vp) {
        Ok(p) => {
            row.raster_ms = ms(t);
            row.png_bytes = p.bytes.len();
            row.width = p.width;
            row.height = p.height;
            row.status = "ok".into();
            let _ = fs::write(out.join(format!("{name}.png")), &p.bytes);
        }
        Err(e) => {
            row.raster_ms = ms(t);
            row.status = "render_error".into();
            row.error = format!("{e:?}");
        }
    }
    row
}

// ---------------------------------------------------------------------------
// Reference B: readany-render
// ---------------------------------------------------------------------------
#[cfg(feature = "ref-b")]
fn run_ref_b(name: &str, path: &Path, out: &Path) -> Row {
    use readany_render::{Options, Rect, rasterise_rect, render};

    let mut row = Row {
        backend: "ref-b".into(),
        file: name.into(),
        ..Row::default()
    };
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            row.status = "read_error".into();
            row.error = e.to_string();
            return row;
        }
    };
    let fname = path.file_name().unwrap().to_string_lossy().to_string();
    let t = Instant::now();
    let doc = match render(
        &bytes,
        &Options {
            filename: Some(&fname),
            sheet_headers: true,
            ..Options::default()
        },
    ) {
        Ok(d) => d,
        Err(e) => {
            row.parse_ms = ms(t);
            row.status = "parse_error".into();
            row.error = format!("{e:?}");
            return row;
        }
    };
    // readany fuses parse and layout; record it as scene time.
    row.scene_ms = ms(t);
    let Some(page) = doc.pages.first() else {
        row.status = "no_pages".into();
        return row;
    };
    for u in &doc.unrendered {
        let s = format!("{u:?}");
        let s = s.split(['{', '(']).next().unwrap_or(&s).trim().to_string();
        if !row.warnings.contains(&s) {
            row.warnings.push(s);
        }
    }
    // Clamp to a comparable window rather than the whole sheet.
    let rect = Rect {
        x: 0.0,
        y: 0.0,
        width: page.size.width.min(1400.0),
        height: page.size.height.min(1000.0),
    };
    let t = Instant::now();
    match rasterise_rect(page, rect, 1.0) {
        Ok(pm) => {
            row.raster_ms = ms(t);
            let t = Instant::now();
            match pm.encode_png() {
            Ok(png) => {
                row.encode_ms = ms(t);
                row.png_bytes = png.len();
                row.width = pm.width;
                row.height = pm.height;
                row.status = "ok".into();
                let _ = fs::write(out.join(format!("{name}.png")), &png);
            }
            Err(e) => {
                row.encode_ms = ms(t);
                row.status = "encode_error".into();
                row.error = format!("{e:?}");
            }
            }
        }
        Err(e) => {
            row.raster_ms = ms(t);
            row.status = "render_error".into();
            row.error = format!("{e:?}");
        }
    }
    row
}

// ---------------------------------------------------------------------------
// Spike D
// ---------------------------------------------------------------------------
#[cfg(feature = "spike-d")]
fn run_spike_d(name: &str, path: &Path, out: &Path) -> Row {
    use spike_d::extract::{Options, extract};

    let mut row = Row {
        backend: format!("spike-d[{}/{}]", spike_d::STACK, spike_d::FONT_STRATEGY),
        file: name.into(),
        ..Row::default()
    };
    let t = Instant::now();
    let book = match umya_spreadsheet::reader::xlsx::read(path) {
        Ok(b) => b,
        Err(e) => {
            row.parse_ms = ms(t);
            row.status = "parse_error".into();
            row.error = format!("{e:?}");
            return row;
        }
    };
    row.parse_ms = ms(t);
    let Some(sheet) = book.get_sheet(&0) else {
        row.status = "no_sheet".into();
        return row;
    };

    let t = Instant::now();
    let scene = extract(&book, sheet, &Options::default());
    row.scene_ms = ms(t);
    row.warnings = scene.warnings.iter().map(|w| format!("{w:?}")).collect();

    let fonts = spike_d::Fonts::new();
    let t = Instant::now();
    match spike_d::raster::render(&scene, &fonts) {
        Ok(pix) => {
            row.raster_ms = ms(t);
            let t = Instant::now();
            match spike_d::raster::to_png(&pix) {
            Ok(png) => {
                row.encode_ms = ms(t);
                row.png_bytes = png.len();
                let (w, h) = png_dims(&png);
                row.width = w;
                row.height = h;
                row.status = "ok".into();
                let _ = fs::write(out.join(format!("{name}.png")), &png);
            }
            Err(e) => {
                row.encode_ms = ms(t);
                row.status = "encode_error".into();
                row.error = format!("{e:?}");
            }
            }
        }
        Err(e) => {
            row.raster_ms = ms(t);
            row.status = "render_error".into();
            row.error = format!("{e:?}");
        }
    }
    row
}
