//! Measures the PNG-encode share of spike D's total time at three zlib levels.
use std::time::Instant;

fn main() {
    let dir = std::env::args().nth(1).unwrap_or("corpus".into());
    let fonts = spike_d::Fonts::new();
    let mut rows = Vec::new();
    for g in std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()) {
        if !g.path().is_dir() {
            continue;
        }
        for f in std::fs::read_dir(g.path()).unwrap().filter_map(|e| e.ok()) {
            let p = f.path();
            if p.extension().and_then(|s| s.to_str()) != Some("xlsx") {
                continue;
            }
            let Ok(book) = umya_spreadsheet::reader::xlsx::read(&p) else {
                continue;
            };
            let Some(sh) = book.get_sheet(&0) else { continue };
            let scene = spike_d::extract::extract(&book, sh, &spike_d::extract::Options::default());
            let Ok(pix) = spike_d::raster::render(&scene, &fonts) else {
                continue;
            };
            let mut rec = vec![];
            for (name, lvl) in [
                ("balanced(default)", png::Compression::Balanced),
                ("fast", png::Compression::Fast),
                ("fastest", png::Compression::Fastest),
            ] {
                let t = Instant::now();
                let b = spike_d::raster::to_png_level(&pix, lvl).unwrap();
                rec.push((name, t.elapsed().as_secs_f64() * 1000.0, b.len()));
            }
            rows.push(rec);
        }
    }
    for i in 0..3 {
        let name = rows[0][i].0;
        let mut ms: Vec<f64> = rows.iter().map(|r| r[i].1).collect();
        let mut by: Vec<usize> = rows.iter().map(|r| r[i].2).collect();
        ms.sort_by(f64::total_cmp);
        by.sort_unstable();
        println!(
            "{name}\tmedian_ms={:.2}\tmean_ms={:.2}\tmedian_bytes={}\ttotal_bytes={}",
            ms[ms.len() / 2],
            ms.iter().sum::<f64>() / ms.len() as f64,
            by[by.len() / 2],
            by.iter().sum::<usize>()
        );
    }
}
