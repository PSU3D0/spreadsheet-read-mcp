//! Reference C: office2pdf (umya -> Typst -> PDF). Own workspace so the heavy
//! Typst dependency graph never touches the spike workspace or the product.
//! PDF is rastered afterwards with pdftoppm at 2x (192 dpi) by the caller.
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = PathBuf::from(args.get(1).cloned().unwrap_or("corpus".into()));
    let out = PathBuf::from(args.get(2).cloned().unwrap_or("out/ref-c".into()));
    let _ = fs::create_dir_all(&out);

    let mut groups: Vec<_> = fs::read_dir(&corpus)
        .expect("corpus")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    groups.sort();
    for g in groups {
        let gname = g.file_name().unwrap().to_string_lossy().to_string();
        let mut fs_: Vec<_> = fs::read_dir(&g)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("xlsx"))
            .collect();
        fs_.sort();
        for f in fs_ {
            let name = format!("{gname}__{}", f.file_stem().unwrap().to_string_lossy());
            let t = Instant::now();
            let r = std::panic::catch_unwind(|| office2pdf::convert(&f));
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            match r {
                Ok(Ok(res)) => {
                    let warns: Vec<String> = res
                        .warnings
                        .iter()
                        .map(|w| {
                            let s = format!("{w:?}");
                            s.split(['{', '(']).next().unwrap_or(&s).trim().to_string()
                        })
                        .collect();
                    let _ = fs::write(out.join(format!("{name}.pdf")), &res.pdf);
                    println!(
                        "{}",
                        serde_json::json!({"backend":"ref-c","file":name,"status":"ok",
                            "convert_ms":(ms*1000.0).round()/1000.0,
                            "pdf_bytes":res.pdf.len(),"warnings":warns})
                    );
                }
                Ok(Err(e)) => println!(
                    "{}",
                    serde_json::json!({"backend":"ref-c","file":name,"status":"convert_error",
                        "convert_ms":(ms*1000.0).round()/1000.0,"error":format!("{e:?}")})
                ),
                Err(_) => println!(
                    "{}",
                    serde_json::json!({"backend":"ref-c","file":name,"status":"panic",
                        "convert_ms":(ms*1000.0).round()/1000.0})
                ),
            }
        }
    }
}
