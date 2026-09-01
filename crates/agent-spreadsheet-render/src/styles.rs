//! The workbook Normal font, read out of `xl/styles.xml`.
//!
//! Ported from office2pdf `extract_normal_font`. Excel derives every column
//! print metric from this font, not from cell fonts, and umya does not expose
//! the stylesheet at all.
//!
//! This is also the fix for the bake-off's fatal non-determinism: spike D
//! inferred the normal font by frequency over `Worksheet::get_cell_collection()`,
//! whose iteration order is not stable, so the same binary produced a
//! different column unit and a different image width between runs of the same
//! fixture. Nothing in this crate may aggregate over that collection.
//!
//! office2pdf reads the part out of the zip archive itself; this crate has no
//! filesystem or archive dependency, so the caller passes the decompressed
//! part bytes in [`crate::RenderOptions::styles_xml`].

/// The Normal font: the `xl/styles.xml` font cells with no style of their own
/// inherit.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalFont {
    pub family: String,
    pub size_pt: f64,
}

impl Default for NormalFont {
    /// Excel's own default, and the documented fallback when the caller passes
    /// no `styles.xml` or the part is unreadable.
    fn default() -> Self {
        Self {
            family: "Calibri".to_string(),
            size_pt: 11.0,
        }
    }
}

/// Read the first `<font>` element of an `xl/styles.xml` part.
///
/// Returns `None` when the part does not parse or declares no font name, in
/// which case the caller uses [`NormalFont::default`].
pub fn extract_normal_font(styles_xml: &[u8]) -> Option<NormalFont> {
    use quick_xml::events::Event;

    let text = std::str::from_utf8(styles_xml).ok()?;
    let mut reader = quick_xml::Reader::from_str(text);
    let mut in_first_font = false;
    let mut name: Option<String> = None;
    let mut size: Option<f64> = None;
    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) if e.local_name().as_ref() == b"font" => {
                in_first_font = true;
            }
            Ok(Event::End(ref e)) if e.local_name().as_ref() == b"font" => break,
            Ok(Event::Empty(ref e)) if in_first_font => {
                let value = e
                    .try_get_attribute("val")
                    .ok()
                    .flatten()
                    .and_then(|a| String::from_utf8(a.value.into_owned()).ok());
                match e.local_name().as_ref() {
                    b"name" => name = value,
                    b"sz" => size = value.and_then(|v| v.parse::<f64>().ok()),
                    _ => {}
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    let family = name?;
    if family.is_empty() {
        return None;
    }
    Some(NormalFont {
        family,
        size_pt: size.filter(|s| *s > 0.0).unwrap_or(11.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const STYLES: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="12"/><name val="Verdana"/></font>
    <font><b/><sz val="18"/><name val="Arial"/></font>
  </fonts>
</styleSheet>"#;

    #[test]
    fn reads_the_first_font_only() {
        let font = extract_normal_font(STYLES.as_bytes()).unwrap();
        assert_eq!(font.family, "Verdana");
        assert_eq!(font.size_pt, 12.0);
    }

    #[test]
    fn missing_size_defaults_to_eleven() {
        let xml = r#"<styleSheet><fonts><font><name val="Calibri"/></font></fonts></styleSheet>"#;
        let font = extract_normal_font(xml.as_bytes()).unwrap();
        assert_eq!(font.size_pt, 11.0);
    }

    #[test]
    fn unparseable_part_falls_back() {
        assert!(extract_normal_font(b"not xml at all <<<").is_none());
        assert_eq!(NormalFont::default().family, "Calibri");
        assert_eq!(NormalFont::default().size_pt, 11.0);
    }
}
