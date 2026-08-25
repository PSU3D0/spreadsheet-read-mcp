use super::address::CellAddress;
use super::sst::Sst;
use anyhow::{Result, anyhow};
use quick_xml::events::{BytesStart, Event};
use quick_xml::reader::Reader;
use std::io::BufRead;

#[derive(Debug)]
pub struct RawCell {
    pub address: CellAddress,
    pub value: Option<String>,
    pub formula: Option<String>,
    pub style_id: Option<u32>,
}

pub struct CellIterator<'a, R: BufRead> {
    reader: Reader<R>,
    sst: Option<&'a Sst>,
    buf: Vec<u8>,
}

impl<'a, R: BufRead> CellIterator<'a, R> {
    pub fn new(reader: R, sst: Option<&'a Sst>) -> Self {
        let reader = Reader::from_reader(reader);
        Self {
            reader,
            sst,
            buf: Vec::new(),
        }
    }

    fn read_text_content(&mut self, end_tag: &[u8]) -> Result<String> {
        let mut text = String::new();
        let mut buf = Vec::new();
        loop {
            match self.reader.read_event_into(&mut buf)? {
                Event::Text(e) => text.push_str(&e.unescape()?),
                Event::CData(e) => text.push_str(&String::from_utf8_lossy(&e)),
                Event::End(e) if e.name().as_ref() == end_tag => break,
                Event::Eof => return Err(anyhow!("Unexpected EOF reading text")),
                _ => (),
            }
            buf.clear();
        }
        Ok(text)
    }

    fn parse_cell(&mut self, e: &BytesStart) -> Result<RawCell> {
        let mut address_str = String::new();
        let mut type_str = String::new();
        let mut style_id: Option<u32> = None;

        for attr in e.attributes() {
            let attr = attr?;
            if attr.key.as_ref() == b"r" {
                address_str = String::from_utf8_lossy(&attr.value).to_string();
            } else if attr.key.as_ref() == b"t" {
                type_str = String::from_utf8_lossy(&attr.value).to_string();
            } else if attr.key.as_ref() == b"s" {
                let s = String::from_utf8_lossy(&attr.value).to_string();
                style_id = s.parse::<u32>().ok();
            }
        }

        if address_str.is_empty() {
            return Err(anyhow!("Cell missing address"));
        }
        let address = CellAddress::parse(&address_str)
            .ok_or_else(|| anyhow!("Invalid cell address: {}", address_str))?;

        let mut value = None;
        let mut formula = None;
        let mut buf = Vec::new();

        loop {
            match self.reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"v" => {
                            let text = self.read_text_content(b"v")?;
                            value = Some(text);
                        }
                        b"f" => {
                            let text = self.read_text_content(b"f")?;
                            formula = Some(text);
                        }
                        b"is" => {
                            // Inline string - just skip for now or try to read text if simple
                            // For now we might lose inline strings, but they are rare in recalc scenarios
                        }
                        _ => {}
                    }
                }
                Ok(Event::End(ref e)) if e.name().as_ref() == b"c" => {
                    break;
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(e.into()),
                _ => {}
            }
            buf.clear();
        }

        // Post-process value based on type
        if type_str == "s"
            && let Some(ref v) = value
            && let Ok(idx) = v.parse::<usize>()
            && let Some(sst) = self.sst
            && let Some(s) = sst.get(idx)
        {
            value = Some(s.to_string());
        }

        Ok(RawCell {
            address,
            value,
            formula,
            style_id,
        })
    }
}

impl<'a, R: BufRead> Iterator for CellIterator<'a, R> {
    type Item = Result<RawCell>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.buf.clear();
            match self.reader.read_event_into(&mut self.buf) {
                Ok(Event::Start(ref e)) if e.name().as_ref() == b"c" => {
                    // Clone event to own the data while parsing children
                    let e_owned = e.to_owned();
                    return Some(self.parse_cell(&e_owned));
                }
                Ok(Event::Eof) => return None,
                Err(e) => return Some(Err(e.into())),
                _ => {}
            }
        }
    }
}
