use anyhow::{Result, anyhow};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use std::io::BufRead;

pub struct Sst {
    strings: Vec<String>,
}

impl Sst {
    pub fn from_reader<R: BufRead>(reader: R) -> Result<Self> {
        let mut reader = Reader::from_reader(reader);

        let mut strings = Vec::new();
        let mut buf = Vec::new();
        let mut current_string = String::new();
        let mut inside_si = false;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    if e.name().as_ref() == b"si" {
                        inside_si = true;
                        current_string.clear();
                    } else if inside_si && e.name().as_ref() == b"t" {
                        let text = read_text_content(&mut reader, b"t")?;
                        current_string.push_str(&text);
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.name().as_ref() == b"si" {
                        inside_si = false;
                        strings.push(current_string.clone());
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(e.into()),
                _ => (),
            }
            buf.clear();
        }

        Ok(Self { strings })
    }

    pub fn get(&self, idx: usize) -> Option<&str> {
        self.strings.get(idx).map(|s| s.as_str())
    }
}

fn read_text_content<R: BufRead>(reader: &mut Reader<R>, end_tag: &[u8]) -> Result<String> {
    let mut text = String::new();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf)? {
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
