use chrono::{DateTime, SecondsFormat, Utc};
use rand::Rng;
use sha2::{Digest, Sha256};
use std::fs::{File, Metadata};
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::SystemTime;

pub fn system_time_to_datetime(time: SystemTime) -> Option<DateTime<Utc>> {
    Some(DateTime::<Utc>::from(time))
}

pub fn system_time_to_rfc3339(time: SystemTime) -> Option<DateTime<Utc>> {
    system_time_to_datetime(time)
}

fn hash_path_metadata_digest(path: &Path, metadata: &Metadata) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(metadata.len().to_le_bytes());
    if let Ok(modified) = metadata.modified()
        && let Some(dt) = system_time_to_datetime(modified)
    {
        hasher.update(dt.to_rfc3339_opts(SecondsFormat::Micros, true).as_bytes());
    }
    hasher.finalize().into()
}

fn hash_path_digest(path: &Path) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.finalize().into()
}

pub fn hash_path_metadata_full(path: &Path, metadata: &Metadata) -> String {
    let digest = hash_path_metadata_digest(path, metadata);
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

const WORKBOOK_ID_TOKEN_LEN: usize = 10;

fn encode_base32_u64_prefix(value: u64, len: usize) -> String {
    let mut out = String::with_capacity(len);
    for i in 0..len {
        let shift = 64 - (i + 1) * 5;
        let idx = ((value >> shift) & 31) as usize;
        out.push(SHORT_ID_ALPHABET[idx] as char);
    }
    out
}

pub fn hash_path_metadata(path: &Path, metadata: &Metadata) -> String {
    let digest = hash_path_metadata_digest(path, metadata);
    workbook_id_from_digest(digest)
}

pub fn hash_path_identity(path: &Path) -> String {
    let digest = hash_path_digest(path);
    workbook_id_from_digest(digest)
}

fn workbook_id_from_digest(digest: [u8; 32]) -> String {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    let value = u64::from_be_bytes(bytes);

    format!(
        "wb-{}",
        encode_base32_u64_prefix(value, WORKBOOK_ID_TOKEN_LEN)
    )
}

pub fn hash_bytes_sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub fn hash_file_sha256_hex(path: &Path) -> std::io::Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

pub fn column_number_to_name(column: u32) -> String {
    let mut column = column;
    let mut name = String::new();
    while column > 0 {
        let rem = ((column - 1) % 26) as u8;
        name.insert(0, (b'A' + rem) as char);
        column = (column - 1) / 26;
    }
    name
}

pub fn cell_address(column: u32, row: u32) -> String {
    format!("{}{}", column_number_to_name(column), row)
}

pub fn make_short_workbook_id(_slug: &str, canonical_id: &str) -> String {
    canonical_id
        .strip_prefix("wb-")
        .unwrap_or(canonical_id)
        .to_string()
}

pub fn path_to_forward_slashes(path: &Path) -> String {
    let raw = path.to_string_lossy();
    if raw.contains('\\') {
        raw.replace('\\', "/")
    } else {
        raw.into_owned()
    }
}

const SHORT_ID_ALPHABET: &[u8] = b"23456789abcdefghijkmnpqrstuvwxyz";

pub fn make_short_random_id(prefix: &str, len: usize) -> String {
    let mut rng = rand::thread_rng();

    let mut out = String::with_capacity(prefix.len() + if prefix.is_empty() { 0 } else { 1 } + len);
    if !prefix.is_empty() {
        out.push_str(prefix);
        out.push('-');
    }

    for _ in 0..len {
        let idx = rng.gen_range(0..SHORT_ID_ALPHABET.len());
        out.push(SHORT_ID_ALPHABET[idx] as char);
    }

    out
}
