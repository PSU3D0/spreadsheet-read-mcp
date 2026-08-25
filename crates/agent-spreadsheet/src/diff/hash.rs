use sha2::{Digest, Sha256};
use std::io::Read;

pub fn compute_hash<R: Read>(mut reader: R) -> std::io::Result<u64> {
    let mut hasher = Sha256::new();
    // Copy to hasher adapter
    std::io::copy(&mut reader, &mut DigestWriter(&mut hasher))?;
    let result = hasher.finalize();
    // Use first 8 bytes for u64 hash
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&result[0..8]);
    Ok(u64::from_le_bytes(buf))
}

struct DigestWriter<'a, D: Digest>(&'a mut D);

impl<'a, D: Digest> std::io::Write for DigestWriter<'a, D> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
