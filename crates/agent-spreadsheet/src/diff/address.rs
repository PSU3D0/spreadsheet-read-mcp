use std::cmp::Ordering;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct CellAddress {
    pub col: u32,
    pub row: u32,
    pub original: String,
}

impl CellAddress {
    pub fn parse(s: &str) -> Option<Self> {
        // Split into letters and numbers
        let split_idx = s.find(|c: char| c.is_ascii_digit())?;
        let (col_str, row_str) = s.split_at(split_idx);

        let row = row_str.parse::<u32>().ok()?;
        let col = col_from_letters(col_str)?;

        Some(Self {
            col,
            row,
            original: s.to_string(),
        })
    }
}

fn col_from_letters(s: &str) -> Option<u32> {
    let mut col = 0;
    for c in s.chars() {
        if !c.is_ascii_alphabetic() {
            return None;
        }
        col = col * 26 + (c.to_ascii_uppercase() as u32 - 'A' as u32 + 1);
    }
    Some(col)
}

impl Ord for CellAddress {
    fn cmp(&self, other: &Self) -> Ordering {
        // Row-major ordering
        match self.row.cmp(&other.row) {
            Ordering::Equal => self.col.cmp(&other.col),
            ord => ord,
        }
    }
}

impl PartialOrd for CellAddress {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordering() {
        let a1 = CellAddress::parse("A1").unwrap();
        let b1 = CellAddress::parse("B1").unwrap();
        let a2 = CellAddress::parse("A2").unwrap();
        let aa1 = CellAddress::parse("AA1").unwrap();

        assert!(a1 < b1);
        assert!(b1 < aa1); // B=2, AA=27
        assert!(aa1 < a2); // Row 1 < Row 2
    }
}
