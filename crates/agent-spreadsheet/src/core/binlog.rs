//! Append-only JSONL binlog writer/reader for OpEvent streams.
//!
//! The binlog is stored as `events.jsonl` within a session directory.
//! Each line is a self-contained JSON-serialized [`OpEvent`].

use crate::core::events::OpEvent;
use anyhow::{Context, Result, anyhow, bail};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// BinlogWriter
// ---------------------------------------------------------------------------

/// Append-only writer for the session event log.
pub struct BinlogWriter {
    path: PathBuf,
}

impl BinlogWriter {
    /// Open or create a binlog file at the given path.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create binlog directory: {}", parent.display())
            })?;
        }
        Ok(Self { path })
    }

    /// Append a single event to the binlog with fsync.
    pub fn append(&self, event: &OpEvent) -> Result<()> {
        let line = serde_json::to_string(event).context("failed to serialize OpEvent to JSON")?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("failed to open binlog: {}", self.path.display()))?;

        writeln!(file, "{}", line).context("failed to write event to binlog")?;

        file.sync_data().context("failed to fsync binlog")?;

        Ok(())
    }

    /// Return the path to the binlog file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ---------------------------------------------------------------------------
// BinlogReader
// ---------------------------------------------------------------------------

/// Reader for replaying events from a binlog file.
pub struct BinlogReader {
    path: PathBuf,
}

impl BinlogReader {
    /// Open an existing binlog file for reading.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if !path.exists() {
            bail!("binlog not found: {}", path.display());
        }
        Ok(Self { path })
    }

    /// Read all events from the binlog.
    pub fn read_all(&self) -> Result<Vec<OpEvent>> {
        let file = File::open(&self.path)
            .with_context(|| format!("failed to open binlog: {}", self.path.display()))?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();

        for (line_number, line_result) in reader.lines().enumerate() {
            let line = line_result
                .with_context(|| format!("failed to read binlog line {}", line_number + 1))?;

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let event: OpEvent = serde_json::from_str(trimmed).with_context(|| {
                format!("corrupted binlog at line {}: invalid JSON", line_number + 1)
            })?;

            events.push(event);
        }

        Ok(events)
    }

    /// Read events starting after a given op_id (for replay from a snapshot).
    pub fn read_after(&self, after_op_id: &str) -> Result<Vec<OpEvent>> {
        let all = self.read_all()?;
        let start_index = all
            .iter()
            .position(|e| e.op_id == after_op_id)
            .map(|i| i + 1)
            .unwrap_or(0);
        Ok(all.into_iter().skip(start_index).collect())
    }

    /// Read events up to and including a given op_id.
    pub fn read_until(&self, until_op_id: &str) -> Result<Vec<OpEvent>> {
        let all = self.read_all()?;
        let end_index = all
            .iter()
            .position(|e| e.op_id == until_op_id)
            .ok_or_else(|| anyhow!("op_id '{}' not found in binlog", until_op_id))?;
        Ok(all.into_iter().take(end_index + 1).collect())
    }

    /// Validate the parent_id chain: each event's parent_id should match the
    /// previous event's op_id (except the first event).
    pub fn validate_lineage(&self) -> Result<Vec<String>> {
        let events = self.read_all()?;
        let mut warnings = Vec::new();

        for (i, event) in events.iter().enumerate() {
            if i == 0 {
                if event.parent_id.is_some() {
                    warnings.push(format!(
                        "first event '{}' has parent_id but is the first in the log",
                        event.op_id
                    ));
                }
                continue;
            }

            let expected_parent = &events[i - 1].op_id;
            match &event.parent_id {
                Some(parent) if parent != expected_parent => {
                    warnings.push(format!(
                        "event '{}' has parent_id '{}' but expected '{}' (branch point or corruption)",
                        event.op_id, parent, expected_parent
                    ));
                }
                None => {
                    warnings.push(format!(
                        "event '{}' at position {} has no parent_id",
                        event.op_id, i
                    ));
                }
                _ => {}
            }
        }

        Ok(warnings)
    }

    /// Validate hash chain integrity (if events have event_hash/prev_event_hash).
    pub fn validate_hash_chain(&self) -> Result<Vec<String>> {
        let events = self.read_all()?;
        let mut warnings = Vec::new();

        for (i, event) in events.iter().enumerate() {
            if i == 0 {
                continue;
            }

            if let (Some(prev_hash), Some(expected_prev)) =
                (&events[i - 1].event_hash, &event.prev_event_hash)
                && prev_hash != expected_prev
            {
                warnings.push(format!(
                    "hash chain broken at event '{}': prev_event_hash '{}' != previous event_hash '{}'",
                    event.op_id, expected_prev, prev_hash
                ));
            }
        }

        Ok(warnings)
    }

    /// Return the last event in the binlog (the current tip).
    pub fn tip(&self) -> Result<Option<OpEvent>> {
        let events = self.read_all()?;
        Ok(events.into_iter().last())
    }

    /// Count events in the binlog.
    pub fn count(&self) -> Result<usize> {
        let file = File::open(&self.path)
            .with_context(|| format!("failed to open binlog: {}", self.path.display()))?;
        let reader = BufReader::new(file);
        let count = reader
            .lines()
            .map_while(Result::ok)
            .filter(|l| !l.trim().is_empty())
            .count();
        Ok(count)
    }
}

// ---------------------------------------------------------------------------
// Snapshot manifest
// ---------------------------------------------------------------------------

/// Index entry for a materialized snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotEntry {
    /// The op_id at which this snapshot was taken.
    pub op_id: String,
    /// Relative path to the snapshot file within the session directory.
    pub file_name: String,
    /// SHA-256 hash of the snapshot file.
    pub file_hash: String,
    /// Timestamp of snapshot creation.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Number of events from base to this snapshot.
    pub event_count: usize,
}

/// Manifest tracking all snapshots for a session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotManifest {
    pub session_id: String,
    pub entries: Vec<SnapshotEntry>,
}

impl SnapshotManifest {
    /// Create a new empty manifest.
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            entries: Vec::new(),
        }
    }

    /// Load a manifest from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read snapshot manifest: {}", path.display()))?;
        serde_json::from_str(&content).context("failed to parse snapshot manifest")
    }

    /// Save the manifest to disk.
    pub fn save(&self, path: &Path) -> Result<()> {
        let content =
            serde_json::to_string_pretty(self).context("failed to serialize snapshot manifest")?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, content)
            .with_context(|| format!("failed to write snapshot manifest: {}", path.display()))
    }

    /// Add a new snapshot entry.
    pub fn add_entry(&mut self, entry: SnapshotEntry) {
        self.entries.push(entry);
    }

    /// Find the nearest snapshot at or before the given op_id.
    /// Returns the entry whose op_id appears earliest in the event order
    /// but is closest to the target.
    pub fn nearest_snapshot(
        &self,
        target_op_id: &str,
        event_order: &[String],
    ) -> Option<&SnapshotEntry> {
        let target_pos = event_order.iter().position(|id| id == target_op_id)?;
        self.entries
            .iter()
            .filter_map(|entry| {
                event_order
                    .iter()
                    .position(|id| id == &entry.op_id)
                    .filter(|&pos| pos <= target_pos)
                    .map(|pos| (pos, entry))
            })
            .max_by_key(|(pos, _)| *pos)
            .map(|(_, entry)| entry)
    }
}

// ---------------------------------------------------------------------------
// Branch metadata
// ---------------------------------------------------------------------------

/// Branch pointer within a session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BranchInfo {
    /// Branch name (e.g. "main", "alt-scenario").
    pub name: String,
    /// The op_id at the tip of this branch.
    pub tip_op_id: Option<String>,
    /// The op_id where this branch forked from its parent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fork_point: Option<String>,
    /// Optional human-readable label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Creation timestamp.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Container for all branch metadata in a session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BranchesFile {
    pub branches: Vec<BranchInfo>,
}

impl BranchesFile {
    pub fn new() -> Self {
        Self {
            branches: vec![BranchInfo {
                name: "main".to_string(),
                tip_op_id: None,
                fork_point: None,
                label: None,
                created_at: chrono::Utc::now(),
            }],
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("failed to read branches file: {}", path.display()))?;
        serde_json::from_str(&content).context("failed to parse branches file")
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)
            .with_context(|| format!("failed to write branches file: {}", path.display()))
    }

    pub fn get_branch(&self, name: &str) -> Option<&BranchInfo> {
        self.branches.iter().find(|b| b.name == name)
    }

    pub fn get_branch_mut(&mut self, name: &str) -> Option<&mut BranchInfo> {
        self.branches.iter_mut().find(|b| b.name == name)
    }

    pub fn add_branch(&mut self, info: BranchInfo) {
        self.branches.push(info);
    }
}

impl Default for BranchesFile {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::events::{Actor, OpEvent, OpKind};
    use serde_json::json;
    use tempfile::TempDir;

    fn test_event(session_id: &str, parent_id: Option<&str>) -> OpEvent {
        OpEvent::new(
            session_id.to_string(),
            parent_id.map(|s| s.to_string()),
            Actor {
                id: "test".to_string(),
                run_id: None,
                source: "test".to_string(),
            },
            OpKind::edit_batch(),
            json!({"cell": "A1", "value": 42}),
        )
    }

    #[test]
    fn binlog_append_and_read() {
        let tmp = TempDir::new().unwrap();
        let log_path = tmp.path().join("events.jsonl");

        let writer = BinlogWriter::open(&log_path).unwrap();
        let event1 = test_event("sess1", None);
        let event2 = test_event("sess1", Some(&event1.op_id));

        writer.append(&event1).unwrap();
        writer.append(&event2).unwrap();

        let reader = BinlogReader::open(&log_path).unwrap();
        let events = reader.read_all().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].op_id, event1.op_id);
        assert_eq!(events[1].op_id, event2.op_id);
    }

    #[test]
    fn binlog_read_after() {
        let tmp = TempDir::new().unwrap();
        let log_path = tmp.path().join("events.jsonl");

        let writer = BinlogWriter::open(&log_path).unwrap();
        let e1 = test_event("sess1", None);
        let e2 = test_event("sess1", Some(&e1.op_id));
        let e3 = test_event("sess1", Some(&e2.op_id));

        writer.append(&e1).unwrap();
        writer.append(&e2).unwrap();
        writer.append(&e3).unwrap();

        let reader = BinlogReader::open(&log_path).unwrap();
        let after = reader.read_after(&e1.op_id).unwrap();
        assert_eq!(after.len(), 2);
        assert_eq!(after[0].op_id, e2.op_id);
    }

    #[test]
    fn binlog_lineage_validation() {
        let tmp = TempDir::new().unwrap();
        let log_path = tmp.path().join("events.jsonl");

        let writer = BinlogWriter::open(&log_path).unwrap();
        let e1 = test_event("sess1", None);
        let e2 = test_event("sess1", Some(&e1.op_id));
        // e3 has wrong parent (simulates branch or corruption)
        let e3 = test_event("sess1", Some("wrong_parent"));

        writer.append(&e1).unwrap();
        writer.append(&e2).unwrap();
        writer.append(&e3).unwrap();

        let reader = BinlogReader::open(&log_path).unwrap();
        let warnings = reader.validate_lineage().unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("wrong_parent"));
    }

    #[test]
    fn snapshot_manifest_nearest() {
        let mut manifest = SnapshotManifest::new("sess1".to_string());
        manifest.add_entry(SnapshotEntry {
            op_id: "op_001".to_string(),
            file_name: "snap_001.xlsx".to_string(),
            file_hash: "sha256:aaa".to_string(),
            created_at: chrono::Utc::now(),
            event_count: 1,
        });
        manifest.add_entry(SnapshotEntry {
            op_id: "op_005".to_string(),
            file_name: "snap_005.xlsx".to_string(),
            file_hash: "sha256:bbb".to_string(),
            created_at: chrono::Utc::now(),
            event_count: 5,
        });

        let order: Vec<String> = (1..=10).map(|i| format!("op_{:03}", i)).collect();

        // Target at op_007 → nearest snapshot is op_005
        let nearest = manifest.nearest_snapshot("op_007", &order);
        assert_eq!(nearest.unwrap().op_id, "op_005");

        // Target at op_003 → nearest snapshot is op_001
        let nearest = manifest.nearest_snapshot("op_003", &order);
        assert_eq!(nearest.unwrap().op_id, "op_001");

        // Target at op_001 → exact match
        let nearest = manifest.nearest_snapshot("op_001", &order);
        assert_eq!(nearest.unwrap().op_id, "op_001");
    }

    #[test]
    fn branches_file_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("branches.json");

        let mut bf = BranchesFile::new();
        bf.add_branch(BranchInfo {
            name: "alt-scenario".to_string(),
            tip_op_id: Some("op_abc".to_string()),
            fork_point: Some("op_005".to_string()),
            label: Some("Alternative Scenario".to_string()),
            created_at: chrono::Utc::now(),
        });

        bf.save(&path).unwrap();
        let loaded = BranchesFile::load(&path).unwrap();
        assert_eq!(loaded.branches.len(), 2);
        assert_eq!(
            loaded
                .get_branch("alt-scenario")
                .unwrap()
                .tip_op_id
                .as_deref(),
            Some("op_abc")
        );
    }
}
