//! File-store checkpoints: before a mutating tool touches a file, snapshot
//! its prior bytes under `$XDG_DATA_HOME/grim/checkpoints/<hash-of-project>/`.
//! Works in non-git directories. One manifest per checkpoint, id = unix nanos.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const MAX_FILE_BYTES: u64 = 1_000_000;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FileSnapshot {
    pub path: PathBuf, // absolute original path
    #[serde(with = "base64_bytes")]
    pub prior_content: Option<Vec<u8>>, // None = file did not exist before
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Checkpoint {
    pub id: u64,
    pub ts: String,
    pub tool: String,
    pub files: Vec<FileSnapshot>,
}

/// In-memory snapshot; persisted only when the tool call succeeds.
pub struct PendingCheckpoint {
    checkpoint: Checkpoint,
}

pub struct CheckpointStore {
    dir: PathBuf,
}

mod base64_bytes {
    use base64::Engine;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(bytes) => {
                s.serialize_some(&base64::engine::general_purpose::STANDARD.encode(bytes))
            }
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<u8>>, D::Error> {
        let opt: Option<String> = Option::deserialize(d)?;
        match opt {
            None => Ok(None),
            Some(enc) => base64::engine::general_purpose::STANDARD
                .decode(enc)
                .map(Some)
                .map_err(serde::de::Error::custom),
        }
    }
}

impl CheckpointStore {
    pub fn open(project_dir: &Path) -> Self {
        let key = project_dir
            .canonicalize()
            .unwrap_or_else(|_| project_dir.to_path_buf());
        let hash = Sha256::digest(key.to_string_lossy().as_bytes());
        let dir = crate::tui::paths::data_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("checkpoints")
            .join(format!("{hash:x}"));
        std::fs::create_dir_all(&dir).ok();
        Self { dir }
    }

    /// Only file-mutating tools with a single known target are checkpointed.
    /// `run_command` effects are unknowable; read-only tools need no rollback.
    fn tool_target(call: &grim_format::ToolCallMsg) -> Option<String> {
        if !matches!(call.name.as_str(), "write_file" | "edit_file") {
            return None;
        }
        let args: serde_json::Value = serde_json::from_str(&call.arguments).ok()?;
        args.get("path")?.as_str().map(str::to_string)
    }

    /// Snapshot every file the tool is about to touch. In-memory only until
    /// `persist`. `None` for non-checkpointable tools or oversized files.
    pub fn capture(
        &self,
        call: &grim_format::ToolCallMsg,
        sandbox: &crate::tui::tools::Sandbox,
    ) -> Option<PendingCheckpoint> {
        let rel = Self::tool_target(call)?;
        let abs = sandbox.resolve(&rel).ok()?;
        let prior = if abs.exists() {
            let meta = std::fs::metadata(&abs).ok()?;
            if meta.len() > MAX_FILE_BYTES {
                return None; // too big to snapshot; tool still runs
            }
            Some(std::fs::read(&abs).ok()?)
        } else {
            None
        };
        let now = chrono::Utc::now();
        Some(PendingCheckpoint {
            checkpoint: Checkpoint {
                id: now.timestamp_nanos_opt().unwrap_or(0).max(0) as u64,
                ts: now.to_rfc3339(),
                tool: call.name.clone(),
                files: vec![FileSnapshot {
                    path: abs,
                    prior_content: prior,
                }],
            },
        })
    }

    pub fn persist(&self, pending: PendingCheckpoint) -> Option<Checkpoint> {
        let path = self.dir.join(format!("{}.json", pending.checkpoint.id));
        let json = serde_json::to_string(&pending.checkpoint).ok()?;
        std::fs::write(path, json).ok()?;
        Some(pending.checkpoint)
    }

    /// Newest first.
    pub fn list(&self) -> Vec<Checkpoint> {
        let mut out: Vec<Checkpoint> = std::fs::read_dir(&self.dir)
            .map(|rd| {
                rd.flatten()
                    .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
                    .filter_map(|e| std::fs::read_to_string(e.path()).ok())
                    .filter_map(|t| serde_json::from_str::<Checkpoint>(&t).ok())
                    .collect()
            })
            .unwrap_or_default();
        out.sort_by(|a, b| b.id.cmp(&a.id));
        out.truncate(200);
        out
    }

    /// Restore every file to its pre-tool state; a file that did not exist
    /// before is deleted. The manifest is consumed on success.
    pub fn restore(&self, id: u64) -> Result<String, String> {
        let cp = self
            .list()
            .into_iter()
            .find(|c| c.id == id)
            .ok_or_else(|| format!("checkpoint {id} not found"))?;
        let mut restored = 0usize;
        for f in &cp.files {
            match &f.prior_content {
                Some(bytes) => {
                    if let Some(parent) = f.path.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                    }
                    std::fs::write(&f.path, bytes).map_err(|e| e.to_string())?;
                    restored += 1;
                }
                None => {
                    if f.path.exists() {
                        std::fs::remove_file(&f.path).map_err(|e| e.to_string())?;
                        restored += 1;
                    }
                }
            }
        }
        std::fs::remove_file(self.dir.join(format!("{id}.json"))).ok();
        Ok(format!(
            "restored {restored} file(s) from {} checkpoint #{}",
            cp.tool, cp.id
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::tools::Sandbox;

    fn write_tool(file: &str, content: &str) -> grim_format::ToolCallMsg {
        grim_format::ToolCallMsg {
            id: "c1".into(),
            name: "write_file".into(),
            arguments: format!(
                "{{\"path\":\"{file}\",\"content\":{}}}",
                serde_json::to_string(content).unwrap()
            ),
        }
    }

    #[test]
    fn capture_persist_restore_roundtrip() {
        let _guard = crate::tui::paths::env_lock();
        let proj = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("XDG_DATA_HOME", data.path()) };
        std::fs::write(proj.path().join("src.txt"), "original").unwrap();

        let sandbox = Sandbox::new(proj.path().to_path_buf());
        let store = CheckpointStore::open(proj.path());
        let cp = store
            .persist(store.capture(&write_tool("src.txt", "replaced"), &sandbox).unwrap())
            .unwrap();

        std::fs::write(proj.path().join("src.txt"), "replaced").unwrap();
        let summary = store.restore(cp.id).unwrap();
        assert!(summary.contains("1 file"));
        assert_eq!(
            std::fs::read_to_string(proj.path().join("src.txt")).unwrap(),
            "original"
        );
        unsafe { std::env::remove_var("XDG_DATA_HOME") };
    }

    #[test]
    fn restore_deletes_file_that_did_not_exist() {
        let _guard = crate::tui::paths::env_lock();
        let proj = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("XDG_DATA_HOME", data.path()) };
        let sandbox = Sandbox::new(proj.path().to_path_buf());
        let store = CheckpointStore::open(proj.path());
        let cp = store
            .persist(
                store
                    .capture(&write_tool("brand_new.txt", "hello"), &sandbox)
                    .unwrap(),
            )
            .unwrap();
        // Simulate the tool succeeding: the file now exists.
        std::fs::write(proj.path().join("brand_new.txt"), "hello").unwrap();
        assert!(proj.path().join("brand_new.txt").exists());
        store.restore(cp.id).unwrap();
        assert!(!proj.path().join("brand_new.txt").exists());
        unsafe { std::env::remove_var("XDG_DATA_HOME") };
    }

    #[test]
    fn read_only_tools_are_not_captured() {
        let proj = tempfile::tempdir().unwrap();
        let sandbox = Sandbox::new(proj.path().to_path_buf());
        let store = CheckpointStore::open(proj.path());
        let call = grim_format::ToolCallMsg {
            id: "c2".into(),
            name: "read_file".into(),
            arguments: "{\"path\":\"x\"}".into(),
        };
        assert!(store.capture(&call, &sandbox).is_none());
    }

    #[test]
    fn list_newest_first() {
        let _guard = crate::tui::paths::env_lock();
        let proj = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("XDG_DATA_HOME", data.path()) };
        let sandbox = Sandbox::new(proj.path().to_path_buf());
        let store = CheckpointStore::open(proj.path());
        let a = store
            .persist(store.capture(&write_tool("a.txt", "1"), &sandbox).unwrap())
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = store
            .persist(store.capture(&write_tool("b.txt", "2"), &sandbox).unwrap())
            .unwrap();
        let list = store.list();
        assert_eq!(list[0].id, b.id);
        assert_eq!(list[1].id, a.id);
        unsafe { std::env::remove_var("XDG_DATA_HOME") };
    }
}
