//! Central session store: autosaved transcripts under
//! `$XDG_DATA_HOME/grim/sessions/`, titled from the first user prompt.

use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct SessionMeta {
    pub path: PathBuf,
    pub title: String,
    pub modified: u64, // unix secs
}

pub fn sessions_dir() -> Option<PathBuf> {
    crate::tui::paths::data_dir().map(|d| d.join("sessions"))
}

/// Filename-safe slug of the first user prompt (max 40 chars).
pub fn slugify(text: &str) -> String {
    let mapped: String = text
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = mapped.trim_matches('-');
    let mut out: String = trimmed.chars().take(40).collect();
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "session".into()
    } else {
        out
    }
}

/// Store path for a new session, titled from its first prompt.
pub fn new_session_path(first_prompt: &str) -> Option<PathBuf> {
    let dir = sessions_dir()?;
    let ts = chrono::Local::now().format("%Y%m%d-%H%M%S");
    Some(dir.join(format!("{ts}-{}.jsonl", slugify(first_prompt))))
}

/// Rewrite the session file from the current transcript (JSONL export format).
pub fn autosave(
    transcript: &crate::tui::transcript::Transcript,
    path: &Path,
) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    crate::tui::export_transcript(transcript, &path.to_string_lossy()).map(|_| ())
}

/// Newest-first listing of the central store plus cwd `*.jsonl` files.
pub fn list_sessions() -> Vec<SessionMeta> {
    let mut out: Vec<SessionMeta> = Vec::new();
    for dir in [sessions_dir(), std::env::current_dir().ok()] {
        let Some(dir) = dir else { continue };
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let path = entry.path();
            if !path.extension().map(|x| x == "jsonl").unwrap_or(false) {
                continue;
            }
            let title = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            out.push(SessionMeta { path, title, modified });
        }
    }
    out.sort_by(|a, b| b.modified.cmp(&a.modified).then(b.title.cmp(&a.title)));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_basic() {
        assert_eq!(
            slugify("Fix the ROCm flash-attn bug!"),
            "fix-the-rocm-flash-attn-bug"
        );
    }

    #[test]
    fn slugify_caps_length_and_strips_edges() {
        let s = slugify(
            "--!!  very long prompt that goes on and on and should be truncated at forty chars  !!--",
        );
        assert!(s.len() <= 40);
        assert!(!s.starts_with('-'));
        assert!(!s.ends_with('-'));
    }

    #[test]
    fn slugify_empty_falls_back() {
        assert_eq!(slugify("!!!"), "session");
    }

    #[test]
    fn list_sessions_reads_store_dir() {
        let _guard = crate::tui::paths::env_lock();
        let tmp = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("XDG_DATA_HOME", tmp.path()) };
        let dir = sessions_dir().unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("20260101-000000-old-one.jsonl"), "{}\n").unwrap();
        std::fs::write(dir.join("20260202-000000-new-two.jsonl"), "{}\n").unwrap();
        let list = list_sessions();
        unsafe { std::env::remove_var("XDG_DATA_HOME") };
        assert!(list.len() >= 2);
        assert!(list.iter().any(|s| s.title == "20260202-000000-new-two"));
        assert!(list.iter().any(|s| s.title == "20260101-000000-old-one"));
    }

    #[test]
    fn autosave_writes_exportable_jsonl() {
        let _guard = crate::tui::paths::env_lock();
        let tmp = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("XDG_DATA_HOME", tmp.path()) };
        let mut tr = crate::tui::transcript::Transcript::new();
        tr.push_user("hello session".into());
        let path = new_session_path("hello session").unwrap();
        autosave(&tr, &path).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("hello session"));
        unsafe { std::env::remove_var("XDG_DATA_HOME") };
    }
}
