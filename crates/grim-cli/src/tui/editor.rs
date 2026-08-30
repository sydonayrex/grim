//! External editor ($EDITOR / $VISUAL) integration.
//!
//! Borrowed from the opencode-dev pattern: suspend the TUI, spawn the user's
//! preferred editor on a temp file, and resume with the edited content.
//! Activated via `/edit` in the chat input.

use std::env;
use std::ffi::OsString;
use std::io::Write;
use std::process::Command;

use grim_tensor::error::{Error, Result};

/// Open the user's external editor with the given initial content.
///
/// Returns the edited content on success, or `None` if the user has no
/// editor configured. The TUI should call this only when stdout is a terminal.
pub fn open_editor(content: &str) -> Result<Option<String>> {
    let editor = env::var_os("VISUAL")
        .or_else(|| env::var_os("EDITOR"))
        .or_else(windows_notepad_fallback);
    let editor = match editor {
        Some(e) => e,
        None => return Ok(None),
    };

    let mut tmp = env::temp_dir();
    tmp.push(format!("grim_tui_edit_{}.md", std::process::id()));
    let path = tmp;

    // Write initial content to the temp file.
    {
        let mut file = std::fs::File::create(&path)
            .map_err(|e| Error::Backend(format!("failed to create temp file: {e}")))?;
        file.write_all(content.as_bytes())
            .map_err(|e| Error::Backend(format!("failed to write temp file: {e}")))?;
    }

    // Parse the editor command (supports "code --wait" style multi-token).
    let editor_str = editor.to_string_lossy().to_string();
    let parts: Vec<&str> = editor_str.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(None);
    }

    // Suspend the TUI and spawn the editor in the foreground.
    let _ = ratatui::restore();
    let status = Command::new(parts[0])
        .args(&parts[1..])
        .arg(&path)
        .status()
        .map_err(|e| Error::Backend(format!("failed to spawn editor: {e}")));

    // Re-enter raw mode regardless of editor outcome.
    let mut term = ratatui::init();
    let _ = term.clear();

    let status = status?;
    if !status.success() {
        // Editor exited non-zero (e.g. user aborted); clean up and return None.
        let _ = std::fs::remove_file(&path);
        return Ok(None);
    }

    // Read back the edited content.
    let edited = std::fs::read_to_string(&path)
        .map_err(|e| Error::Backend(format!("failed to read temp file: {e}")))?;
    let _ = std::fs::remove_file(&path);

    // Trim trailing newline that many editors add.
    let edited = edited.trim_end_matches('\n').trim_end_matches('\r').to_string();
    if edited.is_empty() {
        Ok(None)
    } else {
        Ok(Some(edited))
    }
}

/// On Windows, fall back to notepad if neither VISUAL nor EDITOR is set.
#[cfg(windows)]
fn windows_notepad_fallback() -> Option<OsString> {
    Some(OsString::from("notepad.exe"))
}

#[cfg(not(windows))]
fn windows_notepad_fallback() -> Option<OsString> {
    None
}
