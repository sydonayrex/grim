//! Text clipboard helpers: arboard first (requires display server), then an
//! OSC52 escape-sequence fallback so the terminal emulator can set the
//! clipboard itself.

use std::io::Write;

/// Copy text to the system clipboard. When arboard is unavailable (headless
/// or missing display server) falls back to writing an OSC52 sequence to stdout
/// so the terminal emulator can place the text in the system clipboard.
pub fn copy_to_clipboard(text: &str) {
    // Try arboard first (requires display server).
    if let Ok(mut cb) = arboard::Clipboard::new() {
        if cb.set_text(text.to_string()).is_ok() {
            return;
        }
    }
    // Fallback: OSC52 — encode as base64 and emit "\x1b]52;c;{}\x07".
    let encoded = base64_encoded(text);
    let seq = format!("\x1b]52;c;{}\x07", encoded);
    let _ = std::io::stdout().write_all(seq.as_bytes());
    let _ = std::io::stdout().flush();
}

fn base64_encoded(input: &str) -> String {
    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, input.as_bytes())
}

/// Emit OSC52 copy without attempting arboard (useful for tests).
pub fn osc52_copy(text: &str) -> String {
    let encoded = base64_encoded(text);
    format!("\x1b]52;c;{}\x07", encoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn osc52_encoding_matches_spec() {
        assert_eq!(osc52_copy("hi"), "\x1b]52;c;aGk=\x07");
    }
}
