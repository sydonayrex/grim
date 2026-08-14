use std::io::{IsTerminal, Write};

/// A minimal, dependency-free conversion progress bar.
///
/// Renders `[stage] [####----] 42% (done/total)` to stderr. When stderr is a
/// terminal the bar updates in place with `\r`; otherwise (piped / logged)
/// each update is printed as its own line so the percentage survives in
/// redirects and logs.
pub struct Progress {
    tty: bool,
    last_pct: isize,
}

impl Progress {
    pub fn new() -> Self {
        let tty = std::io::stderr().is_terminal();
        Self { tty, last_pct: -1 }
    }

    /// Report an update for `stage`. `done` is 1-based within `total`.
    pub fn render(&mut self, stage: &str, done: usize, total: usize) {
        if total == 0 {
            return;
        }
        let pct = (done as f64 * 100.0 / total as f64) as isize;
        let width = 24;
        let filled = ((pct as f64 / 100.0) * width as f64).round() as usize;
        let bar: String = std::iter::repeat('#')
            .take(filled)
            .chain(std::iter::repeat('-').take(width - filled))
            .collect();

        if self.tty {
            // Interactive terminal (TTY): update the exact same line in-place using \r + ANSI clear line
            eprint!("\r\x1B[2K[{stage}] [{bar}] {pct:3}% ({done}/{total})");
            let _ = std::io::stderr().flush();
        } else {
            // Non-interactive / log stream: emit a new distinct line whenever percentage changes
            if pct != self.last_pct {
                eprintln!("[{stage}] [{bar}] {pct:3}% ({done}/{total})");
                self.last_pct = pct;
            }
        }
    }

    /// Terminate the current in-place bar line.
    pub fn finish(&mut self) {
        eprintln!();
    }
}

impl Default for Progress {
    fn default() -> Self {
        Self::new()
    }
}
