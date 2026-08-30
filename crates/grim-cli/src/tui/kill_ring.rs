//! Emacs kill-ring buffer for cut and yank cycling.
//!
//! Stores killed text entries. Consecutive kills can accumulate into a single
//! entry. Supports peek and rotate for yank and yank-pop.

/// Options for pushing text into the ring.
#[derive(Debug, Clone, Copy)]
pub struct KillPushOpts {
    /// When accumulating, prepend the new text before the existing entry.
    pub prepend: bool,
    /// Merge into the last entry instead of creating a new one.
    pub accumulate: bool,
}

/// Ring buffer for killed text. Last entry is the most recent kill.
#[derive(Debug, Clone, Default)]
pub struct KillRing {
    ring: Vec<String>,
}

impl KillRing {
    /// Create an empty ring.
    pub fn new() -> Self {
        Self { ring: Vec::new() }
    }

    /// Push `text` into the ring.
    ///
    /// When `opts.accumulate` is true and the ring is non-empty, the new text
    /// is merged into the last entry (prepend or append based on `opts.prepend`).
    /// Otherwise a new entry is created. Empty text is ignored.
    pub fn push(&mut self, text: String, opts: KillPushOpts) {
        if text.is_empty() {
            return;
        }
        if opts.accumulate && !self.ring.is_empty() {
            if let Some(last) = self.ring.last_mut() {
                if opts.prepend {
                    *last = format!("{text}{last}");
                } else {
                    last.push_str(&text);
                }
                return;
            }
        }
        self.ring.push(text);
        if self.ring.len() > 32 {
            self.ring.remove(0);
        }
    }

    /// Most recent entry, if any.
    pub fn peek(&self) -> Option<&str> {
        self.ring.last().map(|s| s.as_str())
    }

    /// Move the last entry to the front, cycling for yank-pop.
    pub fn rotate(&mut self) {
        if self.ring.len() > 1 {
            if let Some(last) = self.ring.pop() {
                self.ring.insert(0, last);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_peek() {
        let mut r = KillRing::new();
        r.push(
            "hello".into(),
            KillPushOpts {
                prepend: false,
                accumulate: false,
            },
        );
        assert_eq!(r.peek(), Some("hello"));
    }

    #[test]
    fn accumulate_merges() {
        let mut r = KillRing::new();
        r.push(
            "foo".into(),
            KillPushOpts {
                prepend: false,
                accumulate: false,
            },
        );
        r.push(
            "bar".into(),
            KillPushOpts {
                prepend: false,
                accumulate: true,
            },
        );
        assert_eq!(r.peek(), Some("foobar"));
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn rotate_cycles() {
        let mut r = KillRing::new();
        r.push(
            "a".into(),
            KillPushOpts {
                prepend: false,
                accumulate: false,
            },
        );
        r.push(
            "b".into(),
            KillPushOpts {
                prepend: false,
                accumulate: false,
            },
        );
        r.rotate();
        assert_eq!(r.peek(), Some("a"));
    }
}
