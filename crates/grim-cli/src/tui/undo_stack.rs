//! Generic bounded undo stack.
//!
//! Stores clones of state snapshots. Oldest entries are dropped when the
//! capacity is exceeded. No redo stack: one undo direction is enough for chat.

// Allow dead_code for len/clear which are used by callers and tests.
#![allow(unused)]

/// Bounded stack of cloned snapshots.
#[derive(Debug, Clone)]
pub struct UndoStack<S: Clone> {
    stack: Vec<S>,
    limit: usize,
}

impl<S: Clone> UndoStack<S> {
    /// Create a stack with capacity `limit`. Oldest entries are dropped first.
    pub fn new(limit: usize) -> Self {
        Self {
            stack: Vec::new(),
            limit: limit.max(1),
        }
    }

    /// Push a clone of `state` onto the stack.
    pub fn push(&mut self, state: S) {
        self.stack.push(state);
        if self.stack.len() > self.limit {
            self.stack.remove(0);
        }
    }

    /// Pop and return the most recent snapshot, if any.
    pub fn pop(&mut self) -> Option<S> {
        self.stack.pop()
    }

    /// Remove all snapshots.
    pub fn clear(&mut self) {
        self.stack.clear();
    }

    pub fn len(&self) -> usize {
        self.stack.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

impl<S: Clone> Default for UndoStack<S> {
    fn default() -> Self {
        Self::new(64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_roundtrip() {
        let mut s: UndoStack<Vec<char>> = UndoStack::new(8);
        s.push(vec!['a', 'b']);
        assert_eq!(s.pop(), Some(vec!['a', 'b']));
        assert_eq!(s.pop(), None);
    }

    #[test]
    fn bounded_drop_oldest() {
        let mut s: UndoStack<i32> = UndoStack::new(2);
        s.push(1);
        s.push(2);
        s.push(3);
        assert_eq!(s.len(), 2);
    }
}
