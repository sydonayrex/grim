//! Queue of user messages typed while the worker is generating.
//! Drained automatically on `WorkerEvent::TurnComplete`.

#[derive(Default)]
pub struct MessageQueue {
    items: std::collections::VecDeque<String>,
}

impl MessageQueue {
    pub fn push(&mut self, text: impl Into<String>) {
        self.items.push_back(text.into());
    }

    pub fn pop(&mut self) -> Option<String> {
        self.items.pop_front()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn items(&self) -> impl Iterator<Item = &str> {
        self.items.iter().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_order() {
        let mut q = MessageQueue::default();
        q.push("first");
        q.push("second");
        assert_eq!(q.len(), 2);
        assert_eq!(q.pop().as_deref(), Some("first"));
        assert_eq!(q.pop().as_deref(), Some("second"));
        assert!(q.is_empty());
    }

    #[test]
    fn pop_empty_is_none() {
        let mut q = MessageQueue::default();
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn clear_drops_all() {
        let mut q = MessageQueue::default();
        q.push("a");
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn items_iterates_without_consuming() {
        let mut q = MessageQueue::default();
        q.push("x");
        assert_eq!(q.items().count(), 1);
        assert_eq!(q.len(), 1);
    }
}
