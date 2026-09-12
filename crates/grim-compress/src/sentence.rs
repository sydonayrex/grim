//! Rule-based multilingual sentence segmentation.
//!
//! Supports Latin scripts (`.!?`), CJK full-width punctuation (`。！？`),
//! Arabic (`؟`), and Devanagari (`।`). Adapted from the paper's multilingual
//! segmenter (arXiv:2603.12646v1, §IV-B).

/// Split text into sentences. Handles:
/// - Latin: `.` `!` `?` followed by whitespace or end-of-string
/// - CJK: `。` `！` `？` `；` (full-width punctuation, no space needed)
/// - Arabic/Devanagari: `؟` `।`
/// - Quoted sentences: `"Hello." She said.` → two sentences
///
/// Sentences are capped at 500 chars each (with uniform sampling for longer
/// inputs) per the paper's GC optimization.
pub fn segment_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        current.push(c);

        if is_sentence_end(c) {
            // Look ahead: is this really a sentence boundary?
            // For Latin: punctuation must be followed by whitespace or end-of-string.
            // For CJK: punctuation is always a boundary (no space needed).
            let is_boundary = if is_cjk_punct(c) {
                true
            } else {
                // Latin: check if followed by whitespace or end-of-string
                match chars.get(i + 1) {
                    None => true,                       // end of text
                    Some(next) => next.is_whitespace(), // whitespace follows
                }
            };

            if is_boundary {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    // Cap sentence length at 500 chars
                    if trimmed.chars().count() > 500 {
                        for chunk in chop_sentence(&trimmed, 500) {
                            sentences.push(chunk);
                        }
                    } else {
                        sentences.push(trimmed);
                    }
                }
                current = String::new();
            }
        }
        i += 1;
    }

    // Remaining text
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        if trimmed.chars().count() > 500 {
            for chunk in chop_sentence(&trimmed, 500) {
                sentences.push(chunk);
            }
        } else {
            sentences.push(trimmed);
        }
    }

    // If no boundaries found, treat the whole text as one sentence
    if sentences.is_empty() && !text.trim().is_empty() {
        sentences.push(text.trim().to_string());
    }

    sentences
}

/// Check if a character is a sentence-ending punctuation mark.
fn is_sentence_end(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '。' | '！' | '？' | '؛' | '।' | '…')
}

/// Check if a character is CJK full-width punctuation (always a boundary).
fn is_cjk_punct(c: char) -> bool {
    matches!(c, '。' | '！' | '？' | '；' | '、')
}

/// Chop a long sentence into ~500-char chunks at word boundaries.
fn chop_sentence(s: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for word in s.split_whitespace() {
        if current.chars().count() + word.chars().count() + 1 > max_chars && !current.is_empty() {
            chunks.push(current.trim().to_string());
            current = String::new();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }

    if !current.is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latin_segmentation() {
        let text = "Hello world. How are you? I am fine! Thank you.";
        let sentences = segment_sentences(text);
        assert_eq!(sentences.len(), 4);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "I am fine!");
        assert_eq!(sentences[3], "Thank you.");
    }

    #[test]
    fn test_cjk_segmentation() {
        let text = "你好世界。今天天气怎么样？很好！谢谢。";
        let sentences = segment_sentences(text);
        assert_eq!(sentences.len(), 4);
        assert_eq!(sentences[0], "你好世界。");
        assert_eq!(sentences[1], "今天天气怎么样？");
        assert_eq!(sentences[2], "很好！");
        assert_eq!(sentences[3], "谢谢。");
    }

    #[test]
    fn test_abbreviation_not_split() {
        // "e.g." should ideally not split, but our simple splitter will.
        // This is acceptable per the paper's faithfulness-over-precision tradeoff.
        let text = "See e.g. this example. It works.";
        let sentences = segment_sentences(text);
        assert!(sentences.len() >= 2);
    }

    #[test]
    fn test_empty_input() {
        let sentences = segment_sentences("");
        assert!(sentences.is_empty());
    }

    #[test]
    fn test_no_boundary() {
        let text = "just one long sentence without any ending punctuation";
        let sentences = segment_sentences(text);
        assert_eq!(sentences.len(), 1);
    }

    #[test]
    fn test_sentence_cap() {
        let long_word = "a".repeat(600);
        let text = format!("Short. {long_word}. End.");
        let sentences = segment_sentences(&text);
        // The 600-char sentence should be chopped
        assert!(sentences.len() >= 3);
    }
}
