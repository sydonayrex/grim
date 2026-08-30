//! Markdown → ratatui rendering for assistant messages.
//!
//! Uses pulldown-cmark to parse CommonMark and syntect for syntax
//! highlighting inside fenced code blocks. Produces styled Lines that
//! the transcript widget can render directly.

use pulldown_cmark::{Event, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

const SYNTAX_THEME: &str = "base16-ocean.dark";

/// Render a markdown string to styled Lines.
pub fn render_markdown(src: &str) -> Vec<Line<'static>> {
    let parser = Parser::new(src);
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut current_line: Vec<Span<'static>> = Vec::new();
    let mut in_code_block = false;
    let mut code_lang = String::new();
    let mut code_buf = String::new();
    let mut bold = false;
    let mut italic = false;
    let mut heading_level = 0u8;

    let flush_line = |current: &mut Vec<Span<'static>>, lines: &mut Vec<Line<'static>>| {
        if current.is_empty() {
            lines.push(Line::from(""));
        } else {
            lines.push(Line::from(std::mem::take(current)));
        }
    };

    for event in parser {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                heading_level = level as u8;
            }
            Event::End(TagEnd::Heading(_)) => {
                let style = match heading_level {
                    1 => Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                    2 => Style::default().fg(Color::Cyan),
                    _ => Style::default().fg(Color::White),
                };
                let text: String = current_line.iter().map(|s| s.content.as_ref()).collect();
                current_line = vec![Span::styled(text, style)];
                flush_line(&mut current_line, &mut lines);
                heading_level = 0;
            }
            Event::Start(Tag::Paragraph) => {}
            Event::End(TagEnd::Paragraph) => {
                flush_line(&mut current_line, &mut lines);
            }
            Event::Start(Tag::CodeBlock(kind)) => {
                in_code_block = true;
                code_lang = match kind {
                    pulldown_cmark::CodeBlockKind::Fenced(lang) => lang.to_string(),
                    pulldown_cmark::CodeBlockKind::Indented => String::new(),
                };
                code_buf.clear();
            }
            Event::End(TagEnd::CodeBlock) => {
                in_code_block = false;
                lines.push(Line::from(vec![Span::styled(
                    format!("  ┌── [{}]", if code_lang.is_empty() { "code" } else { &code_lang }),
                    Style::default().fg(Color::DarkGray),
                )]));
                for hl_line in highlight_code(&code_buf, &code_lang) {
                    lines.push(hl_line);
                }
                lines.push(Line::from(vec![Span::styled(
                    "  └────".to_string(),
                    Style::default().fg(Color::DarkGray),
                )]));
            }
            Event::Start(Tag::Emphasis) => italic = true,
            Event::End(TagEnd::Emphasis) => italic = false,
            Event::Start(Tag::Strong) => bold = true,
            Event::End(TagEnd::Strong) => bold = false,
            Event::Text(text) => {
                if in_code_block {
                    code_buf.push_str(&text);
                } else {
                    let mut style = Style::default().fg(Color::White);
                    if bold {
                        style = style.add_modifier(Modifier::BOLD);
                    }
                    if italic {
                        style = style.add_modifier(Modifier::ITALIC);
                    }
                    current_line.push(Span::styled(text.to_string(), style));
                }
            }
            Event::Code(inline) => {
                current_line.push(Span::styled(
                    inline.to_string(),
                    Style::default().fg(Color::Yellow),
                ));
            }
            Event::SoftBreak => {
                flush_line(&mut current_line, &mut lines);
            }
            Event::HardBreak => {
                flush_line(&mut current_line, &mut lines);
            }
            _ => {}
        }
    }
    flush_line(&mut current_line, &mut lines);
    lines
}

/// Highlight a code string using syntect. Falls back to plain white on error.
fn highlight_code(code: &str, lang: &str) -> Vec<Line<'static>> {
    use syntect::easy::HighlightLines;
    use syntect::highlighting::{Style as SynStyle, ThemeSet};
    use syntect::parsing::SyntaxSet;

    let ss = SyntaxSet::load_defaults_newlines();
    let ts = ThemeSet::load_defaults();
    let syntax = ss
        .find_syntax_by_token(lang)
        .unwrap_or_else(|| ss.find_syntax_plain_text());
    let mut h = HighlightLines::new(syntax, &ts.themes[SYNTAX_THEME]);

    let mut out = Vec::new();
    for line in code.lines() {
        let ranges: Vec<(SynStyle, &str)> = match h.highlight_line(line, &ss) {
            Ok(r) => r,
            Err(_) => vec![(SynStyle::default(), line)],
        };
        let mut spans = vec![Span::styled(
            "  │ ".to_string(),
            Style::default().fg(Color::DarkGray),
        )];
        for (style, text) in ranges {
            let fg = Color::Rgb(style.foreground.r, style.foreground.g, style.foreground.b);
            spans.push(Span::styled(text.to_string(), Style::default().fg(fg)));
        }
        out.push(Line::from(spans));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_plain_text() {
        let lines = render_markdown("hello world");
        assert!(!lines.is_empty());
        let text: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(text.contains("hello world"));
    }

    #[test]
    fn renders_code_block_with_frame() {
        let lines = render_markdown("```rust\nfn main() {}\n```");
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        // Has frame markers.
        assert!(joined.contains("┌──"));
        assert!(joined.contains("└────"));
        // Has language label.
        assert!(joined.contains("rust"));
        // Has content.
        assert!(joined.contains("fn main"));
    }

    #[test]
    fn renders_bold_and_inline_code() {
        let lines = render_markdown("**bold** and `code`");
        let joined: String = lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(joined.contains("bold"));
        assert!(joined.contains("code"));
    }
}
