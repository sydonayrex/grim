//! Skill discovery and loading for the grim TUI.
//!
//! Skills are directories under `~/.agents/skills/<name>/SKILL.md` (default).
//! Each SKILL.md has YAML frontmatter (`---`) with at minimum `name` and
//! `description` fields, followed by the skill body as markdown.
//!
//! When a skill is activated via `/skill <name>`, the full SKILL.md content
//! (body only, frontmatter stripped) is injected as the system prompt so the
//! model adopts the skill's behavioral context for the session.

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A discovered skill loaded from a SKILL.md file.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Short identifier (directory name, e.g. "caveman").
    pub id: String,
    /// Human-readable name from YAML frontmatter `name:` field.
    pub name: String,
    /// One-line description from YAML frontmatter `description:` field.
    pub description: String,
    /// Full filesystem path to the SKILL.md file.
    pub path: PathBuf,
}

// ---------------------------------------------------------------------------
// Default search root
// ---------------------------------------------------------------------------

/// Return the default skills directory: `~/.agents/skills/`.
///
/// Falls back to `$HOME/.agents/skills` on Unix. Returns `None` if the
/// home directory cannot be determined.
pub fn default_skills_dir() -> Option<PathBuf> {
    // Try $HOME env var first; fall back to /home/<USER> on Linux.
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("USER")
                .ok()
                .map(|u| PathBuf::from(format!("/home/{u}")))
        })?;
    Some(home.join(".agents").join("skills"))
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Scan `dir` for `<name>/SKILL.md` entries and parse their frontmatter.
///
/// Silently skips entries that are not directories, lack a SKILL.md, or have
/// malformed frontmatter. Result is sorted alphabetically by `id`.
pub fn discover_skills(dir: &Path) -> Vec<Skill> {
    let mut skills = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return skills,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let skill_md = path.join("SKILL.md");
        if !skill_md.is_file() {
            continue;
        }
        let content = match std::fs::read_to_string(&skill_md) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let id = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if id.is_empty() {
            continue;
        }

        let (name, description) = parse_frontmatter(&content);
        skills.push(Skill {
            id,
            name,
            description,
            path: skill_md,
        });
    }

    skills.sort_by(|a, b| a.id.cmp(&b.id));
    skills
}

/// Search for a skill by name (case-insensitive prefix match on `id` and `name`).
///
/// Returns the first match, or `None` if nothing is found.
pub fn find_skill<'a>(skills: &'a [Skill], query: &str) -> Option<&'a Skill> {
    let q = query.trim().to_lowercase();
    // Exact id match first.
    if let Some(s) = skills.iter().find(|s| s.id.to_lowercase() == q) {
        return Some(s);
    }
    // Prefix match on id.
    if let Some(s) = skills.iter().find(|s| s.id.to_lowercase().starts_with(&q)) {
        return Some(s);
    }
    // Prefix match on name from frontmatter.
    skills
        .iter()
        .find(|s| s.name.to_lowercase().starts_with(&q))
}

/// Read a skill's SKILL.md and return the body (frontmatter stripped).
///
/// The body is everything after the closing `---` of the frontmatter block.
/// If there is no frontmatter, the entire file is returned as-is.
pub fn load_skill_body(skill: &Skill) -> std::io::Result<String> {
    let content = std::fs::read_to_string(&skill.path)?;
    Ok(strip_frontmatter(&content).to_string())
}

// ---------------------------------------------------------------------------
// Frontmatter helpers
// ---------------------------------------------------------------------------

/// Parse YAML frontmatter for `name` and `description` fields.
///
/// Returns `(name, description)` with the directory id as fallback for name
/// and an empty string as fallback for description.
fn parse_frontmatter(content: &str) -> (String, String) {
    // Frontmatter must start with `---` on the very first line.
    if !content.starts_with("---") {
        return (String::new(), String::new());
    }
    // Find the closing `---` after line 1.
    let rest = &content[3..];
    let end = rest.find("\n---").unwrap_or(rest.len());
    let fm = &rest[..end];

    let mut name = String::new();
    let mut description = String::new();
    let mut in_description_block = false;
    let mut desc_lines: Vec<String> = Vec::new();

    for line in fm.lines() {
        if line.starts_with("name:") {
            name = line["name:".len()..].trim().to_string();
            // Strip surrounding quotes if present.
            if (name.starts_with('"') && name.ends_with('"'))
                || (name.starts_with('\'') && name.ends_with('\''))
            {
                name = name[1..name.len() - 1].to_string();
            }
            in_description_block = false;
        } else if line.starts_with("description:") {
            let inline = line["description:".len()..].trim();
            if inline == ">" || inline == "|" {
                // Multi-line block scalar — collect subsequent indented lines.
                in_description_block = true;
                desc_lines.clear();
            } else {
                description = inline.to_string();
                in_description_block = false;
            }
        } else if in_description_block {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                // End of block scalar.
                in_description_block = false;
                description = desc_lines.join(" ");
            } else {
                desc_lines.push(trimmed.to_string());
            }
        }
    }

    // Flush any trailing block-scalar description.
    if in_description_block && description.is_empty() {
        description = desc_lines.join(" ");
    }

    (name, description)
}

/// Return the content of `raw` with YAML frontmatter removed.
///
/// Strips the opening and closing `---` fences and everything between them.
fn strip_frontmatter(raw: &str) -> &str {
    if !raw.starts_with("---") {
        return raw;
    }
    // Skip past the first `---\n`.
    let after_open = match raw.find('\n') {
        Some(i) => &raw[i + 1..],
        None => return raw,
    };
    // Find closing `---`.
    if let Some(close) = after_open.find("\n---") {
        let after_close = &after_open[close + 4..]; // skip `\n---`
        // Skip optional trailing newline(s) after the closing `---`.
        let mut trimmed = after_close;
        while trimmed.starts_with('\n') {
            trimmed = &trimmed[1..];
        }
        trimmed
    } else {
        raw
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_frontmatter_simple() {
        let md = "---\nname: my-skill\ndescription: Does a thing\n---\n\nBody here.";
        let (name, desc) = parse_frontmatter(md);
        assert_eq!(name, "my-skill");
        assert_eq!(desc, "Does a thing");
    }

    #[test]
    fn parse_frontmatter_block_scalar_description() {
        let md = "---\nname: caveman\ndescription: >\n  Ultra-compressed mode.\n  Drops filler.\n---\n\nBody.";
        let (name, desc) = parse_frontmatter(md);
        assert_eq!(name, "caveman");
        assert!(desc.contains("Ultra-compressed"), "got: {desc}");
    }

    #[test]
    fn strip_frontmatter_removes_fence() {
        let md = "---\nname: x\n---\n\nActual content.";
        assert_eq!(strip_frontmatter(md), "Actual content.");
    }

    #[test]
    fn strip_frontmatter_no_fence_passthrough() {
        let md = "No frontmatter here.";
        assert_eq!(strip_frontmatter(md), md);
    }

    #[test]
    fn find_skill_exact_and_prefix() {
        let skills = vec![
            Skill {
                id: "caveman".into(),
                name: "Caveman".into(),
                description: "terse".into(),
                path: PathBuf::from("/fake/caveman/SKILL.md"),
            },
            Skill {
                id: "rust-review".into(),
                name: "Rust Review".into(),
                description: "review rust".into(),
                path: PathBuf::from("/fake/rust-review/SKILL.md"),
            },
        ];
        assert_eq!(find_skill(&skills, "caveman").map(|s| &s.id[..]), Some("caveman"));
        assert_eq!(find_skill(&skills, "cav").map(|s| &s.id[..]), Some("caveman"));
        assert_eq!(find_skill(&skills, "rust").map(|s| &s.id[..]), Some("rust-review"));
        assert!(find_skill(&skills, "xyz").is_none());
    }
}
