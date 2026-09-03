//! Slash command descriptors, registry, and autocomplete matching.
//!
//! Provides a single source of truth for command metadata, input parsing,
//! and candidate completion during interactive chat sessions.

/// Metadata descriptor for a slash command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSpec {
    /// Command name without leading slash (e.g. "model").
    pub name: &'static str,
    /// Argument hint (e.g. "<name>").
    pub hint: &'static str,
    /// Human-readable description for autocomplete popup.
    pub description: &'static str,
}

/// Parsed command invocation from user input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedCommand {
    /// Command name (lowercase without slash).
    pub name: String,
    /// Arguments string after command name (trimmed).
    pub args: String,
}

/// Registry holding all known slash commands.
#[derive(Debug, Clone)]
pub struct CommandRegistry {
    commands: Vec<CommandSpec>,
}

impl Default for CommandRegistry {
    fn default() -> Self {
        Self::default_commands()
    }
}

impl CommandRegistry {
    /// Create registry loaded with standard GRIM commands.
    pub fn default_commands() -> Self {
        let mut reg = Self {
            commands: Vec::new(),
        };
        reg.register(CommandSpec {
            name: "model",
            hint: "[name]",
            description: "List local models or load/hot-swap a model by name",
        });
        reg.register(CommandSpec {
            name: "temp",
            hint: "<value>",
            description: "Set temperature parameter (e.g. /temp 0.7)",
        });
        reg.register(CommandSpec {
            name: "topp",
            hint: "<value>",
            description: "Set top-p nucleus sampling parameter (e.g. /topp 0.9)",
        });
        reg.register(CommandSpec {
            name: "ctx",
            hint: "<limit|auto>",
            description: "Set context token limit override (e.g. /ctx 8192)",
        });
        reg.register(CommandSpec {
            name: "system",
            hint: "[prompt]",
            description: "View or update system prompt for chat context",
        });
        reg.register(CommandSpec {
            name: "load",
            hint: "<path>",
            description: "Load prior chat session from JSONL file",
        });
        reg.register(CommandSpec {
            name: "clear",
            hint: "",
            description: "Reset session history and clear transcript",
        });
        reg.register(CommandSpec {
            name: "save",
            hint: "<path>",
            description: "Export current chat transcript to JSONL or text file",
        });
        reg.register(CommandSpec {
            name: "help",
            hint: "",
            description: "Show available commands and keybindings",
        });
        reg.register(CommandSpec {
            name: "skill",
            hint: "[name|off]",
            description: "Activate a skill by name, or open the skill picker",
        });
        reg.register(CommandSpec {
            name: "skills",
            hint: "",
            description: "List all discovered skills",
        });
        reg.register(CommandSpec {
            name: "project",
            hint: "<path>",
            description: "Set the project directory (sandbox root for tools)",
        });
        reg.register(CommandSpec {
            name: "cd",
            hint: "<path>",
            description: "Alias for /project — set the working directory",
        });
        reg.register(CommandSpec {
            name: "pwd",
            hint: "",
            description: "Print the current project directory",
        });
        reg.register(CommandSpec {
            name: "think",
            hint: "[off|low|medium|high]",
            description: "Set thinking/reasoning effort level (Ctrl+T to cycle)",
        });
        reg.register(CommandSpec {
            name: "plan",
            hint: "[on|off]",
            description: "Toggle plan mode: read-only tools, model proposes a plan first",
        });
        reg.register(CommandSpec {
            name: "compact",
            hint: "",
            description: "Summarize older context now to free tokens (auto-runs at 85% context)",
        });
        reg.register(CommandSpec {
            name: "backend",
            hint: "[rocm|cuda|metal|cpu|auto]",
            description: "Select inference backend (auto-detect if unset)",
        });
        reg.register(CommandSpec {
            name: "exit",
            hint: "",
            description: "Quit GRIM TUI",
        });
        reg
    }

    /// Add a command descriptor to the registry.
    pub fn register(&mut self, spec: CommandSpec) {
        self.commands.push(spec);
    }

    /// List all registered commands.
    pub fn all_commands(&self) -> &[CommandSpec] {
        &self.commands
    }

    /// Find command candidates matching a typed prefix (e.g. "/m" -> ["model"]).
    pub fn find_completions(&self, prefix: &str) -> Vec<&CommandSpec> {
        let query = prefix.strip_prefix('/').unwrap_or(prefix).trim_start();
        self.commands
            .iter()
            .filter(|cmd| cmd.name.starts_with(query))
            .collect()
    }

    /// Parse an input line into a command name and argument string if it starts with '/'.
    pub fn parse(&self, line: &str) -> Option<ParsedCommand> {
        let trimmed = line.trim();
        if !trimmed.starts_with('/') {
            return None;
        }
        let content = trimmed[1..].trim();
        if content.is_empty() {
            return Some(ParsedCommand {
                name: String::new(),
                args: String::new(),
            });
        }
        let (name, args) = match content.split_once(char::is_whitespace) {
            Some((n, a)) => (n.to_lowercase(), a.trim().to_string()),
            None => (content.to_lowercase(), String::new()),
        };
        Some(ParsedCommand { name, args })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocomplete_candidates() {
        let registry = CommandRegistry::default_commands();
        let matches = registry.find_completions("/m");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "model");

        let all = registry.find_completions("/");
        assert!(all.len() >= 5);
    }

    #[test]
    fn test_parse_arguments() {
        let registry = CommandRegistry::default_commands();
        let cmd = registry.parse("/temp 0.85").unwrap();
        assert_eq!(cmd.name, "temp");
        assert_eq!(cmd.args, "0.85");

        let empty = registry.parse("hello world");
        assert!(empty.is_none());
    }
}
