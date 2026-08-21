//! Catalog of known chat-template families mapped to render.

use grim_format::tokenizer::{ChatMessage, render_chat_template};

#[derive(Debug, Clone, Copy)]
pub struct TemplateFamily {
    pub name: &'static str,
    pub jinja: &'static str,
    pub description: &'static str,
}

pub struct TemplateRegistry;

impl TemplateRegistry {
    pub fn default() -> Vec<TemplateFamily> {
        vec![
            TemplateFamily {
                name: "chatml",
                jinja: "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
                description: "ChatML format used by Qwen, Yi, and InternLM.",
            },
            TemplateFamily {
                name: "llama3",
                jinja: "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
                description: "Llama 3 / 3.1 / 3.2 instruction template.",
            },
            TemplateFamily {
                name: "qwen",
                jinja: "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
                description: "Qwen 2 / 2.5 chat template.",
            },
            TemplateFamily {
                name: "mistral",
                jinja: "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<s>[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '</s>' }}{% endif %}{% endfor %}",
                description: "Mistral / Mixtral instruction format.",
            },
            TemplateFamily {
                name: "gemma",
                jinja: "{% for message in messages %}{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n{% else %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}",
                description: "Gemma 2 / Gemma 3 chat format.",
            },
        ]
    }

    pub fn lookup(name: &str) -> Option<TemplateFamily> {
        Self::default().into_iter().find(|f| f.name == name)
    }
}

/// Render a family against messages JSON.
pub fn render_family(family: &str, messages_val: serde_json::Value) -> Result<String, String> {
    let f = TemplateRegistry::lookup(family)
        .ok_or_else(|| format!("unknown template family '{family}'"))?;
    let msgs_array = messages_val
        .as_array()
        .ok_or("input must be a JSON array of message objects")?;

    let mut chat_messages = Vec::with_capacity(msgs_array.len());
    for m in msgs_array {
        let role = m
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("user")
            .to_string();
        let content = m
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        chat_messages.push(ChatMessage {
            role,
            content,
            name: None,
            tool_call_id: None,
            tool_calls: None,
        });
    }

    render_chat_template(f.jinja, &chat_messages, true, "", "", None, None)
        .map_err(|e| format!("render failed: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lists_known_families() {
        let list = TemplateRegistry::default();
        let names: Vec<&str> = list.iter().map(|f| f.name).collect();
        assert!(names.contains(&"chatml"));
        assert!(names.contains(&"llama3"));
        assert!(names.contains(&"qwen"));
        assert!(names.contains(&"mistral"));
        assert!(names.contains(&"gemma"));
    }

    #[test]
    fn render_chatml_works() {
        let json_val = serde_json::json!([
            {"role": "user", "content": "Hello"}
        ]);
        let rendered = render_family("chatml", json_val).unwrap();
        assert!(rendered.contains("<|im_start|>user\nHello<|im_end|>"));
    }
}
