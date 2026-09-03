//! Context compaction planning: pure functions, no engine access.
//!
//! Strategy (locked): summarize older turns with the SAME engine on the
//! worker thread; keep the original system preamble and the last
//! `keep_turns` complete turns verbatim. Cuts only at user-message
//! boundaries so tool_call/tool_result pairs are never split.

use grim_format::ChatMessage;

#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPlan {
    /// messages[0..cut] are summarized; messages[cut..] kept verbatim.
    pub cut: usize,
}

fn is_turn_start(m: &ChatMessage) -> bool {
    m.role == "user" && m.tool_call_id.is_none()
}

/// Find a safe cut point: the user message that starts the
/// `keep_turns`-th turn from the end (so exactly the last `keep_turns`
/// turns stay verbatim). `None` when there is nothing to summarize.
/// `keep_turns` must be >= 1.
pub fn plan(messages: &[ChatMessage], keep_turns: usize) -> Option<CompactionPlan> {
    if keep_turns == 0 {
        return None;
    }
    let turn_starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| is_turn_start(m))
        .map(|(i, _)| i)
        .collect();
    if turn_starts.len() <= keep_turns {
        return None;
    }
    let cut = turn_starts[turn_starts.len() - keep_turns];
    if cut == 0 {
        return None;
    }
    Some(CompactionPlan { cut })
}

/// Build the summarization exchange for the engine (same engine, quiet run).
pub fn summary_messages(to_summarize: &[ChatMessage]) -> Vec<ChatMessage> {
    let mut digest = String::from(
        "Summarize this excerpt of a coding-assistant conversation. Preserve: the user's goals \
         and constraints, decisions made, files modified (with paths), commands run and their \
         outcomes, and any unresolved problems. Under 400 words. Output only the summary.\n\n",
    );
    for m in to_summarize {
        let body: String = m.content.chars().take(2000).collect();
        if !body.trim().is_empty() {
            digest.push_str(&format!("[{}] {body}\n", m.role));
        }
        if let Some(calls) = &m.tool_calls {
            for c in calls {
                let args: String = c.arguments.chars().take(500).collect();
                digest.push_str(&format!("[tool_call] {} {args}\n", c.name));
            }
        }
    }
    vec![
        ChatMessage {
            role: "system".into(),
            content: "You compress conversation history for an assistant with a small context window. Output only the summary.".into(),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
        ChatMessage {
            role: "user".into(),
            content: digest,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    ]
}

/// Apply a summary: original system preamble stays first, then the summary
/// as a user-role message (safe for any chat template), then the kept tail.
pub fn apply(messages: &[ChatMessage], plan: &CompactionPlan, summary: &str) -> Vec<ChatMessage> {
    let mut out: Vec<ChatMessage> = Vec::new();
    out.extend(
        messages[..plan.cut]
            .iter()
            .take_while(|m| m.role == "system")
            .cloned(),
    );
    out.push(ChatMessage {
        role: "user".into(),
        content: format!("[Conversation summary — earlier context]\n{summary}"),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    });
    out.extend(messages[plan.cut..].iter().cloned());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(t: &str) -> ChatMessage {
        ChatMessage { role: "user".into(), content: t.into(), tool_calls: None, tool_call_id: None, name: None }
    }
    fn assistant(t: &str) -> ChatMessage {
        ChatMessage { role: "assistant".into(), content: t.into(), tool_calls: None, tool_call_id: None, name: None }
    }
    fn system(t: &str) -> ChatMessage {
        ChatMessage { role: "system".into(), content: t.into(), tool_calls: None, tool_call_id: None, name: None }
    }
    fn tool_msg(id: &str) -> ChatMessage {
        ChatMessage { role: "tool".into(), content: "out".into(), tool_calls: None, tool_call_id: Some(id.into()), name: None }
    }

    #[test]
    fn plan_keeps_last_two_turns_and_cuts_at_user_boundary() {
        let msgs = vec![
            system("be helpful"),
            user("t1"),
            assistant("a1"),
            user("t2"),
            assistant("a2"),
            user("t3"),
            assistant("a3"),
            user("t4"),
        ];
        let plan = plan(&msgs, 2).unwrap();
        assert_eq!(plan.cut, 5); // turns t3+t4 kept verbatim
        assert_eq!(msgs[plan.cut].content, "t3");
    }

    #[test]
    fn plan_none_when_few_turns() {
        let msgs = vec![user("only"), assistant("one turn")];
        assert!(plan(&msgs, 2).is_none());
    }

    #[test]
    fn plan_cut_never_inside_tool_turn() {
        // Single old turn whose cut point would be index 0 -> nothing to summarize.
        let with_calls = ChatMessage {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: Some(vec![grim_format::ToolCallMsg {
                id: "1".into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }]),
            tool_call_id: None,
            name: None,
        };
        let msgs = vec![
            user("t1"),
            with_calls,
            tool_msg("1"),
            assistant("done"),
            user("t2"),
        ];
        // The whole t1 turn (incl. its tool exchange) gets summarized; t2 kept.
        assert_eq!(plan(&msgs, 1), Some(CompactionPlan { cut: 4 }));
    }

    #[test]
    fn plan_two_tool_turns_cut_at_second_user() {
        let with_calls = |id: &str| ChatMessage {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: Some(vec![grim_format::ToolCallMsg {
                id: id.into(),
                name: "read_file".into(),
                arguments: "{}".into(),
            }]),
            tool_call_id: None,
            name: None,
        };
        let msgs = vec![
            user("t1"),
            with_calls("1"),
            tool_msg("1"),
            assistant("done1"),
            user("t2"),
            with_calls("2"),
            tool_msg("2"),
            assistant("done2"),
            user("t3"),
        ];
        let plan = plan(&msgs, 1).unwrap();
        assert_eq!(msgs[plan.cut].content, "t3"); // only last turn kept
    }

    #[test]
    fn apply_preserves_system_head_and_keeps_tail_verbatim() {
        let msgs = vec![system("sys"), user("t1"), assistant("a1"), user("t2")];
        let plan = plan(&msgs, 1).unwrap();
        let out = apply(&msgs, &plan, "summary text");
        assert_eq!(out[0].role, "system");
        assert_eq!(out[0].content, "sys");
        assert!(out[1].content.contains("summary text"));
        assert_eq!(out[1].role, "user");
        assert_eq!(&out[2..], &[user("t2")]);
    }

    #[test]
    fn summary_messages_digest_includes_roles_and_tool_calls() {
        let mut a = assistant("working");
        a.tool_calls = Some(vec![grim_format::ToolCallMsg {
            id: "1".into(),
            name: "read_file".into(),
            arguments: "{\"path\":\"x.rs\"}".into(),
        }]);
        let msgs = summary_messages(&[user("do it"), a]);
        assert_eq!(msgs[0].role, "system");
        assert!(msgs[1].content.contains("[user] do it"));
        assert!(msgs[1].content.contains("read_file"));
        assert!(msgs[1].content.contains("x.rs"));
    }
}
