# 10x Analysis: grim-cli TUI
Session 1 | Date: 2026-09-03

## Current Value

grim TUI = fully-local coding agent terminal. ratatui 0.30 + crossterm, ~11.7k LOC in `crates/grim-cli/src/tui/`. In-process `grim_engine::Engine` (worker thread, `worker.rs:193-219`) — no HTTP hop. GGUF models, ROCM/CUDA/Metal/CPU backends, agentic tool loop (7 sandboxed tools, max 10 iterations), streaming markdown + syntect highlighting, tool-approval with LCS diff previews, plan mode, session resume, command palette, skills from `~/.agents/skills`, thinking-level control, live diagnostics (TTFT/tok/s/KV/context).

**Who**: privacy-conscious devs running local models on their own GPU. The moat is *local + private + agentic* — no competitor does all three well.

**Core action**: type prompt → watch stream → approve/reject file edits. Users spend most time waiting on generation and judging tool calls.

**The unique constraint that shapes everything**: local models = small context windows + slower tok/s than frontier APIs. UX must compensate for both.

## The Question

What would make this 10x more valuable?

---

## Massive Opportunities

### 1. Checkpoints — undo any agent file edit
**What**: Snapshot workspace state (shadow git index or file-content store) before each mutating tool call. One keybind (`Ctrl+R`? `/undo`) lists turns with edits → restore. Diff previews already exist (`diff.rs`); this is the missing rollback half.
**Why 10x**: Kills the #1 anxiety of agentic tools: "what did it just do to my files." Trust = adoption. User lets the agent run because mistakes are reversible.
**Unlocks**: Confident long autonomous runs; "rewind and redirect" workflow (undo turn 3, edit prompt, re-run).
**Effort**: High
**Risk**: Snapshot storage bloat on big repos; interplay with user's own git state.
**Score**: 🔥

### 2. MCP client support
**What**: grim-plugin crate exists but is unwired; zero MCP anywhere in grim-cli/src or grim-server/src. Implement MCP client (stdio + HTTP transport) so local models can use filesystem/git/browser servers.
**Why 10x**: Turns TUI from a tool with 7 built-in tools into a platform riding the entire MCP ecosystem. This is the "make competitors nervous" move — no local-first agent has solid MCP yet.
**Unlocks**: Web search, DB access, browser control — all local, all private.
**Effort**: High
**Risk**: Local small models are weaker at tool selection across many servers; needs tool-result truncation discipline for small ctx.
**Score**: 🔥 (strategic bet)

### 3. Multi-session: background runs + session tabs
**What**: One session per process today. Add detached/background agent runs + tab switching inside one TUI process.
**Why 10x**: Agent runs are slow (local tok/s). Parallelism is the only cure for wall-clock time. "Kick off refactor in tab 2, keep working in tab 1."
**Unlocks**: grim becomes a workspace, not a conversation.
**Effort**: Very High
**Risk**: GPU contention between two in-process engines; worker thread design assumes one Engine.
**Score**: 🤔

---

## Medium Opportunities

### 1. Message queueing while streaming
**What**: Input during generation is rejected: "generation in progress; Esc to cancel first" (`mod.rs:1730-1734`). Instead: buffer typed input, auto-send when turn completes (with Esc to edit/clear the queue).
**Why 10x**: Highest-frequency friction in the whole app. Every single turn ends with the user having waited to type. Local models are slower → wait is longer → friction multiplied.
**Impact**: Turns stop feeling like lock-step requests; feels like a conversation with a busy colleague.
**Effort**: Medium (state machine: composing→queued→sent; interacts with Esc-cancel and composer history)
**Score**: 🔥

### 2. Session autosave + central store + cross-session search
**What**: Transcripts exist only if user runs `/save`; resume scans cwd only (`mod.rs:1394-1410`). Autosave every turn to `~/.local/share/grim/sessions/`, auto-title from first prompt, Ctrl+O browser gains full-text search + fuzzy + frecency sort.
**Why 10x**: Continuity compounds: every past conversation becomes retrievable memory. Today a crash or forgotten `/save` = total loss. "Continue yesterday's debug session" must be zero-friction.
**Impact**: Resume goes from "if I remembered to save" to guaranteed; browser becomes real history.
**Effort**: Medium
**Score**: 🔥

### 3. Context auto-compaction
**What**: Local models = tight context. When KV/context nears limit mid-agentic-run, auto-summarize older turns + tool outputs and continue, with a visible "compacted N→M tok" marker. `/compact` as manual fallback.
**Why 10x**: Long agentic runs currently die or degrade at ctx wall — the exact use case (agent editing files for 10 iterations) this TUI exists for. Without it the 10-iteration loop is a trap.
**Impact**: Unlocks long runs on small models; directly compensates for the local-model constraint.
**Effort**: Medium-High (summarization uses own engine; careful with tool_call/tool_result pairing on truncate)
**Score**: 🔥

### 4. Ctrl+R fuzzy history search + persisted frecency
**What**: Composer history = linear 100-entry ring (`composer.rs:50`); frecency is in-memory only, dies on restart (`frecency.rs`).
**Why 10x**: Power users re-run prompts constantly (`r` exists but only for the last one). Search over all past prompts (and with #2, all sessions) = muscle-memory feature.
**Effort**: Low
**Score**: 🔥

### 5. Image paste into prompt (multimodal)
**What**: `src/multimodal.rs` exists but TUI never references it; clipboard is text-only (`mod.rs:2256-2280`). Wire arboard image → attach → send to VLM catalog models; render thumbnail placeholder in composer. Kitty graphics protocol optional later.
**Why 10x**: "Screenshot → fix this UI bug" is a killer local-VLM demo; hardware exists (catalog includes VLMs like chameleon/dots).
**Effort**: Medium
**Score**: 👍

### 6. Themes + config file for TUI
**What**: Hardcoded neon-purple palette (`mod.rs:2304-2313`); no config surface.
**Why 10x**: Honestly not 10x — polish. Low priority vs. workflow features, but cheap once god-module refactored.
**Effort**: Low-Medium
**Score**: 🤔

---

## Small Gems

### 1. Auto-title sessions for resume browser
**What**: First user prompt (first ~40 chars) becomes session filename/title. Browser lists filenames only today (`mod.rs:2919-2946`).
**Why powerful**: "2026-09-03-14-22-01.jsonl" vs "fix-rocm-flash-attn" — resume pickability changes completely. Nearly free.
**Effort**: Low
**Score**: 🔥

### 2. Copy-last-code-block keybind (e.g. `y` on empty composer)
**What**: Extract last fenced code block from transcript → clipboard (clipboard plumbing already exists).
**Why powerful**: The single most repeated manual action in any coding chat is copying the code out. One key instead of mouse selection.
**Effort**: Low
**Score**: 🔥

### 3. Esc-Esc: rewind conversation to a previous message (edit & branch)
**What**: Select an earlier user message → truncate history there → load into composer.
**Why powerful**: Turns a linear transcript into an editable loop; pairs with #1 Massive (checkpoints) later.
**Effort**: Low-Medium
**Score**: 👍

### 4. Toast on "generation done" already exists; add failure-differentiated notification
**What**: Desktop notification distinguishes "done" vs "needs approval" vs "errored" (different sound/urgency).
**Why powerful**: Agent waiting on approval silently = wasted minutes. Approval-pending is the highest-value notification.
**Effort**: Low
**Score**: 👍

### 5. `/stats` per-session cumulative summary
**What**: Total tokens, turns, tool calls, avg tok/s for the session, on demand in a toast/panel.
**Why powerful**: Zero cost (data already tracked in stats), feeds the "my local rig is fast" pride loop.
**Effort**: Low
**Score**: 👍

---

## Recommended Priority

### Do Now (quick wins, days each)
1. **Message queueing while streaming** — Why: removes highest-frequency friction in the app. Impact: every turn feels faster.
2. **Session autosave + auto-titles** — Why: ends silent transcript loss; makes Ctrl+O actually useful. Impact: continuity guaranteed.
3. **Persisted frecency + Ctrl+R history search** — Why: trivial on top of autosave store. Impact: power-user speed.
4. **Copy-last-code-block keybind** — Why: one function, used many times daily.

### Do Next (high leverage, weeks)
1. **Context auto-compaction** — Why: unblocks the long agentic runs the tool exists for; compensates for the small-ctx local model constraint. Unlocks: 10-iteration loop becomes survivable → reliable.
2. **Checkpoint/undo of agent edits** — Why: trust is the adoption gate for agentic tools; diffs (done) + rollback (missing) = full safety story. Unlocks: hands-off runs.
3. **Image paste for VLM models** — Why: hardware + models already in catalog; unique local demo.

### Explore (strategic bets)
1. **MCP client** — Why: platform move; rides ecosystem instead of competing with it. Risk: small models fumble big tool sets — mitigate with per-server enable + tool-result truncation. Upside: category-defining local agent.
2. **Background/parallel sessions** — Why: local tok/s makes wall-clock parallelism the only speedup that matters. Risk: GPU contention, single-Engine worker design. Upside: tool → workspace.

### Backlog
1. **Themes/config** — Why later: polish; do after god-module `mod.rs` (4,279 LOC) refactor makes palettes pluggable.
2. **Vim mode** — Why later: vocal minority; Emacs-grade composer already strong (kill ring, undo, yank-pop).
3. **Sub-agents / parallel tool calls** — Why later: depends on compaction + GPU scheduling work.

## Questions

### Answered
- **Q**: Does TUI use the server? **A**: No — fully in-process engine (`worker.rs:193-219`).
- **Q**: Any MCP/plugin wiring? **A**: None; grim-plugin crate exists, unreferenced by TUI.
- **Q**: Is undo scoped to anything real today? **A**: Composer text only (`undo_stack.rs`); file edits unrolled-backable.

### Blockers
- **Q**: Compaction summarization budget — dedicated small model, or same engine? (affects UX while compacting on slow GPUs)
- **Q**: Checkpoints — shadow-git (requires repo) vs file-store (works anywhere)? Prefer file-store for non-git dirs?

## Next Steps
- [ ] Validate: instrument how often "generation in progress" rejection fires (queue evidence)
- [ ] Prototype: autosave store format (reuse JSONL export schema from `mod.rs:2038-2106`)
- [ ] Decide: checkpoint mechanism (file-store vs shadow-git)
- [ ] Research: MCP client Rust crates vs hand-roll; tool-result budget policy for small ctx
