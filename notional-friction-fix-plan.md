# Notional Friction Fix Plan — Top 5 from `notional-usability-results.md`

Status: draft · Owner: eng + UX · Input: notional-usability-results.md (top-5 cross-persona friction)
Scope: feature-complete Grim vision. Treat each item as a surface-level integration, not a backend
rewrite, unless a later spike says otherwise.

---

## Guiding rules

1. One canonical surface per question, surfaced in more than one place (CLI + API/dashboard) only
   when the user's context differs enough to justify it.
2. Teach at the point of choice, not in a separate doc page the user has to remember to open.
3. Prefer few, well-named knobs and fields over broad surface expansion. Do not add metrics for
   their own sake.
4. Every fix above stubs-only is paired with a verification artifact: a targeted behavioral test in
   the relevant crate and a one-paragraph notional re-score of the affected personas so we can tell
   later whether the fix changed the verdict.

---

## 1. Memory / KV / tier single surface

**Problem (from results §1).** Total memory, KV-cache split, and KV-tier occupancy are reachable but
fragmented across `/metrics`, CLI, and a dashboard that needs a filter. Users discover the state
reactively, not as a single live view.

**Intent.** One canonical view that answers: *how much of my memory/VRAM is the model, how much is
KV, and where is overflow going right now?*

**What we build**

- A single "memory / KV / tier" summary surface. Primary home: `grim-garage`. Secondary: a concise
  CLI summary (e.g. `grim memory` / `grim status --memory` or a `/memory` endpoint) so the same
  answer is reachable headless.
- The view shows, in one place:
  - total memory/VRAM in use
  - KV-cache portion vs. model-weight portion
  - current KV-tier occupancy (GPU / RAM / NVMe) and which tier is active
  - the active spill threshold as a visible, editable knob
- The summary reflects the relationship between model-weight memory and KV memory, not just two
  separate numbers.

**Suggested sequence**

1. Spike: confirm where the per-request/server memory and KV stats already live and what is missing
   to present them together. Do not add new metric collection yet — reuse what exists.
2. Wire the canonical summary in `grim-garage` first (the dashboard is where P2/P3/P16 already look
   for state).
3. Expose a headless equivalent (CLI and/or HTTP) so the same answer is reachable without the
   dashboard. Keep it concise; this is a summary, not a metrics dump.
4. Make the spill threshold editable from the same surface where it is visible.

**Verification**

- Behavioral test in the dashboard/metrics crate that the summary shows total, KV, and active tier,
  and that editing the threshold is reflected.
- Notional re-score: P1 memory attribution moves toward PASS on D1/D3; P3 and P11 move toward PASS on
  KV-tier live visibility.

**Rough shape**

- Small/medium. Mostly wiring and UI layout plus one headless summary path. Risk is mostly about which
  stats are already exposed and consistent, not about adding a new backend.

---

## 2. Scheduler waiting reason as a first-class field

**Problem (from results §2).** The scheduler exposes queue/active/admit state, but the *reason* a
request waited is found in logs, not as a labeled field an operator can read or alert on.

**Intent.** A first-class "waiting reason" the user can read — which admission rule held the request
— without dropping into logs.

**What we build**

- A labeled waiting-reason field on scheduler/request state that is readable in `grim-garage` and
  reachable headless (structured stats the engineer can scrape).
- The reason should map to the admission rule that held the request (batch full, queue depth, tier
  pressure, etc.), not just "waiting."
- Prefer a small, well-defined set of reason labels over an open-ended narrative.

**Suggested sequence**

1. Spike: inventory the admission decisions the scheduler already makes and choose a compact set of
   reason labels that cover the real cases. Do not build a generic log-to-widget pipeline; pick the
   actual rules.
2. Attach the reason to the scheduler/request state where it is already tracked.
3. Surface it in `grim-garage` as a labeled status on waiting requests/queues.
4. Expose it headless (stats/endpoint) for Grafana-style scraping and on-call use.

**Verification**

- Behavioral test that, under a controlled waiting condition, the reason field reflects the actual
  admission rule.
- Notional re-score: P2 moves toward PASS on D1/D4 (scheduler-wait explainability). Weaker echo for P3.

**Rough shape**

- Small/medium. The scheduler already knows the decision; the work is labeling and surfacing it
  consistently. Main risk is choosing a reason taxonomy that stays stable as admission rules evolve.

---

## 3. Adapter-family taxonomy with in-context explanations

**Problem (from results §3).** LoRA/QLoRA/Vera/SoulEater/QGaLore/PISSA/OLORA read as a wall of
options until used; the same names appear across fine-tune selection, runtime load, and reload, but
the per-family meaning and trade-off are not shown at the point of choice.

**Intent.** Short, in-context explanations next to each adapter family where it is selected, plus a
consistent adapter identity across the surfaces that mention it.

**What we build**

- In-context one-line explanation per family at the point of selection (what the family does and the
  primary trade-off, e.g. quantized vs. full-precision adapter, memory vs. fidelity).
- Keep it short; this is a label, not a tutorial.
- Make the same adapter name mean the same thing on fine-tune, load, and reload surfaces so the user
  is not re-learning the name each time.

**Suggested sequence**

1. Decide the canonical short description for each family (one line, trade-off focus). This is a UX/
   content decision, not a code change, so do it first.
2. Put the explanation in the fine-tune selection surface where the family is chosen.
3. Carry the same label/identity into runtime adapter load and reload surfaces.
4. If any surface lists families without explanation today, fix that surface for consistency rather
   than treating it as a separate problem.

**Verification**

- Behavioral test that each family shown at selection carries its one-line explanation and that the
  same identity appears on load and reload.
- Notional re-score: P4 moves toward PASS on D1 (adapter-taxonomy clarity). Weaker echo for P2 and P15
  on adapter identity consistency.

**Rough shape**

- Small. Mostly content + UI labels plus consistency work across a few surfaces. Risk is keeping the
  descriptions accurate and stable, not a technical risk.

---

## 4. Tool-call loop teachable at the point of use

**Problem (from results §4).** The API follows `tool_calling_spec` and is correct, but a non-expert
OpenAI consumer cannot confidently state the client loop from the schema alone. The feature that should
make Grim "just an OpenAI drop-in" requires a tutorial for the very consumer we target.

**Intent.** Make the tool-call loop teachable at the point of use for a non-expert consumer.

**What we build**

- A minimal worked example in the API surface: one request/response pair that shows tool call → tool
  result → follow-up turn, not just the schema.
- A client-side hint in the path where it helps: when a tool call is returned and no result has been
  posted, surface a short, obvious next-step hint rather than leaving the consumer to infer the loop.
- Keep the schema correct; this is about the teachable path around it, not changing the contract.

**Suggested sequence**

1. Write the minimal worked example (single turn that exercises tool call → result → follow-up). Keep
   it the smallest possible correct example.
2. Place it where a consumer hits the feature: API docs / request example surface, not buried in a
   separate tool-calling doc.
3. Add the short next-step hint for the "model called a tool, now what?" moment, if that surface exists
   and can carry it without over-explaining.
4. Confirm the example is correct for the OpenAI-compat path and, where relevant, the Ollama-compat
   path, without duplicating effort.

**Verification**

- Behavioral test that the worked example is a correct, minimal tool-call loop for the targeted
  consumer path.
- Notional re-score: P7 moves toward PASS on D1/D5 (tool-call loop discoverability for non-experts). Weaker
  echo for P8.

**Rough shape**

- Small. Docs/example + a small hint surface. Main risk is keeping the example minimal and correct
  rather than thorough.

---

## 5. Single model-trust verdict command

**Problem (from results §5).** A user can determine how to trust a GGUF and produce a checksum + config
trace, but there is no single command that packages "I want to assert this checkpoint is from a known
toolchain and here is the evidence" into one trust verdict. Trust is a first-class concern for P18 and
a provenance concern for P10.

**Intent.** One surface that turns the existing provenance data into a compact trust verdict a
gatekeeper can act on.

**What we build**

- A single `grim verify-trust` (or equivalent) flow that takes a checkpoint and returns a compact trust
  verdict: checksum, the toolchain/source it was produced by (as recorded in metadata), and the config
  trace connecting the artifact to a known pipeline.
- The command is a verdict surface, not a new provenance collector; it assembles what already exists
  into one place and one decision.
- Keep the verdict compact; let the user drill in only if they want the underlying evidence.

**Suggested sequence**

1. Spike: confirm where checksum, metadata/source, and config trace currently live and how much is
   already machine-readable. Do not invent new provenance data yet.
2. Define the compact verdict shape: what is the header claim, and what is the drill-down evidence.
3. Implement the command as assembly over existing data, with the compact verdict as the primary output.
4. Treat the same flow as a provenance check for quantization artifacts (P10) where it applies, without
   building two separate things.

**Verification**

- Behavioral test that the command returns a compact verdict over a checkpoint whose provenance is known,
  and that the underlying evidence is reachable on drill-down.
- Notional re-score: P18 moves toward PASS on D1/D4 (model-trust single-command flow). Weaker echo for P10
  on artifact provenance.

**Rough shape**

- Small/medium. Mostly assembly and UX of the verdict, contingent on how machine-readable the existing
  provenance data already is. Main risk is gap-filling missing provenance fields, not the command itself.

---

## Cross-cutting sequencing

Do these in parallel where teams are independent, but order them so the hardest dependency is not at the
end:

1. **Content/UX decisions first** for items 3 (adapter one-liners) and 4 (tool-call worked example).
   These are largely content + surface placement and unblock the code work.
2. **Spikes next** for items 1, 2, and 5 to confirm what data already exists and what is missing before
   wiring surfaces. None of these should start as a backend expansion; they start as "what do we already
   have."
3. **Wire the canonical surfaces** (1: memory/KV/tier; 2: waiting reason; 5: trust verdict) after the
   spikes, with `grim-garage` as the primary home where a dashboard makes sense and a headless path
   where the user is script/headless-driven.
4. **Consistency pass last** across adapter surfaces (item 3) and the trust flow for quantization
   artifacts (item 5) so the same identity/verdict reads the same way on every surface.

---

## What "done" looks like for each item

- **Item 1:** one canonical memory/KV/tier summary reachable in dashboard and headless; spill threshold
  visible and editable in the same place.
- **Item 2:** a labeled waiting reason on scheduler/request state, readable in dashboard and scrapeable
  headless, mapped to the actual admission rule.
- **Item 3:** each adapter family shown at selection with a one-line trade-off explanation, and the same
  adapter identity on fine-tune, load, and reload.
- **Item 4:** a minimal correct worked example at the point of use plus a short next-step hint for the
  tool-call result moment, aimed at the non-expert consumer.
- **Item 5:** one `verify-trust`-style command that returns a compact verdict over existing provenance
  data, with drill-down available.

---

## Known risks / open questions

- **Item 1:** the biggest unknown is which memory/KV stats are already consistent and exposed; the plan
  assumes reuse, not new collection. If reuse is weak, the item grows.
- **Item 2:** the reason taxonomy must stay stable as admission rules evolve; keep it small and named.
- **Item 3:** adapter descriptions must be accurate and stable; if the field is fluid, freeze the
  canonical one-liners before shipping labels.
- **Item 4:** the worked example must stay minimal and correct; if tool-calling semantics shift, the
  example is a maintenance surface, not a one-time doc.
- **Item 5:** if provenance data is not yet machine-readable enough, the command is only as good as the
  data it assembles; that is a data gap, not a command design gap, and should be surfaced early in the
  spike.

---

*End of draft plan. This is a planning document derived from notional results; treat it as a starting
point for real implementation scoping, not as a committed roadmap.*
