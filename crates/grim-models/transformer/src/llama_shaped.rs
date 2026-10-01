//! The Llama-shaped architectures, as data instead of as twelve files.
//!
//! ## What qualifies
//!
//! An architecture belongs here when its reference in
//! `old/repo/llama.cpp-master/src/models/` is genuinely Llama-shaped, by
//! four measurements taken against that reference:
//!
//! * it creates exactly the tensor set `llama.cpp` creates -- no
//!   `ATTN_GATE`, no `NEXTN_*`, no `FFN_NORM_EXPS`, no factored output;
//! * it has no router (`FFN_GATE_INP` / `n_expert`);
//! * it has no sliding-window pattern;
//! * it never calls `build_norm(..., LLM_NORM, ...)`, i.e. it is RMS.
//!
//! All four are required. Two earlier versions of this classification were
//! wrong: one generalised `gptneox`'s tensor count to all 84 candidates,
//! and one counted `grep -c` lines as operations. Both over-counted, and
//! both would have deleted files recording real deviations. See
//! `plans/PLAN-macho-merge.md` section 0.
//!
//! ## What does NOT qualify
//!
//! 47 of the 84 original candidates create tensors a Llama cannot express,
//! and keep their own files: `afmoe` (`ATTN_GATE`), `bitnet`
//! (`ATTN_SUB_NORM`), `bailingmoe2` (fused `ATTN_QKV` plus `NEXTN_*`),
//! `arctic` (`FFN_NORM_EXPS`), and others. Those files are the only record
//! of the gap, so they stay.
//!
//! ## What this replaces
//!
//! Twelve ~2,900-byte files, each a `struct` whose `forward` forwarded to
//! an inner `Llama`. They were 96% word-identical, and **none was ever
//! constructed**: the loader routes these architectures through shared
//! arms, and the only reference to these types was an
//! `impl_llama_wrapper_graph!` macro that delegated every method back to
//! `self.inner`. So each file existed only to be deleted.
//!
//! The `reference` and `why` fields exist because "no deviation" is a claim
//! about someone else's code, and a claim with no citation rots silently.

use grim_core::architecture::ModelArchitecture;

/// One architecture whose reference is genuinely Llama-shaped.
pub struct LlamaShaped {
    /// The `ModelArchitecture` this entry stands in for.
    pub arch: ModelArchitecture,
    /// The reference `.cpp` the verdict came from, e.g. `baichuan.cpp`.
    /// Checked by `llama_shaped_table.rs`, so it cannot name a file that is
    /// not there.
    pub reference: &'static str,
    /// Why this qualifies -- one sentence, so a reader can disagree with it.
    pub why: &'static str,
}

/// Every Llama-shaped architecture. Twelve entries, readable in one screen;
/// the twelve files they replaced were not.
pub const LLAMA_SHAPED: &[LlamaShaped] = &[
    LlamaShaped {
        arch: ModelArchitecture::Baichuan,
        reference: "baichuan",
        why: "reference creates exactly llama.cpp's tensor set; RMS; no router; no SWA",
    },
    LlamaShaped {
        arch: ModelArchitecture::ChatGlm,
        reference: "chatglm",
        why: "reference creates exactly llama.cpp's tensor set; RMS; no router; no SWA",
    },
    LlamaShaped {
        arch: ModelArchitecture::Dream,
        reference: "dream",
        why: "reference creates exactly llama.cpp's tensor set; RMS; no router; no SWA",
    },
    LlamaShaped {
        arch: ModelArchitecture::Eurobert,
        reference: "eurobert",
        why: "reference creates exactly llama.cpp's tensor set; RMS; no router; no SWA",
    },
    LlamaShaped {
        arch: ModelArchitecture::HunyuanDense,
        reference: "hunyuan-dense",
        why: "195-byte reference: build_arch_graph only; inherits load_arch_tensors from llama_model_hunyuan_vl",
    },
    LlamaShaped {
        arch: ModelArchitecture::LlamaEmbed,
        reference: "llama-embed",
        why: "199-byte reference: build_arch_graph only; inherits load_arch_tensors from llama_model_llama",
    },
    LlamaShaped {
        arch: ModelArchitecture::Mistral4,
        reference: "mistral4",
        why: "190-byte reference: build_arch_graph only; inherits load_arch_tensors from llama_model_deepseek2 (models.h:1393)",
    },
    LlamaShaped {
        arch: ModelArchitecture::PaddleOcr,
        reference: "paddleocr",
        why: "vision encoder (build_inp_embd/build_cvec), not a causal LM; tensor set matches llama.cpp's",
    },
    LlamaShaped {
        arch: ModelArchitecture::Plamo,
        reference: "plamo",
        why: "reference creates exactly llama.cpp's tensor set; RMS; no router; no SWA",
    },
    LlamaShaped {
        arch: ModelArchitecture::Qwen2,
        reference: "qwen2",
        why: "reference creates exactly llama.cpp's tensor set; RMS; no router; no SWA",
    },
    LlamaShaped {
        arch: ModelArchitecture::SmolLm3,
        reference: "smollm3",
        why: "reference creates exactly llama.cpp's tensor set; RMS; no router; no SWA",
    },
    LlamaShaped {
        arch: ModelArchitecture::Xverse,
        reference: "xverse",
        why: "reference creates exactly llama.cpp's tensor set; RMS; no router; no SWA",
    },
];

/// The entry for `arch`, if it is Llama-shaped.
pub fn lookup(arch: ModelArchitecture) -> Option<&'static LlamaShaped> {
    LLAMA_SHAPED.iter().find(|e| e.arch == arch)
}

/// True when `arch` is served by this table rather than its own module.
pub fn is_llama_shaped(arch: ModelArchitecture) -> bool {
    lookup(arch).is_some()
}
