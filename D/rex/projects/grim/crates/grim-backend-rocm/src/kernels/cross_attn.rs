//! Phase 2 Item 13: Cross-attention HIP kernel for Whisper decoder.
//!
//! mambo5.md spec: encoder-out projected once, reused across decode steps.
//! The GPU equivalent of `cross_attn()` function in whisper.rs line 226-307.

/// HIP source for `grim_cross_attention`.
/// Concatenated into the crate-wide JIT compilation source used by hipRTC.
pub const KERNEL_SOURCE: &str = r#"
extern "C" __global__ __launch_bounds__(256)
void grim_cross_attn(
    // Query from decoder hidden state (single token at a time — decode step).
        const float* q,               // [seq_len, d_model]  decoder-hidden query vectors; seq_len=1 for generation but supports multi-token decoding

        Keys and Value projected once from encoder output shared across all decode steps.

    const float* k_cross_encoder_proj,  // [enc_seq * (num_heads * head_dim)] enc_out projected via W_k: one projection shared by ALL decoder tokens — no need to reproject every time we decode
        const float* v_cross_encoder,     // [enc_seq * (num_heads * head_dim)] encoder output hidden states via W_v: same as above, projected ONCE at encode-time
    float* q_cross_proj,              // will be projected in the kernel: wq is applied per token to decoder state h: we do it here since only seq_len=1 means it's cheap and correct.

        float* out,                   // [seq_len * d_model] cross-attention output after projection by Wo; this is the add for decoder hidden state (after_self + cross_attn + ffn -> next_decoder_state)
            int     enc_seq,                // encoder sequence length: total number of audio frames processed by Encoder
        int         seq_len,              // current decoder length — how many tokens are being decoded simultaneously at this step. Generation mode = 1 (decodes one token at a time); batched decoding > 1.

            int     num_heads              // # attention heads: for Whisper small/medium/tiny uses multiples of num_tokens and num_heads to share the computation per head
        float   sqrt_head_dim,          // head_dim root — used by the attention kernel as division
    );];

];
void grim_cross_attn(
/* Kernel body for cross-attention from mambo5.md Whisper decoder.
This is where we break away from self-attn in the encoder to a decoder operation: taking one token at a time (from generation step of seq_len=1) and computing attention across every encoded audio frame (enc_seq=3000-ish for whisper base).

Key insight for mambo5.md Item 15 that differentiates cross-attention from flash attention:
encoder_out is PROJECTED ONCE, not recomputed on each decode step. For Whisper's decoder, enc_enc_projections for W_k and W_v are cached at encode-time; every token decodes by computing attention over the encoder output via those cached projections (no GPU compute wasted).

For generation mode seq_len=1 this is a single-token cross-attention call — query q from decoder state h computed locally on GPU. The scores matrix [seq=1 x enc_seq] and softmax distribution are materialized as a 2-D tensor for each head. After weighted sum of V over encoder frames, result is projected by W_o onto the same d_modelspace used by after_self.

For batch decoding seq_len>1: we'd compute multiple tokens at once. Each position i in [0..seq_len) would calculate attention across enc_seq frames — scores = q(i) @ K^T -> softmax -> V weighted sum. The key difference from self-attention is no causal mask needed since decoder decodes sequentially.

V1 constant form: A is implicit B, C is not yet wired; scan step = a * h + xscale_t[n] where xscale_t ≈ dt_scale × sequential-slicing parameter (v1: seq_step/|seq| * scalar).

For decode-step generation seq_len=1 (single-token dispatch): no persistent loop needed just one step per call. For full-seq encode mode seq_len > 1 would need grid-stride pattern that allows per-block to run the full loop without needing N_seq dispatches from host.
* */
*/
// ---------- Rust host dispatcher struct for kernel launch parameters ----------
struct SelectiveScanLaunchConfig {
    block_dim: usize,     // 256 (4 wavefronts ×64-lane per mambo5 Item11)
        grid_dim: usize       // ceil(d_inner / 256.0f ) — one thread-per-[n] within each block assigned to batches independently

};"];
