#[cfg(test)]
mod tests {

    #[test]
    fn test_rope_yarn_and_window_lo_kernel_presence() {
        let src = crate::kernels::KERNELS_SOURCE;
        assert!(
            src.contains("grim_rope_yarn"),
            "KERNELS_SOURCE must declare grim_rope_yarn"
        );
        assert!(
            src.contains("inv_freq"),
            "grim_rope_yarn must take a pre-computed inv_freq buffer"
        );
        assert!(
            src.contains("mscale"),
            "grim_rope_yarn must take an mscale (attention_factor) parameter"
        );
        assert!(
            src.contains("rotary_half"),
            "grim_rope_yarn must take a rotary_half (partial-rotary) parameter"
        );
        assert!(
            src.contains("window_lo"),
            "grim_qkv_attention must take a window_lo parameter for SWA"
        );
        assert!(
            src.contains("softcap"),
            "grim_qkv_attention must take a softcap parameter for Gemma-2"
        );
        assert!(
            src.contains("alibi_slopes"),
            "grim_qkv_attention must take an alibi_slopes parameter"
        );
        assert!(
            src.contains("grim_gelu_tanh_mul"),
            "KERNELS_SOURCE must declare grim_gelu_tanh_mul"
        );
        assert!(
            src.contains("grim_qkv_attention_paged"),
            "KERNELS_SOURCE must declare grim_qkv_attention_paged"
        );
        assert!(
            src.contains("grim_tree_attention"),
            "KERNELS_SOURCE must declare grim_tree_attention"
        );
        assert!(
            src.contains("grim_kda_gated_delta_rule_step"),
            "KERNELS_SOURCE must declare grim_kda_gated_delta_rule_step"
        );
        assert!(
            src.contains("grim_mla_q_kv_norm_split"),
            "KERNELS_SOURCE must declare grim_mla_q_kv_norm_split"
        );
    }
}
