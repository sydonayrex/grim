use grim_backend_metal::caps::MetalCaps;

#[test]
fn test_metal_iq4nl_kernel_and_dispatch() {
    // Task 6: IQ4NL parity — kernel present in MSL + dispatch path wired in lib.rs
    let msl_source = include_str!("../src/kernels.msl");
    assert!(
        msl_source.contains("grim_dequant_iq4nl"),
        "MSL shader source must contain kernel declaration for grim_dequant_iq4nl"
    );

    // The dispatch path: pipeline field + get_pipeline init + match arm in dequant_gate
    let lib_source = include_str!("../src/lib.rs");
    assert!(
        lib_source.contains("dequant_iq4nl: Retained"),
        "MetalPipelines must contain dequant_iq4nl pipeline field"
    );
    assert!(
        lib_source.contains("get_pipeline(\"grim_dequant_iq4nl\")"),
        "MetalPipelines init must load grim_dequant_iq4nl pipeline"
    );
    assert!(
        lib_source.contains("\"iq4nl\" => &ctx.pipelines.dequant_iq4nl"),
        "dequant_gate must dispatch iq4nl to the Metal pipeline"
    );
    assert!(
        lib_source.contains("\"iq4nl\" => grim_quant::dequant_iq4nl"),
        "dequant_gate must have an iq4nl host fallback in the match arms"
    );

    // Caps gating: IQ4NL dequant runs on any GPU family capable of compute.
    // Uses gpu_family >= 1 as the gate (any Apple Silicon can run it).
    let caps = MetalCaps::probe_default(1001, "Apple M3 Max".into(), 9);
    assert!(
        caps.gpu_family >= 1,
        "IQ4NL dequant requires Apple Silicon GPU (any family)"
    );
}
