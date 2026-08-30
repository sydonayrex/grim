// matmul_lm_head — LM-head projection GEMM alias, mirrors ROCm's matmul_lm_head (roc_device.rs:13879).
// Routes through matmul_with_op(GemmOp::LmHead), which selects the TLOLog shape class in the Metal autotune.
// No new kernel is required on Metal; this is a convenience API shim.
