# Troubleshooting

## Panics and Unwraps

*   **Shape Mismatch Panic**: Occurs in the `grim_autograd` or backend layers when tensors of incompatible dimensions are multiplied or concatenated.
    *   *Symptom*: Thread panic pointing to `ops.rs` or backend GEMM.
    *   *Fix*: Ensure input dimensions map precisely to model architecture.
*   **OOM Panic**: Thrown by the underlying device allocator.
    *   *Symptom*: Execution halts with CUDA or Metal memory allocation error.
    *   *Fix*: Reduce context size, offload layers to CPU, or select a smaller quantized format.

## Configuration Failures

*   **Missing Tokenizer/Config**: 
    *   *Symptom*: "ERROR: failed to load safetensors model: config.json not found."
    *   *Fix*: Provide `config.json` and `tokenizer.json` adjacent to the `.safetensors` file.
*   **Invalid GGUF Header**:
    *   *Symptom*: Parsing fails at load time.
    *   *Fix*: Re-download the file or ensure it complies with GGUF v2/v3 spec.

## Build Failures

*   **CUDA/HIP Missing**:
    *   *Symptom*: Linker fails to find `nvcc` or `hipcc` during `cargo build --features cuda`.
    *   *Fix*: Install the relevant toolkit and verify environment paths (e.g., `CUDA_PATH`).