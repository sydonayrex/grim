# Glossary

*   **Autograd**: The subsystem responsible for automatic differentiation and reverse-mode gradient calculation, defined primarily in `grim_autograd`.
*   **GGUF**: A binary format used to store model weights, tensors, and hyperparameters in a single file.
*   **Safetensors**: A fast, secure file format for saving and loading machine learning tensors.
*   **Tensor**: The fundamental N-dimensional array representing data and weights. Managed within `grim_core`.
*   **Ollama Protocol**: A standardized API specification for interacting with local LLMs, replicated by the `grim_server` or Axum integration.
*   **Quantization**: Compressing tensor values from FP16 or FP32 to lower bit representations (like Q4_K) to conserve memory.