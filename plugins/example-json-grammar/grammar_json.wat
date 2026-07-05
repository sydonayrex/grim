// Example JSON Grammar Sampler Plugin
// This is an example WASI-compatible WASM component that implements a sampler
// which constrains output to valid JSON grammar.
//
// Compile with: `wat2wasm grammar_json.wat -o grammar_json.wasm`
// Or use `cargo build` with a WASM target.

// Note: This is placeholder documentation. A real implementation would be in WIT/WASM.
// The engine expects a module that exports:
// - `sampler()`: (logits_ptr: i32, logits_len: i32) -> token: i32

(module
  ;; Minimal WASM module structure
  (type $t0 (func (param i32 i32) (result i32)))
  (func $sampler (type $t0) (param $logits_ptr i32) (param $logits_len i32) (result i32)
    ;; Placeholder: return a fixed token
    ;; Real implementation would read logits and apply grammar constraints
    i32.const 1)
  (export "sampler" (func $sampler)))