# How to Run Inference

## One-Shot Inference

```bash
# Simple prompt
grim run llama3 --prompt "Hello, world!"

# With token limit
grim run llama3 --prompt "Write a poem about AI" --max-tokens 200

# Interactive mode
grim run llama3
# Then type prompts and press Enter

# Streaming enabled by default
```

## Start HTTP Server

```bash
# Default port 11434
grim serve

# Custom address
grim serve --address 0.0.0.0:8080

# With config file
grim serve --config /etc/grim/grim.toml
```

## Convert for GPU Optimization

### For ROCm (AMD GPUs)

```bash
# Convert and optimize for your GPU
grim oxidizer convert -i model.gguf -o model.grim

# With ROCm profile
grim oxidizer convert -i model.gguf -o model.grim --profile gfx1100

# Calibrate and convert
grim oxidizer calibrate -m model.gguf -o model
grim oxidizer search scores.json -o model
grim oxidizer convert -i model.gguf -o model.grim
```

### For CUDA (NVIDIA GPUs)

```bash
grim convert -i model.gguf -o model.grim
```

## Server as OpenAI-Compatible API

```bash
# Start server
grim serve --address 127.0.0.1:11434

# Test with curl
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Benchmark Performance

```bash
# Smoke test
grim bench

# With custom parameters
grim bench --tokens 512 --concurrency 4

# With specific model
grim bench --model llama3 --tokens 256
```