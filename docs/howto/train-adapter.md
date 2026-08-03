# How to Train LoRA Adapters

Grim supports SFT (Supervised Fine-Tuning) and QLoRA training of adapter weights.

## Quick Start

```bash
# Train a QLoRA adapter
grim train \
  --model granite-3.1-8b \
  --dataset ./data/alpaca.jsonl \
  --output adapter.grim.train

# With custom parameters
grim train \
  --model llama3 \
  --dataset ./data/train.jsonl \
  --output my-adapter.grim.train \
  --epochs 5 \
  --lr 2e-4 \
  --rank 32 \
  --alpha 64.0
```

## Dataset Format

Training data should be JSONL with the following schema:

```json
{"instruction": "Your instructions", "input": "Optional input", "output": "Desired response"}
```

Or for direct text:

```json
{"text": "Chat conversation or text to learn from"}
```

## Parameters

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 3 | Number of training epochs |
| `--lr` | 2e-4 | Learning rate |
| `--rank` | 16 | LoRA rank dimension |
| `--alpha` | 32.0 | Scaling factor |
| `--device` | cpu | Training device (cpu, cuda, rocm) |
| `--mode` | qlora | Training mode (qlora, soul-eater) |

## Training Process

1. Load base model
2. Apply LoRA adapters to attention layers
3. Compute gradient loss
4. Update adapter weights
5. Save `.grim.train` sidecar

## Serve with Trained Adapter

The trained adapter will be automatically loaded when running inference with the model:

```bash
grim run granite-3.1-8b --prompt "Your question"
```

## Adapter Features

- **Adapter-only backward pass**: Only adapter weights are trained, base weights remain frozen and quantized
- **Batch training**: Multiple samples processed in parallel
- **Deterministic**: RNG seeded by epoch and rank