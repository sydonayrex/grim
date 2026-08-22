# How to Train LoRA Adapters

## Goal

Train or fine-tune LoRA adapters (SFT QLoRA) on a base model using a custom dataset, and serve the result.

## Prerequisites

- Grim is installed.
- A base model is available locally (e.g., downloaded via `grim pull`).
- You have a training dataset in JSONL format.

## Steps

1. **Prepare the Dataset**
   Ensure your data is formatted correctly as JSONL. It can follow the instruction format or direct text format:

   ```json
   {"instruction": "Your instructions", "input": "Optional input", "output": "Desired response"}
   ```
   Or:
   ```json
   {"text": "Chat conversation or text to learn from"}
   ```

2. **Run Quick LoRA Training (`--quick`)**
   For fast experimentation without tweaking dozens of hyperparameter flags, use `--quick`:

   ```bash
   grim train --quick --model granite-3.1-8b --dataset ./data/alpaca.jsonl
   ```
   This automatically configures a lightweight LoRA preset (`rank=8`, `alpha=16.0`, `epochs=1`, `device=cpu`, output=`adapter.grim.train`).

3. **Train with Custom Hyperparameters**
   You can adjust the learning rate, epochs, LoRA rank, and alpha for better convergence.

   ```bash
   grim train \
     --model llama3 \
     --dataset ./data/train.jsonl \
     --output my-adapter.grim.train \
     --epochs 5 \
     --lr 2e-4 \
     --rank 32 \
     --alpha 64.0 \
     --seed 42 \
     --device rocm
   ```

4. **Choosing a Training Mode**
   Grim supports several fine-tuning and alignment modes via `--mode`:
   - **SFT & Adapter Modes**:
     - `qlora` (default): Quantized LoRA with frozen 4-bit base weights and trainable adapter matrices (lowest VRAM).
     - `lora`: Standard LoRA on full-precision / half-precision base weights.
     - `full-bf16` / `full-fp16`: Full parameter fine-tuning in BF16 or FP16.
     - `soul-eater`: spectral QLoRA initialization (the dedicated SoulEater adapter/optimizer is not yet wired to this mode).
     - `oft`: Orthogonal Fine-Tuning preserving representation norms.
   - **Preference Alignment Modes**:
     - `dpo`: Direct Preference Optimization using paired chosen and rejected target sequences with analytical VJP gradients.
     - `kto`: Kahneman-Tversky Optimization applying prospect theory to unpaired or paired preferences.
     - `simpo`: Simple reference-free preference optimization with target margin and length normalization.
     - `orpo`: Odds Ratio Preference Optimization penalizing rejected sequence probability odds.
     - `grpo`: Group Relative Policy Optimization with clipped surrogate loss and reward advantage normalization.

5. **Multi-GPU Data-Parallel (DP) Training**
   Scale training across multiple AMD GPUs using RCCL all-reduce
   (single-process multi-device gradient sync; per-GPU model replicas are on
   the roadmap — today each batch is computed once and gradients sync across
   devices):
   ```bash
   grim train \
     --model llama3 \
     --dataset ./data/train.jsonl \
     --num-gpus 2 \
     --mode dpo \
     --batch-size 2048
   ```

6. **Bake the Adapter (Optional)**
   You can permanently merge the trained adapter sidecar into a base model using the `merge` command:

   ```bash
   grim merge --model llama3 --adapter my-adapter.grim.train --output merged_model.grim
   ```

## Expected Output

During training, Grim will stream loss metrics and progress per epoch:
```
[grim] Epoch 1/5 | Loss: 1.45 | LR: 0.0002
...
[grim] saved adapter to adapter.grim.train
```

## What Can Go Wrong

- **Loss divergence (NaN)**: The learning rate might be too high or the batch size too large. **Recovery**: Reduce `--lr` (e.g., to `1e-5`) or increase `--gradient-accumulation-steps`.
- **Out of memory (OOM)**: Training requires more memory than inference. **Recovery**: Reduce `--batch-size`, enable `--qat-mxfp4`, or scale across GPUs with `--num-gpus`.