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

2. **Run the Training Process**
   Execute the `train` command to train the adapter. You must specify the base model, dataset, and output path.

   ```bash
   grim train --model granite-3.1-8b --dataset ./data/alpaca.jsonl --output adapter.grim.train
   ```

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
     --device cpu
   ```

4. **Bake the Adapter (Optional)**
   You can permanently merge the trained adapter sidecar into a base model using the `merge` command.

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
- **Out of memory (OOM)**: Training requires more memory than inference. **Recovery**: Reduce `--batch-size` or use `--device cpu` to offload processing to system memory, though this will be slower.