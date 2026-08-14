# How to Convert Models

## Goal

Convert models from GGUF/safetensors to the native `.grim` format optimized for ROCm, optionally calibrating and searching for optimal quantization.

## Prerequisites

- Grim is installed.
- You have downloaded a source model (e.g., `model.gguf`).

## Steps

1. **Basic Conversion**
   You can use the top-level `convert` command with flags for input and output.

   ```bash
   grim convert -i model.gguf -o model.grim
   
   # With a target bits-per-weight and GPU target:
   grim convert -i model.gguf -o model.grim --target-bpw 4.0 --target gfx1100
   ```

2. **Advanced Pipeline: Calibrate**
   For higher quality quantization, calibrate the model to generate importance scores. (Note: These arguments are positional).

   ```bash
   grim oxidizer calibrate model.gguf model_scores.json
   
   # With a custom dataset:
   grim oxidizer calibrate model.gguf model_scores.json --dataset /path/to/datasets
   ```

3. **Advanced Pipeline: Search**
   Run the EvoPress evolutionary search on the pre-computed importance scores. You must provide the scores path and a comma-separated list of tensor sizes.

   ```bash
   grim oxidizer search model_scores.json "4096x4096,4096x11008" --target-bpw 4.0 --generations 50
   ```

4. **Advanced Pipeline: Convert**
   Execute the full conversion pipeline using the `oxidizer convert` subcommand (which takes positional arguments for files and uses `--profile` instead of `--target`).

   ```bash
   grim oxidizer convert model.gguf model.grim --profile gfx1100
   ```

5. **Prepare for Training**
   Create a training-ready artifact from a base checkpoint.

   ```bash
   grim oxidizer prepare model.gguf model.train.grim
   
   # With bf16 format:
   grim oxidizer prepare model.gguf model.train.grim --format bf16
   ```

6. **Fuse for Performance**
   Analyze a checkpoint and bake ROCm fusion hints into the output artifact.

   ```bash
   grim oxidizer fuse model.gguf model.fused.grim
   ```

## Expected Output

When converting, you will see a progress bar and metadata about the conversion:
```
[grim] converting model.gguf -> model.grim
[grim] running EvoPress...
[grim] conversion complete
```

## What Can Go Wrong

- **Invalid model format**: Grim may reject unsupported or corrupted files. **Recovery**: Ensure you are using a valid, intact `.gguf` or supported format file.
- **Out of memory during calibration**: Calibrating large models requires substantial RAM/VRAM. **Recovery**: Use a machine with more memory or enable offloading via `--gpu` if applicable.