# How to Convert Models

Grim converts models from GGUF/safetensors to its native `.grim` format optimized for ROCm.

## Convert GGUF to .grim

```bash
# Basic conversion
grim oxidizer convert -i model.gguf -o model.grim

# With quantization for ROCm
grim oxidizer convert -i model.gguf -o model.grim --target_bpw 4.0

# With generations (EvoPress search)
grim oxidizer convert -i model.gguf -o model.grim --generations 100

# Profile for specific GPU
grim oxidizer convert -i model.gguf -o model.grim --profile gfx1100
```

## Calibrate for Optimization

```bash
# Run importance-matrix calibration
grim oxidizer calibrate -m model.gguf -o model

# With custom dataset
grim oxidizer calibrate -m model.gguf -o model --dataset /path/to/datasets
```

## Run EvoPress Evolution

```bash
# Evolve quantization parameters
grim oxidizer search scores.json -o model --target_bpw 4.0 --generations 50
```

## Prepare Training Artifact

```bash
# Create train-ready artifact
grim oxidizer prepare -i model.gguf -o model.train

# With BF16 precision
grim oxidizer prepare -i model.gguf -o model.train --format bf16
```

## Fuse for Better Performance

```bash
# Analyze and bake fusion hints
grim oxidizer fuse -i model.gguf -o model.fused
```

## ROCm Profiles

| Profile | GPU Family |
|---|---|
| `gfx900` | Vega 10/20 (default) |
| `gfx906` | MI50/MI60 |
| `gfx1030` | RX 6000 series |
| `gfx1100` | RX 7000 series |
| `gfx1101` | RX 7000 XT |
| `gfx1200` | RX 8000 series |
| `gfx1201` | RX 8000 XT |

Auto-detection is used when `--profile` is omitted.