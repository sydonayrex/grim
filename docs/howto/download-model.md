# How to Download Models

Grim supports multiple download sources: Ollama registry, Hugging Face, and direct URLs.

## From Ollama Registry

```bash
# Short name (Ollama tag)
grim pull llama3

# With quantization tag
grim pull llama3:q4_k_m
grim pull llama3:latest
```

## From Hugging Face Hub

```bash
# Direct file specification
grim pull hf:org/repo/model.gguf

# Repository (automated GGUF detection)
grim pull org/repo

# With output path
grim dl hf:org/repo/model.gguf --output /path/to/model.gguf
```

## From Direct URL

```bash
# Any HTTPS URL to a model file
grim pull https://huggingface.co/org/repo/resolve/main/model.gguf

# With output path
grim pull https://example.com/model.gguf --output ./model.gguf
```

## Resolve Model Name

To check if a model exists in your cache:

```bash
grim check
grim list
```

## Model Cache Location

Models are stored in:
- System: `/var/lib/grim/models/`
- User: `~/.grim/models/`

Override with:
```bash
export GRIM_MODELS_DIR=/custom/path
```

## Download Progress

Downloads show real-time progress:
```
  [ 45%] 2.15 / 4.80 GB
[grim] downloading
[grim] success
```