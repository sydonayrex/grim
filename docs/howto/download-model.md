# How to Download Models

## Goal

Download models into your local Grim cache from the Ollama registry, Hugging Face, or direct URLs.

## Prerequisites

- Grim is installed.
- You have network access to the model registries.

## Steps

1. **Pull from the Ollama Registry**
   Use the short name (Ollama tag) to download a model.

   ```bash
   grim pull llama3
   
   # Or with a specific tag:
   grim pull llama3:q4_k_m
   ```

2. **Pull from Hugging Face**
   Specify the `hf:` scheme or directly refer to the repository and file.

   ```bash
   grim pull hf:org/repo/model.gguf
   
   # To specify a custom output path, use -o or --output:
   grim dl hf:org/repo/model.gguf -o /path/to/custom.gguf
   ```

3. **Pull from a Direct URL**
   Provide any HTTPS URL to a model file.

   ```bash
   grim pull https://example.com/model.gguf
   
   # With a custom output path:
   grim pull https://example.com/model.gguf --output ./model.gguf
   ```

4. **Verify the Download**
   Check your local cache to confirm the model was downloaded successfully.

   ```bash
   grim check
   ```

## Expected Output

During the download, you will see real-time progress:
```
  [ 45%] 2.15 / 4.80 GB
[grim] downloading
[grim] success
```
When running `grim check`, the downloaded model will appear in your local cached list.

## What Can Go Wrong

- **Network timeouts**: The download may stall or fail due to network instability. **Recovery**: Re-run the `grim pull` command; Grim will attempt to resume the partial download if the server supports range requests.
- **Out of disk space**: The model may be larger than available storage. **Recovery**: Delete unused models using `grim rm <model>` or specify a different partition using `--output`.