# How to Run Inference

## Goal

Run local inference for one-shot generations, interactive chats, or start an OpenAI-compatible HTTP server.

## Prerequisites

- Grim is installed and available in your PATH.
- You have downloaded a valid model file.

## Steps

1. **Run a simple prompt (One-shot inference)**
   Execute a prompt and exit immediately. Provide the model name and the prompt as positional arguments.

   ```bash
   grim run llama3 "Hello, world!"
   ```

2. **Run an interactive chat session**
   Omit the prompt to enter interactive mode.

   ```bash
   grim run llama3
   ```

3. **Start the HTTP Server**
   To start the OpenAI-compatible HTTP server on the default port (11434):

   ```bash
   grim serve
   ```

   You can specify a custom bind address:

   ```bash
   grim serve --address 0.0.0.0:8080
   ```

4. **Serve a model during inference**
   You can also spin up the server directly from the `run` command.

   ```bash
   grim run llama3 --serve --address 127.0.0.1:11434
   ```

5. **Benchmark Performance**
   Run a smoke test or a customized benchmark to test throughput.

   ```bash
   grim bench --model llama3 --tokens 256 --concurrency 4
   ```

## Expected Output

For one-shot execution, the model will output text directly to standard output.
When running `grim serve`, you will see:
```
[grim] server listening on 127.0.0.1:11434
```
You can then test the server via HTTP:
```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Sampling

Grim supports fine-grained sampling parameter configuration:

- **Server Defaults**: When launching `grim run --serve` or `grim serve`, command-line flags (`--temperature`, `--top-p`, `--top-k`, `--repeat-penalty`, `--seed`) set the default sampling policy for requests that omit these parameters.
- **Per-Request Overrides**: Clients sending requests to `POST /v1/chat/completions` or `POST /api/generate` can override any sampling parameter per request in the JSON payload (e.g. `"temperature": 0.2`, `"top_p": 0.95`).

---

## What Can Go Wrong

- **Model not found**: If the requested model is not downloaded or the path is incorrect, Grim will return a missing model error. **Recovery**: Ensure the model is available by running `grim check` or pull it using `grim dl`.
- **Address in use**: If another service (like Ollama) is already using port 11434, the server will fail to bind. **Recovery**: Use a different port via `--address 127.0.0.1:8080` or stop the conflicting service.