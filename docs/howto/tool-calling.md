# How-To: Tool Calling Loop in Grim

Grim's inference server supports OpenAI-compatible function and tool calling.

## 1. Tool Calling Architecture

1. **Client Request**: Send `tools` and `tool_choice: "auto"` in `POST /v1/chat/completions`.
2. **Model Response**: Model returns `tool_calls` containing function name and JSON arguments.
3. **Execution**: Client runs tool locally.
4. **Follow-up Request**: Send conversation history with tool result under `role: "tool"`.

---

## 2. Step-by-Step Example

### Step 1: Send Request with Tool Definition

```bash
curl -s http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "What is the weather in Austin, TX?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get current weather for a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City and state, e.g. Austin, TX"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

**Expected Response**:
```json
{
  "id": "chatcmpl-1",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\": \"Austin, TX\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

---

### Step 2: Execute Tool and Return Result

Append assistant message and tool response:

```bash
curl -s http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "What is the weather in Austin, TX?"},
      {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\": \"Austin, TX\"}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "name": "get_current_weather",
        "content": "{\"temperature\": 85, \"unit\": \"fahrenheit\", \"condition\": \"Sunny\"}"
      }
    ]
  }'
```

**Final Answer Response**:
```json
{
  "id": "chatcmpl-2",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The weather in Austin, TX is currently sunny and 85°F."
      },
      "finish_reason": "stop"
    }
  ]
}
```
