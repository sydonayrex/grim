import json

log_path = "/home/nelson/.gemini/antigravity/brain/2ff05964-79af-452e-a5d4-3fd76896c606/.system_generated/logs/transcript_full.jsonl"

with open("all_pretty_edits_curr.txt", "w") as out:
    with open(log_path) as f:
        for line_num, line in enumerate(f):
            if "replace_file_content" not in line and "multi_replace_file_content" not in line:
                continue
            obj = json.loads(line)
            for tc in obj.get("tool_calls", []):
                if tc["name"] in ["replace_file_content", "multi_replace_file_content"]:
                    out.write(f"=== TOOL CALL {line_num}: {tc['name']} ===\n")
                    out.write(f"Description: {tc['args'].get('Description')}\n")
                    out.write(f"TargetFile: {tc['args'].get('TargetFile')}\n")
                    
                    if tc["name"] == "replace_file_content":
                        out.write(f"StartLine: {tc['args'].get('StartLine')}\n")
                        out.write(f"EndLine: {tc['args'].get('EndLine')}\n")
                        out.write(f"TargetContent:\n{tc['args'].get('TargetContent')}\n")
                        out.write(f"ReplacementContent:\n{tc['args'].get('ReplacementContent')}\n\n")
                    else:
                        chunks = tc["args"]["ReplacementChunks"]
                        if isinstance(chunks, str):
                            try:
                                chunks = json.loads(chunks, strict=False)
                            except Exception as e:
                                pass
                        if isinstance(chunks, list):
                            for i, chunk in enumerate(chunks):
                                out.write(f"  --- Chunk {i} ---\n")
                                out.write(f"  StartLine: {chunk.get('StartLine')}\n")
                                  # Fix indentation to match:
                                out.write(f"  EndLine: {chunk.get('EndLine')}\n")
                                out.write(f"  TargetContent:\n{chunk.get('TargetContent')}\n")
                                out.write(f"  ReplacementContent:\n{chunk.get('ReplacementContent')}\n\n")
