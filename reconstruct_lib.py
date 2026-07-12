import json
import shutil

lib_path = "crates/grim-backend-rocm/src/lib.rs"
backup_path = "crates/grim-backend-rocm/unzipped/grim-backend-rocm/src/lib.rs"
log2_path = "/home/nelson/.gemini/antigravity/brain/2ff05964-79af-452e-a5d4-3fd76896c606/.system_generated/logs/transcript_full.jsonl"

# Copy backup to lib.rs
shutil.copyfile(backup_path, lib_path)

with open(lib_path) as f:
    content = f.read()

print("Initial backup lib.rs size:", len(content))

# Apply edits from current session (log2) up to line 688
with open(log2_path) as f:
    for line_num, line in enumerate(f):
        if line_num >= 689:
            break
        if "replace_file_content" not in line and "multi_replace_file_content" not in line:
            continue
        obj = json.loads(line)
        for tc in obj.get("tool_calls", []):
            if tc["name"] in ["replace_file_content", "multi_replace_file_content"] and "lib.rs" in tc["args"].get("TargetFile", ""):
                desc = tc['args'].get('Description')
                print(f"Applying log2 line {line_num}: {desc}")
                if tc["name"] == "replace_file_content":
                    target = tc["args"]["TargetContent"]
                    repl = tc["args"]["ReplacementContent"]
                    if target in content:
                        content = content.replace(target, repl, 1)
                    else:
                        print(f"  Warning: TargetContent not found! Start: {repr(target[:60])}")
                else:
                    chunks = tc["args"]["ReplacementChunks"]
                    if isinstance(chunks, str):
                        chunks = json.loads(chunks, strict=False)
                    for idx, chunk in enumerate(chunks):
                        target = chunk["TargetContent"]
                        repl = chunk["ReplacementContent"]
                        if target in content:
                            content = content.replace(target, repl, 1)
                        else:
                            print(f"  Warning: Chunk {idx} target not found! Start: {repr(target[:60])}")

print("Final lib.rs size:", len(content))

with open(lib_path, "w") as f:
    f.write(content)
print("Reconstructed lib.rs successfully written.")
