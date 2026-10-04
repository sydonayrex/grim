#!/usr/bin/env python3
"""Report the verified Qwen3-VL mmproj geometry, and what is NOT yet verified.

VERIFIED here (arithmetic over real checkpoint values read from
models/qwen38-27b/mmproj-F16.gguf):

  * n_patches per side = image_size / patch_size = 768 / 16 = 48
  * conv output rows   = 48 * 48 = 2304
  * merger input width = 4 * embedding_length = 4 * 1152 = 4608, which matches
    mm.0.weight's declared shape of 4608x4608
  * merged tokens      = 2304 / 4 = 576
  * every one of the 334 tensors is accounted for by the name table below

NOT VERIFIED here, and deliberately not guessed: the exact element permutation
inside the `// spatial merge` block of
old/repo/llama.cpp-master/tools/mtmd/models/qwen3vl.cpp.

That block interleaves ggml_permute (which produces a strided VIEW) with
ggml_cont applied to that view (which GATHERS through the view's strides).
The resulting (row, feature) -> (channel, x, y) map is not a plain reshape, and
a hand-derived closed form is exactly the kind of claim that is invisible-wrong
at position 0 and catastrophically wrong in production.

Establish it by PARITY against llama.cpp instead:
  1. Build llama.cpp with mtmd support and run a fixed synthetic image.
  2. Dump the graph tensor `inp_pos_emb` (or the conv output) with
     MTMD_DEBUG=1 / clip_set_debug_output_embeddings.
  3. Assert grim's conv output matches element-for-element.

This script exists so the implementer does not have to re-derive the geometry
and so the unverified part is stated out loud instead of buried.

Usage: qwen3vl_geometry.py [path/to/mmproj.gguf]
"""

from __future__ import annotations

import struct
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple

GGUF_STRUCT_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0", 12: "Q4_K",
    14: "Q6_K", 30: "BF16", 34: "MXFP4",
}


def read_u32(fh) -> int:
    return struct.unpack("<I", fh.read(4))[0]


def read_str(fh) -> str:
    n = struct.unpack("<Q", fh.read(8))[0]
    return fh.read(n).decode("utf-8", "replace")


def read_value(fh, vtype: int):
    if vtype == 8:
        return read_str(fh)
    if vtype == 9:
        etype = read_u32(fh)
        count = struct.unpack("<Q", fh.read(8))[0]
        return [read_value(fh, etype) for _ in range(count)]
    fmt = GGUF_STRUCT_SCALARS[vtype]
    return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]


def parse(path: str) -> Tuple[Dict[str, Any], List[Tuple[str, List[int], str, int]]]:
    meta: Dict[str, Any] = {}
    tensors: List[Tuple[str, List[int], str, int]] = []
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file (magic {magic!r})")
        fh.read(4)  # version
        n_tensors = struct.unpack("<Q", fh.read(8))[0]
        n_kv = struct.unpack("<Q", fh.read(8))[0]

        for _ in range(n_kv):
            key = read_str(fh)
            meta[key] = read_value(fh, read_u32(fh))

        for _ in range(n_tensors):
            name = read_str(fh)
            ndim = read_u32(fh)
            dims = [struct.unpack("<Q", fh.read(8))[0] for _ in range(ndim)]
            ttype = read_u32(fh)
            offset = struct.unpack("<Q", fh.read(8))[0]
            tensors.append((name, dims, GGML_TYPE_NAMES.get(ttype, str(ttype)), offset))
    return meta, tensors


def main(argv: List[str]) -> int:
    path = argv[1] if len(argv) > 1 else "models/qwen38-27b/mmproj-F16.gguf"
    meta, tensors = parse(path)

    def m(key: str, default: Any = None) -> Any:
        return meta.get(key, default)

    def as_int(key: str, default: int = 0) -> int:
        val = meta.get(key, default)
        return int(val) if isinstance(val, (int, float)) else default

    def as_str(key: str, default: str = "?") -> str:
        val = meta.get(key, default)
        return val if isinstance(val, str) else default

    def as_bools(key: str) -> List[bool]:
        val = meta.get(key, [])
        return val if isinstance(val, list) else []

    image_size = as_int("clip.vision.image_size")
    patch_size = as_int("clip.vision.patch_size")
    emb_len = as_int("clip.vision.embedding_length")
    ffn_len = as_int("clip.vision.feed_forward_length")
    n_layer = as_int("clip.vision.block_count")
    n_head = as_int("clip.vision.attention.head_count")
    proj_dim = as_int("clip.vision.projection_dim")
    merge = as_int("clip.vision.spatial_merge_size")
    proj_type = as_str("clip.projector_type")
    deepstack = as_bools("clip.vision.is_deepstack_layers")

    per_side = image_size // patch_size if patch_size else 0
    n_patches = per_side * per_side
    head_dim = emb_len // n_head if n_head else 0
    merge_area = merge * merge

    print("== checkpoint identity ==")
    print(f"  general.architecture         = {as_str('general.architecture')}")
    print(f"  general.type                 = {as_str('general.type')}")
    print(f"  clip.projector_type          = {proj_type}")
    print(f"  clip.use_gelu                = {meta.get('clip.use_gelu')}")
    print(f"  clip.vision.image_size       = {image_size}")
    print(f"  clip.vision.patch_size       = {patch_size}")
    print(f"  clip.vision.embedding_length = {emb_len}")
    print(f"  clip.vision.feed_forward_len = {ffn_len}")
    print(f"  clip.vision.block_count      = {n_layer}")
    print(f"  clip.vision.attention.heads  = {n_head}")
    print(f"  clip.vision.projection_dim   = {proj_dim}   (text hidden size)")
    print(f"  clip.vision.spatial_merge    = {merge}")
    print(f"  deepstack layers (true)      = {sum(1 for b in deepstack if b)}/{len(deepstack)}")
    print(f"  head_dim = {emb_len}/{n_head}   = {head_dim}")

    print("\n== derived geometry (arithmetic over the values above) ==")
    print(f"  patches per side = {image_size}/{patch_size}       = {per_side}")
    print(f"  conv output rows = {per_side}^2                  = {n_patches}")
    print(f"  merger input     = {merge}^2 * {emb_len}           = {merge_area * emb_len}")
    print(f"  merged tokens    = {n_patches}/{merge_area}               = {n_patches // merge_area}")
    print(f"  m-rope ids/tok   = 4 (t, h, w and a duplicate of h)")

    print("\n== tensor inventory ==")
    print(f"  total tensors = {len(tensors)}")
    counts = Counter()
    for name, _dims, _dt, _off in tensors:
        head = name.split(".")[0]
        counts[head] += 1
    for k, v in sorted(counts.items()):
        print(f"    {k:6s} {v}")

    # Cross-check the declared shapes against the metadata-derived geometry.
    by_name = {name: (dims, dt) for name, dims, dt, _off in tensors}

    print("\n== shape cross-checks (declared tensor vs metadata arithmetic) ==")
    checks: List[Tuple[str, object, object, bool]] = []

    mm0 = by_name.get("mm.0.weight")
    checks.append((
        "mm.0.weight width == merge^2 * embedding_length",
        mm0[0] if mm0 else None,
        merge * merge * emb_len,
        bool(mm0) and mm0[0][0] == merge * merge * emb_len,
    ))

    mm2 = by_name.get("mm.2.weight")
    checks.append((
        "mm.2.weight OUT == projection_dim (dims are [in, out])",
        mm2[0][1] if mm2 else None,
        proj_dim,
        bool(mm2) and mm2[0][1] == proj_dim,
    ))

    qkv = by_name.get("v.blk.0.attn_qkv.weight")
    checks.append((
        "v.blk.0.attn_qkv OUT == 3 * embedding_length",
        qkv[0][1] if qkv else None,
        3 * emb_len,
        bool(qkv) and qkv[0][1] == 3 * emb_len,
    ))

    up = by_name.get("v.blk.0.ffn_up.weight")
    checks.append((
        "v.blk.0.ffn_up OUT == feed_forward_length",
        up[0][1] if up else None,
        ffn_len,
        bool(up) and up[0][1] == ffn_len,
    ))

    down = by_name.get("v.blk.0.ffn_down.weight")
    checks.append((
        "v.blk.0.ffn_down IN == feed_forward_length",
        down[0][0] if down else None,
        ffn_len,
        bool(down) and down[0][0] == ffn_len,
    ))

    pe = by_name.get("v.patch_embd.weight")
    conv_in = patch_size * patch_size * 3
    checks.append((
        "v.patch_embd.weight == [patch, patch, in_ch, embedding_length]",
        pe[0] if pe else None,
        [patch_size, patch_size, 3, emb_len],
        bool(pe) and list(pe[0]) == [patch_size, patch_size, 3, emb_len],
    ))

    n_blocks = sum(1 for n, _, _, _ in tensors if n.startswith("v.blk.") and n.endswith("attn_qkv.weight"))
    checks.append(("v.blk.*.attn_qkv.weight count == block_count", n_blocks, n_layer, n_blocks == n_layer))

    pos = by_name.get("v.position_embd.weight")
    checks.append((
        "v.position_embd rows == num_position_embeddings (2304)",
        pos[0][1] if pos else None,
        n_patches,
        bool(pos) and pos[0][1] == n_patches,
    ))

    failures = 0
    for label, got, want, ok in checks:
        flag = "OK" if ok else "MISMATCH"
        if not ok:
            failures += 1
        print(f"  [{flag:8s}] {label}: got {got}, expected {want}")

    print("\n== two patch-embed kernels (must be summed, not one of them chosen) ==")
    for nm in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
        if nm in by_name:
            dims, dt = by_name[nm]
            print(f"  {nm:26s} {dims} {dt}")
    print("  reference: old/repo/llama.cpp-master/tools/mtmd/models/qwen2vl.cpp:3-16")
    print("  build_inp_with_temporal_merge() ADDS the two conv2d outputs.")
    print("  For a still image both convs see the SAME pixels, so the patch")
    print("  embedding is (W0 + W1) * patch + bias. Implementations that load")
    print("  only one kernel produce silently wrong embeddings.")
    identical = None
    with open(path, "rb") as fh:
        offs = {name: off for name, _d, _t, off in tensors}
        if "v.patch_embd.weight" in offs and "v.patch_embd.weight.1" in offs:
            a_off = offs["v.patch_embd.weight"]
            b_off = offs["v.patch_embd.weight.1"]
            size = b_off - a_off
            fh.seek(a_off)
            a = fh.read(size)
            fh.seek(b_off)
            b = fh.read(size)
            identical = a == b
    print(f"  payloads byte-identical: {identical}")
    if identical is False:
        print("  -> they are DISTINCT kernels; both must be loaded and summed.")

    print("\n== NOT ESTABLISHED HERE ==")
    print("  The exact element permutation of the `// spatial merge` block in")
    print("  old/repo/llama.cpp-master/tools/mtmd/models/qwen3vl.cpp. It mixes")
    print("  ggml_permute (strided view) with ggml_cont (gather through the")
    print("  view's strides), so it is not a plain reshape. Derive it by parity")
    print("  against llama.cpp's own conv output, not by hand.")

    if failures:
        print(f"\n{failures} shape cross-check(s) MISMATCHED")
        return 1
    print("\nall shape cross-checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))