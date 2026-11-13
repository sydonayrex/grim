#!/usr/bin/env python3
"""Extract a small single-GPU probe model from a big Qwen35 GGUF.

Keeps the real tokenizer-facing metadata (vocab/hidden/head config) and the
first N decoder blocks (KDA + attention paths both exercised by the interval-4
pattern), drops the rest and any nextn block, overrides block_count, and
rewrites a spec-aligned GGUF v3. The parity/window probes then run entirely in
VRAM on ONE GPU — no managed-memory fallback, no allocator pressure — instead
of forcing a 16 GB model onto a 16 GB card.

Usage: extract_probe_model.py <src.gguf> <dst.gguf> [keep_blocks=8]
"""

import struct
import sys

GGUF_TYPE_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
GGUF_TYPE_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
GGML_TYPE_SIZE = {
    0: 4, 1: 2, 2: 18, 3: 24, 6: 22, 7: 26, 8: 34, 9: 36,
    10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 15: 196,
}


def read_header(f):
    magic = f.read(4)
    assert magic == b"GGUF", magic
    ver, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
    return ver, n_tensors, n_kv


def read_string(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def read_value(f, t):
    if t == 8:
        return read_string(f)
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        (n,) = struct.unpack("<Q", f.read(8))
        if et == 8:
            vals = [read_string(f) for _ in range(n)]
            return ("arr_str", et, vals)
        sz = GGUF_TYPE_SIZE[et]
        raw = f.read(n * sz)
        return ("arr", et, raw)
    fmt = GGUF_TYPE_FMT[t]
    sz = struct.calcsize(fmt)
    return ("scalar", t, f.read(sz))


def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    keep_blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    keep_carry_nextn = (len(sys.argv) > 4 and sys.argv[4] == "nextn")

    with open(src_path, "rb") as f:
        ver, n_tensors, n_kv = read_header(f)

        kvs = []  # (name, type, raw_value_bytes_after_type_field)
        block_count_key = None
        for _ in range(n_kv):
            name = read_string(f)
            (t,) = struct.unpack("<I", f.read(4))
            start = f.tell()
            read_value(f, t)
            end = f.tell()
            f.seek(start)
            val_bytes = f.read(end - start)
            f.seek(end)
            kvs.append((name, t, val_bytes))
            if "block_count" in name:
                block_count_key = name

        tensors = []
        for _ in range(n_tensors):
            name = read_string(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            (ttype,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            tensors.append((name, nd, dims, ttype, off))

        pos = f.tell()
        # alignment KV (default 32 per spec)
        alignment = 32
        for (name, t, raw) in kvs:
            if name == "general.alignment" and t == 4:
                alignment = struct.unpack("<I", raw)[0]

        def align_up(x, a):
            return (x + a - 1) // a * a

        data_start = align_up(pos, alignment)

        # Real per-tensor byte sizes from the offset layout: tensors are
        # contiguous (aligned) in the data section, so each tensor's size
        # including its alignment padding is (next_offset - offset). No type
        # tables needed and no quant-ratio mistakes.
        import os
        file_size = os.path.getsize(src_path)
        by_offset = sorted(tensors, key=lambda t: t[4])
        sizes = {}
        for i, (name, nd, dims, ttype, off) in enumerate(by_offset):
            if i + 1 < len(by_offset):
                sizes[name] = by_offset[i + 1][4] - off
            else:
                sizes[name] = file_size - (data_start + off)
        kept = []  # (dest_name, nd, dims, ttype, src_off, src_name)
        for (name, nd, dims, ttype, off) in tensors:
            if name in ("token_embd.weight", "output_norm.weight", "output.weight"):
                kept.append((name, nd, dims, ttype, off, name))
            elif name.startswith("blk."):
                idx = int(name.split(".")[1])
                if idx < keep_blocks:
                    kept.append((name, nd, dims, ttype, off, name))
                elif idx == 64 and keep_carry_nextn:
                    # Carry the source's nextn/MTP block, renamed to sit right
                    # after the kept trunk, so the loader sees has_mtp() and
                    # the MTP gates can run on the probe model.
                    kept.append((name.replace("blk.64.", f"blk.{keep_blocks}."), nd, dims, ttype, off, name))
        total_bytes = sum(sizes[src] for (_, _, _, _, _, src) in kept)
        print(f"kept {len(kept)}/{n_tensors} tensors, {total_bytes/2**30:.2f} GiB")

        # New tensor table: compacted offsets, sizes carry the source's
        # alignment padding so the destination stays aligned for free.
        new_infos = []
        cur = 0
        for (name, nd, dims, ttype, off, src) in kept:
            nbytes = sizes[src]
            new_infos.append((name, nd, dims, ttype, cur, nbytes))
            cur += nbytes

        with open(dst_path, "wb") as o:
            o.write(b"GGUF")
            o.write(struct.pack("<IQQ", ver, len(new_infos), n_kv))
            for (name, t, raw) in kvs:
                nb = name.encode("utf-8")
                o.write(struct.pack("<Q", len(nb)))
                o.write(nb)
                o.write(struct.pack("<I", t))
                o.write(raw)
            for (name, nd, dims, ttype, off, _) in new_infos:
                nb = name.encode("utf-8")
                o.write(struct.pack("<Q", len(nb)))
                o.write(nb)
                o.write(struct.pack("<I", nd))
                o.write(struct.pack(f"<{nd}Q", *dims))
                o.write(struct.pack("<I", ttype))
                o.write(struct.pack("<Q", off))
            # pad to alignment, then copy tensor data
            end_infos = o.tell()
            pad = data_start_pos = align_up(end_infos, alignment)
            o.write(b"\0" * (pad - end_infos))
            with open(src_path, "rb") as sf:
                for (name, nd, dims, ttype, off, nbytes) in new_infos:
                    sf.seek(data_start + off)
                    remaining = nbytes
                    chunk = 1 << 24
                    while remaining > 0:
                        b = sf.read(min(chunk, remaining))
                        assert b, f"short read at {name}"
                        o.write(b)
                        remaining -= len(b)

        print(f"wrote {dst_path}")

        # fix block_count in the KV raw bytes? The block_count KV is a scalar
        # u32 inside kvs raw bytes — rewrite it in place.
    # Post-fix block_count by regenerating: simplest is to re-run with an
    # override — handled here by patching the raw bytes before write. To keep
    # the code simple we patch the destination file: find the block_count KV.
    # (It is a u32 scalar; its byte offset in kvs raw is deterministic.)
    # Instead of fragile patching, we re-serialize KVs with the override.
    # NOTE: the block above already wrote; do a second pass rewrite.
    with open(dst_path, "r+b") as o:
        # locate block_count KV by re-parsing the destination header
        o.seek(0)
        data = o.read()
        i = 12  # magic + version + tensor_count + kv_count = 4+4+8+8 = 24? fix:
        i = 4 + 4 + 8 + 8
        for _ in range(n_kv):
            (nl,) = struct.unpack("<Q", data[i:i + 8])
            name = data[i + 8:i + 8 + nl].decode()
            t_off = i + 8 + nl
            (t,) = struct.unpack("<I", data[t_off:t_off + 4])
            v_off = t_off + 4
            effective_blocks = keep_blocks + (1 if keep_carry_nextn else 0)
            if name == block_count_key and t == 4:
                o.seek(v_off)
                o.write(struct.pack("<I", effective_blocks))
            # skip value
            if t == 8:
                (sl,) = struct.unpack("<Q", data[v_off:v_off + 8])
                i = v_off + 8 + sl
            elif t == 9:
                et = struct.unpack("<I", data[v_off:v_off + 4])[0]
                (an,) = struct.unpack("<Q", data[v_off + 4:v_off + 12])
                if et == 8:
                    j = v_off + 12
                    for _ in range(an):
                        (sl,) = struct.unpack("<Q", data[j:j + 8])
                        j += 8 + sl
                    i = j
                else:
                    i = v_off + 12 + an * GGUF_TYPE_SIZE[et]
            else:
                i = v_off + GGUF_TYPE_SIZE[t]
    print(f"block_count -> {keep_blocks} ({block_count_key})")


if __name__ == "__main__":
    main()
