#!/usr/bin/env python3
"""Dump a GGUF file's metadata and tensor table without any third-party deps.

Usage: gguf-tensor-dump.py <file.gguf> [--tensors]

Prints KV metadata grouped and, with --tensors, the full tensor table with
layer indices collapsed so repeating block structure is visible.
"""

from __future__ import annotations

import re
import struct
import sys
from collections import Counter

GGUF_TYPE_NAMES = {
    0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
    7: "bool", 8: "str", 9: "arr", 10: "u64", 11: "i64", 12: "f64",
}
GGUF_TYPE_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    30: "BF16", 34: "MXFP4",
}


class Reader:
    """Sequential little-endian GGUF reader."""

    def __init__(self, path: str) -> None:
        self.fh = open(path, "rb")
        self.path = path

    def scalar(self, vtype: int):
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype = struct.unpack("<I", self.fh.read(4))[0]
            count = struct.unpack("<Q", self.fh.read(8))[0]
            return [self.scalar(etype) for _ in range(count)]
        fmt = GGUF_TYPE_FMT[vtype]
        return struct.unpack(fmt, self.fh.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = struct.unpack("<Q", self.fh.read(8))[0]
        return self.fh.read(n).decode("utf-8", "replace")

    def close(self) -> None:
        self.fh.close()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    show_tensors = "--tensors" in argv
    reader = Reader(argv[1])
    fh = reader.fh

    magic = fh.read(4)
    version = struct.unpack("<I", fh.read(4))[0]
    n_tensors = struct.unpack("<Q", fh.read(8))[0]
    n_kv = struct.unpack("<Q", fh.read(8))[0]
    print(f"file   {argv[1]}")
    print(f"magic  {magic!r} version {version}")
    print(f"tensors {n_tensors}  kv {n_kv}\n")

    print("== metadata ==")
    for _ in range(n_kv):
        key = reader.string()
        vtype = struct.unpack("<I", fh.read(4))[0]
        value = reader.scalar(vtype)
        if isinstance(value, list) and len(value) > 8:
            shown = f"[{len(value)} items] {value[:4]}"
        else:
            shown = str(value)
        print(f"  {key} = {shown}")

    print("\n== tensor table ==")
    rows = []
    for _ in range(n_tensors):
        name = reader.string()
        ndim = struct.unpack("<I", fh.read(4))[0]
        dims = [struct.unpack("<Q", fh.read(8))[0] for _ in range(ndim)]
        ttype = struct.unpack("<I", fh.read(4))[0]
        offset = struct.unpack("<Q", fh.read(8))[0]
        rows.append((name, dims, GGML_TYPE_NAMES.get(ttype, str(ttype)), offset))
    reader.close()

    if not show_tensors:
        counts = Counter(re.sub(r"\.\d+\.", ".N.", name) for name, _, _, _ in rows)
        for pattern, count in sorted(counts.items()):
            print(f"  {count:4d}  {pattern}")
        return 0

    for name, dims, dtype, offset in rows:
        shape = "x".join(str(d) for d in dims)
        print(f"  {name:66s} {shape:>18s} {dtype:8s} off={offset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))