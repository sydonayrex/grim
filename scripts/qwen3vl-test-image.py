#!/usr/bin/env python3
"""Write a deterministic, asymmetric test PNG for the Qwen3-VL parity oracle.

Reproducibility is the point: the same bytes must reach llama.cpp and grim, or a
mismatch cannot be attributed to the spatial-merge permutation.

The image is deliberately NOT a solid colour. A constant field is invariant under
any spatial permutation, so a permuted 2x2 merge would still agree with the
oracle and WI-2 would pass while the encoder were wrong. This emits an
asymmetric low-frequency ramp in all three channels at different rates.

Usage: qwen3vl-test-image.py <out.png> [size]
"""

import struct
import sys
import zlib


def chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    out = argv[1]
    size = int(argv[2]) if len(argv) > 2 else 768
    if size < 2:
        print("size must be >= 2", file=sys.stderr)
        return 2

    raw = bytearray()
    last = float(size - 1)
    for y in range(size):
        raw.append(0)  # PNG filter type 0 (None)
        for x in range(size):
            raw.append(int(x * 255 / last))
            raw.append(int(y * 255 / last))
            raw.append(int((x + y) * 255 / (2 * last)))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    with open(out, "wb") as fh:
        fh.write(png)
    print(f"{out}: {size}x{size} RGB, {len(png)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
