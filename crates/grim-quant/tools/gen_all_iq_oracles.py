#!/usr/bin/env python3
import os
import re
import subprocess
import sys

REPO = os.environ.get(
    "LLAMA_CPP_REF",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "old", "repo", "llama.cpp-master"),
)
GGML_SRC = os.path.join(REPO, "ggml", "src")
OUT_DIR = "/tmp/iq_oracles"

def read(path: str) -> str:
    with open(os.path.join(GGML_SRC, path)) as f:
        return f.read()

def extract_function(src: str, signature: str) -> str:
    i = src.find(signature)
    if i < 0:
        raise SystemExit(f"signature not found: {signature!r}")
    start = src.index("{", i)
    depth = 0
    for j in range(start, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i : j + 1]
    raise SystemExit(f"unbalanced braces for: {signature!r}")

def extract_table(src: str, name: str) -> str:
    m = re.search(rf"GGML_TABLE_BEGIN\((\w+), {name}, (\d+)\)", src)
    if m:
        start = src.index("\n", m.start()) + 1
        end = src.index("GGML_TABLE_END()", start)
        return f"static const {m.group(1)} {name}[{m.group(2)}] = {{{src[start:end]}}};"
    raise SystemExit(f"table not found: {name}")

def parse_table_entries(src: str, name: str):
    m = re.search(rf"GGML_TABLE_BEGIN\((\w+), {name}, (\d+)\)", src)
    if not m:
        raise SystemExit(f"table not found: {name}")
    start = src.index("\n", m.start()) + 1
    end = src.index("GGML_TABLE_END()", start)
    raw = src[start:end]
    entries = [s.strip() for s in raw.split(",") if s.strip()]
    return m.group(1), int(m.group(2)), entries

quants_c = read("ggml-quants.c")
impl_h = read("ggml-impl.h")
common_h = read("ggml-common.h")

fp16_bits = "".join(
    extract_function(impl_h, sig)
    for sig in (
        "static inline float fp32_from_bits(",
        "static inline uint32_t fp32_to_bits(",
        "static inline float ggml_compute_fp16_to_fp32(",
        "static inline ggml_fp16_t ggml_compute_fp32_to_fp16(",
    )
)

os.makedirs(OUT_DIR, exist_ok=True)

# Generate Rust table code
rust_tables_path = os.path.join(OUT_DIR, "extracted_tables.rs")
with open(rust_tables_path, "w") as f:
    for tbl in ["ksigns_iq2xs", "iq2xxs_grid", "iq2xs_grid", "iq3xxs_grid", "iq3s_grid"]:
        c_type, count, entries = parse_table_entries(common_h, tbl)
        rust_type = {
            "uint8_t": "u8",
            "uint16_t": "u16",
            "uint32_t": "u32",
            "uint64_t": "u64",
            "int8_t": "i8",
        }[c_type]
        f.write(f"pub static {tbl.upper()}: [{rust_type}; {count}] = [\n")
        line = "    "
        for i, e in enumerate(entries):
            line += f"{e}, "
            if (i + 1) % 4 == 0 or i == len(entries) - 1:
                f.write(line + "\n")
                line = "    "
        f.write("];\n\n")

print(f"Generated {rust_tables_path}")

# Common C header for oracles
c_header = f"""#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32((ggml_fp16_t)(x))
#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16((float)(x))
#define QK_K 256
#define QK4_NL 32
#define GGML_RESTRICT __restrict__
typedef uint16_t ggml_fp16_t;
typedef uint16_t ggml_half_t;

typedef struct {{ ggml_half_t d; uint16_t qs[QK_K/8]; }} block_iq2_xxs;
typedef struct {{ ggml_half_t d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; }} block_iq2_xs;
typedef struct {{ ggml_half_t d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; }} block_iq2_s;
typedef struct {{ ggml_half_t d; uint8_t qs[3*QK_K/8]; }} block_iq3_xxs;
#define IQ3S_N_SCALE QK_K/64
typedef struct {{ ggml_half_t d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t signs[QK_K/8]; uint8_t scales[IQ3S_N_SCALE]; }} block_iq3_s;
typedef struct {{ ggml_half_t d; uint8_t qs[QK4_NL/2]; }} block_iq4_nl;
typedef struct {{ ggml_half_t d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; }} block_iq4_xs;

{fp16_bits}
{extract_table(common_h, 'kmask_iq2xs')}
{extract_table(common_h, 'ksigns_iq2xs')}
{extract_table(common_h, 'iq2xxs_grid')}
{extract_table(common_h, 'iq2xs_grid')}
{extract_table(common_h, 'iq2s_grid')}
{extract_table(common_h, 'iq3xxs_grid')}
{extract_table(common_h, 'iq3s_grid')}
{extract_table(common_h, 'kvalues_iq4nl')}
"""

formats = [
    ("iq4_xs", "block_iq4_xs", 136, "void dequantize_row_iq4_xs("),
    ("iq3_xxs", "block_iq3_xxs", 98, "void dequantize_row_iq3_xxs("),
    ("iq3_s", "block_iq3_s", 110, "void dequantize_row_iq3_s("),
    ("iq2_xxs", "block_iq2_xxs", 66, "void dequantize_row_iq2_xxs("),
    ("iq2_xs", "block_iq2_xs", 74, "void dequantize_row_iq2_xs("),
]

for name, block_type, block_size, sig in formats:
    dequant_fn = extract_function(quants_c, sig)
    main_code = f"""
int main(void) {{
    unsigned char * in = malloc(1 << 24);
    float * y = malloc(1 << 24);
    size_t len = fread(in, 1, 1 << 24, stdin);
    size_t nblocks = len / {block_size};
    if (nblocks == 0 || len % {block_size} != 0) {{
        fprintf(stderr, "bad input len %zu for block_size {block_size}\\n", len);
        return 1;
    }}
    for (size_t i = 0; i < nblocks; ++i)
        {sig.split('(')[0].split()[-1]}((const {block_type} *)(in + i*{block_size}), y + i*256, 256);
    fwrite(y, 4, nblocks*256, stdout);
    return 0;
}}
"""
    c_src = c_header + "\n" + dequant_fn + "\n" + main_code
    cpath = os.path.join(OUT_DIR, f"{name}_oracle.c")
    bpath = os.path.join(OUT_DIR, f"{name}_oracle")
    with open(cpath, "w") as f:
        f.write(c_src)
    subprocess.run(["cc", "-O2", "-o", bpath, cpath, "-lm"], check=True)
    print(f"Built oracle {bpath}")
