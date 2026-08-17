#!/usr/bin/env python3
"""Extract remaining tuna2 PDFs not yet processed."""
import os, subprocess, glob

TUNA2 = "/D/rex/projects/grim/old/tuna2"
OUTDIR = "/D/rex/projects/grim/old/tuna2_txt"

EXCLUDE = [
    "3730584", "3804601.3804607",
    "62_Hawkeye_Hardware_Aware_GPU_",
    "Characterizing_the_Performance_and_Usability_of_GPU_JIT_Compilation_Interfaces_using_Proteus",
    "j.issn.1000-565X.240498",
    "OptimizingStandardConvolutionforDiversePrecisiononDCU",
    "1887_4301430-Chapter 7",
    "2606.11357v2",
]

os.makedirs(OUTDIR, exist_ok=True)

for pdf in glob.glob(os.path.join(TUNA2, "*.pdf")):
    name = os.path.basename(pdf)
    stem = os.path.splitext(name)[0]
    if any(x in stem for x in EXCLUDE):
        continue
    out = os.path.join(OUTDIR, stem + ".txt")
    if os.path.exists(out):
        print("skip", stem)
        continue
    print("extract", stem)
    r = subprocess.run(
        ["pdftotext", "-layout", pdf, out],
        capture_output=True, text=True,
    )
    size = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"  -> {size} bytes", "FAIL" if r.returncode != 0 else "")
