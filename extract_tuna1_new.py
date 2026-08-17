#!/usr/bin/env python3
"""Extract the two new PDFs added to old/tuna1/ since last run."""
import os, subprocess

TUNA1 = "/D/rex/projects/grim/old/tuna1"
OUTDIR = "/D/rex/projects/grim/old/tuna1_txt"
os.makedirs(OUTDIR, exist_ok=True)

for pdf_name in ["2511.15503v4.pdf", "2604.18616v1.pdf"]:
    pdf = os.path.join(TUNA1, pdf_name)
    name = pdf_name.replace(".pdf", "")
    out = os.path.join(OUTDIR, name + ".txt")
    if os.path.exists(out) and os.path.getsize(out) > 500:
        print(f"SKIP {name}", flush=True)
        continue
    try:
        res = subprocess.run(["pdftotext", "-layout", pdf, out],
                            capture_output=True, text=True, timeout=180)
        if res.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 500:
            print(f"OK pdftotext {name}: {os.path.getsize(out)} bytes", flush=True)
            continue
    except Exception as e:
        print(f"pdftotext fail {name}: {e}", flush=True)
    try:
        import fitz
        doc = fitz.open(pdf)
        text = "\n".join(p.get_text() for p in doc)
        with open(out, "w") as f:
            f.write(text)
        print(f"OK pymupdf {name}: {len(text)} chars", flush=True)
    except Exception as e:
        print(f"ALL FAILED {name}: {e}", flush=True)
        with open(out, "w") as f:
            f.write(f"## FAILED TO EXTRACT: {pdf}\n")
