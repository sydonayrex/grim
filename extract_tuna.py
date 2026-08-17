#!/usr/bin/env python3
"""Extract text from all PDFs in old/tuna/ using pdftotext, fallback to PyMuPDF."""
import os, glob, sys, subprocess, tempfile

TUNA = "/D/rex/projects/grim/old/tuna"
OUTDIR = "/D/rex/projects/grim/old/tuna_txt"
os.makedirs(OUTDIR, exist_ok=True)

pdfs = sorted(glob.glob(os.path.join(TUNA, "*.pdf")))
print(f"Found {len(pdfs)} PDFs", file=sys.stderr)

for pdf in pdfs:
    name = os.path.basename(pdf).replace(".pdf", "")
    out = os.path.join(OUTDIR, name + ".txt")
    if os.path.exists(out) and os.path.getsize(out) > 500:
        print(f"SKIP {name} (already extracted)", file=sys.stderr)
        continue
    # Try pdftotext first
    try:
        res = subprocess.run(["pdftotext", "-layout", pdf, out],
                            capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 500:
            print(f"OK pdftotext {name}: {os.path.getsize(out)} bytes", file=sys.stderr)
            continue
    except Exception as e:
        print(f"pdftotext failed {name}: {e}", file=sys.stderr)
    # Fallback PyMuPDF
    try:
        import fitz
        doc = fitz.open(pdf)
        text = "\n".join(p.get_text() for p in doc)
        with open(out, "w") as f:
            f.write(text)
        print(f"OK pymupdf {name}: {len(text)} chars", file=sys.stderr)
    except Exception as e:
        print(f"ALL FAILED {name}: {e}", file=sys.stderr)
        # write empty marker
        with open(out, "w") as f:
            f.write(f"## FAILED TO EXTRACT: {pdf}\n")
