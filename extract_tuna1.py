#!/usr/bin/env python3
"""Extract text from all PDFs in old/tuna1/; also dump the HTML."""
import os, glob, subprocess

TUNA1 = "/D/rex/projects/grim/old/tuna1"
OUTDIR = "/D/rex/projects/grim/old/tuna1_txt"
os.makedirs(OUTDIR, exist_ok=True)

pdfs = sorted(glob.glob(os.path.join(TUNA1, "*.pdf")))
print(f"Found {len(pdfs)} PDFs", flush=True)
for pdf in pdfs:
    name = os.path.basename(pdf).replace(".pdf", "")
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

# HTML
html = os.path.join(TUNA1, "0000-0003-2303-9736.html")
if os.path.exists(html):
    out = os.path.join(OUTDIR, "0000-0003-2303-9736.html.txt")
    import re
    raw = open(html, encoding="utf-8", errors="ignore").read()
    text = re.sub(r"<script.*?</script>", " ", raw, flags=re.S|re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S|re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    with open(out, "w") as f:
        f.write(text)
    print(f"OK html {os.path.basename(html)}: {len(text)} chars", flush=True)
