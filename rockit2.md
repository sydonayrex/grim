# rockit2.md — grim-backend-rocm kernel audit + old/tuna1/ holistic synthesis

Scope: grim-backend-rocm kernel implementations (reviewed from source) cross-referenced
against all research papers in `old/tuna1/` (25 PDFs + 1 HTML), focused on means to
extract maximum performance by composing multiple methods into one optimization surface.
This is an audit + systems-synthesis document, not a patch.

Corpus inventory (`old/tuna1/`, sorted by relevance to the composition goal):

1. 2601.16294v2 — Space Filling Curves, Communication-Avoiding GEMM (Intel)
2...[truncated]