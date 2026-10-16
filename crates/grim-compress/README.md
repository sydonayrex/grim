# grim-compress

Inference-free extractive prompt compression engine implementing composite 4-signal ranking (TextRank, U-shaped position weighting, TF-IDF density, and novelty scoring).

[Back to project README](../../README.md)

## Dependencies

```mermaid
graph LR
    grim_compress["grim-compress"]
    grim_server["grim-server"]
    grim_server --> grim_compress
    style grim_compress stroke-width:3px
```

External dependencies: `thiserror`.

## Flamegraph

Not applicable - library crate (no binary).

## Source Layout

| Folder | Purpose |
|---|---|
| `src/` | Sentence segmentation, TextRank graph centrality, TF-IDF statistics, and prompt compressor |

## Key Responsibilities

- Neural-inference-free extractive text compression reducing prompt length without model evaluation.
- Composite 4-signal scoring combining graph centrality, position weighting, lexical density, and centroid divergence.
- Verbatim token preservation maintaining original text boundaries and discourse coherence.
