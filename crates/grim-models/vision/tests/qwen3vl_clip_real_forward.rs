//! The real checkpoint, end to end through the tower (Task 1.3 + 2.2).
//!
//! This is the first time grim runs the actual 927 MB Qwen3-VL mmproj. Everything
//! before it was a tiny fixture, so this is where a geometry mistake in the real
//! head width (72, not a power of two) or in the section split (9 pairs per
//! section) would show up.
//!
//! Run explicitly:
//!   cargo test -p grim-models-vision --test qwen3vl_clip_real_forward \
//!       -- --ignored --nocapture
//!
//! It is `#[ignore]`d so CI stays fast, and it PANICS rather than skipping when
//! the checkpoint is absent: a real-file test that passes by not running is
//! indistinguishable from one that passes.
//!
//! COST, stated so nobody mistakes a timeout for a pass: at 768x768 the tower is
//! ~330 GFLOP, of which ~96% is the 27 full 2304-token attention layers, computed
//! in scalar f32 with no BLAS. That is minutes, not seconds. The device path
//! (Task 3.1) exists precisely to make this cheap; until then the CPU forward is
//! a correctness oracle, not something to run in a loop.

use grim_models_vision::qwen3vl_clip::{Qwen3VlClip, Qwen3VlClipConfig, ROPE_FREQ_BASE};

fn mmproj_path() -> std::path::PathBuf {
    let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for _ in 0..4 {
        let candidate = dir.join("models/qwen38-27b/mmproj-F16.gguf");
        if candidate.exists() {
            return candidate;
        }
        dir = dir.join("..");
    }
    panic!(
        "mmproj-F16.gguf not found walking up from {}; this test must find the \
         checkpoint or fail, never pass by absence",
        env!("CARGO_MANIFEST_DIR")
    );
}

/// The real tower encodes a real-sized image and emits the shape the chat
/// template reserves slots for: 576 tokens of 5120 features.
// SLOW BY DESIGN: ~330 GFLOP at 768x768, ~96% of it in 27 full 2304-token
// attention layers, in scalar f32 with no BLAS. Budget minutes, and run
// with --test-threads=1 in a release build.
#[test]
#[ignore = "requires models/qwen38-27b/mmproj-F16.gguf; CPU forward is minutes-slow"]
fn real_tower_encodes_a_full_image() {
    use grim_format::tprov::GgufProvider;

    let path = mmproj_path();
    println!("real mmproj: {}", path.display());
    let provider = GgufProvider::open(path.to_str().expect("utf-8 path")).expect("mmproj opens");
    let cfg = Qwen3VlClipConfig::from_provider(&provider).expect("geometry parses");
    cfg.ensure_supported()
        .expect("no deepstack in this checkpoint");

    // Geometry the rope depends on.
    assert_eq!(cfg.embedding_length, 1152);
    assert_eq!(cfg.num_heads, 16);
    assert_eq!(cfg.head_dim(), 72, "1152 / 16");
    assert_eq!(cfg.patch_size, 16);
    assert_eq!(cfg.spatial_merge_size, 2);
    assert_eq!(cfg.merged_tokens(), 576, "48*48 / 4");
    assert_eq!(cfg.merger_in(), 4608);
    assert_eq!(cfg.projection_dim, 5120);
    // The vision rope base, which is NOT the text tower's.
    assert_eq!(ROPE_FREQ_BASE, 10000.0);

    let clip = Qwen3VlClip::load(&provider, cfg.clone()).expect("real tower loads");
    assert_eq!(clip.blocks.len(), 27);
    assert_eq!(clip.position_grid, 48, "sqrt(2304)");

    // A deterministic, non-degenerate image at the checkpoint's own resolution.
    let side = cfg.image_size;
    let pixels: Vec<f32> = (0..(cfg.in_channels * side * side))
        .map(|i| ((i * 2654435761usize) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    let out = clip
        .forward_cpu(&pixels, side, side)
        .expect("the real tower must encode");

    assert_eq!(
        out.len(),
        576 * 5120,
        "576 merged tokens x 5120 projection dim"
    );
    assert!(
        out.iter().all(|v| v.is_finite()),
        "the real tower produced a non-finite embedding; that is NaN through the \
         graph, not a rounding artefact"
    );
    let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(norm > 0.0, "embedding must not be identically zero");
    println!(
        "encoded {} tokens x {} features, L2 norm {norm:.4}, \
         first value {:.6}",
        576, 5120, out[0]
    );
}

/// A different image must give a different embedding. On the real weights this
/// also proves the conv actually reads the pixels: a zeroed or mis-gathered conv
/// would still produce a finite, correctly-shaped, non-zero output.
// SLOW BY DESIGN: ~330 GFLOP at 768x768, ~96% of it in 27 full 2304-token
// attention layers, in scalar f32 with no BLAS. Budget minutes, and run
// with --test-threads=1 in a release build.
#[test]
#[ignore = "requires models/qwen38-27b/mmproj-F16.gguf; CPU forward is minutes-slow"]
fn real_tower_distinguishes_two_images() {
    use grim_format::tprov::GgufProvider;

    let path = mmproj_path();
    let provider = GgufProvider::open(path.to_str().expect("utf-8 path")).expect("mmproj opens");
    let cfg = Qwen3VlClipConfig::from_provider(&provider).expect("geometry parses");
    let clip = Qwen3VlClip::load(&provider, cfg.clone()).expect("real tower loads");

    let side = cfg.image_size;
    let n = cfg.in_channels * side * side;
    let a: Vec<f32> = (0..n).map(|i| (i % 251) as f32 / 251.0 - 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| 0.5 - (i % 199) as f32 / 199.0).collect();

    let ea = clip.forward_cpu(&a, side, side).expect("image A");
    let eb = clip.forward_cpu(&b, side, side).expect("image B");

    let max_diff = ea
        .iter()
        .zip(eb.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "two different images produced embeddings differing by at most {max_diff}; \
         the conv is not reading its input"
    );
    println!("max per-element difference between two images: {max_diff:.6}");
}
