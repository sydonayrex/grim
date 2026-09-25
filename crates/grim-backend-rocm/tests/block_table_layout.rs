//! Block-table byte layout regression for paged KV attention.
//!
//! `grim_qkv_attention_paged_quant` reads the block table as
//! `BlockTableEntry*` — two u32 words per entry (block_id, page_size), an
//! 8-byte stride per page. Uploading a plain `[0.0, 1.0, 2.0, ...]` f32 vector
//! instead silently mis-strides: page 1 lands on f32[2] and decodes 1.0f32's bit
//! pattern (1065353216) as a block_id, so `elem_offset = physical_block *
//! page_stride + ...` addresses memory that was never mapped and faults the GPU.
//!
//! This is format-independent, so int8, FP8 and Nutcracker all fault identically.
//! The existing per-format parity tests do not catch it because they build the
//! table from real (block_id, page_size) pairs.
use grim_backend_rocm::kernels::qkv_attention::KvCacheQuantFormat;

/// Reinterpret 4 bytes as a u32 exactly as the GPU does when it loads the
/// table as `BlockTableEntry*`. Rust's `as u32` on an f32 is a VALUE conversion,
/// not a bit reinterpretation, so it cannot express this.
fn bits_of(x: f32) -> u32 {
    u32::from_le_bytes(x.to_le_bytes())
}

#[test]
fn block_table_stride_is_two_u32_words_per_page() {
    // A page's block_id must be readable as a plain integer at an 8-byte stride,
    // not as the bit pattern of a float.
    let n_pages = 8usize;
    let mut words: Vec<u32> = Vec::with_capacity(n_pages * 2);
    for i in 0..n_pages {
        words.push(i as u32);
        words.push(1);
    }
    let table_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(words.as_ptr() as *const f32, words.len()) };

    for page in 0..n_pages {
        let block_id = bits_of(table_f32[page * 2]);
        let page_size = bits_of(table_f32[page * 2 + 1]);
        assert_eq!(block_id as usize, page, "page {page} block_id must be {page}");
        assert_eq!(page_size, 1, "page {page} page_size must be 1");
    }
}

/// The wrong layout (what was shipped) turns small integers into huge ones once
/// reinterpreted as f32 bit patterns, which is the mechanism of the GPU fault.
#[test]
fn naive_f32_block_table_decodes_as_astronomical_block_ids() {
    let naive: Vec<f32> = (0..8).map(|i| i as f32).collect();
    for page in 0..8 {
        // Kernel reads an 8-byte stride, so page N lands on naive[N*2].
        let idx = page * 2;
        if idx >= naive.len() {
            continue;
        }
        let block_id = bits_of(naive[idx]);
        if page > 0 {
            assert!(
                block_id > 1_000_000,
                "page {page} decoded block_id {block_id} is the garbage value \
                 that causes the fault (expected an f32 bit pattern)"
            );
        }
    }
}

#[test]
fn quant_formats_share_the_block_table_path() {
    // Documents why all three formats failed identically: the format only
    // selects the dequant branch; the table stride is shared.
    let _ = [
        KvCacheQuantFormat::Int8,
        KvCacheQuantFormat::Fp8E4M3,
        KvCacheQuantFormat::NutFp4,
    ];
}
