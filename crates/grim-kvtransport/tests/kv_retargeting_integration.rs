//! Integration test for KV block position retargeting in LocalSpillManager.

use grim_kvtransport::LocalSpillManager;
use tempfile::tempdir;

#[test]
fn test_local_spill_manager_retarget_block_positions() {
    let dir = tempdir().unwrap();
    let tokens_per_block = 4;
    let head_dim = 8;
    let num_heads = 2;
    let block_elems = tokens_per_block * num_heads * head_dim;

    let mut mgr = LocalSpillManager::new(dir.path().to_path_buf(), block_elems).unwrap();

    let mut k_init = vec![0.0f32; block_elems];
    let v_init = vec![1.0f32; block_elems];
    for (i, slot) in k_init.iter_mut().enumerate() {
        *slot = ((i as f32 + 1.0) * 0.05).sin();
    }

    let block_id = 42;
    mgr.demote_to_host(block_id, k_init.clone(), v_init.clone()).unwrap();

    // Retarget from old_pos=0 to new_pos=100
    mgr.retarget_block_positions(
        block_id,
        0,
        100,
        tokens_per_block,
        head_dim,
        num_heads,
        10000.0,
    )
    .expect("retargeting should succeed");

    let (k_retargeted, v_retargeted) = mgr.retrieve(block_id).unwrap().unwrap();
    assert_eq!(v_retargeted, v_init, "V vectors must remain untouched by Re-RoPE");
    assert_eq!(k_retargeted.len(), block_elems);
    assert_ne!(k_retargeted, k_init, "K vectors must be re-rotated to new positions");
}
