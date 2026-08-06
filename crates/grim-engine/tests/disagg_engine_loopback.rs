//! Engine-level disaggregation loopback test.
//!
//! Exercises the full prefill → network transfer → decode lifecycle through
//! `Engine::tick()` / `drive_decode_with_outcome`, verifying that:
//!
//! 1. `drive_prefill` (Prefill role) streams real KV blocks from the pool to
//!    the decode node's `KvReceiverServer` via TCP.
//! 2. `drive_decode_with_outcome` (Decode role) correctly skips blocks that
//!    have already arrived via the receiver server — including a genuinely
//!    all-zero KV block, which the old non-zero-content sniff would have
//!    incorrectly re-fetched.
//! 3. The `block_is_received` bitset (Issue 1 fix) correctly tracks arrival
//!    state regardless of data content.

use std::sync::Arc;

use grim_core::model::CausalLm;
use grim_core::session::DeterminismMode;
use grim_disagg::{DisaggConfig, DisaggRouter, PoolRole};
use grim_engine::{Engine, EngineConfig};
use grim_kvtransport::{KvBlockStore, NetworkKvClient};
use grim_memory::BLOCK_SIZE;
use grim_models_transformer::{Llama, LlamaConfig};
use grim_tensor::Device;

/// Find a free TCP port for loopback tests.
fn find_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("must bind to find free port");
    listener.local_addr().expect("must get local addr").port()
}

/// Small Llama model matching the engine test config in lib.rs.
fn small_llama() -> Box<dyn CausalLm> {
    Box::new(Llama::random(
        Device::Cpu,
        LlamaConfig {
            vocab_size: 256,
            hidden_size: 32,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 16,
            num_layers: 1,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_seq_len: 64,
        },
    ))
}

/// Build a minimal EngineConfig with the given pool dimensions and optional
/// disaggregation config.
fn make_config(
    num_kv_heads: usize,
    head_dim: usize,
    block_pool_capacity: usize,
    disagg_router: Option<Arc<DisaggRouter>>,
    disagg_config: Option<DisaggConfig>,
) -> EngineConfig {
    EngineConfig {
        max_batched_tokens: 4096,
        max_num_seqs: 8,
        block_pool_capacity,
        num_kv_heads,
        head_dim,
        target_ttft_ms: 2000,
        target_itl_ms: 100,
        determinism_mode: DeterminismMode::Relaxed,
        kv_compressor: None,
        tp_size: 0,
        tp_gpus: vec![],
        max_tool_calls_per_conversation: 20,
        max_messages_per_request: 200,
        disagg_router,
        disagg_config,
    }
}

/// Test that `block_is_received` correctly tracks all-zero KV blocks.
/// The old non-zero-content sniff would fail here because the block data
/// is all zeros — it would be treated as "not received" and re-fetched.
#[test]
fn test_all_zero_kv_block_marked_received() {
    let port = find_free_port();
    let decode_addr = format!("127.0.0.1:{port}");

    // Start a decode engine — its Engine::new will spawn a KvReceiverServer
    // on decode_addr.
    let disagg_config = DisaggConfig {
        role: PoolRole::Decode,
        prefill_addr: format!("127.0.0.1:{port}"),
        decode_addr: decode_addr.clone(),
    };
    let config = make_config(2, 16, 8, None, Some(disagg_config));
    let engine = Engine::new(config);

    // Push an all-zero KV block via TCP — this is the edge case that the
    // non-zero sniff would fail on.
    let block_elems = 2 * 16 * 16; // num_heads * head_dim * BLOCK_SIZE
    let k_data = vec![0.0f32; block_elems];
    let v_data = vec![0.0f32; block_elems];

    let client = NetworkKvClient::new("127.0.0.1".to_string());
    client
        .send_block_remote(0, 0, &k_data, &v_data, &decode_addr)
        .expect("send_block_remote must succeed against live receiver");

    // Give the receiver thread time to process.
    std::thread::sleep(std::time::Duration::from_millis(300));

    // Verify the block was received and marked — the core of Issue 1.
    let pool = engine.block_pool.lock().unwrap();
    assert!(
        pool.block_is_received(0),
        "all-zero KV block must be marked as received (not re-fetched)"
    );
}

/// Full prefill → network transfer → decode loopback test.
///
/// Sets up a prefill engine and a decode engine communicating over TCP
/// loopback. The prefill engine runs a prefill step and transfers KV blocks
/// to the decode engine. The decode engine's fetch loop should see the
/// blocks as already received and skip fetching.
#[test]
fn test_disagg_prefill_to_decode_loopback() {
    let prefill_port = find_free_port();
    let decode_port = find_free_port();
    let prefill_addr = format!("127.0.0.1:{prefill_port}");
    let decode_addr = format!("127.0.0.1:{decode_port}");

    // --- Decode engine: listens for incoming KV blocks ---
    let decode_config = DisaggConfig {
        role: PoolRole::Decode,
        prefill_addr: prefill_addr.clone(),
        decode_addr: decode_addr.clone(),
    };
    let decode_router = Arc::new(DisaggRouter::new(
        &prefill_addr,
        &decode_addr,
        PoolRole::Decode,
    ));
    let decode_engine_config = make_config(2, 16, 8, Some(decode_router), Some(decode_config));
    let mut decode_engine = Engine::new(decode_engine_config);
    decode_engine.register_model("small", small_llama());

    // --- Prefill engine: transfers KV blocks to decode node ---
    let prefill_config = DisaggConfig {
        role: PoolRole::Prefill,
        prefill_addr: prefill_addr.clone(),
        decode_addr: decode_addr.clone(),
    };
    // The prefill router's pool points at the *decode* engine's pool so that
    // transfer_kv_cache_real reads real KV data from the prefill engine's pool.
    // In the real deployment these are separate processes; here we share the
    // decode pool as a stand-in for the prefill pool's content.
    let prefill_router = Arc::new(
        DisaggRouter::new(&prefill_addr, &decode_addr, PoolRole::Prefill)
            .with_pool(decode_engine.block_pool.clone()),
    );
    let prefill_engine_config = make_config(2, 16, 8, Some(prefill_router), Some(prefill_config));
    let mut prefill_engine = Engine::new(prefill_engine_config);
    prefill_engine.register_model("small", small_llama());

    // --- Step 1: Run prefill on the prefill engine ---
    prefill_engine.enqueue_request(grim_scheduler::Request {
        id: 1,
        prompt_tokens: 4,
        priority: 0,
        ..Default::default()
    });
    let prefill_result = prefill_engine.tick();
    assert!(prefill_result.is_ok(), "prefill tick must succeed");

    // The transfer happens in drive_prefill after the forward pass.
    // Give the network a moment to settle.
    std::thread::sleep(std::time::Duration::from_millis(500));

    {
        let pool = decode_engine.block_pool.lock().unwrap();
        for block_id in 0..pool.num_blocks() {
            assert!(
                pool.block_is_received(block_id),
                "decode pool block {block_id} must be marked received after transfer"
            );
        }
    }

    // --- Step 2: Run decode on the decode engine ---
    // drive_decode_with_outcome checks block_is_received for each block.
    // Since all blocks were just received via the receiver server, the
    // fetch loop should skip them (not call fetch_kv_block on any block).
    let _ = decode_engine.enqueue_request(grim_scheduler::Request {
        id: 1,
        prompt_tokens: 4,
        priority: 0,
        ..Default::default()
    });

    // The decode tick should succeed — the fetch loop finds all blocks
    // already received and doesn't attempt network fetches.
    let decode_result = decode_engine.tick();
    assert!(
        decode_result.is_ok(),
        "decode tick must succeed after KV transfer: {:?}",
        decode_result.err()
    );

    // Verify the decode position advanced (session was created + forward ran).
    let pos = decode_engine
        .sessions
        .get(&1)
        .map(|s| s.current_pos())
        .unwrap_or(0);
    assert!(pos >= 1, "decode must advance session position (got {pos})");
}

/// Test that `block_is_received` returns false for never-touched blocks
/// and true after `write_keys`, including all-zero data.
#[test]
fn test_block_received_bitset_semantics() {
    let mut pool = grim_memory::KvBlockPool::new(4, 2, 4);
    assert_eq!(pool.num_blocks(), 4);

    // Fresh pool: no blocks received.
    for i in 0..pool.num_blocks() {
        assert!(
            !pool.block_is_received(i),
            "block {i} should not be received initially"
        );
    }

    // Allocate a block and write all-zero data — the edge case that
    // the old non-zero sniff would fail on.
    let id = pool.alloc().unwrap();
    let elem = 2 * 4; // num_heads * head_dim
    let k = vec![0.0f32; elem * BLOCK_SIZE];
    pool.write_keys(id, &k, BLOCK_SIZE);

    // All-zero data should still be marked received.
    assert!(
        pool.block_is_received(id),
        "block must be marked received even with all-zero KV data"
    );

    // Free the block — received should be cleared (no spill attached).
    pool.free(id);
    assert!(
        !pool.block_is_received(id),
        "block must be unmarked after free (no spill)"
    );
}
