//! Integration test for disaggregated KV cache transfer.
//!
//! Launches a prefill node receiver on 127.0.0.1:9190 and a decode node
//! receiver on 127.0.0.1:9191, seeds the prefill pool with known KV data,
//! transfers it via `NetworkKvClient::send_block_remote`, then verifies the
//! decode node's pool received the exact matching KV float data.

use std::sync::Arc;
use std::sync::Mutex;

use grim_disagg::{DisaggRouter, DisaggRouterT, KvReceiverServer, PoolRole};
use grim_memory::KvBlockPool;

/// Find a free TCP port on loopback.
fn find_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("must bind to find free port");
    let port = listener.local_addr().expect("must get local addr").port();
    drop(listener);
    port
}

#[test]
fn test_disaggregated_kv_transfer_loopback() {
    // ── Set up prefill node pool (source) ──────────────────────────────────
    let num_heads = 4;
    let head_dim = 8;
    let elem_per_token = num_heads * head_dim; // 32

    let mut prefill_pool = KvBlockPool::new(8, num_heads, head_dim);
    // Seed block 0 with known KV data (1 token = elem_per_token elements).
    let block_id = 0usize;
    let k_data: Vec<f32> = (0..elem_per_token).map(|i| (i as f32) + 0.5).collect();
    let v_data: Vec<f32> = (0..elem_per_token).map(|i| (i as f32) * 2.0).collect();
    prefill_pool.write_keys(block_id, &k_data, 1);
    prefill_pool.write_values(block_id, &v_data);

    let prefill_shared = Arc::new(Mutex::new(prefill_pool));

    // ── Start prefill node receiver ────────────────────────────────────────
    let prefill_port = find_free_port();
    let prefill_addr = format!("127.0.0.1:{prefill_port}");
    let _prefill_receiver = KvReceiverServer::new(&prefill_addr, prefill_shared.clone()).unwrap();

    // ── Set up decode node pool (destination) ──────────────────────────────
    let decode_pool = KvBlockPool::new(8, num_heads, head_dim);
    let decode_shared = Arc::new(Mutex::new(decode_pool));

    // ── Start decode node receiver ─────────────────────────────────────────
    let decode_port = find_free_port();
    let decode_addr = format!("127.0.0.1:{decode_port}");
    let decode_receiver = KvReceiverServer::new(&decode_addr, decode_shared.clone()).unwrap();

    // ── Create router on the prefill node and transfer KV ──────────────────
    let router = DisaggRouter::new(&prefill_addr, &decode_addr, PoolRole::Prefill)
        .with_pool(prefill_shared.clone());

    // Use request_id=0 so network block_id = 0 + i = 0 (fits in the 8-block dest pool).
    let transfer_req_id = 0u64;
    let block_ids = vec![block_id];
    router
        .transfer_kv_cache(transfer_req_id, &block_ids)
        .expect("transfer_kv_cache must succeed over loopback");

    // Give the receiver thread time to write.
    std::thread::sleep(std::time::Duration::from_millis(300));

    // ── Verify the decode node received exact matching data ────────────────
    let decode_guard = decode_shared.lock().unwrap();
    let recv_k = decode_guard.read_keys(transfer_req_id as usize);
    let recv_v = decode_guard.read_values(transfer_req_id as usize);

    // read_keys returns the full block (BLOCK_SIZE * elem_per_token = 512),
    // but only elem_per_token (32) elements were actually written.  Compare
    // the written prefix against the sent data.
    assert!(
        recv_k.len() >= k_data.len(),
        "received K slice too short: got {} need at least {}",
        recv_k.len(),
        k_data.len()
    );
    assert_eq!(
        &recv_k[..k_data.len()],
        &k_data[..],
        "keys must match exactly after transfer"
    );
    assert_eq!(
        &recv_v[..v_data.len()],
        &v_data[..],
        "values must match exactly after transfer"
    );

    drop(decode_receiver);
}

#[test]
fn test_disaggregated_kv_transfer_multiple_blocks() {
    let num_heads = 2;
    let head_dim = 4;
    let elem_per_token = num_heads * head_dim; // 8

    let mut src_pool = KvBlockPool::new(16, num_heads, head_dim);
    // Seed blocks 1, 2, 3 with distinct data.
    for &bid in &[1usize, 2, 3] {
        let k_data: Vec<f32> = (0..elem_per_token)
            .map(|i| (i as f32) * (bid as f32))
            .collect();
        let v_data: Vec<f32> = (0..elem_per_token).map(|i| (i as f32) + 100.0).collect();
        src_pool.write_keys(bid, &k_data, 1);
        src_pool.write_values(bid, &v_data);
    }

    let src_shared = Arc::new(Mutex::new(src_pool));

    let port = find_free_port();
    let src_addr = format!("127.0.0.1:{port}");
    let _src_receiver = KvReceiverServer::new(&src_addr, src_shared.clone()).unwrap();

    let dest_pool = KvBlockPool::new(16, num_heads, head_dim);
    let dest_shared = Arc::new(Mutex::new(dest_pool));

    let port2 = find_free_port();
    let dest_addr = format!("127.0.0.1:{port2}");
    let _dest_receiver = KvReceiverServer::new(&dest_addr, dest_shared.clone()).unwrap();

    let router =
        DisaggRouter::new(&src_addr, &dest_addr, PoolRole::Prefill).with_pool(src_shared.clone());

    let request_id = 0u64;
    // Transfer blocks 1, 2, 3 as a batch.
    router
        .transfer_kv_cache(request_id, &[1, 2, 3])
        .expect("batch transfer must succeed");

    std::thread::sleep(std::time::Duration::from_millis(300));

    let dest_guard = dest_shared.lock().unwrap();
    for &bid in [1usize, 2, 3].iter() {
        // transfer_kv_cache now sends the physical block_id (bid) on the
        // wire, not request_id + i, so data arrives at the same block index.
        let recv_k = dest_guard.read_keys(bid);
        let recv_v = dest_guard.read_values(bid);

        // Verify keys were received
        let any_nonzero_k = recv_k.iter().any(|&x| x != 0.0);
        assert!(
            any_nonzero_k,
            "block {bid} keys must be non-zero after transfer"
        );
        let any_nonzero_v = recv_v.iter().any(|&x| x != 0.0);
        assert!(
            any_nonzero_v,
            "block {bid} values must be non-zero after transfer"
        );
    }
}

#[test]
fn test_disaggregated_kv_no_pool_errors() {
    // A DisaggRouter without a pool must return errors (no real KV to extract).
    let port = find_free_port();
    let addr = format!("127.0.0.1:{port}");
    let router = DisaggRouter::new(&addr, &addr, PoolRole::Prefill);

    assert!(router.pool.is_none());

    let result = router.transfer_kv_cache(1, &[0]);
    assert!(result.is_err());

    let result = router.dispatch_prefill(1, &[1, 2, 3], &[0]);
    assert!(result.is_err());
}

#[test]
fn test_receptor_server_write_and_read() {
    // Direct receiver: send a block, verify it arrives in the pool.
    let num_heads = 4;
    let head_dim = 8;
    let elem_per_token = num_heads * head_dim;

    let pool = KvBlockPool::new(8, num_heads, head_dim);
    let shared = Arc::new(Mutex::new(pool));

    let port = find_free_port();
    let addr = format!("127.0.0.1:{port}");
    let _receiver = KvReceiverServer::new(&addr, shared.clone()).unwrap();
    assert_eq!(_receiver.listen_addr(), &addr);

    let client = grim_kvtransport::NetworkKvClient::new("127.0.0.1".to_string());
    let k_data: Vec<f32> = (0..elem_per_token).map(|i| i as f32).collect();
    let v_data: Vec<f32> = (0..elem_per_token).map(|i| (i + 1) as f32).collect();
    let block_id = 5usize;

    client
        .send_block_remote(block_id, 0, &k_data, &v_data, &addr)
        .expect("send_block_remote must succeed");

    std::thread::sleep(std::time::Duration::from_millis(200));

    let guard = shared.lock().unwrap();
    let recv_k = guard.read_keys(block_id);
    let recv_v = guard.read_values(block_id);
    // read_keys returns the full block (BLOCK_SIZE * elem_per_token = 512),
    // but only `elem_per_token` (32) elements were written.  Compare the
    // written prefix.
    assert_eq!(&recv_k[..k_data.len()], &k_data[..]);
    assert_eq!(&recv_v[..v_data.len()], &v_data[..]);
}

#[test]
fn test_layer_pipelined_kv_streamer_and_orchestrator() {
    use grim_disagg::{DisaggConfig, DisaggOrchestrator, LayerPipelinedKvStreamer};

    let prefill_cfg = DisaggConfig {
        role: PoolRole::Prefill,
        prefill_addr: "127.0.0.1:9001".into(),
        decode_addr: "127.0.0.1:9002".into(),
    };
    let prefill_orch = DisaggOrchestrator::new(prefill_cfg);
    assert!(prefill_orch.handles_prefill());
    assert!(!prefill_orch.handles_decode());

    let decode_cfg = DisaggConfig {
        role: PoolRole::Decode,
        prefill_addr: "127.0.0.1:9001".into(),
        decode_addr: "127.0.0.1:9002".into(),
    };
    let decode_orch = DisaggOrchestrator::new(decode_cfg);
    assert!(!decode_orch.handles_prefill());
    assert!(decode_orch.handles_decode());

    // Test live streaming of layer blocks
    let pool = KvBlockPool::new(4, 2, 4);
    let shared = Arc::new(Mutex::new(pool));
    let port = find_free_port();
    let addr = format!("127.0.0.1:{port}");
    let _receiver = KvReceiverServer::new(&addr, shared.clone()).unwrap();

    let streamer = LayerPipelinedKvStreamer::new(addr.clone());
    let k_data = vec![1.23f32; 32];
    let v_data = vec![4.56f32; 32];

    streamer
        .stream_layer_block(0, 0, &k_data, &v_data)
        .expect("streaming layer block must succeed");

    std::thread::sleep(std::time::Duration::from_millis(150));
    let guard = shared.lock().unwrap();
    let recv_k = guard.read_keys(0);
    assert_eq!(&recv_k[..32], &k_data[..]);
}

#[test]
fn test_disagg_orchestrator_heartbeat_failover() {
    use grim_disagg::{DisaggConfig, DisaggOrchestrator};

    let decode_cfg = DisaggConfig {
        role: PoolRole::Decode,
        prefill_addr: "127.0.0.1:9001".into(),
        decode_addr: "127.0.0.1:9002".into(),
    };
    let mut orch = DisaggOrchestrator::new(decode_cfg);

    // Initial state with fresh heartbeat
    orch.record_heartbeat(PoolRole::Prefill, 1000);
    assert_eq!(orch.evaluate_failover(1200, 500), PoolRole::Decode);

    // Time elapsed exceeds timeout -> failover to Colocated
    assert_eq!(orch.evaluate_failover(2000, 500), PoolRole::Colocated);
}
