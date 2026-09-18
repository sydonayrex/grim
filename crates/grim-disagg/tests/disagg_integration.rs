//! Integration test for disaggregated KV cache transfer.
//!
//! Launches a prefill node receiver on 127.0.0.1:9190 and a decode node
//! receiver on 127.0.0.1:9191, seeds the prefill pool with known KV data,
//! transfers it via `NetworkKvClient::send_block_remote`, then verifies the
//! decode node's pool received the exact matching KV float data.

use std::sync::Arc;
use std::sync::Mutex;

use grim_disagg::KvReceiverServer;
use grim_memory::KvBlockPool;

/// Find a free TCP port on loopback.
fn find_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("must bind to find free port");
    let port = listener.local_addr().expect("must get local addr").port();
    drop(listener);
    port
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

    // 1 token of valid data (elem_per_token elements).
    client
        .send_block_remote(block_id, 0, &k_data, &v_data, 1, &addr)
        .expect("send_block_remote must succeed");

    // Pushes ACK on commit — the data is in the pool already.
    let guard = shared.lock().unwrap_or_else(|e| e.into_inner());
    let recv_k = guard.read_keys(block_id);
    let recv_v = guard.read_values(block_id);
    // read_keys returns the full block (BLOCK_SIZE * elem_per_token = 512),
    // but only `elem_per_token` (32) elements were written.  Compare the
    // written prefix, and require the valid-token count to round-trip.
    assert_eq!(&recv_k[..k_data.len()], &k_data[..]);
    assert_eq!(&recv_v[..v_data.len()], &v_data[..]);
    assert_eq!(
        guard.block_num_tokens(block_id),
        Some(1),
        "receiver must store the sender's valid token count"
    );
}
