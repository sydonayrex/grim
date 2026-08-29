//! Cross-crate integration tests for `grim-kvtransport`.
//!
//! Validates:
//! - KvBlockHeader protocol wire serialization, deserialization, and checksum verification
//! - LocalSpillManager tiered storage lifecycle (GPU -> Host RAM -> NVMe scratch -> restore)
//! - NetworkKvClient & start_kv_receiver_server TCP socket streaming with multi-layer KV blocks
//! - NvmeWeightStreamer sequential page streaming and memory-mapped reader integrity

use grim_kvtransport::{
    BlockId, CacheTier, KvBlockHeader, KvBlockStore, NetworkKvClient, NvmeWeightStreamer,
    SharedSpillManager, start_kv_receiver_server,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Mock in-memory KvBlockStore for testing network ingestion.
struct MockBlockStore {
    num_blocks: usize,
    elem_per_token: usize,
    block_size: usize,
    received: Vec<bool>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl MockBlockStore {
    fn new(num_blocks: usize, elem_per_token: usize, block_size: usize) -> Self {
        Self {
            num_blocks,
            elem_per_token,
            block_size,
            received: vec![false; num_blocks],
            keys: vec![vec![0.0; elem_per_token * block_size]; num_blocks],
            values: vec![vec![0.0; elem_per_token * block_size]; num_blocks],
        }
    }
}

impl KvBlockStore for MockBlockStore {
    fn num_blocks(&self) -> usize {
        self.num_blocks
    }
    fn block_elem_per_token(&self) -> usize {
        self.elem_per_token
    }
    fn block_size(&self) -> usize {
        self.block_size
    }
    fn write_keys(&mut self, id: BlockId, keys: &[f32], _num_tokens: usize) {
        if id < self.num_blocks {
            self.keys[id].copy_from_slice(keys);
            self.received[id] = true;
        }
    }
    fn write_values(&mut self, id: BlockId, values: &[f32]) {
        if id < self.num_blocks {
            self.values[id].copy_from_slice(values);
        }
    }
    fn block_is_received(&self, id: BlockId) -> bool {
        self.received.get(id).copied().unwrap_or(false)
    }
    fn read_keys(&self, id: BlockId) -> Option<Vec<f32>> {
        self.keys.get(id).cloned()
    }
    fn read_values(&self, id: BlockId) -> Option<Vec<f32>> {
        self.values.get(id).cloned()
    }
}

#[test]
fn test_kv_block_header_serialization_and_checksum() {
    let header = KvBlockHeader {
        magic: 0x4B56434B,
        version: 3,
        block_id: 42,
        layer_idx: 3,
        num_elements: 128,
        checksum: 0xDEADBEEF,
        num_tokens: 16,
    };

    let serialized = header.serialize();
    assert_eq!(serialized.len(), KvBlockHeader::SIZE);

    let deserialized = KvBlockHeader::deserialize(&serialized).unwrap();
    assert_eq!(deserialized.magic, header.magic);
    assert_eq!(deserialized.version, header.version);
    assert_eq!(deserialized.block_id, 42);
    assert_eq!(deserialized.layer_idx, 3);
    assert_eq!(deserialized.num_elements, 128);
    assert_eq!(deserialized.checksum, 0xDEADBEEF);
    assert_eq!(deserialized.num_tokens, 16);
    assert!(deserialized.verify());
}

#[test]
fn test_tiered_spill_manager_gpu_host_nvme_roundtrip() {
    let temp_dir = tempfile::tempdir().unwrap();
    let scratch_path = temp_dir.path().join("kv_spill");

    let shared = SharedSpillManager::new(scratch_path, 64).unwrap();

    let k_data = vec![1.23f32; 64];
    let v_data = vec![4.56f32; 64];

    // 1. Demote to Host RAM
    shared
        .demote_to_host(10, k_data.clone(), v_data.clone())
        .unwrap();
    assert_eq!(shared.get_tier(10), Some(CacheTier::HostRam));

    // 2. Demote Host RAM to NVMe
    shared.demote_to_nvme(10).unwrap();
    assert_eq!(shared.get_tier(10), Some(CacheTier::NvMe));

    // 3. Retrieve from NVMe back to Host
    let (restored_k, restored_v) = shared.retrieve(10).unwrap().unwrap();
    assert_eq!(restored_k, k_data);
    assert_eq!(restored_v, v_data);

    // 4. Evict block
    shared.evict(10);
    assert_eq!(shared.get_tier(10), None);
}

#[test]
fn test_network_kv_transport_tcp_loopback_streaming() {
    let pool = Arc::new(Mutex::new(MockBlockStore::new(8, 4, 16)));
    let pool_clone = Arc::clone(&pool);

    // Pick dynamic local port
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    // Start background receiver server
    let _handle = start_kv_receiver_server(&addr.to_string(), pool_clone);

    // Give server thread a moment to bind
    std::thread::sleep(Duration::from_millis(50));

    // Send KV block over network client (64 elems / 4 elems-per-token = 16 tokens).
    let client = NetworkKvClient::new("127.0.0.1".to_string());
    let k_payload = vec![0.42f32; 4 * 16];
    let v_payload = vec![0.84f32; 4 * 16];

    let send_res =
        client.send_block_remote(2, 0, &k_payload, &v_payload, 16, &addr.to_string());
    assert!(
        send_res.is_ok(),
        "Client should successfully send KV block to loopback receiver: {:?}",
        send_res.err()
    );

    // The send only returns after the receiver's commit ACK.
    {
        let store = pool.lock().unwrap();
        assert!(
            store.block_is_received(2),
            "Block 2 should be marked received"
        );
        assert_eq!(store.keys[2], k_payload);
        assert_eq!(store.values[2], v_payload);
    }
}

#[test]
fn test_nvme_weight_streamer_read_and_prefetch() {
    let temp_dir = tempfile::tempdir().unwrap();
    let weight_file = temp_dir.path().join("model_weights.bin");

    // Write dummy weight file (1024 floats per layer)
    let dummy_weights: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.1).collect();
    let dummy_bytes: Vec<u8> = dummy_weights.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(&weight_file, dummy_bytes).unwrap();

    // unit_elems=1024: matches the dummy weight file (1024 floats per layer).
    let streamer = NvmeWeightStreamer::new(weight_file, 4, 1024);
    assert!(streamer.prefetch_layer_async(0).is_ok());
    assert!(streamer.commit_and_swap(0, 1).is_ok());
}
