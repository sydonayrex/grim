//! Paged KV cache implementation — `KvBlockPool`, `BlockTable`, and the
//! `KvCache` impl that uses them.
//!
//! Mirrors vLLM's block-table-over-physical-blocks design (§5.1):
//! - Sequences address KV memory through a logical block table.
//! - Physical blocks are allocated/freed from a shared pool.
//! - Prefix caching: identical prompt prefixes share ref-counted blocks.
//! - Speculative decoding: draft-token KV entries use `tentative_append`,
//!   then `commit` or `rollback_to`.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use grim_core::kv_cache::KvCache;
use grim_core::error::{Error, Result};
use grim_tensor::{Device, DType, Shape, Tensor};

pub const BLOCK_SIZE: usize = 16;

pub type BlockId = usize;

/// One physical KV block in the pool.
/// Stores key and value slices for `BLOCK_SIZE` token positions.
struct KvBlock {
    id: BlockId,
    /// Flat `[BLOCK_SIZE, num_kv_heads, head_dim]` for keys.
    key_data: Vec<f32>,
    /// Flat `[BLOCK_SIZE, num_kv_heads, head_dim]` for values.
    value_data: Vec<f32>,
    num_tokens: usize,
}

/// Shared pool of physical blocks, pre-allocated.
pub struct KvBlockPool {
    blocks: Vec<KvBlock>,
    free_list: VecDeque<BlockId>,
    ref_counts: HashMap<BlockId, u32>,
    num_heads: usize,
    head_dim: usize,
}

impl KvBlockPool {
    pub fn new(capacity: usize, num_heads: usize, head_dim: usize) -> Self {
        let block_elem = BLOCK_SIZE * num_heads * head_dim;
        let mut blocks = Vec::with_capacity(capacity);
        let mut free_list = VecDeque::with_capacity(capacity);
        for i in 0..capacity {
            blocks.push(KvBlock {
                id: i,
                key_data: vec![0.0; block_elem],
                value_data: vec![0.0; block_elem],
                num_tokens: 0,
            });
            free_list.push_back(i);
        }
        Self {
            blocks,
            free_list,
            ref_counts: HashMap::new(),
            num_heads,
            head_dim,
        }
    }

    pub fn alloc(&mut self) -> Result<BlockId> {
        let id = self.free_list.pop_front()
            .ok_or_else(|| Error::KvCache("block pool exhausted".into()))?;
        self.ref_counts.insert(id, 1);
        Ok(id)
    }

    pub fn free(&mut self, id: BlockId) {
        if let Some(cnt) = self.ref_counts.get_mut(&id) {
            *cnt -= 1;
            if *cnt == 0 {
                self.ref_counts.remove(&id);
                self.blocks[id].num_tokens = 0;
                self.blocks[id].key_data.fill(0.0);
                self.blocks[id].value_data.fill(0.0);
                self.free_list.push_back(id);
            }
        }
    }

    pub fn add_ref(&mut self, id: BlockId) {
        *self.ref_counts.entry(id).or_insert(1) += 1;
    }

    pub fn write_keys(&mut self, id: BlockId, keys: &[f32], num_tokens: usize) {
        let block = &mut self.blocks[id];
        let n = num_tokens.min(BLOCK_SIZE);
        let elem = self.num_heads * self.head_dim;
        let total = n * elem;
        for i in 0..total.min(keys.len()) {
            block.key_data[i] = keys[i];
        }
        block.num_tokens = n;
    }

    pub fn write_values(&mut self, id: BlockId, values: &[f32]) {
        let block = &mut self.blocks[id];
        let n = block.num_tokens;
        let elem = self.num_heads * self.head_dim;
        for i in 0..(n * elem).min(values.len()) {
            block.value_data[i] = values[i];
        }
    }

    pub fn read_keys(&self, id: BlockId) -> &[f32] {
        &self.blocks[id].key_data
    }

    pub fn read_values(&self, id: BlockId) -> &[f32] {
        &self.blocks[id].value_data
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

/// Logical → physical block mapping for one sequence.
pub struct BlockTable {
    logical_to_physical: Vec<BlockId>,
    pool_id: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            logical_to_physical: Vec::new(),
            pool_id: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.logical_to_physical.len()
    }

    pub fn num_tokens(&self, pool: &KvBlockPool) -> usize {
        let mut total = 0usize;
        for &pid in &self.logical_to_physical {
            total += pool.blocks[pid].num_tokens;
        }
        total
    }

    pub fn physical_ids(&self) -> &[BlockId] {
        &self.logical_to_physical
    }

    pub fn push(&mut self, block_id: BlockId) {
        self.logical_to_physical.push(block_id);
    }

    pub fn truncate(&mut self, len: usize) {
        self.logical_to_physical.truncate(len);
    }
}

/// A `KvCache` implementation backed by a shared `KvBlockPool`.
pub struct PagedKvCache {
    table: BlockTable,
    pool: Arc<Mutex<KvBlockPool>>,
    num_heads: usize,
    head_dim: usize,
    /// Number of "tentative" (speculative-draft) slots at the end.
    tentative_len: usize,
}

impl PagedKvCache {
    pub fn new(
        pool: Arc<Mutex<KvBlockPool>>,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            table: BlockTable::new(),
            pool,
            num_heads,
            head_dim,
            tentative_len: 0,
        }
    }

    /// Token count as computed from the block table.
    fn token_count(&self) -> usize {
        // v1 estimates: each full block = BLOCK_SIZE tokens, partial = whatever
        self.table.len() * BLOCK_SIZE
    }
}

impl KvCache for PagedKvCache {
    fn append_slot(&mut self) -> Result<()> {
        let mut pool = self.pool.lock().unwrap();
        let id = pool.alloc()?;
        self.table.push(id);
        Ok(())
    }

    fn tentative_append(&mut self, n: usize) -> Result<()> {
        for _ in 0..n {
            self.append_slot()?;
        }
        self.tentative_len += n;
        Ok(())
    }

    fn commit(&mut self, accepted_len: usize) -> Result<()> {
        let to_drop = self.tentative_len.saturating_sub(accepted_len);
        self.rollback_to(self.token_count().saturating_sub(to_drop))?;
        self.tentative_len = 0;
        Ok(())
    }

    fn rollback_to(&mut self, len: usize) -> Result<()> {
        let current = self.token_count();
        if len >= current {
            return Ok(());
        }
        let to_remove = current - len;
        let blocks_to_pop = (to_remove + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut pool = self.pool.lock().unwrap();
        for _ in 0..blocks_to_pop {
            if let Some(pid) = self.table.logical_to_physical.pop() {
                pool.free(pid);
            }
        }
        self.tentative_len = self.tentative_len.saturating_sub(to_remove);
        Ok(())
    }

    fn len(&self) -> usize {
        self.token_count()
    }

    fn current_k(&self) -> Result<Tensor> {
        Err(Error::KvCache("current_k not implemented in v1 paged cache".into()))
    }

    fn current_v(&self) -> Result<Tensor> {
        Err(Error::KvCache("current_v not implemented in v1 paged cache".into()))
    }
}