# ScytheRing Multi-GPU Universal Dispatch Plan

## Combined Option 2+3: Peer-Direct + Pinned Host Staging

**Status**: peer access confirmed working on all 6 directed device pairs
(3 ROCm devices: gfx1201 + second GPU + gfx1036 APU). enable_peer_access
returned true for every pair. peer_status reports Pcie for all pairs
(no xGMI/Infinity Fabric on consumer hardware — PCIe DMA is the transport).

---

## Architecture

```
                    +------------------------------+
                    |      Host Coordinator         |
                    |  Topology: P2PTopology matrix |
                    |  Routes descriptors per op    |
                    +------+--------+------+------+
                           |        |      |
              +------------v+ +-----v-----------+ +v-------------+
              | GPU[0] ring | | GPU[1] ring     | | APU[2] ring  |
              | (gfx1201)   | | (gfx110x?)      | | (gfx1036)    |
              | wave polls  | | wave polls      | | wave polls   |
              +------+------+ +------+----------+ +------+-------+
                     |               |                  |
              +------v------+ +-----v-------+  +------v------+
              | VRAM 16 GB  | | VRAM 16 GB  |  | GTT 2 GB    |
              +------+------+ +-----+-------+  +------+------+
                     |               |                 |
              =======+===============+=================+=======
                     |      PCIe peer access           |
                     |   (all pairs confirmed Pcie)    |
```

Transport selection per device pair:
- Pcie (peer access enabled) -> peer_ptr in task descriptor -> direct remote
  write from the persistent wave (no host round-trip)
- HostBounce -> pinned host staging buffer -> peer reads via host-coordinated
  hipMemcpyPeerAsync (fallback for pairs where peer access fails)

On this system: ALL pairs are Pcie, so peer-direct is available everywhere.

---

## Phases

### Phase MG-1: Topology discovery + peer access enablement (prerequisite)

At RocmDevice construction (or at first ring channel creation):
1. Build P2PTopology via peer_access::build_topology_matrix(&[&devices])
2. Call enable_peer_access(src, dst) for every pair where peer_status != Host
3. Store the LinkType per pair in the RingChannel (or a per-device field)

Changes: scythe_route.rs -- channel_for() calls peer_access::enable_peer_access
for all pairs before returning the channel. Cache the P2PTopology result.

Verification: p2p_probe example prints enable_peer_access = true for all pairs
(already confirmed on this system).

---

### Phase MG-2: Cross-device descriptor routing

Extend RingChannel to route descriptors to peer device rings. Two patterns:

Pattern A (host-coordinated): The host coordinator writes descriptors to the
target device's ring channel. The route_gemm function takes a target_ordinal
parameter:
```rust
pub(crate) fn route_gemm_to(
    device: &RocmDevice,       // device that owns the output buffer
    target: &RocmDevice,       // device that executes the GEMM
    ...
)
```
If device.ordinal() != target.ordinal(), the descriptor is written to the
TARGET device's ring, and the input pointers use peer-access-enabled
addresses (which ROCm provides automatically when peer access is enabled).

Pattern B (device-initiated): Device A's persistent wave, after completing an
op, writes the next descriptor directly into device B's ring slots (via peer
memory write). This requires:
- peer_ptr in the descriptor = address of device B's slot array
- A peer atomic store to advance device B's head counter
- No host involvement for chained ops on different devices

Changes: scythe_route.rs -- add route_gemm_to(), extend RingChannel
with peer_link: Option<LinkType>, add peer descriptor routing.

Verification: Route a GEMM on GPU[0] whose input is on GPU[1] (peer access
enabled). Output must match CPU reference.

---

### Phase MG-3: OP_COMMFUSE as cross-device tensor pipe

The existing OP_COMMFUSE opcode (scythe_persistent.rs:263) already reads
src, writes peer_dst, and writes local_out in one kernel execution.
Fill peer_ptr with the remote device's buffer address:

```c
// OP_COMMFUSE: src -> peer_dst + local_out (fused copy/broadcast)
const float* src = (const float*)desc->input_ptr;
float* peer_dst = (float*)desc->peer_ptr;       // remote device memory
float* local_out = (float*)desc->output_ptr;    // local device memory
for (idx ...) {
    float val = src[idx];
    if (peer_dst) peer_dst[idx] = val;          // write to peer via PCIe
    if (local_out) local_out[idx] = val;        // write locally
}
```

Model-level use: after GPU[0]'s persistent wave completes a GEMM whose
consumer is on GPU[1], emit an OP_COMMFUSE descriptor on GPU[0]'s ring with:
- input_ptr = GPU[0] local output buffer
- peer_ptr = GPU[1] destination buffer (peer-access-enabled address)
- local_out = 0 (no local copy needed if the tensor is fully forwarded)

Changes: scythe_route.rs -- add route_commfuse() function. parallel_comm.rs
-- the HostStagingRing becomes the fallback when peer access is unavailable.

Verification: Write a tensor on GPU[0], pipe it to GPU[1] via OP_COMMFUSE,
read it back on GPU[1], compare against CPU reference.

---

### Phase MG-4: New multi-GPU opcodes

Add opcodes for multi-GPU collectives that the persistent wave executes inline:

| Opcode | Name              | Semantics |
|--------|-------------------|-----------|
| 8      | OP_PEER_REDUCE    | Ring-allreduce across N device rings (or RCCL if available) |
| 9      | OP_PEER_BROADCAST | One-to-all broadcast from root device to N peers |
| 10     | OP_PEER_GATHER    | Gather partial results from N peers into one device |

OP_PEER_REDUCE design (ring-allreduce, works over PCIe):
- Each device's persistent wave computes a partial sum into its local buffer
- The wave writes partials to peer buffers via peer_ptr (ring topology: A->B->C->A)
- Each wave receives peer partials, adds them, and forwards
- After N-1 rounds, every device has the full sum

Alternative for 2-device setup (simpler, sufficient for GPU+APU):
- Device A writes its partial to peer_ptr (device B's buffer)
- Device B's wave reads device A's partial (via peer access), adds its own,
  writes the sum back
- Total: 2 PCIe transfers per all-reduce (one per direction)

Changes: scythe_persistent.rs -- add 3 opcode arms. scythe_route.rs --
add descriptor packing functions for each.

---

### Phase MG-5: Cross-device dependency tracking

When device A's output feeds device B's input, device B's wave must wait for
device A's completion. Options (in order of preference for this system):

5a. Peer flag polling (simplest, works with peer access):
- Device A's wave writes ST_COMPLETE to the descriptor's status field
- Device B's wave polls the peer status field via peer-access memory read
- No host involvement; works over PCIe peer access

5b. Host-mediated barrier (fallback for non-P2P pairs):
- The host coordinator tracks completion via hipStreamSynchronize per ring
- Routes the next descriptor only after the dependency is satisfied
- Adds host latency but works universally

5c. hipEvent cross-device (zero-copy):
- Record an event on device A's stream after quantize
- Device B's stream waits on the event before launching the consumer kernel
- Requires hipEventRecord + hipStreamWaitEvent (supported across devices
  when peer access is enabled)

Changes: scythe_route.rs -- add dependency tracking to the ring coordinator.
scythe_persistent.rs -- add peer status polling for option 5a.

---

### Phase MG-6: Model-level wiring

Tensor-parallel decode (the primary use case for this system):
- QKV projection: split across GPU[0] + GPU[1], each computes half the heads
  -> all-reduce via OP_PEER_REDUCE after both complete
- FFN: split across GPU[0] + APU[2] (row-parallel)
  -> all-reduce via OP_PEER_REDUCE
- KV cache: replicated or sharded per device

Pipeline-parallel decode (alternative):
- GPU[0] handles layers 0..8
- GPU[1] handles layers 8..16
- APU[2] handles layers 16..24 (if MoE or small layers)
- OP_COMMFUSE transfers activations between pipeline stages

Changes: model.rs (or a new parallel_model.rs) -- a multi-device model
wrapper that routes ops to the owning device's ScytheRing ring. The session
holds ParallelTopology with the device assignment.

---

## Verification plan

| Step | Test                           | Device pairs                   | Expected                |
|------|--------------------------------|--------------------------------|-------------------------|
| MG-1 | p2p_probe example              | all pairs                      | enable_peer_access=true |
| MG-2 | Route GEMM to peer device      | GPU[0]->GPU[1], GPU[0]->APU[2] | matches CPU reference   |
| MG-3 | OP_COMMFUSE peer pipe          | GPU[0]->GPU[1], GPU[0]->APU[2] | data arrives intact     |
| MG-4 | OP_PEER_REDUCE (2-device)      | GPU[0]+GPU[1]                  | sum matches CPU         |
| MG-5 | Dependency tracking            | chained ops across devices     | ordering preserved      |
| MG-6 | Tensor-parallel Llama decode   | GPU[0]+GPU[1] or GPU[0]+APU[2] | tokens match single-dev |

---

## Hardware-specific considerations (9800X3D + RX 9070 XT + gfx1036 APU)

| Pair            | Link                            | P2P   | Bandwidth   | Notes |
|-----------------|---------------------------------|-------|-------------|-------|
| GPU[0]<->GPU[1] | PCIe 4.0 x16-x16                | Pcie  | ~32 GB/s    | Both dedicated GPUs, same root complex |
| GPU[0]<->APU[2] | PCIe 4.0 + Infinity Fabric      | Pcie  | ~16-32 GB/s | Different memory pools (VRAM vs GTT) |
| APU[2]<->Host   | Infinity Fabric                 | N/A   | ~50-100 GB/s| APU's unified memory advantage |

APU advantage: The gfx1036 APU shares system RAM with the CPU via Infinity
Fabric. A pinned host buffer is accessible to the APU at near-local speed and
to the dedicated GPU via PCIe DMA. This makes pinned host memory the natural
staging area for GPU<->APU data exchange, even without peer access.

APU limitation: 2 GB GTT is small. A 350M model's weights (Q8_0 = 350 MB) fit,
but activations + KV cache for long contexts may exceed the budget.
The APU is best used for small layers, embedding, or the router gate.

---

## Dependency graph

MG-1 (topology) --> MG-2 (descriptor routing) --> MG-3 (COMMFUSE pipe)
                                              --> MG-4 (new opcodes)
                                              --> MG-5 (dependency tracking)
                                                        |
                                                        v
                                              MG-6 (model wiring)

MG-1 is the prerequisite. MG-2 through MG-5 can proceed in parallel once
MG-1 confirms peer access. MG-6 depends on MG-2+MG-3+MG-5.
