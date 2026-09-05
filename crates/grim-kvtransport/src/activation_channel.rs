//! Point-to-point activation frames over TCP for multi-rank VPP (R3).
//!
//! VPP rank threads exchange f32 activation tensors at chunk fold points.
//! Unlike the KV-block protocol (block-store semantics, pull by block id),
//! activations are push-only, ordered per (channel, tag), and sized by the
//! sender — a separate minimal wire format keeps the two contracts from
//! drifting. Latency budget: prefill chunks are large, so plain TCP is
//! acceptable (see `gpu-followup-workitems.md`, work item 2, transport
//! options).

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use grim_core::error::{Error, Result};

/// Wire magic for the activation frame format (`VPAX`).
const ACTIVATION_MAGIC: u32 = 0x5650_4158;

/// Header: magic, channel id, tag, element count.
const HEADER_BYTES: usize = 16;

/// A peer that stops reading (crashed rank, wedged scheduler) must surface
/// as an error instead of parking the receiving rank forever.
const IO_TIMEOUT: Duration = Duration::from_secs(60);

/// How long a rank waits for its peer's activation before declaring the
/// schedule wedged.
const ACCEPT_DEADLINE: Duration = Duration::from_secs(30);

/// Per-rank TCP transport for VPP activation exchange.
///
/// One [`TcpActivationTransport`] owns a listener per rank so a single-node,
/// multi-rank job (one process, N GPUs) can run all ranks over loopback;
/// a multi-node deployment constructs it with only the local rank's listener
/// and `set_peer` entries for the remote ranks. The same call sequence works
/// for both — senders dial the peer's listener, receivers accept on their own.
pub struct TcpActivationTransport {
    listeners: Vec<TcpListener>,
    peers: HashMap<usize, SocketAddr>,
}

impl TcpActivationTransport {
    /// Binds an ephemeral-port listener per rank on loopback.
    pub fn bind(num_ranks: usize) -> Result<Self> {
        let mut listeners = Vec::with_capacity(num_ranks);
        for rank in 0..num_ranks {
            let listener = TcpListener::bind(("127.0.0.1", 0))
                .map_err(|e| Error::KvCache(format!("vpp activation bind rank {rank}: {e}")))?;
            listeners.push(listener);
        }
        Ok(Self {
            listeners,
            peers: HashMap::new(),
        })
    }

    /// The address peers must dial to reach `rank`'s receiver.
    pub fn local_addr(&self, rank: usize) -> Result<SocketAddr> {
        let listener = self.listener(rank)?;
        listener
            .local_addr()
            .map_err(|e| Error::KvCache(format!("vpp activation local_addr rank {rank}: {e}")))
    }

    /// Registers where `rank`'s receiver listens. `set_peer` on the local
    /// rank's own address is valid (same-process loopback).
    pub fn set_peer(&mut self, rank: usize, addr: SocketAddr) {
        self.peers.insert(rank, addr);
    }

    /// Sends one activation frame to `to_rank`. Fire-and-forget: the frame
    /// lands in the peer's listener backlog and the peer's matching
    /// [`Self::recv_activation`] consumes it.
    pub fn send_activation(
        &self,
        to_rank: usize,
        channel_id: u32,
        tag: u32,
        data: &[f32],
    ) -> Result<()> {
        let addr = *self.peers.get(&to_rank).ok_or_else(|| {
            Error::KvCache(format!(
                "vpp activation: no peer address for rank {to_rank}"
            ))
        })?;
        let mut stream = TcpStream::connect(addr)
            .map_err(|e| Error::KvCache(format!("vpp activation connect rank {to_rank}: {e}")))?;
        stream
            .set_write_timeout(Some(IO_TIMEOUT))
            .map_err(|e| Error::KvCache(format!("vpp activation set_write_timeout: {e}")))?;

        let count = u32::try_from(data.len()).map_err(|_| {
            Error::KvCache(format!("activation length {} exceeds u32::MAX", data.len()))
        })?;

        let mut frame = Vec::with_capacity(HEADER_BYTES + data.len() * 4);
        frame.extend_from_slice(&ACTIVATION_MAGIC.to_le_bytes());
        frame.extend_from_slice(&channel_id.to_le_bytes());
        frame.extend_from_slice(&tag.to_le_bytes());
        frame.extend_from_slice(&count.to_le_bytes());
        // Explicit little-endian encoding: host-endianness independent,
        // bit-exact for every IEEE-754 payload including NaNs.
        for f in data {
            frame.extend_from_slice(&f.to_le_bytes());
        }

        stream
            .write_all(&frame)
            .map_err(|e| Error::KvCache(format!("vpp activation send rank {to_rank}: {e}")))?;
        stream
            .flush()
            .map_err(|e| Error::KvCache(format!("vpp activation flush rank {to_rank}: {e}")))?;
        Ok(())
    }

    /// Blocks until the next frame for `for_rank` with a matching
    /// (channel_id, tag) arrives, and returns its payload. The frame's own
    /// element count sizes the output; callers validate it against the
    /// expected activation shape.
    pub fn recv_activation(&self, for_rank: usize, channel_id: u32, tag: u32) -> Result<Vec<f32>> {
        let listener = self.listener(for_rank)?;
        let deadline = Instant::now() + ACCEPT_DEADLINE;
        loop {
            listener
                .set_nonblocking(true)
                .map_err(|e| Error::KvCache(format!("vpp activation set_nonblocking: {e}")))?;
            match listener.accept() {
                Ok((mut stream, _peer)) => return Self::read_frame(&mut stream, channel_id, tag),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        return Err(Error::KvCache(format!(
                            "vpp activation recv rank {for_rank}: no frame within {ACCEPT_DEADLINE:?}"
                        )));
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(e) => {
                    return Err(Error::KvCache(format!(
                        "vpp activation accept rank {for_rank}: {e}"
                    )));
                }
            }
        }
    }

    fn listener(&self, rank: usize) -> Result<&TcpListener> {
        self.listeners.get(rank).ok_or_else(|| {
            Error::KvCache(format!(
                "vpp activation: rank {rank} out of range ({} listeners)",
                self.listeners.len()
            ))
        })
    }

    fn read_frame(stream: &mut TcpStream, channel_id: u32, tag: u32) -> Result<Vec<f32>> {
        stream
            .set_read_timeout(Some(IO_TIMEOUT))
            .map_err(|e| Error::KvCache(format!("vpp activation set_read_timeout: {e}")))?;

        let mut header = [0u8; HEADER_BYTES];
        stream
            .read_exact(&mut header)
            .map_err(|e| Error::KvCache(format!("vpp activation header read: {e}")))?;
        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        if magic != ACTIVATION_MAGIC {
            return Err(Error::KvCache(format!(
                "vpp activation: protocol mismatch magic={magic:#x} expected={ACTIVATION_MAGIC:#x}"
            )));
        }
        let wire_channel = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        let wire_tag = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
        if wire_channel != channel_id || wire_tag != tag {
            return Err(Error::KvCache(format!(
                "vpp activation: frame (channel {wire_channel}, tag {wire_tag}) does not match \
                 expected (channel {channel_id}, tag {tag})"
            )));
        }
        let count = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;

        let mut bytes = vec![0u8; count * 4];
        stream
            .read_exact(&mut bytes)
            .map_err(|e| Error::KvCache(format!("vpp activation payload read: {e}")))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Binds a two-rank loopback transport with both peers registered.
    fn bound_loopback() -> std::sync::Arc<TcpActivationTransport> {
        let mut transport = TcpActivationTransport::bind(2).expect("bind");
        transport.set_peer(0, transport.local_addr(0).unwrap());
        transport.set_peer(1, transport.local_addr(1).unwrap());
        std::sync::Arc::new(transport)
    }

    #[test]
    fn test_activation_frame_roundtrip_loopback() {
        let transport = bound_loopback();
        let payload: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();

        let receiver = {
            let transport = std::sync::Arc::clone(&transport);
            std::thread::spawn(move || transport.recv_activation(1, 7, 42))
        };
        transport.send_activation(1, 7, 42, &payload).expect("send");
        let got = receiver.join().unwrap().expect("recv");

        assert_eq!(got, payload, "f32 payload must round-trip bit-exact");
    }

    #[test]
    fn test_recv_rejects_mismatched_tag() {
        let transport = bound_loopback();

        let receiver = {
            let transport = std::sync::Arc::clone(&transport);
            std::thread::spawn(move || transport.recv_activation(1, 3, 9))
        };
        transport
            .send_activation(1, 3, 8, &[1.0])
            .expect("send with wrong tag");
        let err = receiver
            .join()
            .unwrap()
            .expect_err("tag mismatch must error");
        assert!(err.to_string().contains("does not match"), "{err}");
    }
}
