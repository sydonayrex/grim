use grim_backend_rocm::device::expert_prefetch_stream::ExpertPrefetchStream;

/// Under a 10-forward stress loop across 8 layers, double-buffered next-layer prefetch
/// (`slot = (l + 1) & 1`) must deliver byte-identical expert weights to every layer and
/// record a completed stream sync before every layer's expert compute.
#[test]
fn astrum_p12_double_buffered_expert_prefetch_stress_loop_bit_exact() {
    let num_layers = 8;
    let mut prefetcher = ExpertPrefetchStream::new(num_layers);
    let mut staging_ring: [Vec<u8>; 2] = [Vec::new(), Vec::new()];

    let make_layer_weights = |layer: usize, expert: usize| -> Vec<u8> {
        (0..64)
            .map(|i| ((layer * 37 + expert * 13 + i) & 0xFF) as u8)
            .collect()
    };

    for _pass in 0..10 {
        // Seed layer 0 prefetch before layer 0 starts
        let w0_e1 = make_layer_weights(0, 1);
        let w0_e2 = make_layer_weights(0, 2);
        prefetcher.enqueue_next_layer_prefetch(
            0,
            vec![1, 2],
            &mut staging_ring,
            &[&w0_e1, &w0_e2],
        );

        for l in 0..num_layers {
            // 1. Sync current layer `l` before consuming its prefetched weights
            let ticket = prefetcher
                .sync_layer_ready(l)
                .expect("layer prefetch ticket must exist");
            assert!(ticket.synced);
            assert_eq!(ticket.ring_slot, l & 1);

            let mut expected = make_layer_weights(l, 1);
            expected.extend_from_slice(&make_layer_weights(l, 2));
            assert_eq!(
                staging_ring[l & 1],
                expected,
                "prefetched expert bytes at layer {l} must be bit-identical to synchronous path"
            );

            // 2. Overlap next layer `l + 1` upload on opposite ring slot `(l + 1) & 1`
            if l + 1 < num_layers {
                let wn_1 = make_layer_weights(l + 1, 1);
                let wn_2 = make_layer_weights(l + 1, 2);
                prefetcher.enqueue_next_layer_prefetch(
                    l + 1,
                    vec![1, 2],
                    &mut staging_ring,
                    &[&wn_1, &wn_2],
                );
            }
        }
    }

    assert_eq!(prefetcher.completed_syncs(), 10 * num_layers);
}
