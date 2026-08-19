    fn dora_backward_sanity() -> Result<(), Box<dyn std::error::Error>> {
        let in_features = 4;
        let out_features = 3;
        let rank = 2;
        let batch = 1;

        let x = tensor(
            (0..in_features).map(|i| (i as f32 + 1.0) / 10.0).collect::<Vec<f32>>(),
            vec![batch, in_features],
        );
        let w_base = tensor(
            (0..out_features * in_features)
                .map(|i| (i as f32 + 1.0) / 100.0)
                .collect::<Vec<f32>>(),
            vec![out_features, in_features],
        );
        let a = tensor(
            (0..rank * in_features)
                .map(|i| (i as f32 + 1.0) / 50.0)
                .collect::<Vec<f32>>(),
            vec![rank, in_features],
        );
        let b = tensor(
            (0..rank * out_features)
                .map(|i| (i as f32 + 1.0) / 50.0)
                .collect::<Vec<f32>>(),
            vec![rank, out_features],
        );
        let m = tensor(
            (0..in_features)
                .map(|i| 0.5 + (i as f32) * 0.1)
                .collect::<Vec<f32>>(),
            vec![batch, in_features],
        );
        let scale = 1.0;
        let out_grad = tensor(
            (0..out_features * batch).map(|i| 1.0).collect::<Vec<f32>>(),
            vec![batch, out_features],
        );

        let (_, grad_w, grad_a, grad_b, grad_m) = dora_backward(&out_grad, &x, &w_base, &a, &b, &m, scale)?;

        // Shape checks
        assert_eq!(grad_w.shape().dims(), vec![out_features, in_features]);
        assert_eq!(grad_a.shape().dims(), vec![rank, in_features]);
        assert_eq!(grad_b.shape().dims(), vec![rank, out_features]);
        assert_eq!(grad_m.shape().dims(), vec![batch, in_features]);

        // Non-zero gradient check (catches catastrophic zeroing bugs)
        let gw = grad_w.to_vec_f32().unwrap();
        let ga = grad_a.to_vec_f32().unwrap();
        let gb = grad_b.to_vec_f32().unwrap();
        let gm = grad_m.to_vec_f32().unwrap();

        assert!(
            gw.iter().any(|&v| v.abs() > 1e-6),
            "grad_w is all zeros — likely a sign/zero bug"
        );
        assert!(
            ga.iter().any(|&v| v.abs() > 1e-6),
            "grad_a is all zeros — likely a sign/zero bug"
        );
        assert!(
            gb.iter().any(|&v| v.abs() > 1e-6),
            "grad_b is all zeros — likely a sign/zero bug"
        );
        assert!(
            gm.iter().any(|&v| v.abs() > 1e-6),
            "grad_m is all zeros — likely a sign/zero bug"
        );
        Ok(())
    }
