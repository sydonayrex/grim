use grim_backend_rocm::kernels::shared_device_fns::KERNEL_SOURCE;

fn reference_shfl_xor_wave32(lanes: &[f32; 32]) -> f32 {
    let mut v = *lanes;
    for mask in [16usize, 8, 4, 2, 1] {
        let prev = v;
        for l in 0..32 {
            v[l] = prev[l] + prev[l ^ mask];
        }
    }
    v[0]
}

fn tsum4_wave32_host(cols: &[[f32; 32]; 4]) -> [f32; 4] {
    let mut s = *cols;
    for c in 0..4 {
        let prev = s[c];
        for l in 0..32 {
            s[c][l] = prev[l] + prev[l ^ 16];
        }
    }
    let mut p01 = [0.0f32; 32];
    let mut p23 = [0.0f32; 32];
    for l in 0..32 {
        let src01 = if (l & 8) == 0 { s[0][l ^ 8] } else { s[1][l ^ 8] };
        let own01 = if (l & 8) == 0 { s[0][l] } else { s[1][l] };
        p01[l] = own01 + src01;

        let src23 = if (l & 8) == 0 { s[2][l ^ 8] } else { s[3][l ^ 8] };
        let own23 = if (l & 8) == 0 { s[2][l] } else { s[3][l] };
        p23[l] = own23 + src23;
    }
    let mut q = [0.0f32; 32];
    for l in 0..32 {
        let src = if (l & 4) == 0 { p01[l ^ 4] } else { p23[l ^ 4] };
        let own = if (l & 4) == 0 { p01[l] } else { p23[l] };
        q[l] = own + src;
    }
    for mask in [2usize, 1] {
        let prev = q;
        for l in 0..32 {
            q[l] = prev[l] + prev[l ^ mask];
        }
    }
    [q[0], q[8], q[4], q[12]]
}

#[test]
fn oracle_tsum4_is_bit_identical_to_shfl_xor_across_random_waves() {
    let mut st = 0xCAFE_BABE_1234_5678u64;
    for _trial in 0..256 {
        let mut cols = [[0.0f32; 32]; 4];
        for c in 0..4 {
            for l in 0..32 {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((st >> 33) as i32 - (1 << 30)) as f32 * 1e-7;
                cols[c][l] = val;
            }
        }
        let got = tsum4_wave32_host(&cols);
        for c in 0..4 {
            let want = reference_shfl_xor_wave32(&cols[c]);
            assert_eq!(got[c].to_bits(), want.to_bits());
        }
    }
}

#[test]
fn shared_device_fns_contains_strata_sync_lds_perm_and_tsum4() {
    assert!(KERNEL_SOURCE.contains("grim_sync_lds"));
    assert!(KERNEL_SOURCE.contains("grim_perm_b32"));
    assert!(KERNEL_SOURCE.contains("grim_tsum4_wave32"));
}
