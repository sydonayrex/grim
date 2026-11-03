use grim_backend_rocm::device::roc_device::RocmDevice;
use grim_backend_rocm::device::util::as_rocm;
use grim_tensor::backend::{BackendStorage, SamplingOps};
use grim_tensor::{CoreTensorOps, DType, MemoryOps, Shape};

#[test]
#[ignore]
fn test_speculative_sampler_parity() {
    let dev = match RocmDevice::try_new(0) {
        Ok(d) => d,
        Err(_) => return,
    };

    let batch_size = 1;
    let num_draft = 4;
    let vocab_size = 1024;

    // Craft draft and target probabilities where tokens 0 and 1 are accepted, token 2 is rejected
    let mut draft_probs_vec = vec![0.0f32; num_draft * vocab_size];
    let mut target_probs_vec = vec![0.0f32; (num_draft + 1) * vocab_size];

    let draft_tokens = vec![100u32, 200, 300, 400];
    let p_draft = vec![0.8f32, 0.7, 0.6, 0.5];
    let p_target = vec![0.9f32, 0.8, 0.1, 0.5];
    let uniform_rands = vec![0.5f32, 0.5, 0.8, 0.1, 0.5];

    for (k, &tok) in draft_tokens.iter().enumerate() {
        draft_probs_vec[k * vocab_size + tok as usize] = p_draft[k];
        target_probs_vec[k * vocab_size + tok as usize] = p_target[k];
    }

    // CPU baseline
    let mut cpu_accepted = Vec::new();
    for i in 0..draft_tokens.len() {
        let p_d = p_draft[i].max(1e-8);
        let p_t = p_target[i];
        let ratio = (p_t / p_d).min(1.0);
        if uniform_rands[i] <= ratio {
            cpu_accepted.push(draft_tokens[i]);
        } else {
            break;
        }
    }

    // GPU evaluation
    let tp_shape = Shape::new(vec![batch_size, num_draft + 1, vocab_size]);
    let dp_shape = Shape::new(vec![batch_size, num_draft, vocab_size]);
    let dt_shape = Shape::new(vec![batch_size, num_draft]);
    let ur_shape = Shape::new(vec![batch_size, num_draft + 1]);
    let at_shape = Shape::new(vec![batch_size, num_draft + 1]);
    let al_shape = Shape::new(vec![batch_size]);

    let dt_bytes: Vec<u8> = draft_tokens.iter().flat_map(|&t| (t as i32).to_le_bytes()).collect();

    let tp_dev = CoreTensorOps::from_cpu(&dev, &target_probs_vec, &tp_shape, DType::F32).unwrap();
    let dp_dev = CoreTensorOps::from_cpu(&dev, &draft_probs_vec, &dp_shape, DType::F32).unwrap();
    let dt_dev = MemoryOps::from_cpu_bytes(&dev, &dt_bytes, &dt_shape, DType::U8).unwrap();
    let ur_dev = CoreTensorOps::from_cpu(&dev, &uniform_rands, &ur_shape, DType::F32).unwrap();
    let u32_dt = grim_tensor::DType {
        arith: grim_tensor::ArithType::U32,
        storage: grim_tensor::Storage::Native,
    };
    let al_dev = MemoryOps::alloc_storage(&dev, &al_shape, u32_dt.clone()).unwrap();
    let at_dev = MemoryOps::alloc_storage(&dev, &at_shape, u32_dt).unwrap();

    let tp_s = as_rocm(tp_dev.as_ref()).unwrap();
    let dp_s = as_rocm(dp_dev.as_ref()).unwrap();
    let dt_s = as_rocm(dt_dev.as_ref()).unwrap();
    let ur_s = as_rocm(ur_dev.as_ref()).unwrap();
    let at_s = as_rocm(at_dev.as_ref()).unwrap();
    let al_s = as_rocm(al_dev.as_ref()).unwrap();

    dev.speculative_rejection_sample(
        tp_s,
        dp_s,
        dt_s,
        ur_s,
        at_s,
        al_s,
        batch_size,
        num_draft,
        vocab_size,
    ).unwrap();
    dev.synchronize();

    let al_vec = al_s.to_cpu_vec_u32().unwrap();
    let gpu_accepted_count = al_vec[0] as usize;

    let at_vec = at_s.to_cpu_vec_u32().unwrap();
    let gpu_tokens: Vec<u32> = at_vec.into_iter().take(gpu_accepted_count).collect();

    // Standard Speculative Decoding (Leviathan et al.):
    // When a draft token k is rejected, the target model samples a replacement token from
    // normalized max(0, p - q). So accepted count is k accepted draft tokens + 1 target replacement token.
    assert_eq!(gpu_accepted_count, cpu_accepted.len() + 1, "Accepted count mismatch");
    assert_eq!(gpu_accepted_count, 3);
    assert_eq!(&gpu_tokens[0..2], &cpu_accepted[0..2]);
}
