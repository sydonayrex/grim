//! Test: Vulkan VkFsdpGroup shard planning.
//!
//! Run with `GRIM_RUN_GPU_TESTS=1 cargo test -p grim-backend-vulkan --test fsdp_parity`.

use grim_tensor::Shape;
use grim_backend_vulkan::collective::VkCommunicator;
use grim_backend_vulkan::fsdp::{VkFsdpConfig, VkFsdpGroup};

#[test]
fn fsdp_shard_shape_splits_first_dim() {
    let config = VkFsdpConfig {
        world_size: 2,
        rank: 0,
        ..Default::default()
    };
    let group = VkFsdpGroup::new(config, None).unwrap();
    let full = Shape::new(vec![1024, 256]);
    let shard = group.shard_shape(&full).unwrap();
    assert_eq!(shard.dims(), &[512, 256]);
}

#[test]
fn fsdp_shard_row_offset_respects_rank() {
    let config = VkFsdpConfig {
        world_size: 4,
        rank: 2,
        ..Default::default()
    };
    let group = VkFsdpGroup::new(config, None).unwrap();
    let full = Shape::new(vec![1024, 256]);
    let offset = group.shard_row_offset(&full).unwrap();
    // rank 2 of 4 over 1024 rows => offset 512
    assert_eq!(offset, 512);
}

#[test]
fn fsdp_rejects_uneven_split() {
    let config = VkFsdpConfig {
        world_size: 3,
        rank: 0,
        ..Default::default()
    };
    let group = VkFsdpGroup::new(config, None).unwrap();
    let full = Shape::new(vec![1024, 256]);
    // 1024 is not divisible by 3
    assert!(group.shard_shape(&full).is_err());
}

#[test]
fn fsdp_rejects_scalar_shard() {
    let config = VkFsdpConfig {
        world_size: 2,
        rank: 0,
        ..Default::default()
    };
    let group = VkFsdpGroup::new(config, None).unwrap();
    let full = Shape::new(vec![]);
    assert!(group.shard_shape(&full).is_err());
}

#[test]
fn fsdp_validates_communicator_topology() {
    let config = VkFsdpConfig {
        world_size: 2,
        rank: 0,
        ..Default::default()
    };
    // Mismatched communicator (rank 1 vs config rank 0)
    let comm = VkCommunicator::new(2, 1).unwrap();
    assert!(VkFsdpGroup::new(config, Some(comm)).is_err());

    // Matching communicator
    let config2 = VkFsdpConfig {
        world_size: 2,
        rank: 1,
        ..Default::default()
    };
    let comm2 = VkCommunicator::new(2, 1).unwrap();
    assert!(VkFsdpGroup::new(config2, Some(comm2)).is_ok());
}

#[test]
fn fsdp_single_gpu_no_communicator() {
    let config = VkFsdpConfig {
        world_size: 1,
        rank: 0,
        ..Default::default()
    };
    let group = VkFsdpGroup::new(config, None).unwrap();
    let full = Shape::new(vec![1024, 256]);
    let shard = group.shard_shape(&full).unwrap();
    // world_size=1 => shard == full
    assert_eq!(shard.dims(), &[1024, 256]);
}
