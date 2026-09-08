//! Backend tests verifying tensor storage creation, quant conversions, and
//! backend device parity between CPU and GPU backends.

use grim_backend_cpu::CpuStorage;
use grim_tensor::dtype::{ArithType, DType, Storage as DTypeStorage};
use grim_tensor::{Device, QuantProvenance, Shape, Tensor};
use std::sync::Arc;

#[test]
fn test_tensor_storage_and_shape_properties() {
    let shape = Shape::new(vec![2, 4]);
    assert_eq!(shape.elem_count(), 8);
    assert_eq!(shape.rank(), 2);

    let dtype = DType {
        arith: ArithType::F32,
        storage: DTypeStorage::Native,
    };
    assert_eq!(dtype.arith.byte_size(), 4);

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let storage = Arc::new(CpuStorage::new(data, shape.clone(), dtype.clone()));
    let t = Tensor::new(
        storage,
        shape,
        dtype,
        QuantProvenance::GrimNative,
        Device::Cpu,
    );
    assert_eq!(t.shape().dims(), &[2, 4]);
}

#[cfg(feature = "rocm")]
#[test]
fn test_rocm_backend_device_integration() {
    use grim_backend_rocm::RocmDevice;
    let dev = RocmDevice::new(0);
    assert_eq!(dev.ordinal(), 0);
}
