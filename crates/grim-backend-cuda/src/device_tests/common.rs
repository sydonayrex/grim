//! Shared test-support helpers for the split CUDA device_tests modules.

use crate::device::cuda_device::CudaDevice;
use crate::memory::storage::CudaStorage;
use grim_tensor::dtype::{ ArithType, DType, Storage as DTypeStorage };
use grim_tensor::{BackendStorage, MemoryOps, Shape};

pub(super) fn dequant_test_device() -> Option<CudaDevice> {
        unsafe { std::env::set_var("GRIM_CUDA_ORDINAL_OVERRIDE", "0") };
        CudaDevice::probe()
            .ok()
            .filter(|d| !d.is_empty())
            .map(|d| d[0].clone())
    }

pub(super) fn assert_dequant_close(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        let max_err = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "{label}: GPU dequant max error {max_err} exceeds 1e-4 \
             (first 4 actual={:?} expected={:?})",
            &actual[..actual.len().min(4)],
            &expected[..expected.len().min(4)],
        );
    }

pub(super) fn upload_packed(
        dev: &CudaDevice,
        bytes: &[u8],
        shape: &Shape,
        storage_kind: DTypeStorage,
    ) -> Box<dyn BackendStorage> {
        let dtype = DType {
            arith: ArithType::U8,
            storage: storage_kind,
        };
        dev.from_cpu_bytes(bytes, shape, dtype)
            .expect("from_cpu_bytes for packed quantized storage")
    }

pub(super) fn as_cuda_storage(s: &dyn BackendStorage) -> &CudaStorage {
        s.as_any()
            .downcast_ref::<CudaStorage>()
            .expect("expected CudaStorage from from_cpu_bytes")
    }

pub(super) fn build_mxfp4_single_buffer(codes: &[u8], exps: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(codes.len() as u64).to_le_bytes());
        buf.extend_from_slice(codes);
        buf.extend_from_slice(&(exps.len() as u64).to_le_bytes());
        buf.extend_from_slice(exps);
        buf
    }

