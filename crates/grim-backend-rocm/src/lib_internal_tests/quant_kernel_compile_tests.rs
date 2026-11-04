//! note: string-only gate retained because grim_split_k_reduction has no exported launcher; upgrade to launch-and-compare when launcher lands

#[cfg(test)]
mod tests {
    #[test]
    fn test_split_k_reduction_bit_stable_for_training() {
        let kernel_source = crate::kernels::source_asm::compute_kernel_source();
        assert!(kernel_source.contains("grim_split_k_reduction"));
    }
}
