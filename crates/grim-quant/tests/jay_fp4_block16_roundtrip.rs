use grim_quant::{dequant_fp4_block16, quant_fp4_block16};

#[test]
fn nvfp4_roundtrip_is_stable() {
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.25 - 2.0).collect();
    let packed = quant_fp4_block16(&data, 16).expect("nvfp4 quant");
    let recovered = dequant_fp4_block16(&packed, data.len()).expect("nvfp4 dequant");
    assert_eq!(recovered.len(), data.len());
    let max_err: f32 = data
        .iter()
        .zip(recovered.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        max_err < 1.5,
        "nvfp4 roundtrip max error too high: {max_err}"
    );
}
