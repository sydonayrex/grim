#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    let mut cursor = Cursor::new(data);
    let _ = grim_format::safetensors::read_safetensors_header(&mut cursor);
});
