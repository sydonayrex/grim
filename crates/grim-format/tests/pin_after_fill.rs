use std::io::Write;
use tempfile::NamedTempFile;
use grim_format::bank::{HostBank, FillFlags};

#[test]
fn test_pin_after_fill_lifecycle() {
    let size = 64 * 1024; // 64KB
    let mut bank = HostBank::mmap_lazy(size).expect("mmap_lazy should succeed");

    assert!(!bank.is_pinned(), "bank should not be pinned initially");
    assert_eq!(bank.len(), size);

    // Create a temporary file with known pattern
    let mut file = NamedTempFile::new().unwrap();
    let pattern: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
    file.write_all(&pattern).unwrap();
    file.flush().unwrap();

    let mut read_file = std::fs::File::open(file.path()).unwrap();
    let read_bytes = bank.fill_from_disk(&mut read_file, FillFlags::Standard).expect("fill_from_disk should succeed");
    assert_eq!(read_bytes, size);

    assert_eq!(bank.as_slice(), pattern.as_slice());

    // Pin after filling data
    bank.pin().expect("pin should succeed");
    assert!(bank.is_pinned());
    assert_eq!(bank.as_slice(), pattern.as_slice());
}

#[test]
fn test_host_bank_fill_from_slice() {
    let size = 4096;
    let mut bank = HostBank::mmap_lazy(size).unwrap();
    let data = vec![42u8; size];
    bank.fill_from_slice(&data).unwrap();
    assert_eq!(bank.as_slice(), data.as_slice());
    bank.pin().unwrap();
    assert!(bank.is_pinned());
}
