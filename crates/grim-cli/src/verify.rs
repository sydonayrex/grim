//! `grim verify` — verify a `.grim` model file for structural integrity,
//! correct offsets/sizes, readable payloads, and QLoRA adapter presence.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use grim_format::format::{GrimFile, GrimHeader, FUCKING_SORCERY};
use grim_format::spec::PayloadCompression;
use grim_tensor::error::{Error, Result};

#[derive(Debug, Default)]
pub struct VerifyReport {
    pub file_size: u64,
    pub magic_ok: bool,
    pub metadata_parsed: bool,
    pub tensor_count: u32,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub adapter_tensors: Vec<AdapterInfo>,
}

#[derive(Debug)]
pub struct AdapterInfo {
    pub tensor_name: String,
    pub backup2_bpw: u8,
    pub backup2_codes_size: u64,
    pub backup2_scale_size: u64,
}

pub fn cmd_verify(path: &str) -> Result<()> {
    let path = Path::new(path);
    println!("=== Grim Verify: {} ===\n", path.display());

    let file = File::open(path).map_err(|e| Error::Io(e))?;
    let file_size = file.metadata().map_err(|e| Error::Io(e))?.len();
    let mut reader = BufReader::new(file);

    let mut report = VerifyReport {
        file_size,
        ..Default::default()
    };

    // 1. Read and validate header
    let header = verify_header(&mut reader, &mut report)?;
    if !report.magic_ok {
        report.errors.push("Invalid magic bytes — not a .grim file".to_string());
        print_report(&report);
        return Ok(());
    }

    // 2. Read and validate metadata
    verify_metadata(&mut reader, header.metadata_len, &mut report)?;

    // 3. Read tensor registry - seek back to start and use GrimFile::read
    reader.seek(SeekFrom::Start(0)).map_err(|e| Error::Backend(e.to_string()))?;
    let grim_file = verify_tensor_registry(&mut reader, header.num_tensors, &mut report)?;

    // 4. Verify each tensor's payload regions
    verify_payload_regions(&mut reader, &grim_file, &mut report);

    // 5. Check for QLoRA adapters in backup2
    check_adapters(&grim_file, &mut report);

    // Print summary
    print_report(&report);

    if !report.errors.is_empty() {
        Err(Error::Backend(format!(
            "Verification failed with {} error(s)",
            report.errors.len()
        )))
    } else {
        Ok(())
    }
}

fn verify_header<R: Read + Seek>(
    reader: &mut R,
    report: &mut VerifyReport,
) -> Result<GrimHeader> {
    let pos_before = reader.stream_position().map_err(|e| Error::Backend(e.to_string()))?;

    let header = GrimHeader::read(reader)?;

    let pos_after = reader.stream_position().map_err(|e| Error::Backend(e.to_string()))?;
    let header_size = pos_after - pos_before;

    // Check magic bytes
    if header.magic == FUCKING_SORCERY {
        report.magic_ok = true;
        println!("[OK]  Magic bytes: GRIM\\x01 (valid)");
    } else {
        report.magic_ok = false;
        report.errors.push(format!(
            "Magic mismatch: expected {:?}, got {:?}",
            FUCKING_SORCERY, header.magic
        ));
        println!("[ERR] Magic bytes: {:?} (INVALID)", header.magic);
    }

    println!("[OK]  Header size: {} bytes", header_size);
    println!("[OK]  Metadata length: {} bytes", header.metadata_len);
    println!("[OK]  Tensor count: {}", header.num_tensors);

    report.tensor_count = header.num_tensors;
    Ok(header)
}

fn verify_metadata<R: Read + Seek>(
    reader: &mut R,
    metadata_len: u64,
    report: &mut VerifyReport,
) -> Result<()> {
    if metadata_len == 0 {
        report.warnings.push("No metadata section present".to_string());
        println!("[WARN] No metadata section (length = 0)");
        return Ok(());
    }

    let pos_before = reader.stream_position().map_err(|e| Error::Backend(e.to_string()))?;

    let mut meta_buf = vec![0u8; metadata_len as usize];
    reader.read_exact(&mut meta_buf)
        .map_err(|e| Error::Backend(format!("Failed to read metadata: {e}")))?;

    let pos_after = reader.stream_position().map_err(|e| Error::Backend(e.to_string()))?;
    let actual_len = pos_after - pos_before;

    if actual_len != metadata_len {
        report.warnings.push(format!(
            "Metadata length mismatch: header says {}, actual read {}",
            metadata_len, actual_len
        ));
    }

    // Parse JSON
    match serde_json::from_slice::<serde_json::Value>(&meta_buf) {
        Ok(_) => {
            report.metadata_parsed = true;
            println!("[OK]  Metadata JSON: valid ({} bytes)", metadata_len);
        }
        Err(e) => {
            report.errors.push(format!("Invalid metadata JSON: {e}"));
            println!("[ERR] Metadata JSON: invalid — {e}");
        }
    }

    Ok(())
}

fn verify_tensor_registry<R: Read + Seek>(
    reader: &mut R,
    num_tensors: u32,
    report: &mut VerifyReport,
) -> Result<GrimFile> {
    let pos_before = reader.stream_position().map_err(|e| Error::Backend(e.to_string()))?;

    // Use GrimFile::read to parse the full file
    let grim_file = GrimFile::read(reader)?;

    let pos_after = reader.stream_position().map_err(|e| Error::Backend(e.to_string()))?;
    let registry_size = pos_after - pos_before;

    if grim_file.tensors.len() != num_tensors as usize {
        report.errors.push(format!(
            "Tensor count mismatch: header says {}, registry has {}",
            num_tensors,
            grim_file.tensors.len()
        ));
    } else {
        println!("[OK]  Tensor registry: {} entries ({} bytes)", num_tensors, registry_size);
    }

    // Print tensor names
    for (i, tensor) in grim_file.tensors.iter().enumerate() {
        println!(
            "      [{:>3}] {} shape={:?} bpw={} payload={} outliers={}",
            i,
            tensor.name,
            tensor.shape,
            tensor.base_bitwidth,
            tensor.payload_size,
            tensor.outlier_count
        );
    }

    Ok(grim_file)
}

fn verify_payload_regions<R: Read + Seek>(
    reader: &mut R,
    grim_file: &GrimFile,
    report: &mut VerifyReport,
) {
    println!("\n--- Payload Region Verification ---");

    for tensor in &grim_file.tensors {
        let mut _tensor_errors = 0;
        let mut _tensor_warnings = 0;

        // Check payload region
        let payload_end = tensor.payload_offset + tensor.payload_size;
        if payload_end > report.file_size {
            report.errors.push(format!(
                "Tensor '{}': payload region exceeds file size (offset={}, size={}, end={}, file={})",
                tensor.name, tensor.payload_offset, tensor.payload_size, payload_end, report.file_size
            ));
        } else {
            // Try to read payload bytes
            if let Err(e) = reader.seek(SeekFrom::Start(tensor.payload_offset)) {
                report.errors.push(format!(
                    "Tensor '{}': failed to seek to payload offset {}: {e}",
                    tensor.name, tensor.payload_offset
                ));
            } else {
                let mut buf = vec![0u8; tensor.payload_size as usize];
                if reader.read_exact(&mut buf).is_err() {
                    report.errors.push(format!(
                        "Tensor '{}': failed to read payload at offset {} size {}",
                        tensor.name, tensor.payload_offset, tensor.payload_size
                    ));
                } else {
                    // Try to decompress if needed
                    if let Some(ext) = grim_file.metadata.get_tensor_ext(&tensor.name) {
                        if ext.compression == PayloadCompression::Zstd {
                            match zstd::decode_all(buf.as_slice()) {
                                Ok(decompressed) => {
                                    println!(
                                        "      [OK]  {}: payload readable (zstd decompressed: {} bytes)",
                                        tensor.name,
                                        decompressed.len()
                                    );
                                }
                                Err(e) => {
                                    report.errors.push(format!(
                                        "Tensor '{}': zstd decompression failed: {e}",
                                        tensor.name
                                    ));
                                }
                            }
                        } else {
                            println!(
                                "      [OK]  {}: payload readable ({} bytes, raw)",
                                tensor.name, tensor.payload_size
                            );
                        }
                    } else {
                        println!(
                            "      [OK]  {}: payload readable ({} bytes, raw)",
                            tensor.name, tensor.payload_size
                        );
                    }
                }
            }
        }

        // Check outliers region
        if tensor.outlier_count > 0 {
            let outlier_end = tensor.outlier_offset + tensor.outlier_count as u64 * 6; // 6 bytes per outlier
            if outlier_end > report.file_size {
                report.errors.push(format!(
                    "Tensor '{}': outliers region exceeds file size (offset={}, count={}, end={}, file={})",
                    tensor.name, tensor.outlier_offset, tensor.outlier_count, outlier_end, report.file_size
                ));
            } else {
                // Try to read outliers
                if let Err(e) = reader.seek(SeekFrom::Start(tensor.outlier_offset)) {
                    report.errors.push(format!(
                        "Tensor '{}': failed to seek to outliers offset {}: {e}",
                        tensor.name, tensor.outlier_offset
                    ));
                } else {
                    let outlier_bytes = tensor.outlier_count as usize * 6;
                    let mut buf = vec![0u8; outlier_bytes];
                    if reader.read_exact(&mut buf).is_err() {
                        report.errors.push(format!(
                            "Tensor '{}': failed to read outliers at offset {}",
                            tensor.name, tensor.outlier_offset
                        ));
                    } else {
                        // Validate each outlier record
                        let mut valid_outliers = 0;
                        for chunk in buf.chunks_exact(6) {
                            if grim_format::format::GrimOutlier::decode(chunk).is_ok() {
                                valid_outliers += 1;
                            }
                        }
                        if valid_outliers == tensor.outlier_count as usize {
                            println!(
                                "      [OK]  {}: outliers readable ({} records)",
                                tensor.name, tensor.outlier_count
                            );
                        } else {
                            report.warnings.push(format!(
                                "Tensor '{}': outlier decode validation — {}/{} records valid",
                                tensor.name, valid_outliers, tensor.outlier_count
                            ));
                        }
                    }
                }
            }
        } else {
            println!("      [OK]  {}: no outliers", tensor.name);
        }

        // Check KV blob region
        if tensor.kv_present != 0 && tensor.kv_compressed_size > 0 {
            let kv_end = tensor.kv_compressed_offset + tensor.kv_compressed_size;
            if kv_end > report.file_size {
                report.errors.push(format!(
                    "Tensor '{}': KV blob region exceeds file size (offset={}, size={}, end={}, file={})",
                    tensor.name, tensor.kv_compressed_offset, tensor.kv_compressed_size, kv_end, report.file_size
                ));
            } else {
                if let Err(e) = reader.seek(SeekFrom::Start(tensor.kv_compressed_offset)) {
                    report.errors.push(format!(
                        "Tensor '{}': failed to seek to KV blob offset {}: {e}",
                        tensor.name, tensor.kv_compressed_offset
                    ));
                } else {
                    let mut buf = vec![0u8; tensor.kv_compressed_size as usize];
                    if reader.read_exact(&mut buf).is_err() {
                        report.errors.push(format!(
                            "Tensor '{}': failed to read KV blob at offset {}",
                            tensor.name, tensor.kv_compressed_offset
                        ));
                    } else {
                        println!(
                            "      [OK]  {}: KV blob readable ({} bytes, bits_k={}, bits_v={})",
                            tensor.name, tensor.kv_compressed_size, tensor.kv_bits_k, tensor.kv_bits_v
                        );
                    }
                }
            }
        }

        // Check element count consistency
        let total_elements: usize = tensor.shape.iter().product();
        if let Some(ext) = grim_file.metadata.get_tensor_ext(&tensor.name) {
            if ext.row_count > 0 && ext.row_stride > 0 {
                let ext_elements = (ext.row_count * ext.row_stride) as usize;
                if ext_elements != total_elements {
                    report.warnings.push(format!(
                        "Tensor '{}': shape elements ({}) != ext row_count*row_stride ({})",
                        tensor.name, total_elements, ext_elements
                    ));
                }
            }
        }
    }
}

fn check_adapters(grim_file: &GrimFile, report: &mut VerifyReport) {
    println!("\n--- QLoRA Adapter Check (backup2) ---");

    let mut found = false;
    for tensor in &grim_file.tensors {
        if let Some(ext) = grim_file.metadata.get_tensor_ext(&tensor.name) {
            if ext.backup2.is_present() {
                found = true;
                let info = AdapterInfo {
                    tensor_name: tensor.name.clone(),
                    backup2_bpw: ext.backup2.bpw,
                    backup2_codes_size: ext.backup2.codes_size,
                    backup2_scale_size: ext.backup2.scale_size,
                };
                report.adapter_tensors.push(info);
                println!(
                    "[ADAPTER] {}: backup2 present — bpw={}, codes={} bytes, scales={} bytes",
                    tensor.name, ext.backup2.bpw, ext.backup2.codes_size, ext.backup2.scale_size
                );
            }
        }
    }

    if !found {
        println!("No QLoRA adapters found in backup2 slots.");
        report.warnings.push("No tensors have backup2 provisioned (no QLoRA adapters)".to_string());
    } else {
        println!("Found {} tensor(s) with backup2 adapter capacity.", report.adapter_tensors.len());
    }
}

fn print_report(report: &VerifyReport) {
    println!("\n=== Verification Summary ===");
    println!("File size:         {} bytes", report.file_size);
    println!("Magic:             {}", if report.magic_ok { "VALID" } else { "INVALID" });
    println!("Metadata JSON:     {}", if report.metadata_parsed { "VALID" } else { "INVALID/MISSING" });
    println!("Tensor count:      {}", report.tensor_count);
    println!("Adapter tensors:   {}", report.adapter_tensors.len());
    println!("Errors:            {}", report.errors.len());
    println!("Warnings:          {}", report.warnings.len());

    if !report.errors.is_empty() {
        println!("\nErrors:");
        for e in &report.errors {
            println!("  - {e}");
        }
    }

    if !report.warnings.is_empty() {
        println!("\nWarnings:");
        for w in &report.warnings {
            println!("  - {w}");
        }
    }

    if report.errors.is_empty() && report.adapter_tensors.is_empty() {
        println!("\nStatus:  PASS (no adapters in backup2)");
    } else if report.errors.is_empty() {
        println!("\nStatus:  PASS ({} adapter(s) verified in backup2)", report.adapter_tensors.len());
    } else {
        println!("\nStatus:  FAIL");
    }
}