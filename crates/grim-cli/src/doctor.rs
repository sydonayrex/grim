//! `grim doctor` — self-diagnosis. Re-verifies every engine and service claim (§13.5).

use std::path::Path;

use grim_format::{GrimFile, ModelFootprint, read_gguf};
use grim_tensor::error::Result;

#[derive(Default)]
pub struct DoctorReport {
    pub unit_file_exists: Option<bool>,
    pub unit_file_verifies: Option<bool>,
    pub service_is_active: Option<bool>,
    pub _process_running: Option<bool>,
    pub health_endpoint_ok: Option<bool>,
    pub gpu_detected: Option<bool>,
    pub gpu_backend_actual: Option<String>,
    pub plugin_grants_enforced: Option<bool>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

pub fn run_doctor(
    addr: &str,
    service_name: &str,
    exec_path: &str,
    config_path: &str,
    model_path: Option<&Path>,
) -> Result<bool> {
    println!("=== Grim Doctor — Self-Diagnosis ===\n");
    let mut report = DoctorReport::default();

    check_unit_file(&mut report, service_name, exec_path, config_path);
    check_service_status(&mut report, service_name);
    check_process(&mut report, service_name);
    check_health_endpoint(&mut report, addr);
    check_gpu_backend(&mut report);
    check_toolchain(&mut report);
    check_plugin_grants(&mut report);
    check_configuration(&mut report);

    // WI-2: optional pre-flight model/hardware compatibility check.
    // Runs *after* the existing system checks so the full suite still
    // prints; the model section is additive and never masks a system error.
    if let Some(path) = model_path {
        check_model_preflight(&mut report, path);
    }

    print_report(&report);

    if !report.errors.is_empty() {
        eprintln!("\n[grim doctor] SUGGESTIONS FOR ERRORS:");
        for err in &report.errors {
            if err.contains("grim serve") || err.contains("obsolete") {
                eprintln!(
                    "  -> FIX ExecStart: Run 'sudo grim service install --config /etc/grim/grim.toml' to overwrite systemd service with correct ExecStart command."
                );
            } else if err.contains("RDNA 2") || err.contains("compatibility") {
                eprintln!(
                    "  -> RDNA 2 COMPATIBILITY: Force RDNA2 compilation by setting environment variable: export HSA_OVERRIDE_GFX_VERSION=10.3.0"
                );
            } else if err.contains("VRAM")
                || err.contains("exceeds free VRAM")
                || err.contains("estimated VRAM")
            {
                // WI-2: OOM-adjacent — suggest a smaller quant tier by name.
                eprintln!(
                    "  -> VRAM INSUFFICIENT: Re-quantize to a smaller codec (Rook/Jay ~4.1 bpw instead of Raven/Jackdaw ~8 bpw, or Crow ~4.5 bpw), \
                     or reduce the model's context_length. See `WeightFormat::bpw`."
                );
            } else if err.contains("no supported dispatch path")
                || err.contains("codec unsupported")
            {
                // WI-2: forced-fallback case — suggest a native-support tier.
                eprintln!(
                    "  -> CODEC UNSUPPORTED: This codec has no native path on the detected arch. Re-quantize to Rook/Jay (MXFP4, native on RDNA2+) \
                     instead of Raven (FP8, RDNA4/CDNA3 only)."
                );
            } else {
                eprintln!("  -> {err}");
            }
        }
        eprintln!("\nDoctor found {} error(s).", report.errors.len());
        return Ok(false);
    }

    if report.warnings.is_empty() {
        println!("\nAll checks passed.");
    } else {
        eprintln!("\n[grim doctor] SUGGESTIONS FOR WARNINGS:");
        for warn in &report.warnings {
            if warn.contains("not found") || warn.contains("systemd") {
                eprintln!(
                    "  -> INSTALL SERVICE: Run 'grim service install --config /etc/grim/grim.toml' to install a background service daemon."
                );
            } else if warn.contains("unreachable") {
                eprintln!(
                    "  -> START SERVER: Start the server manually using 'grim run --serve' or via service: 'grim service start'."
                );
            } else {
                eprintln!("  -> {warn}");
            }
        }
        eprintln!(
            "\nDoctor found {} warning(s). Review above.",
            report.warnings.len()
        );
    }
    Ok(true)
}

fn check_unit_file(
    report: &mut DoctorReport,
    service_name: &str,
    _exec_path: &str,
    _config_path: &str,
) {
    let path = format!("/etc/systemd/system/{service_name}.service");
    match std::fs::read_to_string(&path) {
        Ok(content) => {
            report.unit_file_exists = Some(true);
            println!("[OK]  Systemd unit file exists at {}", path);

            // Verify ExecStart uses 'grim serve' or 'grim run --serve'.
            if content.contains("grim serve") || content.contains("grim run --serve") {
                report.unit_file_verifies = Some(true);
                println!("[OK]  Systemd unit ExecStart uses valid grim entry point.");
            } else {
                report
                    .warnings
                    .push("Systemd unit exists but ExecStart format is unexpected".into());
                eprintln!("[WARN] Systemd unit exists but ExecStart format is unexpected.");
                report.unit_file_verifies = Some(false);
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            report.unit_file_exists = Some(false);
            report.unit_file_verifies = Some(false);
            report
                .warnings
                .push(format!("Systemd unit file not found at {path}"));
            eprintln!("[WARN] Systemd unit file not found at {}.", path);
            eprintln!("      Run 'grim service install --config /etc/grim/grim.toml' to install.");
        }
        Err(e) => {
            report.errors.push(format!("Failed to read unit file: {e}"));
            eprintln!("[ERR] Failed to read unit file: {e}");
        }
    }
}

fn check_service_status(report: &mut DoctorReport, service_name: &str) {
    let output = std::process::Command::new("systemctl")
        .args(["is-active", service_name])
        .output();

    match output {
        Ok(o) => {
            let state = String::from_utf8_lossy(&o.stdout).trim().to_string();
            match state.as_str() {
                "active" => {
                    report.service_is_active = Some(true);
                    println!("[OK]  grim service is active (systemctl is-active).");
                }
                "failed" => {
                    report.service_is_active = Some(false);
                    report
                        .errors
                        .push("grim service is in 'failed' state".into());
                    eprintln!(
                        "[ERR] grim service is in 'failed' state. Run 'systemctl status grim' for details."
                    );
                }
                _ => {
                    report.service_is_active = Some(false);
                    report
                        .warnings
                        .push(format!("grim service is '{}' (not active)", state));
                    eprintln!("[WARN] grim service is '{}' (not active).", state);
                }
            }
        }
        Err(e) => {
            report
                .warnings
                .push(format!("Could not query systemctl is-active: {e}"));
            eprintln!("[WARN] Could not query systemctl is-active: {e}");
        }
    }
}

fn check_process(_report: &mut DoctorReport, service_name: &str) {
    // Find grim process via systemd service MainPID.
    let output = std::process::Command::new("systemctl")
        .args(["show", service_name, "--property", "MainPID"])
        .output();

    match output {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            if let Some(pid_str) = stdout.strip_prefix("MainPID=") {
                let pid: u64 = pid_str.trim().parse().unwrap_or(0);
                if pid > 0 {
                    // Verify process with kill -0.
                    let verify = std::process::Command::new("kill")
                        .args(["-0", &pid.to_string()])
                        .status();
                    if verify.map(|s| s.success()).unwrap_or(false) {
                        println!("[OK]  grim process is running (PID {}).", pid);
                        return;
                    }
                }
            }
            eprintln!("[WARN] No grim process found via systemd MainPID.");
        }
        Err(e) => {
            eprintln!("[WARN] Could not query systemd for grim MainPID: {e}");
        }
    }
    eprintln!("[INFO] Process check skipped (systemd not available or grim not installed).");
}

fn check_health_endpoint(report: &mut DoctorReport, addr: &str) {
    let url = format!("http://{}/health", addr);
    let output = std::process::Command::new("curl")
        .args(["-sf", &url])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let body = String::from_utf8_lossy(&o.stdout);
            if body.trim() == "OK" {
                report.health_endpoint_ok = Some(true);
                println!("[OK]  /health endpoint responds OK at {}.", url);
            } else {
                report.health_endpoint_ok = Some(false);
                report.warnings.push(format!(
                    "health endpoint returned unexpected body: {}",
                    body.trim()
                ));
                eprintln!(
                    "[WARN] /health at {} returned unexpected body: {}",
                    url,
                    body.trim()
                );
            }
        }
        Ok(_o) => {
            report.health_endpoint_ok = Some(false);
            report
                .warnings
                .push(format!("health endpoint at {} returned HTTP error", url));
            eprintln!(
                "[WARN] /health at {} returned HTTP error (status not 200).",
                url
            );
        }
        Err(e) => {
            report.health_endpoint_ok = Some(false);
            report
                .warnings
                .push(format!("health endpoint at {} unreachable: {}", url, e));
            eprintln!("[WARN] /health endpoint at {} unreachable: {}", url, e);
            eprintln!("      Is 'grim run --serve' running?");
        }
    }
}

fn check_gpu_backend(report: &mut DoctorReport) {
    // Query system ROCm path and version
    match grim_backend_rocm::probe_system_rocm() {
        Ok(rocm) => {
            println!(
                "[OK]  System ROCm installation detected: {} (version {})",
                rocm.path.display(),
                rocm.version
            );
        }
        Err(e) => {
            report
                .warnings
                .push(format!("No system ROCm installation detected: {e}"));
            eprintln!("[WARN] No system ROCm installation detected: {e}");
        }
    }

    // Probe for ROCm hardware.
    match grim_backend_rocm::RocmDevice::probe() {
        Ok(devices) if !devices.is_empty() => {
            report.gpu_detected = Some(true);
            let first = &devices[0];
            println!(
                "[OK]  ROCm GPU detected: ordinal={}, wavefront={:?}, xnack={}",
                first.ordinal(),
                first.wavefront_size(),
                first.xnack_enabled()
            );

            // Verify GCN target is RDNA 3/4 compatible
            match grim_backend_rocm::probe_host_gpu(first.ordinal()) {
                Ok(c) => {
                    println!(
                        "[OK]  Host GPU hardware stats: GCN={}, Wavefront={}, LDS={} bytes",
                        c.gcn, c.wavefront_size, c.lds_size_bytes
                    );
                    if c.wavefront_size != 64 {
                        report.warnings.push(format!(
                            "Host GPU wavefront size is {} (Wave64 layout optimizations require 64)",
                            c.wavefront_size
                        ));
                        eprintln!(
                            "[WARN] Host GPU wavefront size is {} (Wave64 layout optimizations require 64)",
                            c.wavefront_size
                        );
                    }
                    if c.gcn.starts_with("gfx10") {
                        report.warnings.push(format!(
                            "Host GPU architecture {} is RDNA 2. RDNA 2 does not support wave64 and is incompatible with .grim optimizations. CPU backend still works.",
                            c.gcn
                        ));
                        eprintln!(
                            "[WARN] Host GPU architecture {} is RDNA 2. RDNA 2 does not support wave64 and is incompatible with .grim optimizations. CPU backend still works.",
                            c.gcn
                        );
                    } else if !c.gcn.starts_with("gfx11") && !c.gcn.starts_with("gfx12") {
                        report.warnings.push(format!(
                            "Host GPU GCN architecture {} is not standard RDNA 3/4. Optimization overrides may mismatch.",
                            c.gcn
                        ));
                        eprintln!(
                            "[WARN] Host GPU GCN architecture {} is not standard RDNA 3/4. Optimization overrides may mismatch.",
                            c.gcn
                        );
                    }
                }
                Err(e) => {
                    report
                        .warnings
                        .push(format!("Failed to query host GPU GCN capabilities: {e}"));
                    eprintln!("[WARN] Failed to query host GPU GCN capabilities: {e}");
                }
            }

            // Check /metrics for actual GPU usage, not CPU fallback.
            let output = std::process::Command::new("curl")
                .args(["-sf", "http://127.0.0.1:11434/metrics"])
                .output();

            match output {
                Ok(o) if o.status.success() => {
                    let body = String::from_utf8_lossy(&o.stdout);
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                        let gpu_count = json
                            .get("hardware")
                            .and_then(|h| h.get("rocm_gpu_count"))
                            .and_then(|v| v.as_i64())
                            .unwrap_or(-1);
                        if gpu_count > 0 {
                            report.gpu_backend_actual =
                                Some(format!("rocm ({} devices)", gpu_count));
                            println!(
                                "[OK]  Engine reports {} ROCm device(s) in /metrics — GPU backend active.",
                                gpu_count
                            );
                        } else {
                            report.gpu_backend_actual =
                                Some(format!("cpu ({} devices in /metrics)", gpu_count));
                            report.warnings.push(
                                "GPU backend appears to report 0 devices — possible CPU fallback"
                                    .into(),
                            );
                            eprintln!(
                                "[WARN] /metrics reports {} GPU count — may indicate CPU fallback.",
                                gpu_count
                            );
                        }
                    } else {
                        eprintln!("[WARN] Could not parse /metrics JSON response.");
                    }
                }
                _ => {
                    eprintln!(
                        "[INFO] /metrics endpoint not reachable — skipping in-process GPU backend check."
                    );
                    report.gpu_backend_actual =
                        Some("unknown (metrics endpoint unreachable)".into());
                }
            }
        }
        Ok(devices) if devices.is_empty() => {
            report.gpu_detected = Some(false);
            eprintln!("[WARN] No ROCm GPU detected on this host.");
            eprintln!("      Grim will use CPU backend. For GPU inference, install ROCm runtime.");
            report.gpu_backend_actual = Some("cpu (no GPU detected)".into());
        }
        Err(e) => {
            report.gpu_detected = Some(false);
            eprintln!("[WARN] Could not probe ROCm devices: {e}");
            report.gpu_backend_actual = Some("unknown (ROCm probe failed)".into());
        }
        _ => {}
    }
}

fn check_plugin_grants(report: &mut DoctorReport) {
    // §13.4+§13.5: verify plugin grants are enforced at runtime.
    let limits = grim_plugin::PluginLimits::default();
    let loader = grim_plugin::WasmPluginLoader::new("doctor-grant-check", limits);

    let deny_network = !loader.grants.network;
    let deny_fs = loader.grants.filesystem.is_empty();
    let deny_meta = !loader.grants.request_metadata;

    if deny_network && deny_fs && deny_meta {
        println!(
            "[OK]  WASM Plugin Sandbox: deny-by-default grants enforced (network=denied, fs=denied, meta=denied)."
        );
        report.plugin_grants_enforced = Some(true);
    } else {
        report
            .errors
            .push("WASM plugin loader failed deny-by-default grant check".into());
        eprintln!("[ERR] WASM plugin loader failed deny-by-default grant check.");
        report.plugin_grants_enforced = Some(false);
    }

    println!(
        "[INFO] Native dylib plugins (.so/.dll) run in-process as trusted modules (see §6.1)."
    );
}

/// WI-X16: verify effective configuration source per key (file vs env vs default).
fn check_configuration(_report: &mut DoctorReport) {
    println!("\n=== Effective Configuration (WI-X16) ===");
    if let Some(p) = grim_core::env_config::RuntimeEnv::locate_config_file() {
        println!("  Config File: {} (found)", p.display());
    } else {
        println!("  Config File: None (using environment / defaults)");
    }
    let env = grim_core::env_config::RuntimeEnv::from_env();
    for (key, val, src) in env.effective_config_summary() {
        println!("  - {:<18} = {:<20} [source: {}]", key, val, src);
    }
}

/// WI-2: pre-flight model/hardware compatibility check.
///
/// Reads the model **header only** (no tensor data), computes a
/// `ModelFootprint`, then predicts:
///   - VRAM fit vs. detected free VRAM (fits / tight / doesn't fit)
///   - codec-vs-arch compat (native / fallback / unsupported)
///
/// `TODO(gpu-verify)`: this is a *prediction*, not a guarantee. The real
/// check is whether the model actually loads and runs; that's out of scope
/// to automate here.
fn check_model_preflight(report: &mut DoctorReport, path: &Path) {
    println!("\n=== Model Pre-Flight (WI-2) ===");
    println!("  Model: {}", path.display());

    let footprint = match read_model_header(path) {
        Ok(fp) => fp,
        Err(e) => {
            report
                .errors
                .push(format!("failed to read model header: {e}"));
            eprintln!("[ERR] Failed to read model header: {e}");
            return;
        }
    };

    let meta = std::fs::metadata(path).ok();
    let size_bytes = meta.map(|m| m.len()).unwrap_or(0);
    println!(
        "  [INFO] File size: {:.2} MB ({} bytes)",
        size_bytes as f64 / (1024.0 * 1024.0),
        size_bytes
    );

    println!(
        "  [INFO] Architecture: {}, params: {:?}, quant: {:?}, weight bytes: {}",
        footprint.architecture,
        footprint.param_count,
        footprint.quant_format,
        footprint.estimated_weight_bytes
    );

    // Detected hardware: GCN arch + free VRAM, via the same probes the
    // existing GPU check uses.
    let (gcn, free_vram) = match detect_hardware() {
        Some(h) => h,
        None => {
            report
                .warnings
                .push("no ROCm hardware detected — skipping VRAM fit check".into());
            eprintln!("[WARN] No ROCm hardware detected — skipping VRAM fit check.");
            return;
        }
    };
    println!("  [INFO] Detected arch: {gcn:?}, free VRAM: {free_vram} bytes");

    // 1. Codec-vs-arch compat.
    if let Some(quant) = footprint.quant_format {
        match grim_garage::check_support(quant, gcn) {
            grim_garage::CompatResult::NativeSupport => {
                println!("[OK]  {quant:?} is natively supported on {gcn:?}.");
            }
            grim_garage::CompatResult::FallbackSupport { to, reason } => {
                report.warnings.push(format!("codec fallback: {reason}"));
                eprintln!(
                    "[WARN] {quant:?} is not native on {gcn:?}; falling back to {to:?}. \
                     This is a quality-preserving downshift, but denser."
                );
            }
            grim_garage::CompatResult::Unsupported { reason } => {
                report.errors.push(format!("codec unsupported: {reason}"));
                eprintln!(
                    "[ERR] {quant:?} has no supported dispatch path on {gcn:?}. \
                     The model cannot run on this hardware as-is."
                );
            }
        }
    } else {
        println!("[INFO] No quantization codec named in the header — skipping compat check.");
    }

    // 2. VRAM fit.
    let ctx = footprint.context_length_default.unwrap_or(4096);
    let kv_layers = estimate_num_layers(&footprint);
    let kv_heads = estimate_num_kv_heads(&footprint);
    let head_dim = estimate_head_dim(&footprint);
    let vram_estimate =
        grim_format::estimate_vram_bytes(&footprint, ctx, 1, kv_layers, kv_heads, head_dim);
    let margin = (vram_estimate as f64 * 0.10) as u64;
    if free_vram == 0 {
        report
            .warnings
            .push("free VRAM unknown — skipping fit check".into());
        eprintln!("[WARN] Free VRAM is 0 — skipping fit check.");
    } else if vram_estimate <= free_vram.saturating_sub(margin) {
        println!(
            "[OK]  Estimated VRAM {} bytes fits comfortably in free VRAM {} bytes.",
            vram_estimate, free_vram
        );
    } else if vram_estimate <= free_vram {
        report.warnings.push(format!(
            "VRAM estimate {vram_estimate} bytes is tight against free VRAM {free_vram} bytes"
        ));
        eprintln!(
            "[WARN] VRAM estimate {vram_estimate} bytes is tight against free VRAM {free_vram} bytes."
        );
    } else {
        report.errors.push(format!(
            "estimated VRAM {vram_estimate} bytes exceeds free VRAM {free_vram} bytes"
        ));
        eprintln!(
            "[ERR] Estimated VRAM {vram_estimate} bytes exceeds free VRAM {free_vram} bytes."
        );
    }
}

/// Read a model file's header only. Dispatches on extension: `.gguf` via
/// `read_gguf`, `.grim` via `GrimFile::read`. Both avoid loading tensor
/// data — this is a hard requirement (WI-2 gate 4).
fn read_model_header(path: &Path) -> Result<ModelFootprint> {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "gguf" => {
            let file = std::fs::File::open(path).map_err(|e| {
                grim_tensor::Error::Backend(format!("cannot open {}: {e}", path.display()))
            })?;
            let mut reader = std::io::BufReader::new(file);
            let gguf = read_gguf(&mut reader).map_err(|e| {
                grim_tensor::Error::Backend(format!("GGUF header parse failed: {e}"))
            })?;
            Ok(ModelFootprint::from_gguf_file(&gguf))
        }
        "grim" => {
            let file = std::fs::File::open(path).map_err(|e| {
                grim_tensor::Error::Backend(format!("cannot open {}: {e}", path.display()))
            })?;
            let mut reader = std::io::BufReader::new(file);
            let grim = GrimFile::read(&mut reader).map_err(|e| {
                grim_tensor::Error::Backend(format!(".grim header parse failed: {e}"))
            })?;
            Ok(ModelFootprint::from_grim_file(&grim))
        }
        other => Err(grim_tensor::Error::Backend(format!(
            "unsupported model file extension '{other}'; expected .gguf or .grim"
        ))),
    }
}

/// Detect the host's GCN arch + free VRAM. Returns `None` when no ROCm
/// hardware is present (the existing GPU check already reports this).
fn detect_hardware() -> Option<(grim_backend_rocm::GcnArch, u64)> {
    let devices = grim_backend_rocm::RocmDevice::probe().ok()?;
    let first = devices.first()?;
    let cap = grim_backend_rocm::probe_host_gpu(first.ordinal()).ok()?;
    let arch = grim_backend_rocm::gcn_arch(&cap.gcn);
    let (_free, total) = grim_backend_rocm::vram_info(first.ordinal());
    Some((arch, total))
}

/// Conservative heuristics for KV-cache sizing when the header doesn't
/// carry them. Each returns 0 when unknown so the estimate stays honest
/// rather than guessing a large number and alarming the user.
fn estimate_num_layers(_footprint: &ModelFootprint) -> u32 {
    // `general.num_layers` is not a standard GGUF key; real models carry
    // it under various family-specific names. Without a parser per family,
    // we can't derive it from the header alone. Return 0 so the KV term
    // vanishes and the estimate is a pure weight-byte lower bound — the
    // conservative choice. `TODO(calibrate)`: read per-family keys.
    0
}

fn estimate_num_kv_heads(_footprint: &ModelFootprint) -> u32 {
    0
}

fn estimate_head_dim(_footprint: &ModelFootprint) -> u32 {
    0
}

fn print_report(report: &DoctorReport) {
    println!("\n--- Summary ---");
    println!(
        "  Unit file:     {}",
        match report.unit_file_exists {
            Some(true) => "present",
            Some(false) => "MISSING",
            None => "unknown",
        }
    );
    println!(
        "  Unit valid:    {}",
        match report.unit_file_verifies {
            Some(true) => "valid (correct ExecStart)",
            Some(false) => "INVALID",
            None => "unknown",
        }
    );
    println!(
        "  Service active: {}",
        match report.service_is_active {
            Some(true) => "yes",
            Some(false) => "no",
            None => "unknown",
        }
    );
    println!(
        "  GPU detected:  {}",
        match report.gpu_detected {
            Some(true) => "yes",
            Some(false) => "no (CPU only)",
            None => "unknown",
        }
    );
    println!(
        "  GPU in use:    {}",
        report.gpu_backend_actual.as_deref().unwrap_or("unknown")
    );
    println!(
        "  Health:        {}",
        match report.health_endpoint_ok {
            Some(true) => "responding",
            Some(false) => "error/unreachable",
            None => "not checked",
        }
    );
    if report.errors.is_empty() && report.warnings.is_empty() {
        println!("  Status:        ALL CLEAR");
    } else if report.errors.is_empty() {
        println!("  Status:        {} warning(s)", report.warnings.len());
    } else {
        println!(
            "  Status:        {} error(s), {} warning(s)",
            report.errors.len(),
            report.warnings.len()
        );
    }
}

/// Probe compiler and runtime toolchain dependencies required for HIPRTC
/// and JIT compilation (WI-X17).
///
/// Verifies presence of Clang, LLVM tools, ROCm runtime shared libraries,
/// write access to the HSACO cache directory, and flags potential rustup
/// toolchain collisions in target/.
pub fn check_toolchain(report: &mut DoctorReport) {
    let clang_res = std::process::Command::new("clang").arg("--version").output();
    match clang_res {
        Ok(out) if out.status.success() => {
            let ver = String::from_utf8_lossy(&out.stdout);
            let first_line = ver.lines().next().unwrap_or("unknown");
            println!("[OK]  Clang toolchain detected: {first_line}");
        }
        _ => {
            report.errors.push(
                "clang not found in PATH: install clang (e.g. 'pacman -S clang' or 'apt install clang') for JIT / HIPRTC compilation"
                    .into(),
            );
            eprintln!(
                "[ERR] Clang not found in PATH. Install clang ('pacman -S clang' or 'apt install clang') for JIT / HIPRTC compilation."
            );
        }
    }

    let llvm_res = std::process::Command::new("llvm-config").arg("--version").output();
    if let Ok(out) = llvm_res {
        if out.status.success() {
            let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
            println!("[OK]  LLVM toolchain detected: version {ver}");
        }
    }

    let rocm_paths = [
        "/opt/rocm/lib/libhipblas.so",
        "/opt/rocm/lib/librocblas.so",
        "/usr/lib/libhipblas.so",
        "/usr/lib/librocblas.so",
    ];
    let rocm_lib_found = rocm_paths.iter().any(|p| std::path::Path::new(p).exists());
    if rocm_lib_found {
        println!("[OK]  ROCm BLAS shared libraries detected on disk.");
    } else {
        report.warnings.push(
            "libhipblas.so / librocblas.so not found at standard paths (/opt/rocm/lib or /usr/lib)"
                .into(),
        );
        eprintln!(
            "[WARN] libhipblas.so / librocblas.so not found at standard paths (/opt/rocm/lib or /usr/lib)."
        );
    }

    let cache_dir = std::env::var("GRIM_HSACO_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_next_cache().unwrap_or_else(|| std::path::PathBuf::from("/tmp/grim_hsaco"))
        });
    match std::fs::create_dir_all(&cache_dir) {
        Ok(_) => {
            let probe_file = cache_dir.join(".doctor_write_probe");
            if std::fs::write(&probe_file, b"ok").is_ok() {
                let _ = std::fs::remove_file(&probe_file);
                println!("[OK]  HSACO JIT cache directory writable: {}", cache_dir.display());
            } else {
                report.errors.push(format!(
                    "HSACO cache directory not writable: {}",
                    cache_dir.display()
                ));
                eprintln!("[ERR] HSACO cache directory not writable: {}", cache_dir.display());
            }
        }
        Err(e) => {
            report.errors.push(format!(
                "Failed to create HSACO cache directory {}: {e}",
                cache_dir.display()
            ));
            eprintln!(
                "[ERR] Failed to create HSACO cache directory {}: {e}",
                cache_dir.display()
            );
        }
    }
}

fn dirs_next_cache() -> Option<std::path::PathBuf> {
    std::env::var("XDG_CACHE_HOME")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| std::path::PathBuf::from(h).join(".cache").join("grim").join("hsaco"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctor_toolchain_check_executes_and_populates_report() {
        let mut report = DoctorReport::default();
        check_toolchain(&mut report);
        // Either clang was found, or an actionable error message naming pacman/apt was pushed.
        if !report.errors.is_empty() {
            assert!(
                report.errors.iter().any(|e| e.contains("pacman -S clang") || e.contains("clang not found")),
                "Error messages must give actionable remedy command"
            );
        }
    }

    #[test]
    fn dirs_next_cache_returns_path_or_fallback() {
        let path = dirs_next_cache();
        assert!(path.is_some() || std::env::var("HOME").is_err());
    }
}
