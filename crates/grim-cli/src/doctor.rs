//! `grim doctor` — self-diagnosis subcommand.
//!
//! §13.5: re-verifies every claim the engine and its services make about
//! themselves. This is the diagnostic of last resort for exactly the
//! failure mode §13.1–§13.4 prevent in the first place.

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

pub fn run_doctor(addr: &str, service_name: &str, exec_path: &str, config_path: &str) -> Result<bool> {
    println!("=== Grim Doctor — Self-Diagnosis ===\n");
    let mut report = DoctorReport::default();

    check_unit_file(&mut report, service_name, exec_path, config_path);
    check_service_status(&mut report, service_name);
    check_process(&mut report, service_name);
    check_health_endpoint(&mut report, addr);
    check_gpu_backend(&mut report);
    check_plugin_grants(&mut report);

    print_report(&report);

    if !report.errors.is_empty() {
        eprintln!("\nDoctor found {} error(s). Run 'grim service install' and ensure ROCm is available.", report.errors.len());
        return Ok(false);
    }

    if report.warnings.is_empty() {
        println!("\nAll checks passed.");
    } else {
        eprintln!("\nDoctor found {} warning(s). Review above.", report.warnings.len());
    }
    Ok(true)
}

fn check_unit_file(report: &mut DoctorReport, service_name: &str, _exec_path: &str, _config_path: &str) {
    let path = format!("/etc/systemd/system/{service_name}.service");
    match std::fs::read_to_string(&path) {
        Ok(content) => {
            report.unit_file_exists = Some(true);
            println!("[OK]  Systemd unit file exists at {}", path);

            // Verify it contains the correct ExecStart (not the old non-existent 'grim serve').
            if content.contains("grim serve") {
                report.errors.push(format!(
                    "Systemd unit at {} contains obsolete 'grim serve' in ExecStart — \
                     should be 'grim run --serve --config'",
                    path
                ));
                eprintln!(
                    "[ERR] Systemd unit at {} contains obsolete 'grim serve' in ExecStart.",
                    path
                );
                report.unit_file_verifies = Some(false);
            } else if content.contains("grim run --serve") {
                report.unit_file_verifies = Some(true);
                println!("[OK]  Systemd unit ExecStart uses 'grim run --serve'.");
            } else {
                report.warnings.push("Systemd unit exists but ExecStart format is unexpected".into());
                eprintln!("[WARN] Systemd unit exists but ExecStart format is unexpected.");
                report.unit_file_verifies = Some(false);
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            report.unit_file_exists = Some(false);
            report.unit_file_verifies = Some(false);
            report.warnings.push("Systemd unit file not found at /etc/systemd/system/grim.service".into());
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
                    report.errors.push("grim service is in 'failed' state".into());
                    eprintln!("[ERR] grim service is in 'failed' state. Run 'systemctl status grim' for details.");
                }
                _ => {
                    report.service_is_active = Some(false);
                    report.warnings.push(format!("grim service is '{}' (not active)", state));
                    eprintln!("[WARN] grim service is '{}' (not active).", state);
                }
            }
        }
        Err(e) => {
            report.warnings.push(format!("Could not query systemctl is-active: {e}"));
            eprintln!("[WARN] Could not query systemctl is-active: {e}");
        }
    }
}

fn check_process(_report: &mut DoctorReport, service_name: &str) {
    // Find grim process by checking pids from the systemd service.
    let output = std::process::Command::new("systemctl")
        .args(["show", service_name, "--property", "MainPID"])
        .output();

    match output {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            if let Some(pid_str) = stdout.strip_prefix("MainPID=") {
                let pid: u64 = pid_str.trim().parse().unwrap_or(0);
                if pid > 0 {
                    // Process is running — verify with kill -0.
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
                report.warnings.push(format!("health endpoint returned unexpected body: {}", body.trim()));
                eprintln!("[WARN] /health at {} returned unexpected body: {}", url, body.trim());
            }
        }
        Ok(_o) => {
            report.health_endpoint_ok = Some(false);
            report.warnings.push(format!("health endpoint at {} returned HTTP error", url));
            eprintln!("[WARN] /health at {} returned HTTP error (status not 200).", url);
        }
        Err(e) => {
            report.health_endpoint_ok = Some(false);
            report.warnings.push(format!("health endpoint at {} unreachable: {}", url, e));
            eprintln!("[WARN] /health endpoint at {} unreachable: {}", url, e);
            eprintln!("      Is 'grim run --serve' running?");
        }
    }
}

fn check_gpu_backend(report: &mut DoctorReport) {
    // Query system ROCm path and version info
    match grim_backend_rocm::device::probe::probe_system_rocm() {
        Ok(rocm) => {
            println!(
                "[OK]  System ROCm installation detected: {} (version {})",
                rocm.path.display(),
                rocm.version
            );
        }
        Err(e) => {
            report.warnings.push(format!("No system ROCm installation detected: {e}"));
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

            // Verify if GCN target is compatible with RDNA 3 or RDNA 4
            match grim_backend_rocm::device::probe::probe_host_gpu(first.ordinal()) {
                Ok(c) => {
                    println!(
                        "[OK]  Host GPU hardware stats: GCN={}, Wavefront={}, LDS={} bytes",
                        c.gcn,
                        c.wavefront_size,
                        c.lds_size_bytes
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
                        report.errors.push(format!(
                            "Host GPU architecture {} is RDNA 2. RDNA 2 does not support wave64 and is incompatible with .grim optimizations",
                            c.gcn
                        ));
                        eprintln!(
                            "[ERR] Host GPU architecture {} is RDNA 2. RDNA 2 does not support wave64 and is incompatible with .grim optimizations",
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
                    report.warnings.push(format!("Failed to query host GPU GCN capabilities: {e}"));
                    eprintln!("[WARN] Failed to query host GPU GCN capabilities: {e}");
                }
            }

            // Check if the running engine is actually using it, not falling back to CPU.
            // We check the /metrics endpoint for rocm_gpu_count.
            let output = std::process::Command::new("curl")
                .args(["-sf", "http://127.0.0.1:8080/metrics"])
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
                            report.gpu_backend_actual = Some(format!("rocm ({} devices)", gpu_count));
                            println!(
                                "[OK]  Engine reports {} ROCm device(s) in /metrics — GPU backend active.",
                                gpu_count
                            );
                        } else {
                            report.gpu_backend_actual = Some("cpu (hardcoded 0 in /metrics)".into());
                            report.warnings.push(
                                "GPU backend appears to report 0 devices — possible CPU fallback".into(),
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
                    report.gpu_backend_actual = Some("unknown (metrics endpoint unreachable)".into());
                }
            }
        }
        Ok(devices) if devices.is_empty() => {
            report.gpu_detected = Some(false);
            eprintln!(
                "[WARN] No ROCm GPU detected on this host."
            );
            eprintln!(
                "      Grim will use CPU backend. For GPU inference, install ROCm runtime."
            );
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
    // §13.4 + §13.5: verify that plugin grants are actually enforced at runtime.
    // We do this by probing a synthetic denied capability — if it's refused, enforcement exists.
    //
    // For a basic check, we look at whether the plugin WASM loader is configured to gate
    // the 'network' import when network=false is declared. A true enforcement test would
    // require a test plugin with network access and a deny-granted policy.
    //
    // The check we can do without a test plugin: verify the plugin manifest schema includes
    // grants and that loading fails closed when a manifest is absent.
    // This is a shallow check — runtime enforcement requires an integration test.

    println!("[INFO] Plugin grant enforcement check: requires integration test (Phase 5).");
    println!("      To verify enforcement manually:");
    println!("        1. Load a plugin with 'network = false' in its manifest");
    println!("        2. Attempt an outbound HTTP request from within the plugin");
    println!("        3. If the request is blocked — grants ARE enforced");
    println!("        4. If the request succeeds — grants are NOT enforced (see §13.4)");
    report.plugin_grants_enforced = None;
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
        println!("  Status:        {} error(s), {} warning(s)", report.errors.len(), report.warnings.len());
    }
}