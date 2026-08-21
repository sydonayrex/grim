use grim_core::error::Result;

/// Query live scheduler queue and memory tier status from the running server.
pub async fn cmd_scheduler(addr: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| grim_core::Error::Config(format!("failed to build client: {e}")))?;
    let mut val: Option<serde_json::Value> = None;
    for scheme in &["https", "http"] {
        let url = format!("{scheme}://{addr}/status");
        if let Ok(res) = client.get(&url).send().await {
            if res.status().is_success() {
                val = res.json().await.ok();
                break;
            }
        }
    }

    let val = match val {
        Some(v) => v,
        None => {
            println!("No running Grim server found on {addr}.");
            println!("Start the server with 'grim run --serve' or 'grim service start'.");
            return Ok(());
        }
    };

    println!("=== Grim Scheduler & KV Tier Status ===");
    println!("Server Address    : {addr}");
    println!("Engine State      : {}", val["engine_state"].as_str().unwrap_or("unknown"));
    println!("Backend           : {}", val["backend"].as_str().unwrap_or("unknown"));

    if let Some(sched) = val.get("scheduler") {
        println!("\n--- Scheduler Queues ---");
        println!("  Active / Running: {}", sched["active_requests"].as_u64().unwrap_or(0));
        println!("  Waiting Queue   : {}", sched["waiting_requests"].as_u64().unwrap_or(0));
        println!("  Admitted Total  : {}", sched["admitted_requests"].as_u64().unwrap_or(0));
        println!("  Paused Requests : {}", sched["paused_requests"].as_u64().unwrap_or(0));
    }

    if let Some(kv) = val.get("kv_cache") {
        println!("\n--- KV Block Pool Telemetry ---");
        let used_bytes = kv["used_bytes"].as_u64().unwrap_or(0);
        let total_bytes = kv["total_bytes"].as_u64().unwrap_or(0);
        let blocks_used = kv["blocks_used"].as_u64().unwrap_or(0);
        let blocks_total = kv["blocks_total"].as_u64().unwrap_or(0);

        let used_gb = used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let pct = if total_bytes > 0 { (used_bytes as f64 / total_bytes as f64) * 100.0 } else { 0.0 };

        println!("  Memory Usage    : {:.2} GB / {:.2} GB ({:.1}%)", used_gb, total_gb, pct);
        println!("  Block Count     : {} used / {} total", blocks_used, blocks_total);

        if let Some(tiers) = kv.get("tiers") {
            let gpu_bytes = tiers["gpu_bytes"].as_u64().unwrap_or(0);
            let ram_bytes = tiers["host_ram_bytes"].as_u64().unwrap_or(0);
            let nvme_bytes = tiers["nvme_bytes"].as_u64().unwrap_or(0);
            println!("  Tiers (VRAM/RAM): {:.2} GB GPU / {:.2} GB Host / {:.2} GB NVMe",
                gpu_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                nvme_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
    }

    if let Some(spec) = val.get("speculation") {
        println!("\n--- Speculative Decoding ---");
        println!("  Enabled         : {}", spec["enabled"].as_bool().unwrap_or(false));
        println!("  Strategy        : {}", spec["strategy"].as_str().unwrap_or("none"));
    }

    println!();
    Ok(())
}
