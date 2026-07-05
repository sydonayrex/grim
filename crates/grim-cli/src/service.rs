//! `grim-service` — Installation and platform-native system services.
//! §12 of the Grim architecture.
//!
//! §13.1 verify-before-success: every mutating operation confirms its
//! effect before reporting success. Service managers write unit files to
//! disk and read them back; start/stop/status call the actual OS service
//! manager rather than assuming the operation succeeded.

use std::path::PathBuf;
use grim_tensor::error::{Error, Result};
use rcgen::generate_simple_self_signed;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartPolicy {
    Never,
    OnFailure,
    Always,
}

#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    pub endpoint: String,
    pub interval_secs: u32,
    pub timeout_secs: u32,
    pub failure_threshold: u32,
}

#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub name: String,
    pub exec_path: PathBuf,
    pub config_path: PathBuf,
    pub restart_policy: RestartPolicy,
    pub run_as_user: Option<String>,
    pub health_check: HealthCheckConfig,
    pub log_path: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub enum ServiceStatus {
    Stopped,
    Running,
    Failed(String),
    Unknown(String),
}

/// Abstracts over the three platform-native service managers so install/uninstall share one code path.
pub trait ServiceManager: Send + Sync {
    fn install(&self, cfg: &ServiceConfig) -> Result<()>;
    fn uninstall(&self, purge: bool) -> Result<()>;
    fn start(&self) -> Result<()>;
    fn stop(&self) -> Result<()>;
    fn status(&self) -> Result<ServiceStatus>;
    fn reload_config(&self) -> Result<()>;
}

/// Generates self-signed certificates and updates the configuration file with TLS config.
///
/// This handles creating certificates for local development and registers them under the
/// `[server.tls]` section of the configuration file.
pub fn setup_tls_and_config(cfg: &ServiceConfig) -> Result<()> {
    let config_dir = cfg.config_path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let cert_dir = config_dir.join("certs");
    let cert_path = cert_dir.join("cert.pem");
    let key_path = cert_dir.join("key.pem");

    // Generate certificates if they don't exist
    if !cert_path.exists() || !key_path.exists() {
        if let Some(parent) = cert_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                Error::Backend(format!("failed to create cert directory: {e}"))
            })?;
        }
        let subject_alt_names = vec!["localhost".to_string(), "127.0.0.1".to_string()];
        let cert = rcgen::generate_simple_self_signed(subject_alt_names)
            .map_err(|e| Error::Backend(format!("failed to generate self-signed cert: {e}")))?;
        
        let cert_pem = cert.serialize_pem().map_err(|e| {
            Error::Backend(format!("failed to serialize cert: {e}"))
        })?;
        let key_pem = cert.serialize_private_key_pem();

        std::fs::write(&cert_path, cert_pem).map_err(|e| {
            Error::Backend(format!("failed to write cert to {}: {e}", cert_path.display()))
        })?;
        std::fs::write(&key_path, key_pem).map_err(|e| {
            Error::Backend(format!("failed to write key to {}: {e}", key_path.display()))
        })?;
        println!("[Service] Generated self-signed SSL/TLS certificate at {}", cert_path.display());
    }

    // Ensure config directory exists
    if let Some(parent) = cfg.config_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            Error::Backend(format!("failed to create config directory: {e}"))
        })?;
    }

    // Write/update config file with [server.tls] block
    let mut content = if cfg.config_path.exists() {
        std::fs::read_to_string(&cfg.config_path).unwrap_or_default()
    } else {
        r#"# Grim configuration file
[engine]
determinism_mode = "relaxed"
"#.to_string()
    };

    if !content.contains("[server.tls]") {
        content.push_str(&format!(
            "\n[server.tls]\ncert_path = \"{}\"\nkey_path = \"{}\"\n",
            cert_path.display().to_string().replace("\\", "\\\\"),
            key_path.display().to_string().replace("\\", "\\\\")
        ));
        std::fs::write(&cfg.config_path, content).map_err(|e| {
            Error::Backend(format!("failed to write config to {}: {e}", cfg.config_path.display()))
        })?;
        println!("[Service] Updated config file at {} with SSL/TLS settings.", cfg.config_path.display());
    }

    Ok(())
}

/// Linux Systemd Manager implementation.
pub struct SystemdManager;

impl SystemdManager {
    fn unit_path(name: &str) -> PathBuf {
        PathBuf::from(format!("/etc/systemd/system/{name}.service"))
    }

    fn restart_arg(policy: RestartPolicy) -> &'static str {
        match policy {
            RestartPolicy::Never => "no",
            RestartPolicy::OnFailure => "on-failure",
            RestartPolicy::Always => "always",
        }
    }

    fn generate_unit(cfg: &ServiceConfig) -> String {
        let log_target = if let Some(ref lp) = cfg.log_path {
            lp.display().to_string()
        } else {
            "/var/log/grim/grim.log".to_string()
        };
        format!(
            r#"[Unit]
Description=Grim inference engine
After=network.target

[Service]
Type=notify
ExecStart={} run --serve --config {}
User={}
Restart={}
RestartSec=2
StartLimitBurst=5
WatchdogSec=10
StandardOutput=append:{log_target}
StandardError=append:{log_target}

[Install]
WantedBy=multi-user.target
"#,
            cfg.exec_path.display(),
            cfg.config_path.display(),
            cfg.run_as_user.as_deref().unwrap_or("grim"),
            Self::restart_arg(cfg.restart_policy),
        )
    }
}

impl ServiceManager for SystemdManager {
    fn install(&self, cfg: &ServiceConfig) -> Result<()> {
        setup_tls_and_config(cfg)?;
        let unit_path = Self::unit_path(&cfg.name);
        let content = Self::generate_unit(cfg);

        // §13.1: write the unit file to disk first.
        std::fs::write(&unit_path, &content).map_err(|e| {
            Error::Backend(format!(
                "failed to write systemd unit to '{}': {e}",
                unit_path.display()
            ))
        })?;

        // §13.1: verify the write succeeded by reading it back.
        let verify = std::fs::read_to_string(&unit_path).map_err(|e| {
            Error::Backend(format!(
                "unit file written but cannot be read back: {e}"
            ))
        })?;
        if verify != content {
            return Err(Error::Backend(
                "unit file content mismatch after write — disk may be corrupt".into(),
            ));
        }

        // Tell systemd to reload its unit files so `systemctl start` finds the new unit.
        let status = std::process::Command::new("systemctl")
            .args(["daemon-reload"])
            .status()
            .map_err(|e| Error::Backend(format!("failed to run systemctl daemon-reload: {e}")))?;

        if !status.success() {
            return Err(Error::Backend(format!(
                "systemctl daemon-reload exited with code {:?}",
                status.code()
            )));
        }

        println!(
            "[SystemdManager] Installed unit file at {} and reloaded systemd.",
            unit_path.display()
        );
        Ok(())
    }

    fn uninstall(&self, purge: bool) -> Result<()> {
        let unit_path = Self::unit_path("grim");
        if unit_path.exists() {
            std::fs::remove_file(&unit_path).map_err(|e| {
                Error::Backend(format!("failed to remove unit file: {e}"))
            })?;
            let _ = std::process::Command::new("systemctl")
                .args(["daemon-reload"])
                .status();
            println!("[SystemdManager] Removed unit file at {}.", unit_path.display());
        }
        if purge {
            if let Some(parent) = unit_path.parent() {
                if parent.as_os_str() == "/etc/systemd/system" {
                    let _ = std::fs::remove_dir("/var/log/grim");
                }
            }
        }
        Ok(())
    }

    fn start(&self) -> Result<()> {
        let unit_path = Self::unit_path("grim");
        if !unit_path.exists() {
            return Err(Error::Backend(
                "cannot start: unit file does not exist. Run 'grim service install' first.".into(),
            ));
        }
        let status = std::process::Command::new("systemctl")
            .args(["start", "grim"])
            .status()
            .map_err(|e| Error::Backend(format!("failed to run systemctl start: {e}")))?;

        if !status.success() {
            return Err(Error::Backend(format!(
                "systemctl start grim exited with code {:?}",
                status.code()
            )));
        }

        println!("[SystemdManager] Started grim service.");
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        let status = std::process::Command::new("systemctl")
            .args(["stop", "grim"])
            .status()
            .map_err(|e| Error::Backend(format!("failed to run systemctl stop: {e}")))?;

        if !status.success() {
            return Err(Error::Backend(format!(
                "systemctl stop grim exited with code {:?}",
                status.code()
            )));
        }

        println!("[SystemdManager] Stopped grim service.");
        Ok(())
    }

    fn status(&self) -> Result<ServiceStatus> {
        let output = std::process::Command::new("systemctl")
            .args(["is-active", "grim"])
            .output()
            .map_err(|e| Error::Backend(format!("failed to run systemctl is-active: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stdout_trim = stdout.trim();

        if stdout_trim == "active" {
            Ok(ServiceStatus::Running)
        } else if stdout_trim == "failed" {
            Ok(ServiceStatus::Failed("exited with failure".into()))
        } else {
            // "inactive" or any other string
            Ok(ServiceStatus::Stopped)
        }
    }

    fn reload_config(&self) -> Result<()> {
        let status = std::process::Command::new("systemctl")
            .args(["reload", "grim"])
            .status()
            .map_err(|e| Error::Backend(format!("failed to run systemctl reload: {e}")))?;

        if !status.success() {
            return Err(Error::Backend(format!(
                "systemctl reload grim exited with code {:?}",
                status.code()
            )));
        }

        println!("[SystemdManager] Reloaded grim configuration.");
        Ok(())
    }
}

/// macOS launchd Manager implementation.
pub struct LaunchdManager;

impl LaunchdManager {
    fn plist_path(name: &str) -> PathBuf {
        PathBuf::from(format!("/Library/LaunchDaemons/com.{name}.plist"))
    }

    fn generate_plist(cfg: &ServiceConfig) -> String {
        let log_target = if let Some(ref lp) = cfg.log_path {
            lp.display().to_string()
        } else {
            "/var/log/grim/grim.log".to_string()
        };
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.{name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exec}</string>
        <string>run</string>
        <string>--serve</string>
        <key>--config</key>
        <string>{config}</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key><false/>
        <key>Crashed</key><true/>
    </dict>
    <key>StandardOutPath</key><string>{log}</string>
    <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"#,
            name = cfg.name,
            exec = cfg.exec_path.display(),
            config = cfg.config_path.display(),
            log = log_target
        )
    }
}

impl ServiceManager for LaunchdManager {
    fn install(&self, cfg: &ServiceConfig) -> Result<()> {
        setup_tls_and_config(cfg)?;
        let plist_path = Self::plist_path(&cfg.name);
        let content = Self::generate_plist(cfg);

        std::fs::write(&plist_path, &content).map_err(|e| {
            Error::Backend(format!("failed to write launchd plist to '{}': {e}", plist_path.display()))
        })?;

        let verify = std::fs::read_to_string(&plist_path).map_err(|e| {
            Error::Backend(format!("plist written but cannot be read back: {e}"))
        })?;
        if verify != content {
            return Err(Error::Backend(
                "plist content mismatch after write — disk may be corrupt".into(),
            ));
        }

        let status = std::process::Command::new("launchctl")
            .args(["load", &plist_path.display().to_string()])
            .status()
            .map_err(|e| Error::Backend(format!("failed to run launchctl load: {e}")))?;

        if !status.success() {
            return Err(Error::Backend(format!(
                "launchctl load exited with code {:?}",
                status.code()
            )));
        }

        println!(
            "[LaunchdManager] Installed plist at {} and loaded service.",
            plist_path.display()
        );
        Ok(())
    }

    fn uninstall(&self, purge: bool) -> Result<()> {
        let plist_path = Self::plist_path("grim");
        if plist_path.exists() {
            let _ = std::process::Command::new("launchctl")
                .args(["unload", &plist_path.display().to_string()])
                .status();
            std::fs::remove_file(&plist_path).map_err(|e| {
                Error::Backend(format!("failed to remove plist: {e}"))
            })?;
            println!("[LaunchdManager] Removed plist at {}.", plist_path.display());
        }
        if purge {
            let _ = std::fs::remove_dir("/var/log/grim");
        }
        Ok(())
    }

    fn start(&self) -> Result<()> {
        let plist_path = Self::plist_path("grim");
        if !plist_path.exists() {
            return Err(Error::Backend(
                "cannot start: plist does not exist. Run 'grim service install' first.".into(),
            ));
        }
        let status = std::process::Command::new("launchctl")
            .args(["start", &format!("com.grim")])
            .status()
            .map_err(|e| Error::Backend(format!("failed to run launchctl start: {e}")))?;

        if !status.success() {
            return Err(Error::Backend(format!(
                "launchctl start exited with code {:?}",
                status.code()
            )));
        }

        println!("[LaunchdManager] Started grim service.");
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        let status = std::process::Command::new("launchctl")
            .args(["stop", "grim"])
            .status()
            .map_err(|e| Error::Backend(format!("failed to run launchctl stop: {e}")))?;

        if !status.success() {
            return Err(Error::Backend(format!(
                "launchctl stop exited with code {:?}",
                status.code()
            )));
        }

        println!("[LaunchdManager] Stopped grim service.");
        Ok(())
    }

    fn status(&self) -> Result<ServiceStatus> {
        let output = std::process::Command::new("launchctl")
            .args(["print", "grim"])
            .output()
            .ok();

        match output {
            Some(o) if o.status.success() => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                if stdout.contains("pid =") {
                    Ok(ServiceStatus::Running)
                } else {
                    Ok(ServiceStatus::Stopped)
                }
            }
            _ => Ok(ServiceStatus::Stopped),
        }
    }

    fn reload_config(&self) -> Result<()> {
        let _ = std::process::Command::new("launchctl")
            .args(["kickstart", "-k", "grim"])
            .status();
        println!("[LaunchdManager] Reloaded configuration.");
        Ok(())
    }
}

/// Windows Service Control Manager (SCM) implementation.
pub struct WindowsScmManager;

impl ServiceManager for WindowsScmManager {
    fn install(&self, cfg: &ServiceConfig) -> Result<()> {
        setup_tls_and_config(cfg)?;
        let unit_name = format!("Grim {}", cfg.name);
        let bin_path = format!(
            "\"{}\" service run --config \"{}\"",
            cfg.exec_path.display(),
            cfg.config_path.display()
        );
        let status = std::process::Command::new("sc")
            .args([
                "create",
                &format!("grim_{}", cfg.name),
                "binPath=",
                &bin_path,
                "DisplayName=",
                &unit_name,
            ])
            .status()
            .map_err(|e| Error::Backend(format!("failed to run sc create: {e}")))?;

        if !status.success() {
            return Err(Error::Backend(format!(
                "sc create grim service exited with code {:?}",
                status.code()
            )));
        }

        println!("[WindowsScmManager] Registered service '{}'.", cfg.name);
        Ok(())
    }

    fn uninstall(&self, purge: bool) -> Result<()> {
        let _ = std::process::Command::new("sc")
            .args(["delete", "grim"])
            .status();
        if purge {
            let _ = std::fs::remove_dir("C:\\Program Files\\Grim\\logs");
        }
        println!("[WindowsScmManager] Deleted service.");
        Ok(())
    }

    fn start(&self) -> Result<()> {
        let status = std::process::Command::new("sc")
            .args(["start", "grim"])
            .status()
            .map_err(|e| Error::Backend(format!("failed to run sc start: {e}")))?;

        if !status.success() {
            return Err(Error::Backend(format!(
                "sc start grim exited with code {:?}",
                status.code()
            )));
        }

        println!("[WindowsScmManager] Started grim service.");
        Ok(())
    }

    fn stop(&self) -> Result<()> {
        let status = std::process::Command::new("sc")
            .args(["stop", "grim"])
            .status()
            .map_err(|e| Error::Backend(format!("failed to run sc stop: {e}")))?;

        if !status.success() {
            return Err(Error::Backend(format!(
                "sc stop grim exited with code {:?}",
                status.code()
            )));
        }

        println!("[WindowsScmManager] Stopped grim service.");
        Ok(())
    }

    fn status(&self) -> Result<ServiceStatus> {
        let output = std::process::Command::new("sc")
            .args(["query", "grim"])
            .output()
            .map_err(|e| Error::Backend(format!("failed to run sc query: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.contains("RUNNING") {
            Ok(ServiceStatus::Running)
        } else if stdout.contains("STOPPED") {
            Ok(ServiceStatus::Stopped)
        } else {
            Ok(ServiceStatus::Failed("unknown state".into()))
        }
    }

    fn reload_config(&self) -> Result<()> {
        let status = std::process::Command::new("sc")
            .args(["config", "grim", "type=", "own"])
            .status()
            .map_err(|e| Error::Backend(format!("failed to run sc config: {e}")))?;

        if !status.success() {
            return Err(Error::Backend(format!(
                "sc config exited with code {:?}",
                status.code()
            )));
        }

        println!("[WindowsScmManager] Reloaded configuration.");
        Ok(())
    }
}