#!/bin/bash
# Grim Zero-Configuration Install Script
# Creates /usr/local/bin/grim binary and installs a systemd service
# that starts the inference server on boot. The bind address defaults to
# loopback (127.0.0.1:11434) for an SSRF-safe-by-default posture; set
# GRIM_HOST/GRIM_PORT in /etc/grim/environment to expose it publicly.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRIM_BINARY="${SCRIPT_DIR}/../target/release/grim"

GRIM_INSTALL_DIR="/usr/local/bin"
GRIM_CONFIG_DIR="/etc/grim"
GRIM_LOG_DIR="/var/log/grim"
GRIM_MODELS_DIR="/var/lib/grim/models"
GRIM_ENV_FILE="${GRIM_CONFIG_DIR}/environment"
GRIM_SERVICE_FILE="/etc/systemd/system/grim.service"
GRIM_DEFAULT_PORT="11434"
GRIM_DEFAULT_HOST="127.0.0.1"   # SSRF-safe default; override in $GRIM_ENV_FILE

# ---------------------------------------------------------------------------
# Helper: run a command with sudo if we are not already root
# ---------------------------------------------------------------------------
need_root() {
    if [ "$EUID" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            sudo "$@"
        else
            echo "[grim] ERROR: Must be run as root or have sudo available."
            exit 1
        fi
    else
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# Hardware detection. Sets globals: DETECTED_BACKEND, DETECTED_GPUS,
# DETECTED_PARALLEL, DETECTED_KERNELS_TIMEOUT.
# ---------------------------------------------------------------------------
detect_hardware() {
    DETECTED_BACKEND=""
    DETECTED_GPUS=""
    DETECTED_PARALLEL="No"
    DETECTED_KERNELS_TIMEOUT=""

    # ROCm (AMD Instinct / CDNA)
    if command -v rocminfo >/dev/null 2>&1 && [ -w /dev/dri 2>/dev/null -o -d /sys/class/kfd ]; then
        local gpu_count
        gpu_count=$(rocminfo 2>/dev/null | grep -c "GPU[" || true)
        if [ "$gpu_count" -gt 0 ] 2>/dev/null; then
            DETECTED_BACKEND="rocm"
            DETECTED_GPUS=$(seq 0 $((gpu_count - 1)) | paste -sd, -)
            DETECTED_PARALLEL=$([ "$gpu_count" -gt 1 ] && echo "Yes" || echo "No")
            # MI300/X/MI355 expose shared-memory ROCm limits; surface a timeout
            # hint so the host aborts a wedged kernel.
            DETECTED_KERNELS_TIMEOUT=300
        fi
    fi

    # CUDA (NVIDIA)
    if [ -z "$DETECTED_BACKEND" ] && command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_count
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            DETECTED_BACKEND="cuda"
            DETECTED_GPUS=$(seq 0 $((gpu_count - 1)) | paste -sd, -)
            DETECTED_PARALLEL=$([ "$gpu_count" -gt 1 ] && echo "Yes" || echo "No")
        fi
    fi

    # Apple Metal (single-GPU assumption)
    if [ -z "$DETECTED_BACKEND" ] && [ "$(uname -s)" = "Darwin" ]; then
        if system_profiler SPDisplaysDataType >/dev/null 2>&1; then
            DETECTED_BACKEND="metal"
            DETECTED_GPUS="0"
            DETECTED_PARALLEL="No"
        fi
    fi

    # Vulkan (fallback, any vendor)
    if [ -z "$DETECTED_BACKEND" ]; then
        if command -v vulkaninfo >/dev/null 2>&1; then
            if vulkaninfo 2>/dev/null | grep -q "VkPhysicalDevice"; then
                DETECTED_BACKEND="vulkan"
                DETECTED_GPUS="0"
                DETECTED_PARALLEL="No"
            fi
        fi
    fi

    # Final default: CPU-only
    if [ -z "$DETECTED_BACKEND" ]; then
        DETECTED_BACKEND="cpu"
        DETECTED_GPUS=""
        DETECTED_PARALLEL="No"
    fi

    # GRIM_CONTEXT: pick a sane default context window. 8192 unless the
    # operator overrides in the environment file.
    if [ -z "$DETECTED_KERNELS_TIMEOUT" ]; then
        DETECTED_KERNELS_TIMEOUT="300"
    fi
}

# ---------------------------------------------------------------------------
# Build grim if the release binary is missing
# ---------------------------------------------------------------------------
build_grim() {
    if [ ! -f "$GRIM_BINARY" ]; then
        echo "[grim] Building grim from source (release)..."
        (cd "$(dirname "$SCRIPT_DIR")" && cargo build --release)
        if [ ! -f "$GRIM_BINARY" ]; then
            echo "[grim] ERROR: cargo build failed — binary not found."
            exit 1
        fi
    fi
}

# ---------------------------------------------------------------------------
# Install the binary
# ---------------------------------------------------------------------------
install_binary() {
    echo "[grim] Installing binary to $GRIM_INSTALL_DIR/grim"
    need_root cp "$GRIM_BINARY" "$GRIM_INSTALL_DIR/grim"
    need_root chmod +x "$GRIM_INSTALL_DIR/grim"
}

# ---------------------------------------------------------------------------
# Detect hardware and write the persisted env file
# ---------------------------------------------------------------------------
install_env_file() {
    detect_hardware
    echo "[grim] Hardware detection:"
    echo "[grim]   backend   = $DETECTED_BACKEND"
    echo "[grim]   gpus      = ${DETECTED_GPUS:-(none)}"
    echo "[grim]   parallel  = $DETECTED_PARALLEL"
    echo "[grim]   timeout   = ${DETECTED_KERNELS_TIMEOUT}s"
    echo "[grim]   context   = ${GRIM_CONTEXT:-8192}"

    need_root mkdir -p "$GRIM_CONFIG_DIR"
    # Write idempotent env file. Operators can edit values here; the daemon
    # (and `grim serve`) read them via RuntimeEnv.
    need_root tee "$GRIM_ENV_FILE" > /dev/null <<EOF
# Persisted by dist/install.sh — read by the grim systemd unit via
# `EnvironmentFile` and by `RuntimeEnv::from_env` at process start.
# Edit and restart the service (`systemctl restart grim`) to apply.

# Incoming SSRF posture (§network): bind host. Defaults to loopback.
GRIM_HOST=${GRIM_DEFAULT_HOST}

# Bind port.
GRIM_PORT=${GRIM_DEFAULT_PORT}

# Compute backend: rocm | cuda | vulkan | metal | cpu | auto
GRIM_BACKEND=${DETECTED_BACKEND}

# Comma-separated device ordinals (e.g. 0,1); empty = all visible.
GRIM_GPUS=${DETECTED_GPUS}

# Multi-GPU parallelism hint (Yes=attempt; No=single device).
GRIM_PARALLEL=${DETECTED_PARALLEL}

# Tensor-parallel world size (0 or 1 = single device; >1 requires RCCL/NCCL
# comms init + sharded model layers — SCYTHE-2 WI-6 entry point).
GRIM_TP_SIZE=0

# KV-cache quantization: off | int4 | int8
GRIM_KV_QUANT=off

# Model context-window cap (KV cache length).
GRIM_CONTEXT=8192

# Per-device GPU memory budget cap in MiB (empty = let the backend decide).
# GRIM_MEM_BUDGET_MIB=

# Soft GPU-kernel timeout in seconds before the host aborts a launch.
GRIM_KERNEL_TIMEOUT=${DETECTED_KERNELS_TIMEOUT}
EOF
    need_root chmod 644 "$GRIM_ENV_FILE"
    echo "[grim] Wrote environment to $GRIM_ENV_FILE"
}

# ---------------------------------------------------------------------------
# Create required directories and a default config
# ---------------------------------------------------------------------------
install_config() {
    echo "[grim] Creating config and log directories..."
    need_root mkdir -p "$GRIM_CONFIG_DIR" "$GRIM_LOG_DIR" "$GRIM_MODELS_DIR"
    need_root chmod 755 "$GRIM_LOG_DIR" "$GRIM_MODELS_DIR"

    # Write a minimal default config if one does not already exist
    if [ ! -f "$GRIM_CONFIG_DIR/grim.toml" ]; then
        need_root tee "$GRIM_CONFIG_DIR/grim.toml" > /dev/null <<EOF
# Grim inference server default configuration.
# Bind address is sourced from /etc/grim/environment (GRIM_HOST/GRIM_PORT).
models_dir = "${GRIM_MODELS_DIR}"

[server.log]
level = "info"
file  = "${GRIM_LOG_DIR}/serve.log"
EOF
        echo "[grim] Default config written to $GRIM_CONFIG_DIR/grim.toml"
    else
        echo "[grim] Config already exists — skipping."
    fi
}

# ---------------------------------------------------------------------------
# Write the systemd service unit
# ---------------------------------------------------------------------------
install_service() {
    if ! command -v systemctl >/dev/null 2>&1; then
        echo "[grim] systemctl not found — skipping service registration."
        echo "[grim] Start the server manually: grim serve"
        return
    fi

    echo "[grim] Writing systemd service unit to $GRIM_SERVICE_FILE"
    need_root tee "$GRIM_SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Grim AI Inference Server
Documentation=https://github.com/example/grim
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
# grim serve reads GRIM_HOST/GRIM_PORT from the EnvironmentFile via
# RuntimeEnv::resolve_bind; no --address flag is needed.
EnvironmentFile=${GRIM_ENV_FILE}
ExecStart=${GRIM_INSTALL_DIR}/grim serve --config ${GRIM_CONFIG_DIR}/grim.toml
Restart=on-failure
RestartSec=5
StandardOutput=append:${GRIM_LOG_DIR}/serve.log
StandardError=append:${GRIM_LOG_DIR}/serve.log

# Run as a dedicated system user when available
# Create with: useradd -r -s /usr/sbin/nologin grim
# User=grim
# Group=grim

[Install]
WantedBy=multi-user.target
EOF

    echo "[grim] Reloading systemd daemon..."
    need_root systemctl daemon-reload

    echo "[grim] Enabling grim.service for boot autostart..."
    need_root systemctl enable grim.service

    echo "[grim] Starting grim.service..."
    need_root systemctl restart grim.service || true

    echo "[grim] Service status:"
    systemctl status grim.service --no-pager 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
uninstall_grim() {
    local purge="${1:-}"

    if command -v systemctl >/dev/null 2>&1; then
        echo "[grim] Stopping and disabling grim.service..."
        need_root systemctl stop grim.service 2>/dev/null || true
        need_root systemctl disable grim.service 2>/dev/null || true
        need_root rm -f "$GRIM_SERVICE_FILE"
        need_root systemctl daemon-reload
    fi

    if [ -f "$GRIM_INSTALL_DIR/grim" ]; then
        need_root rm -f "$GRIM_INSTALL_DIR/grim"
        echo "[grim] Removed binary from $GRIM_INSTALL_DIR"
    fi

    if [ "$purge" = "purge" ]; then
        need_root rm -rf "$GRIM_CONFIG_DIR" "$GRIM_LOG_DIR"
        echo "[grim] Purged config ($GRIM_CONFIG_DIR) and logs ($GRIM_LOG_DIR)."
        echo "[grim] Model files at $GRIM_MODELS_DIR were NOT removed. Delete manually if desired."
    fi
}

# ---------------------------------------------------------------------------
# Grim-config: print the runtime config the daemon will pick up
# ---------------------------------------------------------------------------
cmd_config() {
    detect_hardware
    local host="${GRIM_HOST:-${GRIM_DEFAULT_HOST}}"
    local port="${GRIM_PORT:-${GRIM_DEFAULT_PORT}}"
    echo "=== Grim Runtime Configuration ==="
    echo "  bind     = ${host}:${port}"
    echo "  backend  = ${GRIM_BACKEND:-$DETECTED_BACKEND}"
    echo "  gpus     = ${GRIM_GPUS:-${DETECTED_GPUS:-(none)}}"
    echo "  parallel = ${GRIM_PARALLEL:-$DETECTED_PARALLEL}"
    echo "  tp_size  = ${GRIM_TP_SIZE:-0}"
    echo "  kv_quant = ${GRIM_KV_QUANT:-off}"
    echo "  context  = ${GRIM_CONTEXT:-8192}"
    if [ -f "$GRIM_ENV_FILE" ] && [ -r "$GRIM_ENV_FILE" ]; then
        echo "  env file = $GRIM_ENV_FILE (persisted by install.sh)"
    fi
    echo "  binary   = ${GRIM_INSTALL_DIR}/grim"
    echo "  models   = ${GRIM_MODELS_DIR}"
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
ACTION="${1:-install}"

case "$ACTION" in
    install|-i|--install)
        echo "=== Grim Inference Engine Installer ==="
        build_grim
        install_binary
        install_env_file
        install_config
        install_service
        echo ""
        echo "=== Installation Complete ==="
        echo "  Server is listening on ${GRIM_DEFAULT_HOST}:${GRIM_DEFAULT_PORT}"
        echo "  To expose publicly, edit GRIM_HOST in ${GRIM_ENV_FILE} and run:"
        echo "      sudo systemctl restart grim"
        echo "  Logs:    ${GRIM_LOG_DIR}/serve.log"
        echo "  Config:  ${GRIM_CONFIG_DIR}/grim.toml"
        echo "  Env:     ${GRIM_ENV_FILE}"
        echo "  Models:  ${GRIM_MODELS_DIR}"
        echo ""
        echo "  Run 'grim status' to see loaded models."
        echo "  Use 'grim pull <url>' to download a model."
        echo ""
        echo "  Show detected config: dist/install.sh config"
        ;;
    config|cfg)
        cmd_config
        ;;
    uninstall|-u|--uninstall)
        echo "=== Grim Inference Engine Uninstaller ==="
        PURGE="${2:-}"
        uninstall_grim "$PURGE"
        echo "=== Uninstall Complete ==="
        ;;
    *)
        echo "Usage: $0 {install|config|uninstall} [purge]"
        echo "  install   - Build and install grim; register systemd daemon on loopback:${GRIM_DEFAULT_PORT}"
        echo "  config    - Print the runtime config detected from hardware + env"
        echo "  uninstall - Stop and remove grim service and binary"
        echo "  purge     - (uninstall modifier) also removes config and logs"
        exit 1
        ;;
esac
