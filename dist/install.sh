#!/bin/bash
# Grim Zero-Configuration Install Script
# Creates /usr/local/bin/grim binary and installs a systemd service
# that starts the inference server on boot. The bind address defaults to
# loopback (127.0.0.1:11434) for an SSRF-safe-by-default posture; set
# GRIM_HOST/GRIM_PORT in /etc/grim/environment to expose it publicly.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Search locations for pre-built grim binary
if [ -n "$GRIM_BINARY_PATH" ] && [ -f "$GRIM_BINARY_PATH" ]; then
    GRIM_BINARY="$GRIM_BINARY_PATH"
elif [ -f "${PROJECT_ROOT}/target/release/grim" ]; then
    GRIM_BINARY="${PROJECT_ROOT}/target/release/grim"
elif [ -f "./grim" ]; then
    GRIM_BINARY="./grim"
else
    GRIM_BINARY="${PROJECT_ROOT}/target/release/grim"
fi

GRIM_INSTALL_DIR="/usr/local/bin"
GRIM_CONFIG_DIR="/etc/grim"
GRIM_LOG_DIR="/var/log/grim"
GRIM_VAR_DIR="/var/lib/grim"
GRIM_MODELS_DIR="${GRIM_VAR_DIR}/models"
GRIM_PLUGINS_DIR="${GRIM_VAR_DIR}/plugins"
GRIM_CACHE_DIR="${GRIM_VAR_DIR}/cache"
GRIM_ENV_FILE="${GRIM_CONFIG_DIR}/environment"
GRIM_SERVICE_FILE="/etc/systemd/system/grim.service"
GRIM_USER="grim"
GRIM_GROUP="grim"
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
# DETECTED_PARALLEL, DETECTED_TP_SIZE, DETECTED_KERNELS_TIMEOUT.
# ---------------------------------------------------------------------------
detect_hardware() {
    DETECTED_BACKEND=""
    DETECTED_GPUS=""
    DETECTED_PARALLEL="No"
    DETECTED_TP_SIZE="0"
    DETECTED_KERNELS_TIMEOUT=""

    # ROCm (AMD Instinct / CDNA)
    if command -v rocminfo >/dev/null 2>&1 && [ -w /dev/dri 2>/dev/null -o -d /sys/class/kfd ]; then
        local gpu_count
        gpu_count=$(rocminfo 2>/dev/null | grep -c "GPU[" || true)
        if [ "$gpu_count" -gt 0 ] 2>/dev/null; then
            DETECTED_BACKEND="rocm"
            DETECTED_GPUS=$(seq 0 $((gpu_count - 1)) | paste -sd, -)
            if [ "$gpu_count" -gt 1 ]; then
                DETECTED_PARALLEL="Yes"
                DETECTED_TP_SIZE="$gpu_count"
            else
                DETECTED_PARALLEL="No"
                DETECTED_TP_SIZE="0"
            fi
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
            if [ "$gpu_count" -gt 1 ]; then
                DETECTED_PARALLEL="Yes"
                DETECTED_TP_SIZE="$gpu_count"
            else
                DETECTED_PARALLEL="No"
                DETECTED_TP_SIZE="0"
            fi
        fi
    fi

    # Apple Metal (single-GPU assumption)
    if [ -z "$DETECTED_BACKEND" ] && [ "$(uname -s)" = "Darwin" ]; then
        if system_profiler SPDisplaysDataType >/dev/null 2>&1; then
            DETECTED_BACKEND="metal"
            DETECTED_GPUS="0"
            DETECTED_PARALLEL="No"
            DETECTED_TP_SIZE="0"
        fi
    fi

    # Vulkan (fallback, any vendor)
    if [ -z "$DETECTED_BACKEND" ]; then
        if command -v vulkaninfo >/dev/null 2>&1; then
            if vulkaninfo 2>/dev/null | grep -q "VkPhysicalDevice"; then
                DETECTED_BACKEND="vulkan"
                DETECTED_GPUS="0"
                DETECTED_PARALLEL="No"
                DETECTED_TP_SIZE="0"
            fi
        fi
    fi

    # Final default: CPU-only
    if [ -z "$DETECTED_BACKEND" ]; then
        DETECTED_BACKEND="cpu"
        DETECTED_GPUS=""
        DETECTED_PARALLEL="No"
        DETECTED_TP_SIZE="0"
    fi

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
        (cd "$PROJECT_ROOT" && cargo build --release)
        if [ ! -f "$GRIM_BINARY" ]; then
            echo "[grim] ERROR: cargo build failed — binary not found."
            exit 1
        fi
    fi
}

# ---------------------------------------------------------------------------
# Install binary and companion tools
# ---------------------------------------------------------------------------
install_binary() {
    echo "[grim] Installing binary to $GRIM_INSTALL_DIR/grim"
    need_root cp "$GRIM_BINARY" "$GRIM_INSTALL_DIR/grim"
    need_root chmod +x "$GRIM_INSTALL_DIR/grim"

    if [ -f "${SCRIPT_DIR}/grim-config" ]; then
        echo "[grim] Installing helper to $GRIM_INSTALL_DIR/grim-config"
        need_root cp "${SCRIPT_DIR}/grim-config" "$GRIM_INSTALL_DIR/grim-config"
        need_root chmod +x "$GRIM_INSTALL_DIR/grim-config"
    fi
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
    echo "[grim]   tp_size   = $DETECTED_TP_SIZE"
    echo "[grim]   timeout   = ${DETECTED_KERNELS_TIMEOUT}s"
    echo "[grim]   context   = ${GRIM_CONTEXT:-8192}"

    need_root mkdir -p "$GRIM_CONFIG_DIR"
    # Write idempotent env file. Operators can edit values here; the daemon
    # (and `grim serve`) read them via RuntimeEnv.
    need_root tee "$GRIM_ENV_FILE" > /dev/null <<EOF
# Persisted by dist/install.sh — read by the grim systemd unit via
# \`EnvironmentFile\` and by \`RuntimeEnv::from_env\` at process start.
# Edit and restart the service (\`systemctl restart grim\`) to apply.

# System Directories
GRIM_MODELS_DIR=${GRIM_MODELS_DIR}
GRIM_PLUGINS_DIR=${GRIM_PLUGINS_DIR}
GRIM_LOG_DIR=${GRIM_LOG_DIR}
GRIM_HSACO_CACHE_DIR=${GRIM_CACHE_DIR}/hsaco

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

# Tensor-parallel world size (0 or 1 = single device; >1 requires RCCL/NCCL).
GRIM_TP_SIZE=${DETECTED_TP_SIZE}

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
# Create required directories, system user, and default config
# ---------------------------------------------------------------------------
install_config() {
    echo "[grim] Creating system user ($GRIM_USER)..."
    if ! getent group "$GRIM_GROUP" >/dev/null 2>&1; then
        need_root groupadd -r "$GRIM_GROUP" 2>/dev/null || true
    fi
    if ! getent passwd "$GRIM_USER" >/dev/null 2>&1; then
        need_root useradd -r -g "$GRIM_GROUP" -d "$GRIM_VAR_DIR" -s /usr/sbin/nologin "$GRIM_USER" 2>/dev/null || true
    fi

    # Add grim user to video and render groups for GPU device node access
    for grp in video render kvm; do
        if getent group "$grp" >/dev/null 2>&1; then
            need_root usermod -aG "$grp" "$GRIM_USER" 2>/dev/null || true
        fi
    done

    echo "[grim] Creating config, log, cache, and model directories..."
    need_root mkdir -p "$GRIM_CONFIG_DIR" "$GRIM_LOG_DIR" "$GRIM_MODELS_DIR" "$GRIM_PLUGINS_DIR" "$GRIM_CACHE_DIR/hsaco"
    need_root chown -R "${GRIM_USER}:${GRIM_GROUP}" "$GRIM_LOG_DIR" "$GRIM_VAR_DIR"
    need_root chmod 755 "$GRIM_LOG_DIR" "$GRIM_MODELS_DIR" "$GRIM_PLUGINS_DIR" "$GRIM_CACHE_DIR"

    # Write a minimal default config if one does not already exist
    if [ ! -f "$GRIM_CONFIG_DIR/grim.toml" ]; then
        need_root tee "$GRIM_CONFIG_DIR/grim.toml" > /dev/null <<EOF
# Grim inference server default configuration.
# Bind address is sourced from /etc/grim/environment (GRIM_HOST/GRIM_PORT).
models_dir = "${GRIM_MODELS_DIR}"
plugins_dir = "${GRIM_PLUGINS_DIR}"

[server.log]
level = "info"
file  = "${GRIM_LOG_DIR}/serve.log"
EOF
        need_root chown "${GRIM_USER}:${GRIM_GROUP}" "$GRIM_CONFIG_DIR/grim.toml"
        echo "[grim] Default config written to $GRIM_CONFIG_DIR/grim.toml"
    else
        echo "[grim] Config already exists — skipping."
    fi
}

# ---------------------------------------------------------------------------
# Hardware-adaptive kernel JIT tuning and pre-compilation
# ---------------------------------------------------------------------------
tune_kernels() {
    if [ "$DETECTED_BACKEND" != "rocm" ]; then
        echo "[grim] Skipping GPU kernel tuning (detected backend: $DETECTED_BACKEND)."
        return
    fi

    echo "[grim] Running post-install hardware JIT kernel tuning for ROCm..."
    echo "[grim] Sweeping canonical shapes and compiling optimized .hsaco kernels..."
    
    local tune_bin="$GRIM_INSTALL_DIR/grim"
    if [ ! -f "$tune_bin" ]; then
        tune_bin="$GRIM_BINARY"
    fi

    if [ -x "$tune_bin" ]; then
        GRIM_HSACO_CACHE_DIR="${GRIM_CACHE_DIR}/hsaco" \
            "$tune_bin" tune --device 0 --output-dir "${GRIM_CACHE_DIR}/hsaco" 2>&1 | tee -a "${GRIM_LOG_DIR}/tune.log" || {
            echo "[grim] WARNING: Hardware kernel tuning encountered an issue; grim will fallback to lazy runtime JIT."
        }
        need_root chown -R "${GRIM_USER}:${GRIM_GROUP}" "${GRIM_CACHE_DIR}/hsaco" 2>/dev/null || true
    else
        echo "[grim] WARNING: Grim binary not found or not executable for tuning."
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
User=${GRIM_USER}
Group=${GRIM_GROUP}
SupplementaryGroups=video render kvm
EnvironmentFile=${GRIM_ENV_FILE}
ExecStart=${GRIM_INSTALL_DIR}/grim serve --config ${GRIM_CONFIG_DIR}/grim.toml --plugins ${GRIM_PLUGINS_DIR}
Restart=on-failure
RestartSec=5
StandardOutput=append:${GRIM_LOG_DIR}/serve.log
StandardError=append:${GRIM_LOG_DIR}/serve.log

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
    if [ -f "$GRIM_INSTALL_DIR/grim-config" ]; then
        need_root rm -f "$GRIM_INSTALL_DIR/grim-config"
        echo "[grim] Removed helper from $GRIM_INSTALL_DIR"
    fi

    if [ "$purge" = "purge" ]; then
        need_root rm -rf "$GRIM_CONFIG_DIR" "$GRIM_LOG_DIR" "$GRIM_VAR_DIR"
        need_root userdel "$GRIM_USER" 2>/dev/null || true
        need_root groupdel "$GRIM_GROUP" 2>/dev/null || true
        echo "[grim] Purged config ($GRIM_CONFIG_DIR), logs ($GRIM_LOG_DIR), data ($GRIM_VAR_DIR), and system user."
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
    echo "  tp_size  = ${GRIM_TP_SIZE:-$DETECTED_TP_SIZE}"
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
        tune_kernels
        install_service
        echo ""
        echo "=== Installation Complete ==="
        echo "  Server is listening on http://${GRIM_DEFAULT_HOST}:${GRIM_DEFAULT_PORT}"
        echo "  Open http://${GRIM_DEFAULT_HOST}:${GRIM_DEFAULT_PORT} in your web browser to access the dashboard."
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
        echo "  purge     - (uninstall modifier) also removes config, logs, and user"
        exit 1
        ;;
esac
