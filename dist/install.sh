#!/bin/bash
# Grim Zero-Configuration Install Script
# Creates /usr/local/bin/grim binary and installs a systemd service
# that starts the inference server on boot at port 11434.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRIM_BINARY="${SCRIPT_DIR}/../target/release/grim"

GRIM_INSTALL_DIR="/usr/local/bin"
GRIM_CONFIG_DIR="/etc/grim"
GRIM_LOG_DIR="/var/log/grim"
GRIM_MODELS_DIR="/var/lib/grim/models"
GRIM_SERVICE_FILE="/etc/systemd/system/grim.service"
GRIM_DEFAULT_PORT="11434"

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
# See https://github.com/example/grim for full reference.

[server]
bind = "0.0.0.0:${GRIM_DEFAULT_PORT}"
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
ExecStart=${GRIM_INSTALL_DIR}/grim serve \
    --address 0.0.0.0:${GRIM_DEFAULT_PORT} \
    --config ${GRIM_CONFIG_DIR}/grim.toml
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
    need_root systemctl start grim.service || true

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
# Entry point
# ---------------------------------------------------------------------------
ACTION="${1:-install}"

case "$ACTION" in
    install|-i|--install)
        echo "=== Grim Inference Engine Installer ==="
        build_grim
        install_binary
        install_config
        install_service
        echo ""
        echo "=== Installation Complete ==="
        echo "  Server is listening on port ${GRIM_DEFAULT_PORT}"
        echo "  Logs:    ${GRIM_LOG_DIR}/serve.log"
        echo "  Config:  ${GRIM_CONFIG_DIR}/grim.toml"
        echo "  Models:  ${GRIM_MODELS_DIR}"
        echo ""
        echo "  Run 'grim status' to see loaded models."
        echo "  Use 'grim pull <url>' to download a model."
        ;;
    uninstall|-u|--uninstall)
        echo "=== Grim Inference Engine Uninstaller ==="
        PURGE="${2:-}"
        uninstall_grim "$PURGE"
        echo "=== Uninstall Complete ==="
        ;;
    *)
        echo "Usage: $0 {install|uninstall} [purge]"
        echo "  install   - Build and install grim; register systemd daemon on port ${GRIM_DEFAULT_PORT}"
        echo "  uninstall - Stop and remove grim service and binary"
        echo "  purge     - (uninstall modifier) also removes config and logs"
        exit 1
        ;;
esac