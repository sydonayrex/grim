#!/bin/bash
# Grim Zero-Configuration Install Script
# Creates /usr/local/bin/grim binary and provides optional daemon setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRIM_BINARY="${SCRIPT_DIR}/../target/release/grim"

# Check if running as root or using sudo for system-wide install
USING_SUDO=false
if [ "$EUID" -ne 0 ]; then
    # Check if user has permission to write to /usr/local/bin
    if ! touch /usr/local/bin/grim_test 2>/dev/null; then
        USING_SUDO=true
    fi
fi

# Cleanup test file
rm -f /usr/local/bin/grim_test 2>/dev/null || true

install_grim() {
    local target_dir="$1"
    
    # Check if release binary exists, otherwise build it
    if [ ! -f "$GRIM_BINARY" ]; then
        echo "[grim install] Building grim from source..."
        cd "$(dirname "$SCRIPT_DIR")"
        cargo build --release
        
        if [ ! -f "$GRIM_BINARY" ]; then
            echo "[grim install] ERROR: Failed to build grim binary."
            exit 1
        fi
    fi
    
    # Create target directory if needed
    mkdir -p "$target_dir"
    
    # Copy binary
    cp "$GRIM_BINARY" "$target_dir/grim"
    chmod +x "$target_dir/grim"
    
    echo "[grim install] Successfully installed grim to $target_dir/grim"
}

uninstall_grim() {
    local target_dir="$1"
    
    if [ -f "$target_dir/grim" ]; then
        rm -f "$target_dir/grim"
        echo "[grim uninstall] Removed grim from $target_dir/"
    else
        echo "[grim uninstall] No grim binary found at $target_dir/grim"
    fi
    
    # Clean up config and log directories if requested
    if [ "$2" = "purge" ]; then
        rm -rf /etc/grim 2>/dev/null || true
        rm -rf /var/log/grim 2>/dev/null || true
        echo "[grim uninstall] Purged configuration and log directories."
    fi
}

# Main script logic
ACTION="${1:-install}"
TARGET_DIR="${2:-/usr/local/bin}"

if [ "$ACTION" = "install" ] || [ "$ACTION" = "-i" ] || [ "$ACTION" = "--install" ]; then
    echo "=== Grim Inference Engine Installer ==="
    
    if [ "$USING_SUDO" = true ]; then
        echo "[grim install] Requesting sudo privileges for system-wide installation..."
        if command -v sudo >/dev/null 2>&1; then
            mkdir -p "$TARGET_DIR"
            cp "$GRIM_BINARY" "$TARGET_DIR/grim" 2>/dev/null || {
                echo "[grim install] Attempting with sudo..."
                sudo cp "$GRIM_BINARY" "$TARGET_DIR/grim"
                sudo chmod +x "$TARGET_DIR/grim"
            }
        else
            echo "[grim install] ERROR: No sudo command found. Please run as root or use a user-writable directory."
            exit 1
        fi
    else
        install_grim "$TARGET_DIR"
    fi
    
elif [ "$ACTION" = "uninstall" ] || [ "$ACTION" = "-u" ] || [ "$ACTION" = "--uninstall" ]; then
    echo "=== Grim Inference Engine Uninstaller ==="
    PURGE_MODE="${3:-}"
    if [ "$PURGE_MODE" = "purge" ] || [ "$PURGE_MODE" = "-p" ] || [ "$PURGE_MODE" = "--purge" ]; then
        uninstall_grim "$TARGET_DIR" "purge"
    else
        uninstall_grim "$TARGET_DIR" ""
    fi
else
    echo "Usage: $0 {install|uninstall} [target_directory] [--purge]"
    echo "  install  - Install grim to target_directory (default: /usr/local/bin)"
    echo "  uninstall- Remove grim from target_directory"
    echo "  --purge  - Also remove configuration and log directories (uninstall only)"
    exit 1
fi

echo "=== Installation Complete ==="