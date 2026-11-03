#!/usr/bin/env bash
# Build reproducible release tarball and sha256 checksum for Grim
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VERSION="${1:-$(grep '^version' "${REPO_ROOT}/Cargo.toml" | head -n1 | cut -d'"' -f2)}"
TARGET_ARCH="x86_64-unknown-linux-gnu"
DIST_DIR="${REPO_ROOT}/dist"
TARBALL_NAME="grim-${VERSION}-linux-x86_64-rocm.tar.gz"
TARBALL_PATH="${DIST_DIR}/${TARBALL_NAME}"
SHA256_PATH="${TARBALL_PATH}.sha256"

echo "=== Building Grim Release v${VERSION} ==="
cd "${REPO_ROOT}"

# Build CLI and server binaries
cargo build --release -p grim-cli -p grim-server

mkdir -p "${DIST_DIR}"

TMP_STAGE="$(mktemp -d /tmp/grim-release-stage.XXXXXX)"
trap 'rm -rf "${TMP_STAGE}"' EXIT

mkdir -p "${TMP_STAGE}/bin"
cp "${REPO_ROOT}/target/release/grim" "${TMP_STAGE}/bin/"
cp "${REPO_ROOT}/target/release/grim-server" "${TMP_STAGE}/bin/"
if [ -f "${REPO_ROOT}/grim.toml.example" ]; then
    cp "${REPO_ROOT}/grim.toml.example" "${TMP_STAGE}/"
fi
if [ -f "${REPO_ROOT}/dist/grim-config" ]; then
    cp "${REPO_ROOT}/dist/grim-config" "${TMP_STAGE}/bin/"
fi
cp "${REPO_ROOT}/dist/install.sh" "${TMP_STAGE}/"

# Create reproducible tarball
tar --sort=name \
    --mtime='2026-01-01 00:00:00 UTC' \
    --owner=0 --group=0 --numeric-owner \
    -czf "${TARBALL_PATH}" -C "${TMP_STAGE}" .

# Generate sha256
(cd "${DIST_DIR}" && sha256sum "${TARBALL_NAME}" > "${TARBALL_NAME}.sha256")

echo "=== Release package created successfully ==="
echo "Tarball: ${TARBALL_PATH}"
echo "SHA256:  $(cat "${SHA256_PATH}")"
