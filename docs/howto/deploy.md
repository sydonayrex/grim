# How-To: Deploy Grim in Production

Grim can run bare-metal or containerized on Linux systems with ROCm, CUDA, Vulkan, or CPU support.

## 1. Containerized Deployment (Docker / Podman)

Run Grim with a local models directory mounted:

```bash
docker run -d \
  --name grim \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  -p 11434:11434 \
  -v /path/to/models:/models:ro \
  -v /path/to/grim.toml:/etc/grim/grim.toml:ro \
  -e GRIM_HOST=0.0.0.0 \
  -e GRIM_PORT=11434 \
  grim:latest
```

---

## 2. Health and Liveness Probes

Kubernetes / Container health checks:

- **Liveness & Readiness**: `GET /health` returns `200 OK` with `{"status": "ok"}` when the HTTP engine is ready.
- **Metrics Scraping**: `GET /metrics` returns Prometheus metrics for telemetry tracking.
  - To expose metrics on public interfaces without authentication, set `GRIM_ALLOW_PUBLIC_METRICS=1`.

```bash
# Check server health
curl -f http://127.0.0.1:11434/health

# Scrape Prometheus metrics
curl -s http://127.0.0.1:11434/metrics
```

---

## 3. Production Service Management (systemd)

Install and manage Grim as an OS service daemon:

```bash
# Install service definition
sudo grim service install --config /etc/grim/grim.toml

# Start and enable service
sudo grim service start
sudo systemctl enable grim

# Verify status
grim doctor
```
