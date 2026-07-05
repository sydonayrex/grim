# Usability Test Scenario: Transitioning to Local AMD Inference

**Date**: 2026-07-04  
**Target Hardware**: Workstations equipped with AMD Radeon™ RX 9070 XT GPUs (32GB VRAM, RDNA4 architecture).  
**Context**: A software development team migrating from high-latency external cloud APIs to self-hosted local inference using Grim as a drop-in replacement on the Ollama-compatible port `11434`.

---

## 1. Persona Profiles (6 User Types)

| Persona Name | Role | Experience Level | Key Requirement | Pain Point |
|---|---|---|---|---|
| **Alex** | Junior Full-Stack Engineer | Junior (1-2 years) | Fast code completion & simple CLI setup | CLI complexity, slow initial token latency (TTFT) |
| **Sarah** | Senior Backend Architect | Senior (8+ years) | JSON grammar-constrained outputs, low-bit quant safety | Quantization accuracy loss, formatting errors |
| **Marcus** | Platform/Cloud Engineer | Senior (6+ years) | systemd service deployment & performance telemetry | Service uptime monitoring, CPU/GPU handoff metrics |
| **Linus** | Junior DevOps Specialist | Junior (1-2 years) | Automated installation scripts & clean uninstall paths | Unclean configurations, path mismatches |
| **Elena** | AI Research Engineer | Mid (4 years) | Custom sampler plugins & speculative drafting models | Rigid inference engines that block out-of-tree plugins |
| **David** | Engineering Director | Executive (10+ years)| Multi-tenant routing, local security audit trails | External network leaks, cloud subscription costs |

---

## 2. Usability Test Tasks

### Task 1: Zero-Configuration Install & Daemon Setup
*   **Persona**: Linus (Junior DevOps) & Marcus (Senior Platform)
*   **Goal**: Download, install, and run Grim as a background system service on the development workstation.
*   **Steps**:
    1. Run `install.sh` to extract the binary to `/usr/local/bin/grim`.
    2. Register the service daemon via `grim service install --config /etc/grim/grim.toml`.
    3. Start the service with `grim service start`.
    4. Confirm that status is running and logs target `/var/log/grim/grim.log`.

### Task 2: Drop-in Ollama Replacements & Multi-adapter LoRA
*   **Persona**: Alex (Junior Full-Stack) & Sarah (Senior Backend)
*   **Goal**: Wire the developer IDE to point to `http://127.0.0.1:11434` and verify streaming code completions.
*   **Steps**:
    1. Configure the IDE completion plugin to hit port `11434`.
    2. Request a code completion prompting: `def quicksort(arr):`.
    3. Inject a custom developer style adapter: `/v1/chat/completions` with header/payload key `"adapters": ["peft-style-guide"]`.
    4. Verify that tokens stream using SSE and follow the style layout defined in the PEFT adapter.

### Task 3: Capability-Based Model Ingestion
*   **Persona**: Elena (AI Researcher)
*   **Goal**: Load a Q4_K GGUF model and attach a speculative draft bundle to maximize local AMD token throughput.
*   **Steps**:
    1. Import a custom GGUF checkpoint: `grim run --model ./models/llama3-8b-Q4_K.gguf`.
    2. Load companion speculation config json ensuring draft-bundle VRAM residency check passes.
    3. Verify token throughput matches the expected RDNA4 hardware target (>80 tokens/sec decode).

---

## 3. Evaluation Criteria & Success Metrics

1. **Self-Service Install Success**: Junior engineers must complete Task 1 in under 5 minutes without needing supervisor elevation support.
2. **Interactive Time-To-First-Token (TTFT)**: Code completions must stream the first token in under 150ms on the Radeon RX 9070 XT.
3. **Daemon Resilience**: Stopping the systemd/launchd process must invoke recovery policies and restart the service within 2 seconds without corrupting active model cache files.
