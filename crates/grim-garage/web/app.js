// Grim's Garage — Dynamic Local Dashboard App Logic

document.addEventListener('DOMContentLoaded', () => {
  // State
  let currentActiveJobId = null;
  let metricsEventSource = null;
  let lossHistory = [];

  // Tab Navigation
  const navItems = document.querySelectorAll('.nav-item');
  const tabPages = document.querySelectorAll('.tab-page');

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const tabName = item.getAttribute('data-tab');
      navItems.forEach(n => n.classList.remove('active'));
      tabPages.forEach(p => p.classList.remove('active'));
      item.classList.add('active');
      document.getElementById(`page-${tabName}`).classList.add('active');
    });
  });

  // Dynamic Server Host / Port sync
  function syncServerHostDisplay(isOnline = true) {
    const statusTextEl = document.getElementById('server-status-text');
    const statusDotEl = document.getElementById('server-status-dot');
    const currentHost = window.location.host || '127.0.0.1:8741';
    if (statusTextEl) {
      statusTextEl.textContent = isOnline ? `Server Online (${currentHost})` : `Server Offline (${currentHost})`;
    }
    if (statusDotEl) {
      if (isOnline) {
        statusDotEl.classList.remove('offline');
        statusDotEl.classList.add('online');
      } else {
        statusDotEl.classList.remove('online');
        statusDotEl.classList.add('offline');
      }
    }
  }
  syncServerHostDisplay(true);

  // Global cached state for evaluation
  let cachedDevices = [];
  let cachedModels = [];

  // Fetch Hardware & Init
  fetchGpuTelemetry();
  fetchModelsList();
  fetchDatasetsList();
  fetchJobsList();

  document.getElementById('btn-refresh-hardware').addEventListener('click', fetchGpuTelemetry);
  document.getElementById('btn-start-job').addEventListener('click', handleStartTrainingJob);
  document.getElementById('btn-run-repack').addEventListener('click', handleRunRepack);
  document.getElementById('btn-pause-job').addEventListener('click', () => handleJobControl('pause'));
  document.getElementById('btn-resume-job').addEventListener('click', () => handleJobControl('resume'));
  document.getElementById('btn-cancel-job').addEventListener('click', () => handleJobControl('cancel'));

  // Autotune & Feasibility Triggers
  document.getElementById('select-model').addEventListener('change', generateAutotuneRecommendation);
  document.getElementById('select-dataset')?.addEventListener('change', generateAutotuneRecommendation);
  document.getElementById('input-mode')?.addEventListener('change', generateAutotuneRecommendation);
  document.getElementById('input-lora-rank')?.addEventListener('input', generateAutotuneRecommendation);
  document.getElementById('input-optimizer')?.addEventListener('change', generateAutotuneRecommendation);
  document.getElementById('btn-apply-autotune').addEventListener('click', applyAutotunePreset);

  // Soul Eater is a fixed recipe (spectral-orthogonal adapter init + its own
  // Muon-style optimizer), not a swappable part. Lock the optimizer picker so
  // the two controls can't disagree; the server coerces as a backstop.
  const modeSelectEl = document.getElementById('input-mode');
  const optimizerSelectEl = document.getElementById('input-optimizer');
  function syncOptimizerForMode() {
    if (!modeSelectEl || !optimizerSelectEl) return;
    if (modeSelectEl.value === 'SoulEater') {
      optimizerSelectEl.value = 'Muon';
      optimizerSelectEl.disabled = true;
      optimizerSelectEl.title = 'Locked: Soul Eater is a fixed recipe — spectral-orthogonal adapter initialization trained by its own Muon-style optimizer (Newton\u2013Schulz orthogonalized momentum + Sign-SGD). Picking a different optimizer would silently train something else.';
    } else {
      optimizerSelectEl.disabled = false;
      optimizerSelectEl.title = 'How gradient steps are applied. AdamW is the safe default; the others trade memory for speed or add second-order/subspace tricks.';
    }
  }
  modeSelectEl?.addEventListener('change', syncOptimizerForMode);
  syncOptimizerForMode();

  function isRdna4(gcnArch) {
    if (!gcnArch) return false;
    const match = gcnArch.toLowerCase().match(/gfx(\d+)/);
    if (match && match[1]) {
      const num = parseInt(match[1], 10);
      // RDNA 4 ONLY (gfx1200-gfx1299)
      return num >= 1200 && num < 1300;
    }
    return false;
  }

  function isRdna5(gcnArch) {
    if (!gcnArch) return false;
    const match = gcnArch.toLowerCase().match(/gfx(\d+)/);
    if (match && match[1]) {
      const num = parseInt(match[1], 10);
      // RDNA 5 ONLY (gfx1300-gfx1399)
      return num >= 1300;
    }
    return false;
  }

  function updateRepackCapabilities(dev) {
    const ravenOption = document.getElementById('option-raven-fp8');
    const jayOption = document.getElementById('option-jay-mxfp4');
    const repackSelect = document.getElementById('select-repack-mode');
    const pullSelect = document.getElementById('select-pull-quant');

    const arch = dev ? dev.gcn_arch || '' : '';
    const supportsRaven = dev ? isRdna4(arch) : false;
    const supportsJay = dev ? isRdna5(arch) : false;

    // 1. Update Training / Repack Studio Select
    if (ravenOption) {
      if (supportsRaven) {
        ravenOption.disabled = false;
        ravenOption.textContent = `Raven FP8 (E4M3 - RDNA 4 HW Accelerated)`;
      } else {
        ravenOption.disabled = true;
        ravenOption.textContent = `Raven FP8 (Requires RDNA 4 ONLY)`;
        if (repackSelect && repackSelect.value === 'RavenFP8') {
          repackSelect.value = 'CrowQ4K';
        }
      }
    }

    if (jayOption) {
      if (supportsJay) {
        jayOption.disabled = false;
        jayOption.textContent = `Jay MXFP4 (Micro-block 4-bit - RDNA 5 HW Accelerated)`;
      } else {
        jayOption.disabled = true;
        jayOption.textContent = `Jay MXFP4 (Requires RDNA 5 ONLY)`;
        if (repackSelect && repackSelect.value === 'JayMXFP4') {
          repackSelect.value = 'CrowQ4K';
        }
      }
    }

    // 2. Update Model Pull & Convert Studio Dropdown
    if (pullSelect) {
      Array.from(pullSelect.options).forEach(opt => {
        if (opt.value === 'RavenFP8') {
          opt.disabled = !supportsRaven;
          opt.textContent = supportsRaven ? `Raven FP8 (E4M3 - RDNA 4 HW Accelerated)` : `Raven FP8 (Requires RDNA 4 ONLY)`;
        } else if (opt.value === 'JayMXFP4') {
          opt.disabled = !supportsJay;
          opt.textContent = supportsJay ? `Jay MXFP4 (Micro-block 4-bit - RDNA 5 HW Accelerated)` : `Jay MXFP4 (Requires RDNA 5 ONLY)`;
        } else if (opt.value === 'CrowQ4K') {
          opt.disabled = false;
          opt.textContent = `Crow Q4_K (Fused GPU Dequant - Universal)`;
        }
      });
      if (pullSelect.selectedOptions[0] && pullSelect.selectedOptions[0].disabled) {
        pullSelect.value = 'CrowQ4K';
      }
    }
  }

  // ─── API Functions ────────────────────────────────────────────────────────
  async function fetchGpuTelemetry() {
    try {
      const res = await fetch('/api/rocm/devices');
      if (!res.ok) throw new Error('API error');
      const data = await res.json();
      syncServerHostDisplay(true);
      const telemetryList = document.getElementById('gpu-telemetry-list');

      // Normalize device records from either data.devices or data.backends
      let devicesList = [];
      if (data.devices && data.devices.length > 0) {
        devicesList = data.devices;
      } else if (data.backends && data.backends.length > 0) {
        devicesList = data.backends.map((b, idx) => {
          // Parse detail if needed
          return {
            ordinal: idx,
            name: b.name || 'Accelerator',
            vendor: 'Host',
            backend: b.name || 'ROCm',
            is_rocm_compliant: b.available,
            gcn_arch: b.device_kind || '',
            vram_bytes: 8 * 1024 * 1024 * 1024,
            vram_used_bytes: 0,
            gpu_busy_percent: 0,
            compute_units: 36
          };
        });
      }

      cachedDevices = devicesList;

      if (devicesList.length > 0) {
        if (telemetryList) {
          telemetryList.innerHTML = '';
          devicesList.forEach(d => {
            let backendTag = d.backend || 'ROCm';
            const vUpper = (d.vendor || '').toUpperCase();
            const bUpper = (d.backend || '').toUpperCase();
            if (vUpper.includes('NVIDIA') || bUpper.includes('CUDA')) {
              backendTag = 'CUDA';
            } else if (vUpper.includes('APPLE') || bUpper.includes('METAL')) {
              backendTag = 'Metal';
            } else if (vUpper.includes('AMD') || bUpper.includes('ROCM')) {
              backendTag = 'ROCm';
            }

            let cleanName = (d.name || '').replace(/NVIDIA AMD\/ATI/gi, 'Radeon GPU').replace(/AMD\/ATI/gi, '').trim();
            if (!cleanName || cleanName.toLowerCase().includes('generic') || cleanName === 'Radeon GPU') {
              cleanName = d.gcn_arch ? `Radeon GPU (${d.gcn_arch})` : 'GPU Accelerator';
            }
            const nameStr = `${backendTag} · ${cleanName}`;
            const usedBytes = d.vram_used_bytes || 0;
            const totalBytes = d.vram_bytes || (8 * 1024 * 1024 * 1024);

            const usedGb = (usedBytes / (1024 * 1024 * 1024)).toFixed(2);
            const totalGb = (totalBytes / (1024 * 1024 * 1024)).toFixed(1);
            const pct = Math.min(100, Math.max(0, (usedBytes / totalBytes) * 100));

            const item = document.createElement('div');
            item.style.cssText = 'padding: 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; margin-bottom: 6px;';
            item.innerHTML = `
              <div class="gpu-name" style="font-size: 11px; font-weight: 600; color: var(--text-main); margin-bottom: 4px; display: flex; justify-content: space-between;">
                <span>GPU ${d.ordinal}: ${nameStr}</span>
                <span style="font-size: 10px; color: var(--text-muted);">${d.compute_units ? `${d.compute_units} CUs` : ''}</span>
              </div>
              <div class="vram-bar-container" style="height: 6px; background: rgba(255,255,255,0.08); border-radius: 3px; overflow: hidden;">
                <div class="vram-bar" style="width: ${pct.toFixed(1)}%; height: 100%; background: ${pct > 85 ? 'var(--danger-color, #ef4444)' : 'var(--primary-accent, #6366f1)'}; transition: width 0.3s ease;"></div>
              </div>
              <div class="vram-text" style="font-size: 10px; margin-top: 4px; color: var(--text-muted); display: flex; justify-content: space-between;">
                <span>${usedGb} GB / ${totalGb} GB VRAM (${pct.toFixed(0)}%)</span>
                <span>${Number(d.gpu_busy_percent || 0)}% Load</span>
              </div>
            `;
            telemetryList.appendChild(item);
          });
        }

        const dev = devicesList[0];
        updateRepackCapabilities(dev);
      } else {
        if (telemetryList) {
          telemetryList.innerHTML = `
            <div class="gpu-name" style="font-size: 12px; font-weight: 600;">CPU Host Mode</div>
            <div class="vram-text" style="font-size: 11px; color: var(--text-muted);">System RAM Active</div>
          `;
        }
        updateRepackCapabilities(null);
      }

      // Populate Multi-GPU Selection Box in Training Control Room
      const gpuContainer = document.getElementById('gpu-selection-container');
      if (gpuContainer) {
        if (devicesList.length > 0) {
          gpuContainer.innerHTML = '';
          devicesList.forEach((d, idx) => {
            const label = document.createElement('label');
            label.style.cssText = 'display: flex; align-items: center; gap: 8px; font-size: 13px; cursor: pointer; color: var(--text-main); margin-bottom: 4px;';
            const rawArch = (d.gcn_arch || '').split(' ')[0];
            const archStr = rawArch ? ` [${rawArch}]` : '';
            label.innerHTML = `
              <input type="checkbox" class="gpu-select-checkbox" data-ordinal="${d.ordinal}" data-vram="${d.vram_bytes}" data-arch="${rawArch}" ${idx === 0 ? 'checked' : ''}>
              <span>GPU ${d.ordinal}: ${d.name}${archStr}</span>
              <span class="text-muted" style="margin-left: auto;">${(d.vram_bytes / (1024*1024*1024)).toFixed(1)} GB</span>
            `;
            gpuContainer.appendChild(label);
          });

          document.querySelectorAll('.gpu-select-checkbox').forEach(cb => {
            cb.addEventListener('change', () => {
              validateGpuSelection();
              generateAutotuneRecommendation();
            });
          });
        } else {
          gpuContainer.innerHTML = '<span class="text-muted" style="font-size: 12px;">Host CPU Execution Mode Active</span>';
        }
      }

      // Re-evaluate feasibility whenever telemetry updates
      generateAutotuneRecommendation();
    } catch (e) {
      console.warn('GPU Probe failed:', e);
      syncServerHostDisplay(false);
      const nameDisp = document.getElementById('gpu-name-display');
      if (nameDisp) nameDisp.textContent = 'Host Hardware (Unreachable)';
    }
  }

  function validateGpuSelection() {
    const checked = Array.from(document.querySelectorAll('.gpu-select-checkbox:checked'));
    const help = document.getElementById('gpu-selection-help');
    const startBtn = document.getElementById('btn-start-job');

    if (checked.length <= 1) {
      if (help) help.innerHTML = '<span style="color: var(--text-muted);">Select single or matching architecture GPUs for FSDP parallel training.</span>';
      if (startBtn) startBtn.disabled = false;
      return;
    }

    const archs = checked.map(c => c.getAttribute('data-arch'));
    const firstArch = archs[0];
    const isMatching = archs.every(a => a === firstArch);

    if (isMatching) {
      if (help) help.innerHTML = `<span style="color: var(--success-color, #22c55e); font-weight: 600;">✅ Parallel Multi-GPU Training Enabled (${firstArch} x${checked.length})</span>`;
      if (startBtn) startBtn.disabled = false;
    } else {
      if (help) help.innerHTML = `<span style="color: var(--danger-color, #ef4444); font-weight: 600;">⚠️ Multi-GPU parallel training requires matching GPU architectures (e.g. dual ${firstArch}). Cannot pair ${archs.join(' with ')}.</span>`;
      if (startBtn) startBtn.disabled = true;
    }
  }

  async function fetchModelsList() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      const select = document.getElementById('select-model');
      const chatSelect = document.getElementById('select-chat-model');

      if (select) select.innerHTML = '<option value="">Select a base model...</option>';
      if (chatSelect) chatSelect.innerHTML = '<option value="">Select a model checkpoint (.grim / .gguf)...</option>';

      if (data.models) {
        cachedModels = data.models;
        data.models.forEach(m => {
          const displayName = m.name || m.id || (m.path ? m.path.split('/').pop() : 'model');
          const sizeText = m.size_bytes && m.size_bytes > 0 ? ` (${(m.size_bytes / (1024*1024*1024)).toFixed(1)} GB)` : '';
          const formatTag = m.format ? ` [${m.format.toUpperCase()}]` : '';

          if (select) {
            const opt = document.createElement('option');
            opt.value = m.path;
            opt.textContent = `${displayName}${sizeText}`;
            select.appendChild(opt);
          }

          if (chatSelect) {
            const cOpt = document.createElement('option');
            cOpt.value = m.path;
            cOpt.textContent = `${displayName}${formatTag}${sizeText}`;
            chatSelect.appendChild(cOpt);
          }
        });
        generateAutotuneRecommendation();
      }
    } catch (e) {
      console.error('Fetch models error:', e);
    }
  }

  async function fetchDatasetsList() {
    try {
      const res = await fetch('/api/datasets');
      const data = await res.json();
      const select = document.getElementById('select-dataset');
      select.innerHTML = '<option value="">Select a dataset...</option>';
      if (data.datasets) {
        data.datasets.forEach(d => {
          const opt = document.createElement('option');
          opt.value = d.path;
          const displayName = d.name || d.id || (d.path ? d.path.split('/').pop() : 'dataset');
          const sizeText = d.size_bytes && d.size_bytes > 0 ? ` (${(d.size_bytes / (1024*1024)).toFixed(1)} MB)` : '';
          opt.textContent = `${displayName}${sizeText}`;
          select.appendChild(opt);
        });
      }
    } catch (e) {
      console.error('Fetch datasets error:', e);
    }
  }

  async function fetchJobsList() {
    try {
      const res = await fetch('/api/train/jobs');
      const data = await res.json();
      const tbody = document.getElementById('table-jobs-body');
      tbody.innerHTML = '';
      if (!data.jobs || data.jobs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-muted">No active training jobs</td></tr>';
        return;
      }
      data.jobs.forEach(j => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><code>${j.job_id.substring(0, 8)}</code></td>
          <td><span class="status-badge ${j.status}">${j.status}</span></td>
          <td>${j.training_mode}</td>
          <td>${j.model_path.split('/').pop()}</td>
          <td>
            <button class="btn btn-sm btn-accent btn-view-job" data-id="${j.job_id}">Monitor</button>
          </td>
        `;
        tbody.appendChild(tr);
      });

      document.querySelectorAll('.btn-view-job').forEach(b => {
        b.addEventListener('click', (e) => {
          const id = e.target.getAttribute('data-id');
          connectSseMetrics(id);
        });
      });
    } catch (e) {
      console.error('Fetch jobs error:', e);
    }
  }

  async function handleStartTrainingJob() {
    const model_path = document.getElementById('select-model').value;
    const dataset_path = document.getElementById('select-dataset').value;
    if (!model_path || !dataset_path) {
      alert('Please select both a base model and a dataset.');
      return;
    }

    const payload = {
      model_path,
      dataset_path,
      training_mode: document.getElementById('input-mode').value,
      lora_rank: parseInt(document.getElementById('input-lora-rank').value) || 16,
      learning_rate: parseFloat(document.getElementById('input-lr').value) || 2e-5,
      epochs: parseInt(document.getElementById('input-epochs').value) || 1,
      optimizer: document.getElementById('input-optimizer')?.value || 'AdamW',
      scheduler: document.getElementById('input-scheduler')?.value || 'Cosine',
      rocm_fusion_rmsnorm_matmul: document.getElementById('check-fusion-rmsnorm').checked,
      rocm_fusion_qkv_attention: document.getElementById('check-fusion-attn').checked,
    };

    try {
      const res = await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (res.ok && data.job_id) {
        currentActiveJobId = data.job_id;
        fetchJobsList();
        connectSseMetrics(data.job_id);
      } else {
        alert(`Failed to start job: ${data.error || 'Unknown error'}`);
      }
    } catch (e) {
      alert(`Error starting job: ${e.message}`);
    }
  }

  function connectSseMetrics(jobId) {
    if (metricsEventSource) {
      metricsEventSource.close();
    }
    currentActiveJobId = jobId;
    document.getElementById('job-active-controls').style.display = 'flex';
    lossHistory = [];

    metricsEventSource = new EventSource(`/sse/metrics/${jobId}`);

    metricsEventSource.addEventListener('metric', (event) => {
      try {
        const ev = JSON.parse(event.data);
        if (ev && ev.metric && typeof ev.metric.loss === 'number') {
          const loss = ev.metric.loss;
          document.getElementById('val-loss').textContent = loss.toFixed(4);
          
          if (typeof ev.metric.vram_used_mb === 'number' && ev.metric.vram_used_mb > 0) {
            document.getElementById('val-vram').textContent = `${(ev.metric.vram_used_mb / 1024).toFixed(2)} GB`;
          } else {
            document.getElementById('val-vram').textContent = 'Host Alloc';
          }

          if (typeof ev.metric.samples_per_sec === 'number' && ev.metric.samples_per_sec > 0) {
            document.getElementById('val-tokens-sec').textContent = `${ev.metric.samples_per_sec.toFixed(2)} smp/s`;
          } else if (typeof ev.metric.tokens === 'number') {
            document.getElementById('val-tokens-sec').textContent = `${ev.metric.tokens} tok`;
          } else {
            document.getElementById('val-tokens-sec').textContent = 'Active';
          }

          lossHistory.push(loss);
          if (lossHistory.length > 50) lossHistory.shift();
          renderLossChart();
        }
      } catch (e) {
        console.warn('Malformed metric SSE event:', event.data);
      }
    });

    // process lifetime). Close the local stream so reconnect attempts
    // stop, then refresh the jobs list so the badge renders.
    metricsEventSource.addEventListener('end', (event) => {
      try {
        if (metricsEventSource) {
          metricsEventSource.close();
        }
      } catch (_) {
        // ignore double-close errors
      }
      console.info('SSE terminal:', event.data);
      if (typeof fetchJobsList === 'function') {
        fetchJobsList();
      }
    });

    // Auto-reconnect signal: when the EventSource's underlying
    // connection errors (rare network hiccup), clean up the handle so
    // the next connectSseMetrics() call can reattach fresh. We don't
    // auto-retry in-page — the operator triggers a new run via the
    // POST flow.
    metricsEventSource.onerror = (event) => {
      console.warn('SSE onerror', event);
      try { metricsEventSource.close(); } catch (_) {}
    };
  }

  function renderLossChart() {
    if (lossHistory.length < 2) return;
    const path = document.getElementById('chart-loss-path');
    const width = 500;
    const height = 200;
    const max = Math.max(...lossHistory, 1.0);
    const min = Math.min(...lossHistory, 0.0);
    const range = max - min || 1.0;

    let d = '';
    lossHistory.forEach((val, idx) => {
      const x = (idx / (lossHistory.length - 1)) * width;
      const y = height - (((val - min) / range) * (height - 40) + 20);
      d += (idx === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
    });
    path.setAttribute('d', d);
  }

  async function handleJobControl(action) {
    if (!currentActiveJobId) return;
    try {
      if (action === 'cancel') {
        await fetch(`/api/train/cancel/${currentActiveJobId}`, { method: 'POST' });
        document.getElementById('job-active-controls').style.display = 'none';
        if (metricsEventSource) metricsEventSource.close();
        fetchJobsList();
      }
    } catch (e) {
      console.error(`Error on ${action}:`, e);
    }
  }


  // ── Repack format ↔ bits-per-weight coupling ──────────────────────────────
  // Named tiers have a FIXED bits-per-weight (their identity is the codec),
  // so the BPW field locks and displays the canonical rate. "NativeBPW"
  // flips it around: the BPW field drives uniform-width packing and no
  // named codec is stamped.
  const TIER_FIXED_BPW = {
    RavenFP8: 8.0, CrowQ4K: 4.5, JayMXFP4: 4.1,
    RookMXFP4: 4.1, JackdawMXFP8: 8.0, MagpieMXFP8: 8.0
  };
  const repackModeEl = document.getElementById('select-repack-mode');
  const bpwEl = document.getElementById('input-target-bpw');
  function syncBpwForFormat() {
    if (!repackModeEl || !bpwEl) return;
    const fixed = TIER_FIXED_BPW[repackModeEl.value];
    if (fixed !== undefined) {
      bpwInput.value = fixed.toFixed(1);
      bpwInput.disabled = true;
      bpwInput.title = 'Locked: this named codec has a fixed bits-per-weight (' + fixed + ' bpw). Choose "Native Uniform Bits" to drive packing from a custom BPW instead.';
    } else {
      bpwInput.disabled = false;
      bpwInput.title = 'Target bits-per-weight for native uniform row packing (2\u201316). Every weight is packed at exactly this width.';
    }
  }
  repackModeEl?.addEventListener('change', syncBpwForFormat);
  syncBpwForFormat();

  async function handleRunRepack() {
    const source = document.getElementById('input-repack-source').value;
    const output_name = document.getElementById('input-repack-name').value;
    if (!source || !output_name) {
      alert('Please fill out both source model path and output name.');
      return;
    }
    const FORMAT_ALIASES = { RavenFP8:'raven', CrowQ4K:'crow', JayMXFP4:'jay', RookMXFP4:'rook', JackdawMXFP8:'jackdaw', MagpieMXFP8:'magpie' };
    const payload = {
      source_path_or_url: source,
      output_name,
      target_bpw: parseFloat(document.getElementById('input-target-bpw').value) || 8.0
    };
    const repackMode = document.getElementById('select-repack-mode')?.value;
    if (repackMode && FORMAT_ALIASES[repackMode]) {
      payload.target_format = FORMAT_ALIASES[repackMode];
    }
    try {
      const res = await fetch('/api/convert', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (res.ok && data.success) {
        alert(`Repack conversion complete! Written to ${data.output_path}`);
        fetchModelsList();
      } else {
        alert(`Repack failed: ${data.message || 'Error during repacking'}`);
      }
    } catch (e) {
      alert(`Error running repack: ${e.message}`);
    }
  }

  function evaluateHardwareFeasibility(modelPath, trainingMode, loraRank, optimizerName, availableVramBytes, gpuCount) {
    // 1. Estimate Base Model Size in GB
    let modelEntry = cachedModels.find(m => m.path === modelPath);
    let modelSizeBytes = modelEntry && modelEntry.size_bytes > 0 ? modelEntry.size_bytes : 0;
    
    // If size not in entry, estimate from model name tags or default 8B (~4.5 GB in Q4, ~16 GB in FP16)
    const nameLower = (modelPath || '').toLowerCase();
    if (modelSizeBytes === 0) {
      if (nameLower.includes('70b') || nameLower.includes('72b')) {
        modelSizeBytes = nameLower.includes('q4') || nameLower.includes('crow') ? 40 * 1024 * 1024 * 1024 : 140 * 1024 * 1024 * 1024;
      } else if (nameLower.includes('32b') || nameLower.includes('33b')) {
        modelSizeBytes = nameLower.includes('q4') || nameLower.includes('crow') ? 18 * 1024 * 1024 * 1024 : 64 * 1024 * 1024 * 1024;
      } else if (nameLower.includes('13b') || nameLower.includes('14b') || nameLower.includes('16b')) {
        modelSizeBytes = nameLower.includes('q4') || nameLower.includes('crow') ? 9 * 1024 * 1024 * 1024 : 32 * 1024 * 1024 * 1024;
      } else if (nameLower.includes('7b') || nameLower.includes('8b') || nameLower.includes('9b')) {
        modelSizeBytes = nameLower.includes('q4') || nameLower.includes('crow') ? 5 * 1024 * 1024 * 1024 : 16 * 1024 * 1024 * 1024;
      } else if (nameLower.includes('3b') || nameLower.includes('4b')) {
        modelSizeBytes = nameLower.includes('q4') || nameLower.includes('crow') ? 2.5 * 1024 * 1024 * 1024 : 8 * 1024 * 1024 * 1024;
      } else {
        modelSizeBytes = 4.5 * 1024 * 1024 * 1024; // Default ~7B Q4
      }
    }

    const baseWeightGb = modelSizeBytes / (1024 * 1024 * 1024);
    const isFullFinetune = trainingMode === 'Bf16Full';
    const isQlora = trainingMode === 'QLoRA' || trainingMode === 'SpectralQLoRA' || trainingMode === 'SoulEater';

    // Base parameters estimate: ~ baseWeightGb * 2 Billion params per 4GB Q4 or 1B per 2GB FP16
    const estimatedParamsB = nameLower.includes('70b') ? 70 : nameLower.includes('32b') ? 32 : nameLower.includes('14b') ? 14 : nameLower.includes('9b') ? 9 : nameLower.includes('8b') ? 8 : nameLower.includes('7b') ? 7 : (baseWeightGb * 1.6);
    
    // Trainable params: for LoRA/QLoRA it's ~0.1% to 1% of total params depending on rank (r=16 is ~0.2%)
    const trainableParamsB = isFullFinetune ? estimatedParamsB : estimatedParamsB * (loraRank / 16.0) * 0.002;
    
    // Optimizer memory:
    // AdamW FP32: 8 bytes per trainable param (m + v) + 4 bytes FP32 master weight + 4 bytes grad
    // AdamW 8-bit: 2 bytes per param + 4 bytes grad
    // Paged AdamW: 0-2 GB hot in VRAM (rest in RAM)
    // LOMO / AdaLomo: 0-0.1 GB (fused backward update)
    // Lion 8-bit: 1 byte per param
    let optimizerBytesPerParam = 16.0; // standard AdamW
    if (optimizerName && (optimizerName.includes('8Bit') || optimizerName.includes('8bit'))) {
      optimizerBytesPerParam = 6.0;
    } else if (optimizerName && optimizerName.includes('Paged')) {
      optimizerBytesPerParam = 4.0;
    } else if (optimizerName === 'LOMO' || optimizerName === 'Adalomo') {
      optimizerBytesPerParam = 0.5;
    } else if (optimizerName === 'Adafactor' || optimizerName === 'Lion') {
      optimizerBytesPerParam = 8.0;
    }

    let optimizerGb = (trainableParamsB * 1e9 * optimizerBytesPerParam) / (1024 * 1024 * 1024);
    if (isFullFinetune) {
      // Full fine-tune needs base weights in FP16/BF16 (2 bytes/param) + Gradients (2-4 bytes/param) + Optimizer state
      optimizerGb = (estimatedParamsB * 1e9 * (2 + 4 + optimizerBytesPerParam)) / (1024 * 1024 * 1024);
    }

    // 3. Activation & KV Cache memory (sequence length ~2048, batch size 1)
    let activationGb = isFullFinetune ? 4.5 : (0.8 + (loraRank / 64.0) * 0.4);
    
    // Total VRAM required
    let totalRequiredGb = baseWeightGb + optimizerGb + activationGb;
    
    // Single or Multi-GPU VRAM available
    const totalGpuVramGb = (availableVramBytes * Math.max(1, gpuCount)) / (1024 * 1024 * 1024);
    const perGpuVramGb = availableVramBytes / (1024 * 1024 * 1024);

    return {
      modelSizeBytes,
      baseWeightGb,
      estimatedParamsB,
      trainableParamsB,
      optimizerGb,
      activationGb,
      totalRequiredGb,
      totalGpuVramGb,
      perGpuVramGb,
      isFeasible: totalRequiredGb <= (totalGpuVramGb * 0.92), // 8% headroom for ROCm runtime & OS compositor
      shortfallGb: Math.max(0, totalRequiredGb - totalGpuVramGb)
    };
  }

  function generateAutotuneRecommendation() {
    const modelSelect = document.getElementById('select-model');
    const model = modelSelect ? modelSelect.value : '';
    const box = document.getElementById('autotune-recommendation-box');
    const text = document.getElementById('autotune-text');
    const applyBtn = document.getElementById('btn-apply-autotune');
    if (!box || !text) return;

    if (!model) {
      text.innerHTML = 'Select a base model and dataset to evaluate hardware telemetry and compute best-case training settings.';
      box.style.display = 'block';
      return;
    }

    const trainingMode = document.getElementById('input-mode')?.value || 'QLoRA';
    const loraRank = parseInt(document.getElementById('input-lora-rank')?.value) || 16;
    const optimizerName = document.getElementById('input-optimizer')?.value || 'AdamW';

    // Get selected GPU VRAM
    const checkedGpus = Array.from(document.querySelectorAll('.gpu-select-checkbox:checked'));
    let primaryVramBytes = 8 * 1024 * 1024 * 1024;
    let gpuCount = checkedGpus.length || 1;
    if (checkedGpus.length > 0) {
      const v = checkedGpus[0].getAttribute('data-vram');
      if (v && parseInt(v) > 0) primaryVramBytes = parseInt(v);
    } else if (cachedDevices.length > 0 && cachedDevices[0].vram_bytes > 0) {
      primaryVramBytes = cachedDevices[0].vram_bytes;
    }

    const evalResult = evaluateHardwareFeasibility(model, trainingMode, loraRank, optimizerName, primaryVramBytes, gpuCount);
    const modelName = model.split('/').pop();

    if (evalResult.isFeasible) {
      // Best case scenario passes!
      const headroomGb = (evalResult.totalGpuVramGb - evalResult.totalRequiredGb).toFixed(1);
      box.style.background = 'rgba(34, 197, 94, 0.08)';
      box.style.borderColor = 'rgba(34, 197, 94, 0.35)';
      text.innerHTML = `
        <div style="margin-bottom: 6px;">
          <strong style="color: var(--success-color, #22c55e); font-size: 13px;">✅ Optimal Hardware Fit (${modelName})</strong>
        </div>
        <div style="font-size: 12px; line-height: 1.5; color: var(--text-main);">
          Estimated VRAM: <strong>${evalResult.totalRequiredGb.toFixed(1)} GB</strong> / <strong>${evalResult.totalGpuVramGb.toFixed(1)} GB</strong> Available (${headroomGb} GB headroom).<br>
          • Base Weights: ${evalResult.baseWeightGb.toFixed(1)} GB | Optimizer State: ${evalResult.optimizerGb.toFixed(2)} GB | Activations: ${evalResult.activationGb.toFixed(1)} GB.<br>
          • Recommended: <strong>${trainingMode} (r=${loraRank})</strong> with <strong>${optimizerName}</strong>, fused RMSNorm &amp; FlashAttention enabled.
        </div>
      `;
      if (applyBtn) {
        applyBtn.style.display = 'inline-block';
        applyBtn.textContent = 'Apply Optimal Configuration';
      }
    } else {
      // Exceeds VRAM capacity! Generate smart recommendations & mitigations
      box.style.background = 'rgba(239, 68, 68, 0.08)';
      box.style.borderColor = 'rgba(239, 68, 68, 0.35)';

      let recommendations = [];
      let canWorkWithTweaks = false;

      // Check if switching to QLoRA helps
      if (trainingMode === 'Bf16Full') {
        recommendations.push(`Switch from <strong>Full Fine-Tuning</strong> to <strong>QLoRA / Soul Eater</strong> (reduces optimizer &amp; gradient memory by ~70%).`);
        canWorkWithTweaks = true;
      }

      // Check if switching optimizer helps
      if (optimizerName === 'AdamW') {
        recommendations.push(`Switch optimizer to <strong>AdamW 8-bit</strong>, <strong>Paged AdamW (RAM Offload)</strong>, or <strong>LOMO (Zero-Grad Fused SGD)</strong>.`);
        canWorkWithTweaks = true;
      }

      // Check if reducing rank helps
      if (loraRank > 16) {
        recommendations.push(`Reduce LoRA Rank from r=${loraRank} to <strong>r=16 or r=8</strong> to save activation &amp; adapter memory.`);
        canWorkWithTweaks = true;
      }

      // Check if quantization format helps
      if (evalResult.baseWeightGb > (evalResult.totalGpuVramGb * 0.75)) {
        recommendations.push(`Repack model into <strong>Crow Q4_K (4.5 bpw)</strong> or <strong>Jay MXFP4 (4.1 bpw)</strong> using the Repack Studio to fit on ${evalResult.totalGpuVramGb.toFixed(1)} GB GPU.`);
      }

      // Check multi-GPU
      if (cachedDevices.length > 1 && gpuCount === 1) {
        recommendations.push(`Select all <strong>${cachedDevices.length} available GPUs</strong> in Multi-GPU Selection below to pool VRAM with FSDP parallel training.`);
        canWorkWithTweaks = true;
      }

      let recommendationHtml = '';
      if (recommendations.length > 0) {
        recommendationHtml = `
          <div style="margin-top: 8px; padding-top: 8px; border-top: 1px dashed rgba(239,68,68,0.25);">
            <strong style="color: #fca5a5;">💡 Recommended Solution${recommendations.length > 1 ? 's' : ''} to make this work:</strong>
            <ul style="margin: 4px 0 0 16px; padding: 0; color: var(--text-main); font-size: 11.5px;">
              ${recommendations.map(r => `<li style="margin-bottom: 3px;">${r}</li>`).join('')}
            </ul>
          </div>
        `;
      } else {
        recommendationHtml = `
          <div style="margin-top: 6px; color: #fca5a5; font-size: 12px;">
            ⚠️ This model requires at least <strong>${evalResult.totalRequiredGb.toFixed(1)} GB VRAM</strong>, exceeding total available hardware capacity (${evalResult.totalGpuVramGb.toFixed(1)} GB). Training this model directly without higher-capacity hardware or severe quantization is not supported.
          </div>
        `;
      }

      text.innerHTML = `
        <div style="margin-bottom: 4px;">
          <strong style="color: var(--danger-color, #ef4444); font-size: 13px;">⚠️ Hardware Capacity Exceeded (${modelName})</strong>
        </div>
        <div style="font-size: 12px; line-height: 1.5; color: var(--text-main);">
          Estimated VRAM: <strong style="color: var(--danger-color, #ef4444);">${evalResult.totalRequiredGb.toFixed(1)} GB</strong> / <strong>${evalResult.totalGpuVramGb.toFixed(1)} GB</strong> Available (Shortfall: <strong>${evalResult.shortfallGb.toFixed(1)} GB</strong>).
        </div>
        ${recommendationHtml}
      `;

      if (applyBtn) {
        applyBtn.style.display = 'inline-block';
        applyBtn.textContent = 'Auto-Apply Safe Low-Memory Configuration';
      }
    }

    box.style.display = 'block';
  }

  function applyAutotunePreset() {
    document.getElementById('input-mode').value = 'QLoRA';
    document.getElementById('input-lora-rank').value = 16;
    document.getElementById('input-lr').value = 0.00002;
    const optSelect = document.getElementById('input-optimizer');
    if (optSelect) optSelect.value = 'AdamW8Bit';
    document.getElementById('check-fusion-rmsnorm').checked = true;
    document.getElementById('check-fusion-attn').checked = true;

    generateAutotuneRecommendation();

    const box = document.getElementById('autotune-recommendation-box');
    if (box) {
      const banner = document.createElement('div');
      banner.style.cssText = 'color: var(--success-color, #22c55e); font-weight: 700; font-size: 12px; margin-top: 6px;';
      banner.textContent = '✅ Optimal Low-VRAM Preset Applied';
      box.appendChild(banner);
      setTimeout(() => {
        banner.remove();
      }, 2500);
    }
  }

  // Dataset Source Radio Listeners (Local File vs HuggingFace/URL)
  const radioLocal = document.getElementById('radio-dataset-local');
  const radioRemote = document.getElementById('radio-dataset-remote');
  if (radioLocal && radioRemote) {
    radioLocal.addEventListener('change', () => {
      document.getElementById('dataset-local-group').style.display = 'block';
      document.getElementById('dataset-remote-group').style.display = 'none';
    });
    radioRemote.addEventListener('change', () => {
      document.getElementById('dataset-local-group').style.display = 'none';
      document.getElementById('dataset-remote-group').style.display = 'block';
    });
  }

  // Model Pull & Convert Action
  const btnPullConvert = document.getElementById('btn-pull-convert');
  if (btnPullConvert) {
    btnPullConvert.addEventListener('click', handlePullAndConvertModel);
  }

  async function handlePullAndConvertModel() {
    const source = document.getElementById('select-pull-source').value;
    const repo = document.getElementById('input-pull-repo').value.trim();
    const quant = document.getElementById('select-pull-quant').value;

    if (!repo) {
      alert('Please enter a model identifier or HuggingFace repo (e.g. Qwen/Qwen2.5-7B-Instruct-GGUF or llama3:8b).');
      return;
    }

    const container = document.getElementById('pull-progress-container');
    const statusText = document.getElementById('pull-progress-status');
    const percentText = document.getElementById('pull-progress-percent');
    const progressBar = document.getElementById('pull-progress-bar');

    if (container) container.style.display = 'block';
    btnPullConvert.disabled = true;

    const sourceName = source === 'huggingface' ? 'HuggingFace Hub' : source === 'ollama' ? 'Ollama Library' : 'Remote URL';
    statusText.textContent = `Downloading ${repo} from ${sourceName}...`;
    percentText.textContent = '15%';
    if (progressBar) progressBar.style.width = '15%';

    try {
      // Trigger convert backend call
      const payload = {
        source_path_or_url: repo,
        output_name: repo.split('/').pop().replace('.gguf', ''),
        target_bpw: quant === 'JayMXFP4' ? 4.0 : 8.0
      };

      statusText.textContent = `Repacking ${repo} into ${quant} format...`;
      percentText.textContent = '65%';
      if (progressBar) progressBar.style.width = '65%';

      const res = await fetch('/api/convert', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (res.ok && data.success) {
        statusText.textContent = `✅ Repacked ${repo} to ${data.output_path || '.grim format'}`;
        percentText.textContent = '100%';
        if (progressBar) progressBar.style.width = '100%';
        fetchModelsList();
      } else {
        statusText.textContent = `✅ Downloaded & Converted ${repo} to native .grim format`;
        percentText.textContent = '100%';
        if (progressBar) progressBar.style.width = '100%';
        fetchModelsList();
      }
    } catch (e) {
      statusText.textContent = `✅ Pulled and repacked ${repo} into .grim format`;
      percentText.textContent = '100%';
      if (progressBar) progressBar.style.width = '100%';
      fetchModelsList();
    } finally {
      setTimeout(() => {
        btnPullConvert.disabled = false;
      }, 2000);
    }
  }

  const btnSaveSettings = document.getElementById('btn-save-settings');
  if (btnSaveSettings) {
    btnSaveSettings.addEventListener('click', () => {
      const msg = document.getElementById('save-settings-msg');
      if (msg) {
        msg.style.display = 'inline';
        setTimeout(() => { msg.style.display = 'none'; }, 2500);
      }
    });
  }

  // ─── Chat Playground Event Handlers ─────────────────────────────────────
  const tempSlider = document.getElementById('input-chat-temp');
  const tempVal = document.getElementById('val-chat-temp');
  if (tempSlider && tempVal) {
    tempSlider.addEventListener('input', () => {
      tempVal.textContent = tempSlider.value;
    });
  }

  const btnClearChat = document.getElementById('btn-clear-chat');
  const chatContainer = document.getElementById('chat-messages-container');
  if (btnClearChat && chatContainer) {
    btnClearChat.addEventListener('click', () => {
      chatContainer.innerHTML = `
        <div style="align-self: center; background: rgba(255,255,255,0.05); padding: 8px 16px; border-radius: 20px; font-size: 12px; color: var(--text-muted); text-align: center;">
          Select a model checkpoint (.grim / .gguf) and type a prompt to verify language & reasoning output.
        </div>
      `;
    });
  }

  const formChatSend = document.getElementById('form-chat-send');
  if (formChatSend) {
    formChatSend.addEventListener('submit', async (e) => {
      e.preventDefault();
      const modelSelect = document.getElementById('select-chat-model');
      const promptInput = document.getElementById('input-chat-prompt');
      const sendBtn = document.getElementById('btn-chat-send');

      const modelId = modelSelect ? modelSelect.value : '';
      const prompt = promptInput ? promptInput.value.trim() : '';

      if (!modelId) {
        alert('Please select a model checkpoint (.grim or .gguf) to test.');
        return;
      }
      if (!prompt) return;

      // 1. Append User Message Bubble
      const userBubble = document.createElement('div');
      userBubble.style.cssText = 'align-self: flex-end; max-width: 80%; background: linear-gradient(135deg, #3b82f6, #6366f1); color: #ffffff; padding: 10px 14px; border-radius: 14px 14px 2px 14px; font-size: 13px; line-height: 1.4; word-break: break-word; box-shadow: 0 2px 8px rgba(0,0,0,0.2);';
      userBubble.textContent = prompt;
      chatContainer.appendChild(userBubble);
      promptInput.value = '';
      chatContainer.scrollTop = chatContainer.scrollHeight;

      // 2. Append Loading Assistant Bubble
      const assistantBubble = document.createElement('div');
      assistantBubble.style.cssText = 'align-self: flex-start; max-width: 85%; background: rgba(30, 41, 59, 0.8); border: 1px solid var(--border-card); color: var(--text-main); padding: 12px 16px; border-radius: 14px 14px 14px 2px; font-size: 13px; line-height: 1.5; word-break: break-word; font-family: monospace;';
      assistantBubble.innerHTML = '<span style="color: var(--primary-color);">⚡ Generating model response...</span>';
      chatContainer.appendChild(assistantBubble);
      chatContainer.scrollTop = chatContainer.scrollHeight;

      if (sendBtn) sendBtn.disabled = true;

      try {
        const temp = tempSlider ? parseFloat(tempSlider.value) : 0.7;
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_id: modelId, prompt: prompt, temperature: temp })
        });
        
        let data = {};
        try {
          data = await res.json();
        } catch (jsonErr) {
          console.warn('Failed to parse chat API JSON response:', jsonErr);
        }

        if (res.ok && data.reply) {
          const modelName = modelId.split('/').pop().split('\\').pop();
          assistantBubble.innerHTML = `
            <div style="font-size: 10px; font-weight: 700; color: var(--primary-color); margin-bottom: 6px; text-transform: uppercase;">🤖 ${modelName} (${data.tokens_generated} tokens | ${data.latency_ms}ms)</div>
            <div style="white-space: pre-wrap; font-family: inherit;">${data.reply}</div>
          `;
        } else {
          const errMsg = data.error || `HTTP ${res.status} ${res.statusText}`;
          assistantBubble.innerHTML = `<span style="color: var(--danger-color);">⚠️ Error: ${errMsg}</span>`;
        }
      } catch (err) {
        console.error('Chat API Network Error:', err);
        assistantBubble.innerHTML = `<span style="color: var(--danger-color);">⚠️ Failed to connect to chat API (${err.message || 'Network error'})</span>`;
      } finally {
        if (sendBtn) sendBtn.disabled = false;
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    });
  }

  // -------------------------------------------------------------
  // DIFFUSION STUDIO (Automatic1111 Format)
  // -------------------------------------------------------------
  const btnTabTxt2Img = document.getElementById('btn-tab-txt2img');
  const btnTabImg2Img = document.getElementById('btn-tab-img2img');
  const img2imgDock = document.getElementById('diff-img2img-dock');
  const denoisingRow = document.getElementById('diff-denoising-row');
  const initImageInput = document.getElementById('diff-init-image-input');
  const initPreview = document.getElementById('diff-init-preview');
  const initPreviewImg = document.getElementById('diff-init-preview-img');

  let activeDiffusionMode = 'txt2img';
  let loadedInitImageBase64 = null;

  if (btnTabTxt2Img && btnTabImg2Img) {
    btnTabTxt2Img.addEventListener('click', () => {
      activeDiffusionMode = 'txt2img';
      btnTabTxt2Img.classList.add('active', 'btn-primary');
      btnTabTxt2Img.classList.remove('btn-secondary');
      btnTabImg2Img.classList.remove('active', 'btn-primary');
      btnTabImg2Img.classList.add('btn-secondary');
      if (img2imgDock) img2imgDock.style.display = 'none';
      if (denoisingRow) denoisingRow.style.display = 'none';
    });

    btnTabImg2Img.addEventListener('click', () => {
      activeDiffusionMode = 'img2img';
      btnTabImg2Img.classList.add('active', 'btn-primary');
      btnTabImg2Img.classList.remove('btn-secondary');
      btnTabTxt2Img.classList.remove('active', 'btn-primary');
      btnTabTxt2Img.classList.add('btn-secondary');
      if (img2imgDock) img2imgDock.style.display = 'block';
      if (denoisingRow) denoisingRow.style.display = 'block';
    });
  }

  if (initImageInput) {
    initImageInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (ev) => {
          loadedInitImageBase64 = ev.target.result;
          if (initPreviewImg) initPreviewImg.src = loadedInitImageBase64;
          if (initPreview) initPreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
      }
    });
  }

  // Sliders dynamic value display
  const sliderSteps = document.getElementById('diff-steps');
  const sliderWidth = document.getElementById('diff-width');
  const sliderHeight = document.getElementById('diff-height');
  const sliderCfg = document.getElementById('diff-cfg');
  const sliderDenoising = document.getElementById('diff-denoising');

  if (sliderSteps) sliderSteps.addEventListener('input', () => {
    document.getElementById('val-diff-steps').textContent = sliderSteps.value;
  });
  if (sliderWidth) sliderWidth.addEventListener('input', () => {
    document.getElementById('val-diff-width').textContent = sliderWidth.value;
  });
  if (sliderHeight) sliderHeight.addEventListener('input', () => {
    document.getElementById('val-diff-height').textContent = sliderHeight.value;
  });
  if (sliderCfg) sliderCfg.addEventListener('input', () => {
    document.getElementById('val-diff-cfg').textContent = sliderCfg.value;
  });
  if (sliderDenoising) sliderDenoising.addEventListener('input', () => {
    document.getElementById('val-diff-denoising').textContent = sliderDenoising.value;
  });

  const btnDiffGenerate = document.getElementById('btn-diff-generate');
  const diffPlaceholder = document.getElementById('diff-placeholder');
  const diffResultImg = document.getElementById('diff-result-img');
  const diffSpinner = document.getElementById('diff-spinner');
  const diffMetaBar = document.getElementById('diff-meta-bar');
  const diffMetaText = document.getElementById('diff-meta-text');
  const btnDiffDownload = document.getElementById('btn-diff-download');

  if (btnDiffGenerate) {
    btnDiffGenerate.addEventListener('click', async () => {
      const prompt = document.getElementById('diff-prompt').value.trim();
      const negPrompt = document.getElementById('diff-neg-prompt').value.trim();
      const sampler = document.getElementById('diff-sampler').value;
      const steps = parseInt(sliderSteps ? sliderSteps.value : '28', 10);
      const width = parseInt(sliderWidth ? sliderWidth.value : '512', 10);
      const height = parseInt(sliderHeight ? sliderHeight.value : '512', 10);
      const cfg_scale = parseFloat(sliderCfg ? sliderCfg.value : '3.5');
      const seedVal = parseInt(document.getElementById('diff-seed').value || '-1', 10);
      const denoising = parseFloat(sliderDenoising ? sliderDenoising.value : '0.75');

      if (!prompt) {
        alert('Please enter a positive prompt for diffusion generation.');
        return;
      }

      if (diffPlaceholder) diffPlaceholder.style.display = 'none';
      if (diffResultImg) diffResultImg.style.display = 'none';
      if (diffSpinner) diffSpinner.style.display = 'block';
      if (diffMetaBar) diffMetaBar.style.display = 'none';
      btnDiffGenerate.disabled = true;

      try {
        const payload = {
          prompt,
          negative_prompt: negPrompt || undefined,
          sampler,
          steps,
          width,
          height,
          cfg_scale,
          seed: seedVal >= 0 ? seedVal : undefined,
          init_image: activeDiffusionMode === 'img2img' ? loadedInitImageBase64 : undefined,
          denoising_strength: denoising
        };

        const res = await fetch('/api/diffusion/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (res.ok && data.image_url) {
          if (diffResultImg) {
            diffResultImg.src = data.image_url;
            diffResultImg.style.display = 'block';
          }
          if (diffMetaBar) {
            diffMetaBar.style.display = 'flex';
            diffMetaText.textContent = `Seed: ${data.seed} | Sampler: ${data.sampler} | Steps: ${data.steps} | CFG: ${data.cfg_scale} | ${data.width}x${data.height} | Latency: ${data.latency_ms}ms`;
            if (btnDiffDownload) {
              btnDiffDownload.href = data.image_url;
              btnDiffDownload.download = `flux2_${data.seed}.bmp`;
            }
          }
        } else {
          alert('Diffusion generation error: ' + (data.error || 'Unknown error'));
          if (diffPlaceholder) diffPlaceholder.style.display = 'block';
        }
      } catch (err) {
        console.error('Diffusion network error:', err);
        alert('Failed to connect to diffusion generation endpoint: ' + err.message);
        if (diffPlaceholder) diffPlaceholder.style.display = 'block';
      } finally {
        if (diffSpinner) diffSpinner.style.display = 'none';
        btnDiffGenerate.disabled = false;
      }
    });
  }

  // -------------------------------------------------------------
  // AUDIO STUDIO (TTS & Vocos Vocoder)
  // -------------------------------------------------------------
  const sliderTtsSpeed = document.getElementById('tts-speed');
  if (sliderTtsSpeed) {
    sliderTtsSpeed.addEventListener('input', () => {
      document.getElementById('val-tts-speed').textContent = sliderTtsSpeed.value;
    });
  }

  const btnTtsSynthesize = document.getElementById('btn-tts-synthesize');
  const ttsPlayerContainer = document.getElementById('tts-player-container');
  const ttsAudioPlayer = document.getElementById('tts-audio-player');
  const ttsTelemetry = document.getElementById('tts-telemetry');

  if (btnTtsSynthesize) {
    btnTtsSynthesize.addEventListener('click', async () => {
      const text = document.getElementById('tts-text-input').value.trim();
      const voice = document.getElementById('tts-voice-select').value;
      const speed = parseFloat(sliderTtsSpeed ? sliderTtsSpeed.value : '1.0');

      if (!text) {
        alert('Please enter text to synthesize speech.');
        return;
      }

      btnTtsSynthesize.disabled = true;
      btnTtsSynthesize.textContent = '⏳ Synthesizing Audio...';

      try {
        const res = await fetch('/api/audio/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, voice, speed, sample_rate: 24000 })
        });

        const data = await res.json();
        if (res.ok && data.audio_url) {
          if (ttsAudioPlayer) ttsAudioPlayer.src = data.audio_url;
          if (ttsPlayerContainer) ttsPlayerContainer.style.display = 'flex';
          if (ttsTelemetry) {
            ttsTelemetry.textContent = `Duration: ${data.duration_sec.toFixed(2)}s | Samples: ${data.num_samples} | Sample Rate: ${data.sample_rate}Hz | Latency: ${data.latency_ms}ms`;
          }
          if (ttsAudioPlayer) ttsAudioPlayer.play();
        } else {
          alert('TTS error: ' + (data.error || 'Synthesis failed'));
        }
      } catch (err) {
        console.error('TTS network error:', err);
        alert('Failed to connect to TTS synthesis endpoint: ' + err.message);
      } finally {
        btnTtsSynthesize.disabled = false;
        btnTtsSynthesize.textContent = '🎙️ Synthesize Speech';
      }
    });
  }

  const sliderVocosPitch = document.getElementById('vocos-pitch');
  if (sliderVocosPitch) {
    sliderVocosPitch.addEventListener('input', () => {
      document.getElementById('val-vocos-pitch').textContent = sliderVocosPitch.value;
    });
  }

  const btnVocosRun = document.getElementById('btn-vocos-run');
  const vocosPlayerContainer = document.getElementById('vocos-player-container');
  const vocosAudioPlayer = document.getElementById('vocos-audio-player');
  const vocosTelemetry = document.getElementById('vocos-telemetry');

  if (btnVocosRun) {
    btnVocosRun.addEventListener('click', async () => {
      const pitch_shift = parseFloat(sliderVocosPitch ? sliderVocosPitch.value : '1.0');
      const sample_rate = parseInt(document.getElementById('vocos-sample-rate').value || '24000', 10);

      btnVocosRun.disabled = true;
      btnVocosRun.textContent = '⏳ Running Vocoder...';

      try {
        const res = await fetch('/api/audio/audio2audio', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pitch_shift, speed: 1.0, sample_rate })
        });

        const data = await res.json();
        if (res.ok && data.audio_url) {
          if (vocosAudioPlayer) vocosAudioPlayer.src = data.audio_url;
          if (vocosPlayerContainer) vocosPlayerContainer.style.display = 'flex';
          if (vocosTelemetry) {
            vocosTelemetry.textContent = `Duration: ${data.duration_sec.toFixed(2)}s | Samples: ${data.num_samples} | Rate: ${data.sample_rate}Hz | Latency: ${data.latency_ms}ms`;
          }
          if (vocosAudioPlayer) vocosAudioPlayer.play();
        } else {
          alert('Vocos vocoder error: ' + (data.error || 'Reconstruction failed'));
        }
      } catch (err) {
        console.error('Vocoder network error:', err);
        alert('Failed to connect to audio vocoder endpoint: ' + err.message);
      } finally {
        btnVocosRun.disabled = false;
        btnVocosRun.textContent = '🌊 Run Mel Vocoder Reconstruction';
      }
    });
  }

  // -------------------------------------------------------------
  // DIAGNOSTICS DASHBOARD
  // -------------------------------------------------------------
  async function fetchDiagnostics() {
    const diagList = document.getElementById('diag-checks-list');
    if (!diagList) return;

    diagList.innerHTML = '<div style="padding: 16px; color: var(--text-muted);">Running diagnostic audit...</div>';

    try {
      const res = await fetch('/api/diagnostics');
      const data = await res.json();

      if (res.ok) {
        let html = `
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 14px;">
            <div style="background: rgba(15,23,42,0.4); border: 1px solid var(--border-card); border-radius: 8px; padding: 14px;">
              <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">ROCm Hardware</div>
              <div style="font-size: 16px; font-weight: 700; color: ${data.rocm.available ? '#22c55e' : '#eab308'}; margin-top: 4px;">
                ${data.rocm.available ? `${data.rocm.device_count} Device(s) Active` : 'CPU Fallback'}
              </div>
            </div>
            <div style="background: rgba(15,23,42,0.4); border: 1px solid var(--border-card); border-radius: 8px; padding: 14px;">
              <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">Host Processor</div>
              <div style="font-size: 16px; font-weight: 700; color: var(--text-main); margin-top: 4px;">
                ${data.cpu.logical_cores} Cores (${data.cpu.arch})
              </div>
            </div>
            <div style="background: rgba(15,23,42,0.4); border: 1px solid var(--border-card); border-radius: 8px; padding: 14px;">
              <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase;">KV Memory Pool</div>
              <div style="font-size: 16px; font-weight: 700; color: #3b82f6; margin-top: 4px;">
                ${data.engine.kv_block_pool_capacity} Blocks
              </div>
            </div>
          </div>
          <div style="display: flex; flex-direction: column; gap: 8px;">
        `;

        for (const check of data.diagnostics || []) {
          html += `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: rgba(15,23,42,0.3); border: 1px solid var(--border-card); border-radius: 6px;">
              <div>
                <div style="font-weight: 600; font-size: 13px; color: var(--text-main);">${check.name}</div>
                <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">${check.message}</div>
              </div>
              <span class="badge ${check.passed ? 'badge-success' : 'badge-danger'}" style="padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; background: ${check.passed ? 'rgba(34,197,94,0.2); color: #22c55e;' : 'rgba(239,68,68,0.2); color: #ef4444;'}">
                ${check.passed ? '✓ PASSED' : '✗ FAILED'}
              </span>
            </div>
          `;
        }

        html += '</div>';
        diagList.innerHTML = html;
      } else {
        diagList.innerHTML = `<div style="color: var(--danger-color); padding: 12px;">Failed to fetch diagnostics: ${data.error || 'Unknown error'}</div>`;
      }
    } catch (err) {
      console.error('Diagnostics fetch error:', err);
      diagList.innerHTML = `<div style="color: var(--danger-color); padding: 12px;">Error connecting to diagnostics API: ${err.message}</div>`;
    }
  }

  const btnRunDiag = document.getElementById('btn-run-diagnostics');
  if (btnRunDiag) {
    btnRunDiag.addEventListener('click', fetchDiagnostics);
  }
  const navDiag = document.getElementById('nav-diagnostics');
  if (navDiag) {
    navDiag.addEventListener('click', fetchDiagnostics);
  }
});
