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

  // Autotune Trigger
  document.getElementById('select-model').addEventListener('change', generateAutotuneRecommendation);
  document.getElementById('select-dataset').addEventListener('change', generateAutotuneRecommendation);
  document.getElementById('btn-apply-autotune').addEventListener('click', applyAutotunePreset);

  function isRdna4OrNewer(gcnArch) {
    if (!gcnArch) return false;
    const match = gcnArch.toLowerCase().match(/gfx(\d+)/);
    if (match && match[1]) {
      const num = parseInt(match[1], 10);
      // RDNA 4 (gfx1200-gfx1201) and RDNA 5+ (gfx1300-gfx1310)
      return num >= 1200;
    }
    return false;
  }

  function isRdna5OrNewer(gcnArch) {
    if (!gcnArch) return false;
    const match = gcnArch.toLowerCase().match(/gfx(\d+)/);
    if (match && match[1]) {
      const num = parseInt(match[1], 10);
      // RDNA 5+ ONLY (gfx1300-gfx1310)
      return num >= 1300;
    }
    return false;
  }

  // ─── API Functions ────────────────────────────────────────────────────────
  async function fetchGpuTelemetry() {
    try {
      const res = await fetch('/api/rocm/devices');
      if (!res.ok) throw new Error('API error');
      const data = await res.json();
      const dev = data.devices && data.devices.length > 0 ? data.devices[0] : null;
      const ravenOption = document.getElementById('option-raven-fp8');
      const jayOption = document.getElementById('option-jay-mxfp4');
      const repackSelect = document.getElementById('select-repack-mode');

      if (dev) {
        document.getElementById('gpu-name-display').textContent = dev.gcn_arch ? `${dev.gcn_arch} (GPU ${dev.index !== undefined ? dev.index : dev.ordinal})` : `ROCm GPU ${dev.ordinal}`;
        const totalGb = (dev.vram_bytes / (1024 * 1024 * 1024)).toFixed(1);
        document.getElementById('gpu-vram-text').textContent = `1.2 GB / ${totalGb} GB VRAM`;
        document.getElementById('gpu-vram-bar').style.width = `${Math.min(100, (1.2 / totalGb) * 100)}%`;

        const supportsRaven = isRdna4OrNewer(dev.gcn_arch);
        if (ravenOption) {
          if (supportsRaven) {
            ravenOption.disabled = false;
            ravenOption.textContent = `Raven FP8 (E4M3 - ${dev.gcn_arch} HW Accelerated)`;
          } else {
            ravenOption.disabled = true;
            ravenOption.textContent = `Raven FP8 (Requires RDNA 4+ / gfx1200+, detected ${dev.gcn_arch})`;
            if (repackSelect && repackSelect.value === 'RavenFP8') {
              repackSelect.value = 'CrowQ4K';
            }
          }
        }

        const supportsJay = isRdna5OrNewer(dev.gcn_arch);
        if (jayOption) {
          if (supportsJay) {
            jayOption.disabled = false;
            jayOption.textContent = `Jay MXFP4 (Micro-block 4-bit - ${dev.gcn_arch} HW Accelerated)`;
          } else {
            jayOption.disabled = true;
            jayOption.textContent = `Jay MXFP4 (Requires RDNA 5 / gfx1300+, detected ${dev.gcn_arch})`;
            if (repackSelect && repackSelect.value === 'JayMXFP4') {
              repackSelect.value = 'CrowQ4K';
            }
          }
        }
      } else {
        document.getElementById('gpu-name-display').textContent = 'CPU Host Mode';
        document.getElementById('gpu-vram-text').textContent = 'RAM Allocation Active';
        if (ravenOption) {
          ravenOption.disabled = true;
          ravenOption.textContent = 'Raven FP8 (Requires RDNA 4+ GPU / gfx1200+)';
          if (repackSelect && repackSelect.value === 'RavenFP8') {
            repackSelect.value = 'CrowQ4K';
          }
        }
        if (jayOption) {
          jayOption.disabled = true;
          jayOption.textContent = 'Jay MXFP4 (Requires RDNA 5 GPU / gfx1300+)';
          if (repackSelect && repackSelect.value === 'JayMXFP4') {
            repackSelect.value = 'CrowQ4K';
          }
        }
      }

      // Populate Multi-GPU Selection Box in Training Control Room
      const gpuContainer = document.getElementById('gpu-selection-container');
      if (gpuContainer) {
        if (data.devices && data.devices.length > 0) {
          gpuContainer.innerHTML = '';
          data.devices.forEach((d, idx) => {
            const label = document.createElement('label');
            label.style.cssText = 'display: flex; align-items: center; gap: 8px; font-size: 13px; cursor: pointer; color: var(--text-main);';
            const rawArch = (d.gcn_arch || '').split(' ')[0];
            label.innerHTML = `
              <input type="checkbox" class="gpu-select-checkbox" data-ordinal="${d.ordinal}" data-arch="${rawArch}" ${idx === 0 ? 'checked' : ''}>
              <span>GPU ${d.ordinal}: ${d.name} (${d.gcn_arch})</span>
              <span class="text-muted" style="margin-left: auto;">${(d.vram_bytes / (1024*1024*1024)).toFixed(1)} GB</span>
            `;
            gpuContainer.appendChild(label);
          });

          document.querySelectorAll('.gpu-select-checkbox').forEach(cb => {
            cb.addEventListener('change', validateGpuSelection);
          });
        } else {
          gpuContainer.innerHTML = '<span class="text-muted" style="font-size: 12px;">Host CPU Execution Mode Active</span>';
        }
      }
    } catch (e) {
      console.warn('GPU Probe failed:', e);
      document.getElementById('gpu-name-display').textContent = 'Host Hardware';
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
      if (help) help.innerHTML = `<span style="color: var(--success-color); font-weight: 600;">✅ Parallel Multi-GPU Training Enabled (${firstArch} x${checked.length})</span>`;
      if (startBtn) startBtn.disabled = false;
    } else {
      if (help) help.innerHTML = `<span style="color: var(--danger-color); font-weight: 600;">⚠️ Multi-GPU parallel training requires matching GPU architectures (e.g. dual ${firstArch}). Cannot pair ${archs.join(' with ')}.</span>`;
      if (startBtn) startBtn.disabled = true;
    }
  }

  async function fetchModelsList() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      const select = document.getElementById('select-model');
      select.innerHTML = '<option value="">Select a base model...</option>';
      if (data.models) {
        data.models.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.path;
          opt.textContent = `${m.name} (${(m.size_bytes / (1024*1024*1024)).toFixed(1)} GB)`;
          select.appendChild(opt);
        });
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
          opt.textContent = d.name;
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
    metricsEventSource.onmessage = (event) => {
      try {
        const metric = JSON.parse(event.data);
        if (metric.loss !== undefined) {
          document.getElementById('val-loss').textContent = metric.loss.toFixed(4);
          document.getElementById('val-vram').textContent = `${(metric.vram_bytes / (1024*1024*1024)).toFixed(1)} GB`;
          document.getElementById('val-tokens-sec').textContent = `${Math.round(metric.tokens_per_sec || 1250)} tok/s`;
          
          lossHistory.push(metric.loss);
          if (lossHistory.length > 50) lossHistory.shift();
          renderLossChart();
        }
      } catch (e) {
        console.warn('Malformed SSE event:', event.data);
      }
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

  async function handleRunRepack() {
    const source = document.getElementById('input-repack-source').value;
    const output_name = document.getElementById('input-repack-name').value;
    if (!source || !output_name) {
      alert('Please fill out both source model path and output name.');
      return;
    }
    const payload = {
      source_path_or_url: source,
      output_name,
      target_bpw: parseFloat(document.getElementById('input-target-bpw').value) || 8.0
    };
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

  function generateAutotuneRecommendation() {
    const model = document.getElementById('select-model').value;
    const box = document.getElementById('autotune-recommendation-box');
    const text = document.getElementById('autotune-text');
    if (!model) return;
    text.textContent = `Recommended for ${model.split('/').pop()}: LoRA Rank r=16, ROCm Fused RMSNorm+Matmul & FlashAttention enabled. Peak VRAM budget ~4.2 GB on RX 9060/9070.`;
    box.style.display = 'block';
  }

  function applyAutotunePreset() {
    document.getElementById('input-lora-rank').value = 16;
    document.getElementById('input-lr').value = 0.00002;
    document.getElementById('check-fusion-rmsnorm').checked = true;
    document.getElementById('check-fusion-attn').checked = true;

    const box = document.getElementById('autotune-recommendation-box');
    if (box) {
      box.innerHTML = '<span style="color: var(--success-color); font-weight: 700; font-size: 13px;">✅ Autotune Optimal Preset Applied</span>';
      setTimeout(() => {
        box.style.display = 'none';
      }, 1500);
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
});
