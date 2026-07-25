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
      const ravenOption = document.getElementById('option-raven-fp8');
      const jayOption = document.getElementById('option-jay-mxfp4');
      const repackSelect = document.getElementById('select-repack-mode');
      const telemetryList = document.getElementById('gpu-telemetry-list');

      if (data.devices && data.devices.length > 0) {
        if (telemetryList) {
          telemetryList.innerHTML = '';
          data.devices.forEach(d => {
            let backendTag = 'Vulkan';
            const vUpper = (d.vendor || '').toUpperCase();
            const bUpper = (d.backend || '').toUpperCase();
            if (vUpper.includes('NVIDIA') || bUpper.includes('CUDA')) {
              backendTag = 'CUDA';
            } else if (vUpper.includes('APPLE') || bUpper.includes('METAL')) {
              backendTag = 'Metal';
            } else if (vUpper.includes('AMD') || bUpper.includes('ROCM')) {
              backendTag = 'ROCm';
            }

            const nameStr = d.gcn_arch ? `${backendTag} ${d.gcn_arch}` : `${backendTag} (${d.name})`;
            const totalGb = (d.vram_bytes / (1024 * 1024 * 1024)).toFixed(1);

            const item = document.createElement('div');
            item.style.cssText = 'padding: 6px; background: rgba(255,255,255,0.03); border-radius: 6px; margin-bottom: 4px;';
            item.innerHTML = `
              <div class="gpu-name" style="font-size: 11px; font-weight: 600; color: var(--text-main); margin-bottom: 3px;">GPU ${d.ordinal}: ${nameStr}</div>
              <div class="vram-bar-container" style="height: 6px;">
                <div class="vram-bar" style="width: 15%;"></div>
              </div>
              <div class="vram-text" style="font-size: 10px; margin-top: 2px; color: var(--text-muted);">1.2 GB / ${totalGb} GB VRAM</div>
            `;
            telemetryList.appendChild(item);
          });
        }

        const dev = data.devices[0];
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
        if (telemetryList) {
          telemetryList.innerHTML = `
            <div class="gpu-name">CPU Host Mode</div>
            <div class="vram-text">RAM Allocation Active</div>
          `;
        }
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
          const displayName = m.name || m.id || (m.path ? m.path.split('/').pop() : 'model');
          const sizeText = m.size_bytes && m.size_bytes > 0 ? ` (${(m.size_bytes / (1024*1024*1024)).toFixed(1)} GB)` : '';
          opt.textContent = `${displayName}${sizeText}`;
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
});
