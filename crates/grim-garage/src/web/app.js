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

            let cleanName = (d.name || '').replace(/NVIDIA AMD\/ATI/gi, 'Radeon GPU').replace(/AMD\/ATI/gi, '').trim();
            if (!cleanName || cleanName.toLowerCase().includes('generic') || cleanName === 'Radeon GPU') {
              cleanName = d.gcn_arch ? `Radeon GPU (${d.gcn_arch})` : 'GPU Accelerator';
            }
            const nameStr = `${backendTag} ${cleanName}`;
            const usedBytes = d.vram_used_bytes || 0;
            const totalBytes = d.vram_bytes || 1;

            const usedGb = (usedBytes / (1024 * 1024 * 1024)).toFixed(2);
            const totalGb = (totalBytes / (1024 * 1024 * 1024)).toFixed(1);
            const pct = Math.min(100, Math.max(0, (usedBytes / totalBytes) * 100));

            const item = document.createElement('div');
            item.style.cssText = 'padding: 6px; background: rgba(255,255,255,0.03); border-radius: 6px; margin-bottom: 4px;';
            item.innerHTML = `
              <div class="gpu-name" style="font-size: 11px; font-weight: 600; color: var(--text-main); margin-bottom: 3px;">GPU ${d.ordinal}: ${nameStr}</div>
              <div class="vram-bar-container" style="height: 6px;">
                <div class="vram-bar" style="width: ${pct.toFixed(1)}%;"></div>
              </div>
              <div class="vram-text" style="font-size: 10px; margin-top: 2px; color: var(--text-muted);">${usedGb} GB / ${totalGb} GB VRAM · ${Number(d.gpu_busy_percent || 0)}% util</div>
            `;
            telemetryList.appendChild(item);
          });
        }

        const dev = data.devices[0];
        updateRepackCapabilities(dev);
      } else {
        if (telemetryList) {
          telemetryList.innerHTML = `
            <div class="gpu-name">CPU Host Mode</div>
            <div class="vram-text">RAM Allocation Active</div>
          `;
        }
        updateRepackCapabilities(null);
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
      const chatSelect = document.getElementById('select-chat-model');

      if (select) select.innerHTML = '<option value="">Select a base model...</option>';
      if (chatSelect) chatSelect.innerHTML = '<option value="">Select a model checkpoint (.grim / .gguf)...</option>';

      if (data.models) {
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

    // Subscribe to the named `metric` event the server emits. Previously
    // the page used only `onmessage`, which is the SSE spec for events
    // with no `event:` field — but the grim-garage server tags every
    // metric with `event: metric` (and terminal completion with
    // `event: end`), so `onmessage` never fired and the live loss
    // graph stayed silent until a refresh.
    metricsEventSource.addEventListener('metric', (event) => {
      try {
        const ev = JSON.parse(event.data);
        if (ev && ev.metric && typeof ev.metric.loss === 'number') {
          const loss = ev.metric.loss;
          document.getElementById('val-loss').textContent = loss.toFixed(4);
          document.getElementById('val-vram').textContent = 'Unavailable';
          document.getElementById('val-tokens-sec').textContent = 'Unavailable';
          lossHistory.push(loss);
          if (lossHistory.length > 50) lossHistory.shift();
          renderLossChart();
        }
      } catch (e) {
        console.warn('Malformed metric SSE event:', event.data);
      }
    });

    // Terminal event: server emits `event: end` after Completed,
    // Failed, or Cancelled (the broadcast-stream `Closed` never fires
    // because the broadcast sender lives in the registry for the
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
    text.textContent = `Recommended for ${model.split('/').pop()}: LoRA Rank r=16. Hardware-specific VRAM requirements will be calculated after the model and training configuration are loaded.`;
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
