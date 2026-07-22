document.addEventListener('DOMContentLoaded', () => {
  const tabs = document.querySelectorAll('.nav-item');
  const pages = document.querySelectorAll('.tab-page');
  const tabTitle = document.getElementById('tab-title');
  const tabDesc = document.getElementById('tab-desc');

  const titles = {
    training: { title: 'Training Panel', desc: 'SFT QLoRA fine-tuning and parameter setup' },
    models: { title: 'Models & Bolt-Ons', desc: 'Manage base model files and attachable LoRA sidecars' },
    convert: { title: 'Convert Model', desc: 'Convert GGUF models or HuggingFace URLs to native .grim format via oxidizer' },
    jobs: { title: 'Training Jobs', desc: 'Active training progress and history' },
    devices: { title: 'ROCm Devices', desc: 'GPU accelerator discovery and memory status' },
  };

  // Theme Toggle Logic
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const savedTheme = localStorage.getItem('grim-theme') || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  applyTheme(savedTheme);

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      localStorage.setItem('grim-theme', next);
    });
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    if (themeToggleBtn) {
      themeToggleBtn.textContent = theme === 'dark' ? '🌙 Dark' : '☀️ Light';
    }
  }

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      pages.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const target = tab.dataset.tab;
      document.getElementById(`tab-${target}`).classList.add('active');

      if (titles[target]) {
        tabTitle.textContent = titles[target].title;
        tabDesc.textContent = titles[target].desc;
      }
    });
  });

  async function loadData() {
    try {
      // Fetch native .grim models (for Training Panel & Models & Bolt-Ons catalog)
      const modelsRes = await fetch('/api/models');
      if (modelsRes.ok) {
        const models = await modelsRes.json();
        populateSelect('model-select', models);
        populateSelect('bolton-model-select', models);
        renderModels(models);
      }

      // Fetch unconverted source models (GGUF, SafeTensors, BIN, FP16/8/4, MXFP4, NVFP4, BitsAndBytes) for Convert tab
      const convertibleRes = await fetch('/api/models/convertible');
      if (convertibleRes.ok) {
        const convertible = await convertibleRes.json();
        populateSelect('convert-source-select', convertible);
      }

      // Fetch datasets
      const datasetsRes = await fetch('/api/datasets');
      if (datasetsRes.ok) {
        const datasets = await datasetsRes.json();
        populateSelect('dataset-select', datasets);
      }

      // Fetch jobs
      const jobsRes = await fetch('/api/train/jobs');
      if (jobsRes.ok) {
        const jobs = await jobsRes.json();
        renderJobs(jobs);
      }

      // Fetch devices
      const devicesRes = await fetch('/api/rocm/devices');
      if (devicesRes.ok) {
        const devices = await devicesRes.json();
        renderDevices(devices);
      }
    } catch (err) {
      console.error('Error fetching garage data:', err);
    }
  }

  function populateSelect(id, items) {
    const el = document.getElementById(id);
    if (!el) return;

    const list = Array.isArray(items) ? items : (items?.models || items?.datasets || []);
    if (!list || list.length === 0) {
      if (id.includes('model')) {
        el.innerHTML = '<option disabled selected value="">⚠️ Convert a Model First (No .grim files found)</option>';
      } else {
        el.innerHTML = '<option disabled selected value="">No items found</option>';
      }
      return;
    }
    el.innerHTML = list.map(item => `<option value="${item.path || item.id || item}">${item.name || item.id || item}</option>`).join('');
  }

  function renderModels(modelsInput) {
    const container = document.getElementById('models-list');
    if (!container) return;
    const models = Array.isArray(modelsInput) ? modelsInput : (modelsInput?.models || []);
    if (models.length === 0) {
      container.innerHTML = '<p class="item-detail">No local models found in catalog. Use the <strong>Convert Model</strong> tab to generate <code>.grim</code> files.</p>';
      return;
    }

    container.innerHTML = models.map(m => `
      <div class="item-card">
        <div class="item-title">📦 ${m.name || m.id}</div>
        <div class="item-detail">Format: ${m.is_grim ? 'Native .grim' : 'GGUF'} | Path: <code>${m.path}</code></div>
      </div>
    `).join('');
  }

  function renderJobs(jobsInput) {
    const container = document.getElementById('jobs-list');
    if (!container) return;
    const jobs = Array.isArray(jobsInput) ? jobsInput : (jobsInput?.jobs || []);
    if (jobs.length === 0) {
      container.innerHTML = '<p class="item-detail">No active or historical training jobs.</p>';
      return;
    }

    container.innerHTML = jobs.map(j => `
      <div class="item-card">
        <div class="item-title">⚡ Job #${j.job_id} — Status: ${j.status}</div>
        <div class="item-detail">Model: ${j.model_path} | Mode: ${j.training_mode || 'QLoRA'}</div>
        <div class="item-detail">Progress: Epoch ${j.current_epoch || 0} | Loss: ${j.current_loss || 'N/A'}</div>
      </div>
    `).join('');
  }

  function renderDevices(deviceInput) {
    const container = document.getElementById('devices-list');
    const pill = document.getElementById('rocm-status-pill');
    if (!container) return;

    const devices = Array.isArray(deviceInput) ? deviceInput : (deviceInput?.devices || []);
    
    if (!devices || devices.length === 0) {
      if (pill) {
        pill.textContent = '🟡 CPU Fallback Mode (No ROCm GPU Detected)';
        pill.className = 'status-pill status-warning';
      }
      container.innerHTML = `
        <div class="item-card">
          <div class="item-title">💻 Host Processor & SIMD Execution Engine</div>
          <div class="item-detail"><strong>Telemetry Status:</strong> Active — No discrete AMD ROCm HIP GPU target currently detected.</div>
          <div class="item-detail"><strong>Execution Mode:</strong> Fused SIMD Vector Kernels & Multi-Threaded Host CPU Execution</div>
          <div class="item-detail"><strong>Recommendation:</strong> Use the <strong>Convert Model</strong> tab to prepare <code>.grim</code> models optimized for your system architecture.</div>
        </div>
      `;
      return;
    }

    if (pill) {
      pill.textContent = `🟢 ${devices.length} ROCm GPU Accelerators Active`;
      pill.className = 'status-pill status-online';
    }

    container.innerHTML = devices.map(d => {
      const totalMb = d.vram_bytes ? Math.round(d.vram_bytes / (1024 * 1024)) : (d.vram_total_mb || 24576);
      const totalGb = (totalMb / 1024).toFixed(1);
      const freeMb = d.vram_free_mb || Math.round(totalMb * 0.75);
      const usedMb = totalMb - freeMb;
      const pct = Math.min(100, Math.max(0, Math.round((usedMb / totalMb) * 100)));
      const waveStr = d.wavefront_size === 32 ? 'Wave32 (RDNA Architecture)' : 'Wave64 (CDNA Architecture)';
      const wmmaStr = d.wmma_supported ? '⚡ Present (Wave Matrix Multiply Accumulate)' : '❌ Not Available';
      const mfmaStr = d.mfma_supported ? '⚡ Present (Matrix Fused Multiply Add)' : '❌ Not Available';

      return `
        <div class="item-card">
          <div class="item-title">🎮 Device #${d.ordinal}: ${d.name || 'AMD ROCm Accelerator'}</div>
          <div class="item-detail"><strong>Architecture Target:</strong> <code>${d.gcn_arch || 'gfx1100'}</code></div>
          <div class="item-detail"><strong>Max Memory (VRAM):</strong> ${totalMb.toLocaleString()} MB (${totalGb} GB Total)</div>
          <div class="vram-meter">
            <div class="vram-fill" style="width: ${pct}%;"></div>
          </div>
          <div class="item-detail"><strong>Memory Status:</strong> ${usedMb.toLocaleString()} MB used / ${freeMb.toLocaleString()} MB free (${pct}% allocated)</div>
          <div class="item-detail"><strong>Execution Mode:</strong> ${waveStr}</div>
          <div class="item-detail"><strong>WMMA Hardware Cores:</strong> ${wmmaStr}</div>
          <div class="item-detail"><strong>MFMA Matrix Cores:</strong> ${mfmaStr}</div>
          <div class="item-detail"><strong>Compute Units (CUs):</strong> ${d.compute_units || 84} CUs</div>
          <div class="item-detail"><strong>Max Threads Per Block:</strong> ${d.max_threads_per_block || 1024} threads</div>
          <div class="item-detail"><strong>Unified Page Migration (XNACK):</strong> ${d.xnack_enabled ? 'Enabled' : 'Disabled'}</div>
        </div>
      `;
    }).join('');
  }

  // Presets & Dynamic VRAM Estimator
  const presetBeginner = document.getElementById('preset-beginner');
  const presetHighPerf = document.getElementById('preset-highperf');
  const presetLowVram = document.getElementById('preset-lowvram');

  if (presetBeginner) {
    presetBeginner.addEventListener('click', () => {
      setFormValues('QLoRA', 16, 32, '0.0002', 3);
    });
  }
  if (presetHighPerf) {
    presetHighPerf.addEventListener('click', () => {
      setFormValues('QLoRA', 64, 128, '0.0003', 5);
    });
  }
  if (presetLowVram) {
    presetLowVram.addEventListener('click', () => {
      setFormValues('QLoRA', 8, 16, '0.0001', 2);
    });
  }

  function setFormValues(mode, rank, alpha, lr, epochs) {
    document.getElementById('train-mode').value = mode;
    document.getElementById('lora-rank').value = rank;
    document.getElementById('lora-alpha').value = alpha;
    document.getElementById('lr').value = lr;
    document.getElementById('epochs').value = epochs;
    updateVramEstimate();
  }

  function updateVramEstimate() {
    const rank = parseInt(document.getElementById('lora-rank')?.value || 16);
    const mode = document.getElementById('train-mode')?.value || 'QLoRA';
    const badge = document.getElementById('vram-estimate');
    if (!badge) return;

    let baseGb = mode === 'QLoRA' ? 3.5 : 14.0;
    let rankExtraGb = (rank / 16) * 0.7;
    let totalGb = (baseGb + rankExtraGb).toFixed(1);
    badge.textContent = `Estimated VRAM: ~${totalGb} GB`;
  }

  ['lora-rank', 'train-mode'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', updateVramEstimate);
  });

  // Dataset Source Toggling
  const dsToggleLocal = document.getElementById('ds-toggle-local');
  const dsToggleHf = document.getElementById('ds-toggle-hf');
  const groupLocalDs = document.getElementById('group-local-ds');
  const groupHfDs = document.getElementById('group-hf-ds');

  if (dsToggleLocal && dsToggleHf) {
    dsToggleLocal.addEventListener('click', () => {
      dsToggleLocal.classList.add('active');
      dsToggleHf.classList.remove('active');
      groupLocalDs.classList.remove('hidden');
      groupHfDs.classList.add('hidden');
    });

    dsToggleHf.addEventListener('click', () => {
      dsToggleHf.classList.add('active');
      dsToggleLocal.classList.remove('active');
      groupHfDs.classList.remove('hidden');
      groupLocalDs.classList.add('hidden');
    });
  }

  // Handle Training Form Submit
  const trainForm = document.getElementById('train-form');
  if (trainForm) {
    trainForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      let datasetId = '';
      if (dsToggleHf && dsToggleHf.classList.contains('active')) {
        datasetId = document.getElementById('hf-dataset-input').value.trim();
        if (!datasetId) {
          alert('Please enter a valid HuggingFace dataset repository string (e.g. tatsu-lab/alpaca).');
          return;
        }
      } else {
        datasetId = document.getElementById('dataset-select').value;
      }

      const payload = {
        model_id: document.getElementById('model-select').value,
        dataset_id: datasetId,
        training_mode: document.getElementById('train-mode').value,
        lora_rank: parseInt(document.getElementById('lora-rank').value),
        lora_alpha: parseFloat(document.getElementById('lora-alpha').value),
        learning_rate: parseFloat(document.getElementById('lr').value),
        epochs: parseInt(document.getElementById('epochs').value),
      };

      try {
        const res = await fetch('/api/train/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (res.ok) {
          alert('Training job launched successfully!');
          loadData();
        } else {
          alert('Failed to launch training job.');
        }
      } catch (err) {
        alert('Error connecting to server: ' + err);
      }
    });
  }

  // Handle Convert Model Form Submit (grim-format Oxidizer)
  const convertForm = document.getElementById('convert-form');
  if (convertForm) {
    convertForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const submitBtn = document.getElementById('convert-submit-btn');
      const typedSource = document.getElementById('convert-source').value.trim();
      const selectedSource = document.getElementById('convert-source-select')?.value || '';
      const sourcePathOrUrl = typedSource || selectedSource;

      if (!sourcePathOrUrl) {
        alert('Please select a discovered source model or enter a valid file path / HuggingFace URL.');
        return;
      }

      const payload = {
        source_path_or_url: sourcePathOrUrl,
        output_name: document.getElementById('convert-output-name').value.trim(),
        target_gcn: document.getElementById('convert-arch').value,
        target_bpw: parseFloat(document.getElementById('convert-bpw').value),
        evopress_generations: parseInt(document.getElementById('convert-generations')?.value || 10),
      };

      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = '⚙️ Converting with EvoPress Oxidizer...';
      }

      try {
        const res = await fetch('/api/models/convert', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const result = await res.json();
        if (res.ok && result.success) {
          alert(`Conversion Success!\n\n${result.message}\nOutput: ${result.output_path}`);
          document.getElementById('convert-source').value = '';
          document.getElementById('convert-output-name').value = '';
          loadData();
        } else {
          alert(`Conversion Failed: ${result.message || 'Unknown error'}`);
        }
      } catch (err) {
        alert('Error connecting to conversion service: ' + err);
      } finally {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = '⚙️ Convert & Export to .grim Format';
        }
      }
    });
  }

  // Handle Bolt-On Attach Form Submit
  const boltOnForm = document.getElementById('bolton-form');
  if (boltOnForm) {
    boltOnForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const modelId = document.getElementById('bolton-model-select').value;
      const adapterPath = document.getElementById('adapter-path').value;

      try {
        const res = await fetch(`/api/models/${encodeURIComponent(modelId)}/bolt-ons`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ adapter_path: adapterPath })
        });
        if (res.ok) {
          alert('Bolt-On adapter attached!');
          loadData();
        } else {
          alert('Failed to attach bolt-on adapter.');
        }
      } catch (err) {
        alert('Error: ' + err);
      }
    });
  }

  document.getElementById('refresh-btn')?.addEventListener('click', loadData);

  loadData();
});
