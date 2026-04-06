// ── State ────────────────────────────────────────────────────
let currentSlideId = null;
let slides = [];
let viewerOSD = null;      // OpenSeadragon for Viewer page
let analyzeOSD = null;     // OpenSeadragon for Analyze page
let currentChannel = 'original';
let currentModel = 'hovernet';
let currentCell = 'metastasis';
let annLayer = null, heatLayer = null, inferLayer = null;

// Inference state
let currentJobId = null;
let inferencePollingInterval = null;

// ROI drawing state
let drawingMode = false;
let dragStartPoint = null;
let draftROI = null;      // {x, y, width, height} while dragging
let roiRect = null;       // finalized ROI in image coordinates

// ── OSD Custom Controls ──────────────────────────────────────
function injectOSDControls(containerId, osdViewer) {
    // Remove any existing custom controls
    const existing = document.getElementById(containerId + '-custom-controls');
    if (existing) existing.remove();

    const svgZoomIn = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/><line x1="16" y1="16" x2="21" y2="21"/></svg>`;
    const svgZoomOut = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><line x1="8" y1="11" x2="14" y2="11"/><line x1="16" y1="16" x2="21" y2="21"/></svg>`;
    const svgHome = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>`;
    const svgFull = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`;

    const ctrl = document.createElement('div');
    ctrl.id = containerId + '-custom-controls';
    ctrl.className = 'osd-custom-controls';
    ctrl.innerHTML = `
        <button class="osd-ctrl-btn" title="Zoom in">${svgZoomIn}</button>
        <button class="osd-ctrl-btn" title="Zoom out">${svgZoomOut}</button>
        <button class="osd-ctrl-btn" title="Go home">${svgHome}</button>
        <button class="osd-ctrl-btn" title="Toggle full page">${svgFull}</button>
    `;

    const container = document.getElementById(containerId);
    container.style.position = 'relative';
    container.appendChild(ctrl);

    const [btnIn, btnOut, btnHome, btnFull] = ctrl.querySelectorAll('.osd-ctrl-btn');
    btnIn.addEventListener('click', () => osdViewer.viewport.zoomBy(1.5));
    btnOut.addEventListener('click', () => osdViewer.viewport.zoomBy(1 / 1.5));
    btnHome.addEventListener('click', () => osdViewer.viewport.goHome());
    btnFull.addEventListener('click', () => osdViewer.setFullScreen(!osdViewer.isFullPage()));
}

// ── Modals ───────────────────────────────────────────────────
function openModal(id) { document.getElementById(id).classList.add('active'); }
function closeModal(e, id) {
    if (e && e.target !== e.currentTarget) return;
    if (id) document.getElementById(id).classList.remove('active');
    else document.querySelectorAll('.modal-overlay').forEach(m => m.classList.remove('active'));
}

// ── Router ───────────────────────────────────────────────────
function navigateTo(page, slideId) {
    // If navigating to viewer/analyze without a slideId, use the current one
    if (!slideId && (page === 'viewer' || page === 'analyze')) {
        slideId = currentSlideId;
    }
    if (slideId) currentSlideId = slideId;

    // If still no slide for viewer/analyze, redirect to upload
    if (!slideId && (page === 'viewer' || page === 'analyze')) {
        page = 'upload';
    }

    const hash = slideId ? `#/${page}/${slideId}` : `#/${page}`;
    if (window.location.hash !== hash) window.location.hash = hash;
    else renderPage(page, slideId);
}

function renderPage(page, slideId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    const el = document.getElementById('page-' + page);
    if (el) el.classList.add('active');
    const tab = document.querySelector(`.nav-tab[data-page="${page}"]`);
    if (tab) tab.classList.add('active');

    if (slideId) currentSlideId = slideId;

    if (page === 'upload') loadSlideGallery();
    else if (page === 'viewer') {
        if (slideId) initViewer(slideId);
        else navigateTo('upload');
    }
    else if (page === 'analyze') {
        if (slideId) initAnalyze(slideId);
        else navigateTo('upload');
    }
}

window.addEventListener('hashchange', () => {
    const hash = window.location.hash.replace('#/', '');
    const parts = hash.split('/');
    renderPage(parts[0] || 'upload', parts[1] || null);
});

// ── Upload Logic ─────────────────────────────────────────────
function setupUpload() {
    const zone = document.getElementById('upload-zone');
    const input = document.getElementById('file-input');

    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => { if (input.files.length) uploadFile(input.files[0]); });
}

async function uploadFile(file) {
    const prog = document.getElementById('upload-progress');
    const fill = document.getElementById('progress-fill');
    const text = document.getElementById('progress-text');
    prog.style.display = 'block';
    fill.style.width = '0%';
    text.textContent = `Uploading ${file.name}...`;
    text.style.color = '';

    const fd = new FormData();
    fd.append('file', file);

    let pct = 0;
    const iv = setInterval(() => { pct = Math.min(pct + Math.random() * 8, 85); fill.style.width = pct + '%'; }, 300);

    try {
        const res = await fetch('/api/slides/upload', { method: 'POST', body: fd });
        clearInterval(iv);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }
        const data = await res.json();
        fill.style.width = '100%';
        text.textContent = `✓ ${data.display_name || file.name} uploaded successfully!`;
        text.style.color = '#34d399';
        currentSlideId = data.slide_id;
        setTimeout(() => { prog.style.display = 'none'; text.style.color = ''; loadSlideGallery(); }, 2000);
    } catch (err) {
        clearInterval(iv);
        fill.style.width = '0%';
        text.textContent = '✗ Error: ' + err.message;
        text.style.color = '#f87171';
        setTimeout(() => { prog.style.display = 'none'; text.style.color = ''; }, 4000);
    }
}

// ── Slide Gallery ────────────────────────────────────────────
async function loadSlideGallery() {
    try {
        const res = await fetch('/api/slides');
        slides = await res.json();
    } catch { slides = []; }
    const gal = document.getElementById('slide-gallery');
    if (!slides.length) { gal.innerHTML = '<div class="gallery-empty">No slides available. Upload a WSI to begin.</div>'; return; }

    gal.innerHTML = slides.map(s => `
    <div class="slide-card">
      <div class="slide-card-header">
        <h4>${s.display_name || s.filename}</h4>
        <span class="slide-card-badge">Ready</span>
      </div>
      <div class="slide-card-meta">
        <div class="meta-item"><span class="meta-label">Dimensions</span><span class="meta-value">${s.dimensions[0].toLocaleString()} × ${s.dimensions[1].toLocaleString()} px</span></div>
        <div class="meta-item"><span class="meta-label">Magnification</span><span class="meta-value">${s.magnification}×</span></div>
        <div class="meta-item"><span class="meta-label">Size</span><span class="meta-value">${s.file_size_mb} MB</span></div>
        <div class="meta-item"><span class="meta-label">Channels</span><span class="meta-value">${s.channels.length} detected</span></div>
      </div>
      <div class="slide-card-actions">
        <button class="btn-view" onclick="navigateTo('viewer','${s.slide_id}')">🔬 View Channels</button>
        <button class="btn-analyze" onclick="navigateTo('analyze','${s.slide_id}')">📊 Analyze</button>
      </div>
    </div>
  `).join('');
}

// ── Viewer Page (Color Deconvolution) ────────────────────────
async function initViewer(slideId) {
    let info;
    try {
        const res = await fetch(`/api/slides/${slideId}/info`);
        if (!res.ok) throw new Error('Slide not found');
        info = await res.json();
    } catch (e) {
        console.error('Failed to load slide info:', e);
        return;
    }

    document.getElementById('slide-meta').textContent = `${info.display_name} — ${info.dimensions[0].toLocaleString()} × ${info.dimensions[1].toLocaleString()} px`;

    // Metadata panel
    const mp = document.getElementById('viewer-meta-panel');
    mp.innerHTML = `
    <div class="meta-row"><span class="meta-label">Filename</span><span class="meta-value">${info.filename}</span></div>
    <div class="meta-row"><span class="meta-label">Dimensions</span><span class="meta-value">${info.dimensions[0].toLocaleString()} × ${info.dimensions[1].toLocaleString()} px</span></div>
    <div class="meta-row"><span class="meta-label">Magnification</span><span class="meta-value">${info.magnification}×</span></div>
    <div class="meta-row"><span class="meta-label">Vendor</span><span class="meta-value">${info.vendor}</span></div>
    <div class="meta-row"><span class="meta-label">Levels</span><span class="meta-value">${info.level_count}</span></div>
    <div class="meta-row"><span class="meta-label">Annotations</span><span class="meta-value">${info.annotations_count}</span></div>
    <div class="meta-row"><span class="meta-label">Channels</span><span class="meta-value">${info.channels.map(c => c.label.split(' ')[0]).join(', ')}</span></div>
  `;

    // Destroy previous viewer
    if (viewerOSD) { viewerOSD.destroy(); viewerOSD = null; }
    currentChannel = 'original';
    document.querySelectorAll('.channel-option').forEach(o => o.classList.toggle('active', o.dataset.channel === 'original'));
    document.querySelectorAll('.channel-option input').forEach(r => r.checked = (r.value === 'original'));

    const makeSrc = (ch) => ({
        width: info.dimensions[0], height: info.dimensions[1], tileSize: info.tile_size,
        minLevel: 0, maxLevel: info.level_count - 1,
        getTileUrl: ch === 'original'
            ? (l, x, y) => `/api/slides/${slideId}/tile/${l}/${x}_${y}.jpeg`
            : (l, x, y) => `/api/slides/${slideId}/deconvolve/${ch}/tile/${l}/${x}_${y}.png`,
    });

    viewerOSD = OpenSeadragon({
        id: 'viewer-osd', prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/',
        tileSources: makeSrc('original'),
        showNavigator: true, navigatorBackground: '#000',
        showNavigationControl: false,
        minZoomLevel: 0.1, maxZoomLevel: 25, blendTime: 0.3,
    });
    injectOSDControls('viewer-osd', viewerOSD);

    // Channel switching
    document.querySelectorAll('.channel-option').forEach(opt => {
        opt.onclick = () => {
            const ch = opt.dataset.channel;
            if (ch === currentChannel) return;
            currentChannel = ch;
            document.querySelectorAll('.channel-option').forEach(o => o.classList.toggle('active', o === opt));
            opt.querySelector('input').checked = true;
            const vp = viewerOSD.viewport.getBounds();
            viewerOSD.open(makeSrc(ch));
            viewerOSD.addOnceHandler('open', () => viewerOSD.viewport.fitBounds(vp, true));
        };
    });

    document.getElementById('viewer-reset').onclick = () => viewerOSD.viewport.goHome();
}

// ── Analyze Page ─────────────────────────────────────────────
async function initAnalyze(slideId) {
    let info;
    try {
        const res = await fetch(`/api/slides/${slideId}/info`);
        if (!res.ok) throw new Error('Slide not found');
        info = await res.json();
    } catch (e) {
        console.error('Failed to load slide info for analyze:', e);
        return;
    }

    document.getElementById('slide-meta').textContent = `Analyzing: ${info.display_name}`;

    // Clean up inference polling
    if (inferencePollingInterval) {
        clearInterval(inferencePollingInterval);
        inferencePollingInterval = null;
    }
    currentJobId = null;

    // Model cards
    const mc = document.getElementById('model-cards');
    const models = info.models || [];

    if (models.length === 0) {
        mc.innerHTML = '<div class="meta-empty">No models available.</div>';
    } else {
        currentModel = models[0].id;
        mc.innerHTML = models.map((m, i) => `
      <div class="model-card ${i === 0 ? 'active' : ''}" data-model="${m.id}" id="model-${m.id}">
        <div class="model-card-name">
          ${m.label}
          ${m.real ? '<span class="model-badge real">REAL</span>' : '<span class="model-badge mock">MOCK</span>'}
        </div>
        <div class="model-card-desc">${m.description}</div>
      </div>
    `).join('');

        // Bind click handlers
        models.forEach(m => {
            const card = document.getElementById('model-' + m.id);
            if (card) {
                card.addEventListener('click', () => {
                    currentModel = m.id;
                    document.querySelectorAll('.model-card').forEach(c => c.classList.toggle('active', c.dataset.model === m.id));
                    // Update cell types for this model
                    const cs = document.getElementById('cell-select');
                    cs.innerHTML = m.cell_types.map(c => `<option value="${c}">${c.charAt(0).toUpperCase() + c.slice(1)}</option>`).join('');
                    currentCell = m.cell_types[0];

                    // Show/hide sections based on model type
                    const inferSection = document.getElementById('inference-section');
                    const heatControls = document.getElementById('heatmap-controls');
                    const cellTypeSection = document.getElementById('cell-type-section');
                    if (m.real) {
                        inferSection.style.display = '';
                        // Hide mock heatmap controls for real model
                        if (heatControls) heatControls.style.display = 'none';
                        if (cellTypeSection) cellTypeSection.style.display = 'none';
                        // Turn OFF mock heatmap
                        document.getElementById('toggle-heat').checked = false;
                        safeRemoveLayer('heat');
                    } else {
                        inferSection.style.display = 'none';
                        // Show mock heatmap controls for mock model
                        if (heatControls) heatControls.style.display = '';
                        if (cellTypeSection) cellTypeSection.style.display = '';
                        // Turn OFF inference overlay, enable mock heatmap
                        document.getElementById('toggle-inference').checked = false;
                        safeRemoveLayer('inference');
                        document.getElementById('toggle-heat').checked = true;
                        reloadHeat();
                    }
                });
            }
        });

        // Show inference section if first model is real
        const firstModel = models[0];
        const inferSection = document.getElementById('inference-section');
        inferSection.style.display = firstModel.real ? '' : 'none';

        // Hide/show heatmap controls based on model type
        const heatControls = document.getElementById('heatmap-controls');
        const cellTypeSection = document.getElementById('cell-type-section');
        if (firstModel.real) {
            if (heatControls) heatControls.style.display = 'none';
            if (cellTypeSection) cellTypeSection.style.display = 'none';
        }
    }

    // Cell type selector
    const cs = document.getElementById('cell-select');
    const firstModel = models.find(m => m.id === currentModel);
    const cellTypes = firstModel ? firstModel.cell_types : ['metastasis', 'epithelial', 'normal'];
    cs.innerHTML = cellTypes.map(c => `<option value="${c}">${c.charAt(0).toUpperCase() + c.slice(1)}</option>`).join('');
    currentCell = cellTypes[0] || 'metastasis';

    // Destroy previous OSD
    if (analyzeOSD) { analyzeOSD.destroy(); analyzeOSD = null; }
    annLayer = heatLayer = inferLayer = null;

    // Reset inference UI
    document.getElementById('inference-progress').style.display = 'none';
    document.getElementById('inference-results').style.display = 'none';
    document.getElementById('inference-filter-section').style.display = 'none';
    document.getElementById('toggle-inference').checked = false;
    document.getElementById('show-heatmap-btn').style.display = 'none';
    document.getElementById('show-flow-btn').style.display = 'none';
    document.getElementById('show-flow-inline-btn').style.display = 'none';

    // Set smart defaults based on selected model type
    const isRealModel = firstModel && firstModel.real;
    document.getElementById('toggle-heat').checked = false;
    document.getElementById('toggle-heat').disabled = true; // disabled until cell type selected
    document.getElementById('toggle-vector').checked = false;
    document.getElementById('toggle-ann').checked = false;

    const baseSrc = {
        width: info.dimensions[0], height: info.dimensions[1], tileSize: info.tile_size,
        minLevel: 0, maxLevel: info.level_count - 1,
        getTileUrl: (l, x, y) => `/api/slides/${slideId}/tile/${l}/${x}_${y}.jpeg`,
    };
    const annSrc = {
        width: info.dimensions[0], height: info.dimensions[1], tileSize: info.tile_size,
        minLevel: 0, maxLevel: info.level_count - 1,
        getTileUrl: (l, x, y) => `/api/slides/${slideId}/overlay/annotations/tile/${l}/${x}_${y}.png`,
    };
    const densitySrc = (jobId, cellType, vec) => ({
        width: info.dimensions[0], height: info.dimensions[1], tileSize: info.tile_size,
        minLevel: 0, maxLevel: info.level_count - 1,
        getTileUrl: (l, x, y) => {
            let url = `/api/slides/${slideId}/overlay/density/tile/${l}/${x}_${y}.png`;
            const params = [];
            if (jobId) params.push(`job_id=${jobId}`);
            if (cellType) params.push(`cell_type=${encodeURIComponent(cellType)}`);
            if (vec) params.push('vector=true');
            if (params.length) url += '?' + params.join('&');
            return url;
        },
    });
    const inferSrc = (jobId, filterType) => ({
        width: info.dimensions[0], height: info.dimensions[1], tileSize: info.tile_size,
        minLevel: 0, maxLevel: info.level_count - 1,
        getTileUrl: (l, x, y) => {
            let url = `/api/slides/${slideId}/overlay/inference/tile/${l}/${x}_${y}.png`;
            const params = [];
            if (jobId) params.push(`job_id=${jobId}`);
            if (filterType) params.push(`filter_type=${encodeURIComponent(filterType)}`);
            if (params.length) url += '?' + params.join('&');
            return url;
        },
    });

    analyzeOSD = OpenSeadragon({
        id: 'analyze-osd', prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/',
        tileSources: baseSrc, showNavigator: true, navigatorBackground: '#000',
        showNavigationControl: false,
        minZoomLevel: 0.1, maxZoomLevel: 25, blendTime: 0.5,
    });
    injectOSDControls('analyze-osd', analyzeOSD);

    function addLayer(src, opacity, cb) {
        analyzeOSD.addTiledImage({ tileSource: src, opacity, success: e => cb(e.item) });
    }
    function refreshOpacity() {
        if (annLayer) annLayer.setOpacity(+document.getElementById('ann-opacity').value);
        if (heatLayer) heatLayer.setOpacity(+document.getElementById('heat-opacity').value);
        if (inferLayer) inferLayer.setOpacity(+document.getElementById('inference-opacity').value);
    }
    // Safe layer removal helper — handles stale references
    function safeRemoveLayer(which) {
        try {
            if (which === 'ann' && annLayer) {
                analyzeOSD.world.removeItem(annLayer);
                annLayer = null;
            } else if (which === 'heat' && heatLayer) {
                analyzeOSD.world.removeItem(heatLayer);
                heatLayer = null;
            } else if (which === 'inference' && inferLayer) {
                analyzeOSD.world.removeItem(inferLayer);
                inferLayer = null;
            }
        } catch (e) {
            console.warn('[Layer] Failed to remove', which, e);
            if (which === 'ann') annLayer = null;
            if (which === 'heat') heatLayer = null;
            if (which === 'inference') inferLayer = null;
        }
    }

    function toggleLayer(which, on) {
        if (which === 'ann') {
            safeRemoveLayer('ann');
            if (on) addLayer(annSrc, +document.getElementById('ann-opacity').value, i => annLayer = i);
        } else if (which === 'heat') {
            safeRemoveLayer('heat');
            if (on) {
                const cellType = document.getElementById('inference-type-filter').value;
                const vec = document.getElementById('toggle-vector').checked;
                addLayer(densitySrc(currentJobId, cellType, vec), +document.getElementById('heat-opacity').value, i => heatLayer = i);
            }
        } else if (which === 'inference') {
            safeRemoveLayer('inference');
            if (on) {
                const filterType = document.getElementById('inference-type-filter').value;
                addLayer(inferSrc(currentJobId, filterType), +document.getElementById('inference-opacity').value, i => inferLayer = i);
            }
        }
    }

    function reloadHeat() {
        const on = document.getElementById('toggle-heat').checked;
        safeRemoveLayer('heat');
        if (on) {
            const cellType = document.getElementById('inference-type-filter').value;
            const vec = document.getElementById('toggle-vector').checked;
            addLayer(densitySrc(currentJobId, cellType, vec), +document.getElementById('heat-opacity').value, i => heatLayer = i);
        }
    }
    // Make reloadHeat accessible
    window._reloadHeat = reloadHeat;

    function reloadInferenceOverlay() {
        safeRemoveLayer('inference');
        if (document.getElementById('toggle-inference').checked) {
            const filterType = document.getElementById('inference-type-filter').value;
            addLayer(inferSrc(currentJobId, filterType), +document.getElementById('inference-opacity').value, i => inferLayer = i);
        }
    }

    // ── ROI Drawing Logic ────────────────────────────────────
    const svgOverlay = document.getElementById('polygon-svg');
    const roiHitArea = document.getElementById('roi-hit-area');

    function clampImagePoint(pt) {
        return {
            x: Math.max(0, Math.min(info.dimensions[0], Math.round(pt.x))),
            y: Math.max(0, Math.min(info.dimensions[1], Math.round(pt.y))),
        };
    }

    function getImagePointFromClientPosition(clientX, clientY) {
        const rect = roiHitArea.getBoundingClientRect();
        const pixel = new OpenSeadragon.Point(clientX - rect.left, clientY - rect.top);
        const viewportPoint = analyzeOSD.viewport.pointFromPixel(pixel);
        const imagePoint = analyzeOSD.viewport.viewportToImageCoordinates(viewportPoint);
        return clampImagePoint(imagePoint);
    }

    function normalizeROI(a, b) {
        const x1 = Math.min(a.x, b.x);
        const y1 = Math.min(a.y, b.y);
        const x2 = Math.max(a.x, b.x);
        const y2 = Math.max(a.y, b.y);
        return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
    }

    function updateROIButtons() {
        const hasROI = !!roiRect;
        document.getElementById('draw-roi-btn').style.display = '';
        document.getElementById('draw-roi-btn').textContent = hasROI ? '✏️ Redraw ROI' : '✏️ Draw ROI';
        document.getElementById('finish-roi-btn').style.display = drawingMode ? '' : 'none';
        document.getElementById('clear-roi-btn').style.display = (drawingMode || hasROI) ? '' : 'none';
    }

    function updateROIOverlay() {
        const rect = drawingMode && draftROI ? draftROI : roiRect;

        if (!rect || rect.width <= 0 || rect.height <= 0) {
            svgOverlay.innerHTML = '';
            return;
        }

        const topLeftVp = analyzeOSD.viewport.imageToViewportCoordinates(
            new OpenSeadragon.Point(rect.x, rect.y)
        );
        const bottomRightVp = analyzeOSD.viewport.imageToViewportCoordinates(
            new OpenSeadragon.Point(rect.x + rect.width, rect.y + rect.height)
        );
        const topLeftPx = analyzeOSD.viewport.viewportToViewerElementCoordinates(topLeftVp);
        const bottomRightPx = analyzeOSD.viewport.viewportToViewerElementCoordinates(bottomRightVp);

        const x = Math.min(topLeftPx.x, bottomRightPx.x);
        const y = Math.min(topLeftPx.y, bottomRightPx.y);
        const width = Math.abs(bottomRightPx.x - topLeftPx.x);
        const height = Math.abs(bottomRightPx.y - topLeftPx.y);

        svgOverlay.innerHTML = `
            <rect
                x="${x}"
                y="${y}"
                width="${width}"
                height="${height}"
                rx="6"
                ry="6"
                fill="rgba(0,200,255,0.14)"
                stroke="#00c8ff"
                stroke-width="2"
            />
        `;
    }

    function stopDrawingMode() {
        drawingMode = false;
        dragStartPoint = null;
        draftROI = null;
        document.getElementById('analyze-osd').classList.remove('drawing-mode');
        roiHitArea.classList.remove('active');
        analyzeOSD.setMouseNavEnabled(true);
        updateROIButtons();
        updateROIOverlay();
    }

    function startDrawing() {
        drawingMode = true;
        dragStartPoint = null;
        draftROI = null;
        roiRect = null;
        document.getElementById('roi-status').style.display = '';
        document.getElementById('roi-status').textContent = 'Drag on the slide to draw a rectangular ROI.';
        document.getElementById('analyze-osd').classList.add('drawing-mode');
        roiHitArea.classList.add('active');
        analyzeOSD.setMouseNavEnabled(false);
        updateROIButtons();
        updateROIOverlay();
    }

    function cancelDrawing() {
        stopDrawingMode();
        if (roiRect) {
            document.getElementById('roi-status').style.display = '';
            document.getElementById('roi-status').textContent = `ROI set: ${roiRect.width}×${roiRect.height}px region`;
        } else {
            document.getElementById('roi-status').style.display = 'none';
        }
    }

    function clearROI() {
        roiRect = null;
        draftROI = null;
        dragStartPoint = null;
        stopDrawingMode();
        document.getElementById('roi-status').style.display = 'none';
    }

    function finalizeROI(endPoint) {
        const nextROI = normalizeROI(dragStartPoint, endPoint);
        dragStartPoint = null;
        draftROI = null;

        if (nextROI.width < 8 || nextROI.height < 8) {
            document.getElementById('roi-status').style.display = '';
            document.getElementById('roi-status').textContent = 'ROI too small. Drag a larger region.';
            updateROIOverlay();
            return;
        }

        roiRect = nextROI;
        stopDrawingMode();
        document.getElementById('roi-status').style.display = '';
        document.getElementById('roi-status').textContent = `ROI set: ${roiRect.width}×${roiRect.height}px region`;
    }

    updateROIButtons();
    updateROIOverlay();

    analyzeOSD.addHandler('animation', updateROIOverlay);
    analyzeOSD.addHandler('animation-finish', updateROIOverlay);
    analyzeOSD.addHandler('zoom', updateROIOverlay);
    analyzeOSD.addHandler('pan', updateROIOverlay);
    analyzeOSD.addHandler('open', updateROIOverlay);

    roiHitArea.addEventListener('pointerdown', function (event) {
        if (!drawingMode) return;
        event.preventDefault();
        roiHitArea.setPointerCapture(event.pointerId);
        dragStartPoint = getImagePointFromClientPosition(event.clientX, event.clientY);
        draftROI = { x: dragStartPoint.x, y: dragStartPoint.y, width: 0, height: 0 };
        document.getElementById('roi-status').style.display = '';
        document.getElementById('roi-status').textContent = 'Drawing ROI: 0×0px';
        updateROIOverlay();
    });

    roiHitArea.addEventListener('pointermove', function (event) {
        if (!drawingMode || !dragStartPoint) return;
        event.preventDefault();
        const currentPoint = getImagePointFromClientPosition(event.clientX, event.clientY);
        draftROI = normalizeROI(dragStartPoint, currentPoint);
        document.getElementById('roi-status').style.display = '';
        document.getElementById('roi-status').textContent = `Drawing ROI: ${draftROI.width}×${draftROI.height}px`;
        updateROIOverlay();
    });

    roiHitArea.addEventListener('pointerup', function (event) {
        if (!drawingMode || !dragStartPoint) return;
        event.preventDefault();
        finalizeROI(getImagePointFromClientPosition(event.clientX, event.clientY));
    });

    roiHitArea.addEventListener('pointercancel', function () {
        if (!drawingMode) return;
        dragStartPoint = null;
        draftROI = null;
        updateROIOverlay();
    });

    // ── Inference Logic ──────────────────────────────────────
    function getROI() {
        if (roiRect && roiRect.width > 0 && roiRect.height > 0) {
            return roiRect;
        }
        const bounds = analyzeOSD.viewport.getBounds(true);
        const topLeft = analyzeOSD.viewport.viewportToImageCoordinates(bounds.x, bounds.y);
        const bottomRight = analyzeOSD.viewport.viewportToImageCoordinates(
            bounds.x + bounds.width, bounds.y + bounds.height
        );
        let x = Math.max(0, Math.round(topLeft.x));
        let y = Math.max(0, Math.round(topLeft.y));
        let w = Math.round(bottomRight.x - topLeft.x);
        let h = Math.round(bottomRight.y - topLeft.y);
        const maxDim = 2048;
        if (w > maxDim || h > maxDim) {
            const scale = Math.min(maxDim / w, maxDim / h);
            const cx = x + w / 2, cy = y + h / 2;
            w = Math.round(w * scale);
            h = Math.round(h * scale);
            x = Math.round(cx - w / 2);
            y = Math.round(cy - h / 2);
        }
        return { x, y, width: w, height: h };
    }

    async function startInference() {
        const btn = document.getElementById('run-inference-btn');
        const progressPanel = document.getElementById('inference-progress');
        const resultsPanel = document.getElementById('inference-results');
        const statusText = document.getElementById('inference-status-text');
        const progressFill = document.getElementById('inference-progress-fill');
        const elapsedSpan = document.getElementById('inference-elapsed');

        btn.disabled = true;
        btn.textContent = '⏳ Running...';
        progressPanel.style.display = '';
        resultsPanel.style.display = 'none';
        progressFill.style.width = '0%';
        statusText.textContent = 'Starting inference...';

        const roi = getROI();
        const device = document.getElementById('inference-device').value;

        console.log('[Inference] Starting with model:', currentModel, 'slide:', currentSlideId, 'device:', device, 'roi:', roi);

        try {
            const res = await fetch(`/api/slides/${currentSlideId}/inference/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_id: currentModel,
                    roi: roi,
                    device: device,
                }),
            });
            if (!res.ok) {
                const errBody = await res.text();
                console.error('[Inference] Start failed:', res.status, errBody);
                throw new Error('Failed to start inference: ' + errBody);
            }
            const job = await res.json();
            currentJobId = job.job_id;
            console.log('[Inference] Job started:', job);

            // Start polling
            const startTime = Date.now();
            inferencePollingInterval = setInterval(async () => {
                try {
                    const statusRes = await fetch(`/api/slides/${currentSlideId}/inference/status/${currentJobId}`);
                    if (!statusRes.ok) {
                        console.warn('[Inference] Poll failed:', statusRes.status);
                        return;
                    }
                    const status = await statusRes.json();
                    console.log('[Inference] Poll:', status.status, status.progress + '%', status.status_message);

                    progressFill.style.width = status.progress + '%';
                    statusText.textContent = status.status_message || status.status;
                    elapsedSpan.textContent = Math.round((Date.now() - startTime) / 1000) + 's';

                    if (status.status === 'completed') {
                        clearInterval(inferencePollingInterval);
                        inferencePollingInterval = null;
                        btn.disabled = false;
                        btn.textContent = '⚡ Run Inference';
                        progressPanel.style.display = 'none';
                        showInferenceResults(status);
                    } else if (status.status === 'failed' || status.status === 'cancelled') {
                        clearInterval(inferencePollingInterval);
                        inferencePollingInterval = null;
                        btn.disabled = false;
                        btn.textContent = '⚡ Run Inference';
                        statusText.textContent = status.status === 'cancelled' ? 'Cancelled' : `Error: ${status.error || 'Unknown'}`;
                        progressFill.style.width = '0%';
                        setTimeout(() => { progressPanel.style.display = 'none'; }, 3000);
                    }
                } catch (e) {
                    console.error('Polling error:', e);
                }
            }, 2000);

        } catch (err) {
            btn.disabled = false;
            btn.textContent = '⚡ Run Inference on Viewport';
            statusText.textContent = 'Error: ' + err.message;
            setTimeout(() => { progressPanel.style.display = 'none'; }, 3000);
        }
    }

    async function showInferenceResults(jobStatus) {
        const resultsPanel = document.getElementById('inference-results');
        const summaryDiv = document.getElementById('inference-results-summary');
        const filterSection = document.getElementById('inference-filter-section');
        const filterSelect = document.getElementById('inference-type-filter');

        try {
            const res = await fetch(`/api/slides/${currentSlideId}/inference/results/${currentJobId}`);
            if (!res.ok) throw new Error('Failed to fetch results');
            const results = await res.json();

            const total = results.nuclei_count || 0;
            const typeCounts = results.type_counts || {};
            const elapsed = results.elapsed_seconds || 0;
            const device = results.device || 'unknown';

            // Type color mapping for PanNuke
            const typeColors = {
                'neopla': '#ff0000',
                'inflam': '#00ff00',
                'connec': '#0000ff',
                'necros': '#ffff00',
                'no-neo': '#ffa500',
                'nolabe': '#808080',
            };

            let summaryHTML = `
                <div class="results-stat"><span class="results-stat-label">Total Nuclei</span><span class="results-stat-value">${total.toLocaleString()}</span></div>
                <div class="results-stat"><span class="results-stat-label">Time</span><span class="results-stat-value">${elapsed}s</span></div>
                <div class="results-stat"><span class="results-stat-label">Device</span><span class="results-stat-value">${device}</span></div>
                <div class="results-types">
            `;
            for (const [typeName, count] of Object.entries(typeCounts)) {
                const color = typeColors[typeName] || '#888';
                summaryHTML += `<div class="type-badge" style="--type-color:${color}"><span class="type-dot" style="background:${color}"></span>${typeName}: ${count}</div>`;
            }
            summaryHTML += `</div>`;
            summaryDiv.innerHTML = summaryHTML;
            resultsPanel.style.display = '';

            // Populate filter dropdown
            filterSelect.innerHTML = '<option value="">All Types</option>';
            for (const typeName of Object.keys(typeCounts)) {
                filterSelect.innerHTML += `<option value="${typeName}">${typeName}</option>`;
            }
            filterSection.style.display = '';

            // Auto-enable and load the inference overlay
            document.getElementById('toggle-inference').checked = true;
            reloadInferenceOverlay();

            // Enable density heatmap toggle now that results are available
            document.getElementById('toggle-heat').disabled = false;
            document.getElementById('heatmap-hint').textContent = 'Select a cell type filter, then click Show Heatmap.';
            document.getElementById('flow-hint').textContent = 'Show arrows that flow toward the densest regions in the selected heatmap.';

            // Show the heatmap action button in the filter section
            const heatmapActionBtn = document.getElementById('show-heatmap-btn');
            if (heatmapActionBtn) heatmapActionBtn.style.display = '';
            const flowActionBtn = document.getElementById('show-flow-btn');
            const flowInlineBtn = document.getElementById('show-flow-inline-btn');
            if (flowActionBtn) flowActionBtn.style.display = '';
            if (flowInlineBtn) flowInlineBtn.style.display = '';

        } catch (e) {
            console.error('Failed to show results:', e);
            summaryDiv.innerHTML = '<div class="results-error">Failed to load results</div>';
            resultsPanel.style.display = '';
        }
    }

    async function cancelInference() {
        if (currentJobId) {
            try {
                await fetch(`/api/slides/${currentSlideId}/inference/cancel/${currentJobId}`, { method: 'POST' });
            } catch (e) {
                console.error('Cancel error:', e);
            }
        }
    }

    // ── Bind Controls ────────────────────────────────────────
    document.getElementById('toggle-ann').onchange = e => toggleLayer('ann', e.target.checked);
    document.getElementById('toggle-heat').onchange = e => toggleLayer('heat', e.target.checked);
    document.getElementById('toggle-vector').onchange = () => reloadHeat();
    document.getElementById('toggle-inference').onchange = e => toggleLayer('inference', e.target.checked);
    document.getElementById('ann-opacity').oninput = refreshOpacity;
    document.getElementById('heat-opacity').oninput = refreshOpacity;
    document.getElementById('inference-opacity').oninput = refreshOpacity;
    document.getElementById('analyze-reset').onclick = () => analyzeOSD.viewport.goHome();
    document.getElementById('reload-heat').onclick = () => reloadHeat();
    document.getElementById('run-inference-btn').onclick = () => startInference();
    document.getElementById('cancel-inference-btn').onclick = () => cancelInference();
    document.getElementById('draw-roi-btn').onclick = () => startDrawing();
    document.getElementById('finish-roi-btn').onclick = () => cancelDrawing();
    document.getElementById('clear-roi-btn').onclick = () => clearROI();
    document.getElementById('inference-type-filter').onchange = () => {
        reloadInferenceOverlay();
        // Also reload density heatmap if it's already on
        if (document.getElementById('toggle-heat').checked) {
            reloadHeat();
        }
    };
    document.getElementById('show-heatmap-btn').onclick = () => {
        // Re-enable mouse nav in case user was mid-ROI-draw
        analyzeOSD.setMouseNavEnabled(true);
        // Enable the heatmap toggle and reload
        const heatToggle = document.getElementById('toggle-heat');
        heatToggle.disabled = false;
        heatToggle.checked = true;
        reloadHeat();
    };
    function showFlowVectors() {
        analyzeOSD.setMouseNavEnabled(true);
        const heatToggle = document.getElementById('toggle-heat');
        const vectorToggle = document.getElementById('toggle-vector');
        heatToggle.disabled = false;
        heatToggle.checked = true;
        vectorToggle.checked = true;
        reloadHeat();
    }
    document.getElementById('show-flow-btn').onclick = () => showFlowVectors();
    document.getElementById('show-flow-inline-btn').onclick = () => showFlowVectors();

    // Initial layers — only load what's checked (respects smart defaults above)
    if (document.getElementById('toggle-ann').checked) toggleLayer('ann', true);
    if (document.getElementById('toggle-heat').checked) toggleLayer('heat', true);
}

// ── Init ─────────────────────────────────────────────────────
setupUpload();
const hash = window.location.hash.replace('#/', '');
const parts = hash.split('/');
renderPage(parts[0] || 'upload', parts[1] || null);
