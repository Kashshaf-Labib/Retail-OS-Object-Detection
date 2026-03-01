/**
 * ShelfVision AI — Frontend Application Logic
 * Handles tab navigation, image uploads, API calls, and Chart.js visualizations.
 */

const API_BASE = '';

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  TAB NAVIGATION                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(`content-${tab.dataset.tab}`).classList.add('active');
    });
});

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  DRAG & DROP + FILE INPUT                                              ║
// ╚══════════════════════════════════════════════════════════════════════════╝

function setupUploadZone(dropzoneId, inputId, btnId, previewImgId) {
    const dropzone = document.getElementById(dropzoneId);
    const input = document.getElementById(inputId);
    const btn = document.getElementById(btnId);
    let selectedFile = null;

    ['dragenter', 'dragover'].forEach(evt => {
        dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(evt => {
        dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove('dragover'); });
    });

    dropzone.addEventListener('drop', e => {
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });

    dropzone.addEventListener('click', () => input.click());
    input.addEventListener('change', () => { if (input.files[0]) handleFile(input.files[0]); });

    function handleFile(file) {
        if (!file.type.match(/image\/(jpeg|png)/)) {
            alert('Please upload a JPEG or PNG image.');
            return;
        }
        selectedFile = file;
        btn.disabled = false;

        if (previewImgId) {
            const preview = document.getElementById(previewImgId);
            const previewArea = preview.parentElement;
            preview.src = URL.createObjectURL(file);
            previewArea.style.display = 'block';
        }
    }

    return () => selectedFile;
}

const getDetectFile = setupUploadZone('detect-dropzone', 'detect-input', 'detect-btn', 'detect-preview-img');
const getShelfFile = setupUploadZone('shelf-dropzone', 'shelf-input', 'shelf-btn', null);

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  CONFIDENCE SLIDER                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝

const confSlider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');
confSlider.addEventListener('input', () => { confVal.textContent = confSlider.value; });

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  LOADING OVERLAY                                                       ║
// ╚══════════════════════════════════════════════════════════════════════════╝

function showLoading() { document.getElementById('loading').style.display = 'flex'; }
function hideLoading() { document.getElementById('loading').style.display = 'none'; }

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  DETECT TAB                                                            ║
// ╚══════════════════════════════════════════════════════════════════════════╝

document.getElementById('detect-btn').addEventListener('click', async () => {
    const file = getDetectFile();
    if (!file) return;

    showLoading();
    const formData = new FormData();
    formData.append('file', file);

    const conf = confSlider.value;
    try {
        const res = await fetch(`${API_BASE}/api/detect?confidence=${conf}&return_image=true`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) throw new Error('Detection failed');

        const count = res.headers.get('X-Detection-Count') || '0';
        let detections = [];
        try { detections = JSON.parse(res.headers.get('X-Detections') || '[]'); } catch { }

        const blob = await res.blob();
        const imgUrl = URL.createObjectURL(blob);

        // Show results
        document.getElementById('detect-results').style.display = 'block';
        document.getElementById('result-count').textContent = count;
        document.getElementById('result-img').src = imgUrl;

        // Fill table
        const tbody = document.querySelector('#detect-table tbody');
        tbody.innerHTML = '';
        if (detections.length > 0) {
            document.getElementById('detect-table-wrap').style.display = 'block';
            detections.forEach((d, i) => {
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>${i + 1}</td>
                    <td>${d.class_name}</td>
                    <td>${(d.confidence * 100).toFixed(1)}%</td>
                    <td>[${d.bbox.x1.toFixed(0)}, ${d.bbox.y1.toFixed(0)}, ${d.bbox.x2.toFixed(0)}, ${d.bbox.y2.toFixed(0)}]</td>
                `;
            });
        }
    } catch (err) {
        alert('Detection error: ' + err.message);
    } finally {
        hideLoading();
    }
});

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  SHARE OF SHELF TAB                                                    ║
// ╚══════════════════════════════════════════════════════════════════════════╝

let shelfBarChart = null;
let shelfPieChart = null;

document.getElementById('shelf-btn').addEventListener('click', async () => {
    const file = getShelfFile();
    if (!file) return;

    showLoading();
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_BASE}/api/share-of-shelf`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) throw new Error('Analysis failed');
        const data = await res.json();

        document.getElementById('shelf-results').style.display = 'block';
        document.getElementById('shelf-total').textContent = data.total_products;
        document.getElementById('shelf-unique').textContent = data.all_skus.length;

        const labels = data.all_skus.map(s => s.sku);
        const values = data.all_skus.map(s => s.percentage);
        const colors = labels.map((_, i) => `hsl(${(i * 360 / labels.length)}, 70%, 55%)`);

        // Bar chart
        if (shelfBarChart) shelfBarChart.destroy();
        shelfBarChart = new Chart(document.getElementById('shelf-bar-chart'), {
            type: 'bar',
            data: {
                labels,
                datasets: [{ label: 'Share %', data: values, backgroundColor: colors, borderRadius: 4 }],
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(99,122,180,0.1)' } },
                    y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
                },
            },
        });

        // Pie chart (top 10 only)
        const topLabels = data.top_skus.map(s => s.sku);
        const topValues = data.top_skus.map(s => s.percentage);
        const topColors = topLabels.map((_, i) => `hsl(${(i * 360 / topLabels.length)}, 65%, 50%)`);

        if (shelfPieChart) shelfPieChart.destroy();
        shelfPieChart = new Chart(document.getElementById('shelf-pie-chart'), {
            type: 'doughnut',
            data: {
                labels: topLabels,
                datasets: [{ data: topValues, backgroundColor: topColors, borderWidth: 0 }],
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 11 } } },
                    title: { display: true, text: 'Top 10 SKUs', color: '#e2e8f0', font: { size: 14 } },
                },
            },
        });
    } catch (err) {
        alert('Analysis error: ' + err.message);
    } finally {
        hideLoading();
    }
});

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  EXPERIMENTS TAB                                                       ║
// ╚══════════════════════════════════════════════════════════════════════════╝

let expChart = null;

async function loadExperiments() {
    try {
        const res = await fetch(`${API_BASE}/api/metrics`);
        const data = await res.json();
        if (!data.experiments || data.experiments.length === 0) return;

        const tbody = document.getElementById('exp-body');
        tbody.innerHTML = '';

        data.experiments.forEach(exp => {
            const row = tbody.insertRow();
            const isBest = exp.is_best ? ' style="color:#22c55e;font-weight:600"' : '';
            row.innerHTML = `
                <td${isBest}>${exp.model || '—'}</td>
                <td>${exp.imgsz || '—'}</td>
                <td>${exp.dataset || 'original'}</td>
                <td>${typeof exp.precision === 'number' ? exp.precision.toFixed(2) + '%' : exp.precision || '—'}</td>
                <td>${typeof exp.recall === 'number' ? exp.recall.toFixed(2) + '%' : exp.recall || '—'}</td>
                <td>${typeof exp.map50 === 'number' ? exp.map50.toFixed(2) + '%' : exp.map50 || '—'}</td>
                <td>${typeof exp.map50_95 === 'number' ? exp.map50_95.toFixed(2) + '%' : exp.map50_95 || '—'}</td>
                <td>${exp.notes || ''}</td>
            `;
        });

        // Chart
        const validExps = data.experiments.filter(e => typeof e.precision === 'number');
        if (validExps.length > 0) {
            const labels = validExps.map(e => e.model + (e.imgsz ? ` @${e.imgsz}` : ''));
            if (expChart) expChart.destroy();
            expChart = new Chart(document.getElementById('exp-chart'), {
                type: 'bar',
                data: {
                    labels,
                    datasets: [
                        { label: 'Precision', data: validExps.map(e => e.precision), backgroundColor: 'rgba(59,130,246,0.7)', borderRadius: 4 },
                        { label: 'Recall', data: validExps.map(e => e.recall), backgroundColor: 'rgba(34,197,94,0.7)', borderRadius: 4 },
                        { label: 'mAP@50', data: validExps.map(e => e.map50), backgroundColor: 'rgba(168,85,247,0.7)', borderRadius: 4 },
                    ],
                },
                options: {
                    responsive: true,
                    plugins: { legend: { labels: { color: '#94a3b8' } } },
                    scales: {
                        x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
                        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(99,122,180,0.1)' }, max: 100 },
                    },
                },
            });
        }
    } catch (err) {
        console.log('Could not load experiments:', err.message);
    }
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  DRIFT MONITOR TAB                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝

async function loadDriftStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/drift-status`);
        const data = await res.json();

        // Update badge
        const badge = document.getElementById('drift-badge');
        const badgeLabel = document.getElementById('drift-label');
        badge.className = 'drift-badge ' + data.status.toLowerCase().replace('drift_detected', 'drift');
        badgeLabel.textContent = data.status;

        // Update panel
        const statusLabel = document.getElementById('drift-status-label');
        statusLabel.textContent = data.status;
        statusLabel.className = 'drift-status-big ' + data.status.toLowerCase().replace('drift_detected', 'drift');

        document.getElementById('drift-psi').textContent = data.psi_score.toFixed(4);
        document.getElementById('drift-conf').textContent = (data.avg_confidence * 100).toFixed(1) + '%';
        document.getElementById('drift-count').textContent = data.predictions_logged;
        document.getElementById('drift-message').textContent = data.message;
    } catch (err) {
        console.log('Drift status unavailable:', err.message);
    }
}

document.getElementById('drift-refresh').addEventListener('click', loadDriftStatus);

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  INIT                                                                  ║
// ╚══════════════════════════════════════════════════════════════════════════╝

document.addEventListener('DOMContentLoaded', () => {
    loadExperiments();
    loadDriftStatus();
});
