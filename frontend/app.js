document.addEventListener('DOMContentLoaded', () => {

    // --- CONFIG ---
    const API_BASE = 'http://127.0.0.1:8000';

    // Map frontend operation IDs to backend-expected values
    const OP_MAP = {
        'smile':    'smile',
        'eyebrow':  'eyebrow_raise',
        'lip':      'smile',          // lip widen uses smile with wider params
        'slim':     'thin_face',
        'aging':    'aging',
        'deaging':  'de-aging',
        'fft':      'aging',          // generic FFT defaults to aging filter
    };

    // --- STATE VARIABLES ---
    let currentOriginalImage = null; // The uploaded image
    let currentProcessedImage = null; // The returned image after apply (base64 data URI)
    let currentFileObject = null; // The raw File object for FormData
    let currentLandmarks = []; // Real landmarks from backend
    let lastMetrics = null; // Last metrics from backend
    let selectedOperation = 'smile';
    let isSplitMode = true;
    let sliderPos = 50; // percentage
    let isDraggingSlider = false;
    let operationHistory = [];

    // Default User Settings
    let currentLang = 'EN';
    let isDarkMode = true;

    // --- I18N DICTIONARY ---
    const i18n = {
        TR: {
            dropImage: "Resmi buraya bırakın",
            clickBrowse: "veya tıklayıp seçin",
            imageUploaded: "Resim yüklendi",
            opType: "İşlem Türü",
            addSmile: "Gülümseme",
            eyebrowRaise: "Kaş Kaldırma",
            lipWiden: "Dudak Genişletme",
            faceSlim: "İnce Yüz",
            aging: "Yaşlandırma",
            deaging: "Gençleştirme",
            fftFilter: "FFT Filtresi",
            intensity: "Yoğunluk",
            transformStrength: "Dönüşüm gücü",
            smoothing: "Yumuşatma",
            edgeBlending: "Kenar kaynaştırma",
            showLandmarks: "İşaretçileri Göster",
            frequencyView: "Frekans Görünümü",
            opHistory: "İşlem Geçmişi",
            applyTrans: "Dönüşümü Uygula",
            downloadPDF: "Sonuçları PDF Olarak İndir",
            tabVisual: "Görsel",
            tabFrequency: "Frekans Spektrumu",
            tabLandmarks: "İşaretçiler",
            uploadWait: "Başlamak için bir resim yükleyin",
            beforeBadge: "Öncesi",
            afterBadge: "Sonrası",
            processing: "İşleniyor...",
            splitView: "Bölünmüş Görünüm",
            sideBySide: "Yan Yana",
            origFFT: "Orijinal - FFT Büyüklüğü",
            procFFT: "İşlenmiş - FFT Büyüklüğü",
            logScale: "Logaritmik ölçek",
            freqHz: "Frekans (Hz)",
            qualityMetrics: "Kalite Metrikleri",
            compMode: "Karşılaştırma Modu",
            mseDesc: "Ortalama Kare Hatası",
            psnrDesc: "Tepe Sinyal Gürültü Oranı",
            ssimDesc: "Yapısal Benzerlik İndeksi",
            analysisSummary: "Analiz Özeti",
            waitingSummary: "Özet analiz için görüntü işlemenin tamamlanması bekleniyor."
        },
        EN: {
            dropImage: "Drop image here",
            clickBrowse: "or click to browse",
            imageUploaded: "Image uploaded",
            opType: "Operation Type",
            addSmile: "Add Smile",
            eyebrowRaise: "Eyebrow Raise",
            lipWiden: "Lip Widen",
            faceSlim: "Face Slim",
            aging: "Aging",
            deaging: "De-Aging",
            fftFilter: "FFT Filter",
            intensity: "Intensity",
            transformStrength: "Transform strength",
            smoothing: "Smoothing",
            edgeBlending: "Edge blending",
            showLandmarks: "Show Landmarks",
            frequencyView: "Frequency View",
            opHistory: "Operation History",
            applyTrans: "Apply Transformation",
            downloadPDF: "Download Results as PDF",
            tabVisual: "Visual",
            tabFrequency: "Frequency Spectrum",
            tabLandmarks: "Landmarks",
            uploadWait: "Upload an image to begin",
            beforeBadge: "Before",
            afterBadge: "After",
            processing: "Processing...",
            splitView: "Split View",
            sideBySide: "Side by Side",
            origFFT: "Original - FFT Magnitude",
            procFFT: "Processed - FFT Magnitude",
            logScale: "Log scale",
            freqHz: "Frequency (Hz)",
            qualityMetrics: "Quality Metrics",
            compMode: "Comparison Mode",
            mseDesc: "Mean Squared Error",
            psnrDesc: "Peak Signal-to-Noise Ratio",
            ssimDesc: "Structural Similarity Index",
            analysisSummary: "Analysis Summary",
            waitingSummary: "Waiting for image processing to generate summary analysis."
        }
    };

    // --- DOM ELEMENTS ---
    // Top Controls
    const langToggleSpans = document.querySelectorAll('#langToggle span');
    const themeToggleBtn = document.getElementById('themeToggle');
    const themeIcon = document.getElementById('themeIcon');
    const i18nElements = document.querySelectorAll('[data-i18n]');

    // Upload & Sidebar Controls
    const imageUpload = document.getElementById('imageUpload');
    const uploadZone = document.getElementById('uploadZone');
    const thumbnailView = document.getElementById('thumbnailView');
    const thumbnailImg = document.getElementById('thumbnailImg');
    const removeImgBtn = document.getElementById('removeImgBtn');
    const opButtons = document.querySelectorAll('.op-btn');

    const intensitySlider = document.getElementById('intensitySlider');
    const intensityValue = document.getElementById('intensityValue');
    const smoothingSlider = document.getElementById('smoothingSlider');
    const smoothingValue = document.getElementById('smoothingValue');

    const toggleLandmarks = document.getElementById('showLandmarks');
    const toggleFrequency = document.getElementById('frequencyView');
    const historySection = document.getElementById('historySection');
    const historyContainer = document.getElementById('historyContainer');

    const applyBtn = document.getElementById('applyBtn');
    const downloadBtn = document.getElementById('downloadBtn');

    // Tabs & views
    const tabButtons = document.querySelectorAll('.tab-btn');
    const views = document.querySelectorAll('.view-panel');
    const viewModeControls = document.getElementById('viewModeControls');
    const btnSplitView = document.getElementById('btnSplitView');
    const btnSideBySide = document.getElementById('btnSideBySide');

    // Visual Areas
    const visualPreviewArea = document.getElementById('visualPreviewArea');
    const previewPlaceholder = document.getElementById('previewPlaceholder');
    const imageWrapper = document.getElementById('imageWrapper');
    const beforeImg = document.getElementById('beforeImg');
    const afterImg = document.getElementById('afterImg');
    const afterContainer = document.getElementById('afterContainer');
    const splitSlider = document.getElementById('splitSlider');
    const loadingOverlay = document.getElementById('loadingOverlay');

    const landmarksSvg = document.getElementById('landmarksSvg');

    // Landmarks View (isolated)
    const landmarksOnlyImg = document.getElementById('landmarksOnlyImg');
    const landmarksOnlySvg = document.getElementById('landmarksOnlySvg');
    const landmarksPlaceholder = document.getElementById('landmarksPlaceholder');

    // Metrics 
    const mseValue = document.getElementById('mseValue');
    const mseChange = document.getElementById('mseChange');
    const psnrValue = document.getElementById('psnrValue');
    const psnrChange = document.getElementById('psnrChange');
    const ssimValue = document.getElementById('ssimValue');
    const ssimChange = document.getElementById('ssimChange');
    const analysisSummary = document.getElementById('analysisSummary');

    // --- 1. LANGUAGE & THEME SYSTEM ---

    function applyLocalization(lang) {
        currentLang = lang;
        langToggleSpans.forEach(span => {
            if (span.dataset.lang === lang) span.classList.add('active');
            else span.classList.remove('active');
        });

        i18nElements.forEach(el => {
            const key = el.dataset.i18n;
            if (i18n[lang][key]) {
                el.textContent = i18n[lang][key];
            }
        });
    }

    langToggleSpans.forEach(span => {
        span.addEventListener('click', () => {
            applyLocalization(span.dataset.lang);
        });
    });

    themeToggleBtn.addEventListener('click', () => {
        isDarkMode = !isDarkMode;
        if (isDarkMode) {
            document.body.classList.remove('light-mode');
            // Sun Icon (Switch to Light)
            themeIcon.innerHTML = `
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
            `;
        } else {
            document.body.classList.add('light-mode');
            // Moon Icon (Switch to Dark)
            themeIcon.innerHTML = `
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
            `;
        }
    });

    // Initialize default Lang
    applyLocalization(currentLang);

    // --- 2. IMAGE UPLOAD & PREVIEW ---

    function clearImage() {
        currentOriginalImage = null;
        currentProcessedImage = null;
        thumbnailView.style.display = 'none';
        uploadZone.style.display = 'flex';

        previewPlaceholder.style.display = 'block';
        imageWrapper.style.display = 'none';
        viewModeControls.style.display = 'none';

        applyBtn.disabled = true;
        downloadBtn.disabled = true;

        sliderPos = 50;
        updateSplitSlider();

        // Clear Landmarks only view
        landmarksOnlyImg.src = '';
        landmarksOnlyImg.style.display = 'none';
        landmarksOnlySvg.style.display = 'none';
        landmarksPlaceholder.style.display = 'block';
    }

    function setImage(base64Str) {
        currentOriginalImage = base64Str;
        currentProcessedImage = base64Str; // Initially Same

        thumbnailImg.src = base64Str;
        uploadZone.style.display = 'none';
        thumbnailView.style.display = 'block';

        beforeImg.src = base64Str;
        afterImg.src = base64Str; // Processed img is same at beginning

        landmarksOnlyImg.src = base64Str;
        landmarksOnlyImg.style.display = 'block';
        landmarksPlaceholder.style.display = 'none';

        previewPlaceholder.style.display = 'none';
        imageWrapper.style.display = isSplitMode ? 'block' : 'flex';
        if (isSplitMode) {
            imageWrapper.classList.add('split-mode');
            imageWrapper.classList.remove('side-mode');
            sliderPos = 50;
            updateSplitSlider();
        } else {
            imageWrapper.classList.add('side-mode');
            imageWrapper.classList.remove('split-mode');
            afterContainer.style.clipPath = 'none';
        }
        viewModeControls.style.display = 'flex';

        applyBtn.disabled = false;
        downloadBtn.disabled = false;

        // Auto draw landmarks if toggled
        if (toggleLandmarks.checked) generateLandmarks();
    }

    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        currentFileObject = file; // Store raw File for FormData
        const reader = new FileReader();
        reader.onload = (e) => setImage(e.target.result);
        reader.readAsDataURL(file);
    }

    imageUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    removeImgBtn.addEventListener('click', clearImage);

    // Drag Drop
    uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
    uploadZone.addEventListener('dragleave', (e) => { e.preventDefault(); uploadZone.classList.remove('dragover'); });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault(); uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    // --- 3. SIDEBAR CONTROLS ---

    intensitySlider.addEventListener('input', (e) => intensityValue.textContent = e.target.value + '%');
    smoothingSlider.addEventListener('input', (e) => smoothingValue.textContent = e.target.value + '%');

    opButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            opButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedOperation = btn.dataset.op;
        });
    });

    // Toggles
    toggleLandmarks.addEventListener('change', (e) => {
        if (e.target.checked && currentOriginalImage) {
            generateLandmarks();
            landmarksSvg.style.display = 'block';
            landmarksOnlySvg.style.display = 'block';
        } else {
            landmarksSvg.style.display = 'none';
            landmarksOnlySvg.style.display = 'none';
        }
    });

    toggleFrequency.addEventListener('change', (e) => {
        // Just jumping to frequency tab for demo purposes
        if (e.target.checked) switchTab('frequency');
        else switchTab('visual');
    });

    // --- 4. TABS & VIEW PANELS ---

    function switchTab(targetId) {
        tabButtons.forEach(btn => {
            if (btn.dataset.target === targetId) btn.classList.add('active');
            else btn.classList.remove('active');
        });

        views.forEach(v => {
            if (v.id === `view-${targetId}`) v.classList.add('active-view');
            else v.classList.remove('active-view');
        });

        // Sync toggles visually
        if (targetId === 'frequency') toggleFrequency.checked = true;
        if (targetId === 'landmarks') toggleLandmarks.checked = true;
    }

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.target));
    });

    // --- 5. SPLIT VIEW LOGIC ---

    function updateSplitSlider() {
        if (isSplitMode) {
            splitSlider.style.left = `${sliderPos}%`;
            afterContainer.style.clipPath = `inset(0 0 0 ${sliderPos}%)`;
        }
    }

    btnSplitView.addEventListener('click', () => {
        isSplitMode = true;
        btnSideBySide.classList.remove('active');
        btnSplitView.classList.add('active');

        imageWrapper.classList.remove('side-mode');
        imageWrapper.classList.add('split-mode');
        imageWrapper.style.display = 'block';
        afterContainer.style.clipPath = `inset(0 0 0 ${sliderPos}%)`;
    });

    btnSideBySide.addEventListener('click', () => {
        isSplitMode = false;
        btnSplitView.classList.remove('active');
        btnSideBySide.classList.add('active');

        imageWrapper.classList.remove('split-mode');
        imageWrapper.classList.add('side-mode');
        imageWrapper.style.display = 'flex';
        afterContainer.style.clipPath = 'none'; // reset clip
    });

    // Drag Event Listeners
    splitSlider.addEventListener('mousedown', () => isDraggingSlider = true);
    window.addEventListener('mouseup', () => isDraggingSlider = false);

    visualPreviewArea.addEventListener('mousemove', (e) => {
        if (!isDraggingSlider || !isSplitMode) return;
        const rect = imageWrapper.getBoundingClientRect();
        let x = e.clientX - rect.left;

        // Clamp bounds
        if (x < 0) x = 0;
        if (x > rect.width) x = rect.width;

        sliderPos = (x / rect.width) * 100;
        updateSplitSlider();
    });

    // --- 6. LANDMARKS SVG RENDERER (Uses real backend data) ---

    function generateLandmarks() {
        // If we have real landmarks from backend, use them
        if (currentLandmarks && currentLandmarks.length > 0) {
            renderRealLandmarks(currentLandmarks);
            return;
        }
        // No landmarks yet — show nothing
        landmarksSvg.innerHTML = '';
        landmarksOnlySvg.innerHTML = '';
    }

    function renderRealLandmarks(landmarks) {
        if (!landmarks || landmarks.length === 0) return;

        // We need image dimensions to convert pixel coords to percentages
        // Use the afterImg (or beforeImg) natural dimensions
        const imgEl = afterImg.naturalWidth ? afterImg : beforeImg;
        const imgW = imgEl.naturalWidth || 640;
        const imgH = imgEl.naturalHeight || 480;

        let svgContent = '';

        // Draw connection lines for key facial features
        // Jawline connections (approximate MediaPipe indices)
        const jawline = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
        for (let i = 0; i < jawline.length - 1; i++) {
            const idx1 = jawline[i], idx2 = jawline[i+1];
            if (idx1 < landmarks.length && idx2 < landmarks.length) {
                const p1 = landmarks[idx1], p2 = landmarks[idx2];
                const x1 = (p1.x / imgW) * 100, y1 = (p1.y / imgH) * 100;
                const x2 = (p2.x / imgW) * 100, y2 = (p2.y / imgH) * 100;
                svgContent += `<line class="landmark-line" x1="${x1}%" y1="${y1}%" x2="${x2}%" y2="${y2}%" />`;
            }
        }

        // Draw all landmark dots
        landmarks.forEach(p => {
            const px = (p.x / imgW) * 100;
            const py = (p.y / imgH) * 100;
            svgContent += `<circle class="landmark-point" cx="${px}%" cy="${py}%" r="0.5" />`;
        });

        landmarksSvg.innerHTML = svgContent;
        landmarksOnlySvg.innerHTML = svgContent;

        landmarksSvg.style.display = 'block';
        landmarksOnlySvg.style.display = 'block';
    }

    // --- 7. HISTORY MANAGEMENT ---

    window.removeHistory = function (index) {
        operationHistory.splice(index, 1);
        renderHistory();
    };

    function addHistory(opName) {
        operationHistory.push(opName);
        renderHistory();
    }

    function renderHistory() {
        if (operationHistory.length === 0) {
            historySection.style.display = 'none';
            return;
        }
        historySection.style.display = 'block';

        let html = '';
        operationHistory.forEach((op, index) => {
            html += `<div class="chip"><span>${op}</span> <button onclick="removeHistory(${index})">×</button></div>`;
        });
        historyContainer.innerHTML = html;
    }

    // --- 8. APPLY TRANSFORMATION (REAL BACKEND API) ---

    // FFT image elements
    const origFFTImg = document.getElementById('origFFTImg');
    const procFFTImg = document.getElementById('procFFTImg');
    const origFFTPlaceholder = document.getElementById('origFFTPlaceholder');
    const procFFTPlaceholder = document.getElementById('procFFTPlaceholder');

    applyBtn.addEventListener('click', async () => {
        if (!currentFileObject && !currentOriginalImage) return;

        loadingOverlay.style.display = 'flex';
        applyBtn.disabled = true;

        try {
            // Build FormData with the raw file
            const formData = new FormData();

            if (currentFileObject) {
                formData.append('file', currentFileObject);
            } else {
                // Fallback: convert base64 data URI to Blob
                const resp = await fetch(currentOriginalImage);
                const blob = await resp.blob();
                formData.append('file', blob, 'image.jpg');
            }

            const backendOp = OP_MAP[selectedOperation] || 'smile';
            formData.append('operation', backendOp);
            formData.append('intensity', intensitySlider.value);
            formData.append('show_landmarks', toggleLandmarks.checked.toString());

            // Real API call to backend
            const response = await fetch(`${API_BASE}/apply_transformation`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                let errMsg = `Server error (${response.status})`;
                try {
                    const errData = await response.json();
                    errMsg = errData.detail || errMsg;
                } catch (_) {}
                throw new Error(errMsg);
            }

            const data = await response.json();

            // --- Render processed image ---
            if (data.processed_image) {
                currentProcessedImage = data.processed_image;
                afterImg.src = data.processed_image;
            }

            // --- Render FFT spectrum ---
            if (data.fft_spectrum) {
                if (origFFTImg) {
                    origFFTImg.src = data.fft_spectrum;
                    origFFTImg.style.display = 'block';
                    if (origFFTPlaceholder) origFFTPlaceholder.style.display = 'none';
                }
                if (procFFTImg) {
                    procFFTImg.src = data.fft_spectrum;
                    procFFTImg.style.display = 'block';
                    if (procFFTPlaceholder) procFFTPlaceholder.style.display = 'none';
                }
            }

            // --- Store & render real landmarks ---
            if (data.landmarks && data.landmarks.length > 0) {
                currentLandmarks = data.landmarks;
                if (toggleLandmarks.checked) {
                    renderRealLandmarks(currentLandmarks);
                }
            }

            // --- Update real metrics ---
            if (data.metrics) {
                lastMetrics = data.metrics;
                const m = data.metrics;

                mseValue.textContent = (m.mse !== undefined) ? m.mse.toFixed(2) : '0.00';
                psnrValue.textContent = (m.psnr !== undefined) ? (m.psnr === Infinity ? '∞' : m.psnr.toFixed(2)) : '0.00';
                ssimValue.textContent = (m.ssim !== undefined) ? m.ssim.toFixed(4) : '1.000';

                // Compute quality badges based on real values
                updateBadgeFromValue(mseChange, m.mse, 'mse');
                updateBadgeFromValue(psnrChange, m.psnr, 'psnr');
                updateBadgeFromValue(ssimChange, m.ssim, 'ssim');
            }

            const opTitle = i18n[currentLang][document.querySelector(`.op-btn[data-op="${selectedOperation}"]`).dataset.i18n] || selectedOperation;

            analysisSummary.innerHTML = `<strong>Status: Success ✓</strong><br/>Applied <em>${opTitle}</em> with ${intensitySlider.value}% intensity.<br/>MSE: ${data.metrics?.mse?.toFixed(4) || 'N/A'} | PSNR: ${data.metrics?.psnr?.toFixed(2) || 'N/A'} dB | SSIM: ${data.metrics?.ssim?.toFixed(4) || 'N/A'}`;

            // Push History
            addHistory(opTitle);

            // Trigger Split Animation subtly
            if (isSplitMode) {
                sliderPos = 25;
                updateSplitSlider();
            }

        } catch (e) {
            console.error('Error applying transformation', e);
            analysisSummary.innerHTML = `<strong style="color:#ff4d6a;">Error:</strong> ${e.message}`;
        } finally {
            loadingOverlay.style.display = 'none';
            applyBtn.disabled = false;
        }
    });

    function updateBadge(element, changeValue, reverseLogic) {
        element.textContent = (changeValue > 0 ? '+' : '') + changeValue + '%';
        element.className = 'metric-badge';
        if (changeValue === 0) { element.classList.add('neutral'); element.textContent = '~0%'; return; }

        const isPositive = reverseLogic ? changeValue < 0 : changeValue > 0;
        if (isPositive) element.classList.add('positive');
        else element.classList.add('negative');
    }

    function updateBadgeFromValue(element, value, metricType) {
        element.className = 'metric-badge';
        if (metricType === 'mse') {
            if (value < 20) { element.textContent = 'Low'; element.classList.add('positive'); }
            else if (value < 80) { element.textContent = 'Medium'; element.classList.add('neutral'); }
            else { element.textContent = 'High'; element.classList.add('negative'); }
        } else if (metricType === 'psnr') {
            if (value === Infinity || value > 40) { element.textContent = 'Excellent'; element.classList.add('positive'); }
            else if (value > 25) { element.textContent = 'Good'; element.classList.add('positive'); }
            else { element.textContent = 'Low'; element.classList.add('negative'); }
        } else if (metricType === 'ssim') {
            if (value > 0.9) { element.textContent = 'High'; element.classList.add('positive'); }
            else if (value > 0.7) { element.textContent = 'Medium'; element.classList.add('neutral'); }
            else { element.textContent = 'Low'; element.classList.add('negative'); }
        }
    }

    // --- EXPORT: Download processed image as PNG ---
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            if (!currentProcessedImage) return;

            // Extract base64 from data URI
            const base64Data = currentProcessedImage.split(',')[1];
            if (!base64Data) return;

            const byteChars = atob(base64Data);
            const byteArray = new Uint8Array(byteChars.length);
            for (let i = 0; i < byteChars.length; i++) {
                byteArray[i] = byteChars.charCodeAt(i);
            }
            const blob = new Blob([byteArray], { type: 'image/png' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `warped_${selectedOperation}_${Date.now()}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }

    // --- 9. METRICS TOGGLES & COMPARISON MODE ---
    const metricsHeader = document.getElementById('metricsHeader');
    const metricsGrid = document.getElementById('metricsGrid');
    const metricsChevron = document.getElementById('metricsChevron');
    const comparisonModeInput = document.getElementById('comparisonMode');
    
    if (metricsHeader && metricsGrid && metricsChevron) {
        metricsHeader.addEventListener('click', () => {
            metricsGrid.classList.toggle('collapsed');
            if (metricsGrid.classList.contains('collapsed')) {
                metricsChevron.style.transform = 'rotate(-90deg)';
            } else {
                metricsChevron.style.transform = 'rotate(0deg)';
            }
        });
    }

    if (comparisonModeInput) {
        comparisonModeInput.addEventListener('change', (e) => {
            const isChecked = e.target.checked;
            document.querySelectorAll('.metric-badge').forEach(badge => {
                badge.style.display = isChecked ? 'inline-block' : 'none';
            });
        });
        
        // Initialize comparison mode state
        if (!comparisonModeInput.checked) {
            document.querySelectorAll('.metric-badge').forEach(badge => {
                badge.style.display = 'none';
            });
        }
    }

});
