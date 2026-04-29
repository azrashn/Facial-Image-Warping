document.addEventListener('DOMContentLoaded', () => {

    // --- STATE VARIABLES ---
    let currentOriginalImage = null; // The uploaded image
    let currentProcessedImage = null; // The returned image after apply
    let uploadedFile = null;
    let selectedOperation = 'smile';
    let isSplitMode = true;
    let sliderPos = 50; // percentage
    let isDraggingSlider = false;
    let operationHistory = [];
    let previousMetrics = null; // stores last metrics for delta computation

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
            ageEstimation: "Yaş Tahmini",
            ageResult: "Tahmini Yaş:",
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
            ageEstimation: "Age Estimation",
            ageResult: "Estimated Age:",
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
    const origSpectrumImg = document.getElementById('origSpectrumImg');
    const procSpectrumImg = document.getElementById('procSpectrumImg');
    const API_BASE = 'http://127.0.0.1:8000';

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
        uploadedFile = null;
        previousMetrics = null;
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
        setSpectrumImages(null, null);
    }

    function setImage(base64Str, fileObj) {
        currentOriginalImage = base64Str;
        currentProcessedImage = base64Str; // Initially Same
        uploadedFile = fileObj;

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
        setSpectrumImages(null, null);
    }

    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = (e) => setImage(e.target.result, file);
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
        if (targetId === 'landmarks') {
            toggleLandmarks.checked = true;
            if (uploadedFile) generateLandmarks();
        }

        // Re-align landmarks after layout change
        requestAnimationFrame(() => realignLandmarkSvgs());
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
        requestAnimationFrame(() => realignLandmarkSvgs());
    });

    btnSideBySide.addEventListener('click', () => {
        isSplitMode = false;
        btnSplitView.classList.remove('active');
        btnSideBySide.classList.add('active');

        imageWrapper.classList.remove('split-mode');
        imageWrapper.classList.add('side-mode');
        imageWrapper.style.display = 'flex';
        afterContainer.style.clipPath = 'none'; // reset clip
        requestAnimationFrame(() => realignLandmarkSvgs());
    });

    // Drag Event Listeners (mouse)
    splitSlider.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isDraggingSlider = true;
    });
    window.addEventListener('mouseup', () => isDraggingSlider = false);

    visualPreviewArea.addEventListener('mousemove', (e) => {
        if (!isDraggingSlider || !isSplitMode) return;
        e.preventDefault();
        const rect = imageWrapper.getBoundingClientRect();
        let x = e.clientX - rect.left;

        // Clamp bounds
        if (x < 0) x = 0;
        if (x > rect.width) x = rect.width;

        sliderPos = (x / rect.width) * 100;
        updateSplitSlider();
    });

    // Drag Event Listeners (touch)
    splitSlider.addEventListener('touchstart', (e) => {
        e.preventDefault();
        isDraggingSlider = true;
    }, { passive: false });

    window.addEventListener('touchend', () => isDraggingSlider = false);

    visualPreviewArea.addEventListener('touchmove', (e) => {
        if (!isDraggingSlider || !isSplitMode) return;
        e.preventDefault();
        const touch = e.touches[0];
        const rect = imageWrapper.getBoundingClientRect();
        let x = touch.clientX - rect.left;

        if (x < 0) x = 0;
        if (x > rect.width) x = rect.width;

        sliderPos = (x / rect.width) * 100;
        updateSplitSlider();
    }, { passive: false });

    // --- 6. LANDMARKS – Fetch from backend & render as SVG ---

    /**
     * Compute the actual rendered position/size of an <img> using
     * object-fit: contain inside its container.  Returns {x, y, w, h}
     * in px relative to the container.
     */
    function getRenderedImageRect(imgEl) {
        const containerW = imgEl.clientWidth;
        const containerH = imgEl.clientHeight;
        const naturalW = imgEl.naturalWidth || 512;
        const naturalH = imgEl.naturalHeight || 512;

        const scale = Math.min(containerW / naturalW, containerH / naturalH);
        const renderedW = naturalW * scale;
        const renderedH = naturalH * scale;
        const offsetX = (containerW - renderedW) / 2;
        const offsetY = (containerH - renderedH) / 2;

        return { x: offsetX, y: offsetY, w: renderedW, h: renderedH };
    }

    /**
     * Position an SVG overlay so it exactly covers the rendered area of
     * the associated <img> element (which uses object-fit: contain).
     */
    function alignSvgToImage(svgEl, imgEl) {
        if (!imgEl || !svgEl) return;
        const rect = getRenderedImageRect(imgEl);
        svgEl.style.position = 'absolute';
        svgEl.style.left   = rect.x + 'px';
        svgEl.style.top    = rect.y + 'px';
        svgEl.style.width  = rect.w + 'px';
        svgEl.style.height = rect.h + 'px';
    }

    /** Re-align all visible landmark SVGs after layout changes. */
    function realignLandmarkSvgs() {
        if (landmarksSvg.style.display !== 'none') {
            alignSvgToImage(landmarksSvg, afterImg);
        }
        if (landmarksOnlySvg.style.display !== 'none') {
            alignSvgToImage(landmarksOnlySvg, landmarksOnlyImg);
        }
    }

    // Re-align on window resize so landmarks stay pinned to the face
    window.addEventListener('resize', realignLandmarkSvgs);

    /**
     * Fetch 468 MediaPipe FaceMesh landmarks from the backend and render
     * them as turquoise SVG dots over the face image.
     *
     * The backend returns normalised coordinates (0.0 – 1.0). We multiply
     * by 512 (the preprocessed image size) to get pixel positions inside
     * the SVG whose viewBox is "0 0 512 512".
     */
    async function generateLandmarks() {
        if (!uploadedFile) return;

        const CANVAS = 512; // matches SVG viewBox and backend preprocess size

        try {
            const formData = new FormData();
            formData.append('image', uploadedFile);

            const response = await fetch(`${API_BASE}/process/landmarks`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `Server responded with ${response.status}`);
            }

            const data = await response.json();
            const landmarks = data.landmarks;

            if (!Array.isArray(landmarks) || landmarks.length === 0) {
                throw new Error('No landmarks returned from backend.');
            }

            // Use the preprocessed 512×512 image from backend for the
            // Landmarks-only tab so coordinates match pixel-perfectly.
            if (data.image_b64) {
                landmarksOnlyImg.src = data.image_b64;
                landmarksOnlyImg.style.display = 'block';
                landmarksPlaceholder.style.display = 'none';
            }

            // --- Build SVG content ---
            let svgContent = '';

            // Optional triangulation-style lines between consecutive landmarks
            // (keep only a light mesh for the first 60 to stay subtle)
            const lineCount = Math.min(60, landmarks.length - 1);
            for (let i = 0; i < lineCount; i++) {
                const p = landmarks[i];
                const next = landmarks[i + 1];
                const x1 = p.x * CANVAS;
                const y1 = p.y * CANVAS;
                const x2 = next.x * CANVAS;
                const y2 = next.y * CANVAS;
                svgContent += `<line class="landmark-line" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" />`;
            }

            // Dots for every landmark
            landmarks.forEach(lm => {
                const cx = lm.x * CANVAS;
                const cy = lm.y * CANVAS;
                svgContent += `<circle class="landmark-point" cx="${cx}" cy="${cy}" r="1.5" />`;
            });

            landmarksSvg.innerHTML = svgContent;
            landmarksOnlySvg.innerHTML = svgContent;

            landmarksSvg.style.display = 'block';
            landmarksOnlySvg.style.display = 'block';

            // Align SVGs to the actual rendered image area
            requestAnimationFrame(() => realignLandmarkSvgs());

            console.log(`[Landmarks] Rendered ${landmarks.length} points.`);
        } catch (err) {
            console.error('[Landmarks] Error:', err);
            analysisSummary.innerHTML =
                `<strong>Landmarks:</strong> ${err.message || 'Failed to fetch landmarks.'}`;

            // Clear SVGs on error
            landmarksSvg.innerHTML = '';
            landmarksOnlySvg.innerHTML = '';
            landmarksSvg.style.display = 'none';
            landmarksOnlySvg.style.display = 'none';
        }
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

    // --- 8. APPLY TRANSFORMATION & REAL API ---

    function setSpectrumImages(origSpectrumB64, procSpectrumB64) {
        if (origSpectrumB64 && origSpectrumImg) {
            origSpectrumImg.src = origSpectrumB64;
            origSpectrumImg.style.display = 'block';
        } else if (origSpectrumImg) {
            origSpectrumImg.src = '';
            origSpectrumImg.style.display = 'none';
        }

        if (procSpectrumB64 && procSpectrumImg) {
            procSpectrumImg.src = procSpectrumB64;
            procSpectrumImg.style.display = 'block';
        } else if (procSpectrumImg) {
            procSpectrumImg.src = '';
            procSpectrumImg.style.display = 'none';
        }
    }

    function updateMetricsFromApi(metrics) {
        const parsed = {
            mse: Number(metrics?.mse ?? 0),
            psnr: Number(metrics?.psnr ?? 0),
            ssim: Number(metrics?.ssim ?? 0),
        };
        mseValue.textContent = parsed.mse.toFixed(2);
        psnrValue.textContent = parsed.psnr.toFixed(2);
        ssimValue.textContent = parsed.ssim.toFixed(3);

        if (previousMetrics) {
            // Compute percentage delta: ((new - old) / old) * 100
            const mseDelta = previousMetrics.mse !== 0
                ? ((parsed.mse - previousMetrics.mse) / Math.abs(previousMetrics.mse)) * 100 : null;
            const psnrDelta = previousMetrics.psnr !== 0
                ? ((parsed.psnr - previousMetrics.psnr) / Math.abs(previousMetrics.psnr)) * 100 : null;
            const ssimDelta = previousMetrics.ssim !== 0
                ? ((parsed.ssim - previousMetrics.ssim) / Math.abs(previousMetrics.ssim)) * 100 : null;

            updateBadge(mseChange, mseDelta, true);
            updateBadge(psnrChange, psnrDelta, false);
            updateBadge(ssimChange, ssimDelta, false);
        } else {
            // No baseline yet – hide badges instead of showing fake ~0%
            updateBadge(mseChange, null, true);
            updateBadge(psnrChange, null, false);
            updateBadge(ssimChange, null, false);
        }

        // Store current as baseline for next operation
        previousMetrics = { ...parsed };
    }

    applyBtn.addEventListener('click', async () => {
        if (!uploadedFile || !currentOriginalImage) return;

        const isAgeEstimation = selectedOperation === 'age_estimation';
        const isWarpOperation = ['smile', 'eyebrow', 'lip', 'slim'].includes(selectedOperation);
        const endpoint = isAgeEstimation
            ? `${API_BASE}/process/estimate_age`
            : isWarpOperation ? `${API_BASE}/process/warp` : `${API_BASE}/process/age`;
        const formData = new FormData();
        formData.append('image', uploadedFile);
        formData.append('operation', selectedOperation);
        formData.append('intensity', intensitySlider.value);
        formData.append('smoothing', smoothingSlider.value);

        console.log('[Apply] selected operation:', selectedOperation);
        console.log('[Apply] endpoint:', endpoint);

        loadingOverlay.style.display = 'flex';

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData,
            });
            const payload = await response.json();
            console.log('[Apply] response:', payload);

            if (!response.ok) {
                throw new Error(payload?.detail || 'Processing failed.');
            }

            // --- Age Estimation: JSON-only response (no image) ---
            if (isAgeEstimation) {
                const ageLabel = i18n[currentLang]?.ageResult || 'Estimated Age:';
                const age = payload.estimated_age ?? '—';
                const bucket = payload.age_bucket || '';
                analysisSummary.innerHTML =
                    `<div style="text-align:center;padding:12px 0;">`
                    + `<span style="font-size:1rem;opacity:0.8;">${ageLabel}</span><br/>`
                    + `<span style="font-size:2.8rem;font-weight:800;color:#00e5ff;line-height:1.2;">${age}</span>`
                    + (bucket ? `<br/><span style="font-size:0.9rem;opacity:0.6;">${bucket}</span>` : '')
                    + `</div>`;
                addHistory(i18n[currentLang]?.ageEstimation || 'Age Estimation');
                loadingOverlay.style.display = 'none';
                return; // Skip image rendering
            }

            if (!payload?.image_b64) {
                throw new Error('Missing image_b64 in response.');
            }

            currentProcessedImage = payload.image_b64;
            afterImg.src = currentProcessedImage;
            landmarksOnlyImg.src = currentProcessedImage;

            updateMetricsFromApi(payload.metrics || { mse: 0, psnr: 0, ssim: 0 });
            setSpectrumImages(payload.orig_spectrum_b64 || null, payload.proc_spectrum_b64 || null);

            const opButton = document.querySelector(`.op-btn[data-op="${selectedOperation}"]`);
            const opTitle = opButton
                ? (i18n[currentLang][opButton.dataset.i18n] || selectedOperation)
                : selectedOperation;
            analysisSummary.innerHTML = `<strong>Status: Success</strong><br/>Applied ${opTitle} with ${intensitySlider.value}% intensity.`;
            addHistory(opTitle);

            if (isSplitMode) {
                sliderPos = 25;
                updateSplitSlider();
            }
        } catch (e) {
            console.error('[Apply] CORS or Network error:', e);
            analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>${e.message || 'Transformation failed due to Network/CORS or API error.'}`;
        } finally {
            loadingOverlay.style.display = 'none';
        }
    });

    function updateBadge(element, changeValue, reverseLogic) {
        element.className = 'metric-badge';

        // No baseline available – hide badge entirely
        if (changeValue === null || changeValue === undefined) {
            element.textContent = '';
            element.style.display = 'none';
            return;
        }

        element.style.display = '';
        const rounded = Math.round(changeValue * 10) / 10;

        if (rounded === 0) {
            element.textContent = '0%';
            element.classList.add('neutral');
            return;
        }

        element.textContent = (rounded > 0 ? '+' : '') + rounded.toFixed(1) + '%';
        const isPositive = reverseLogic ? rounded < 0 : rounded > 0;
        if (isPositive) element.classList.add('positive');
        else element.classList.add('negative');
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
