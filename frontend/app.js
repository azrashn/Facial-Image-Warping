document.addEventListener('DOMContentLoaded', () => {

    // --- STATE VARIABLES ---
    let currentOriginalImage = null; // The uploaded image
    let currentProcessedImage = null; // The returned image after apply
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
        const reader = new FileReader();
        reader.onload = (e) => setImage(e.target.result);
        reader.readAsDataURL(file);
    }

    imageUpload.addEventListener('click', function () {
        this.value = "";
    });

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

    // --- 6. LANDMARKS SVG GENERATOR (Mocked Algortihm from React) ---

    function generateLandmarks() {
        if (!currentOriginalImage) return;

        const centerX = 50; const centerY = 45;
        const landmarks = [];

        // Jaw line (17 points)
        for (let i = 0; i < 17; i++) {
            const angle = Math.PI + (i / 16) * Math.PI;
            landmarks.push({ x: centerX + Math.cos(angle) * 25, y: centerY + Math.sin(angle) * 30 });
        }
        // Eyebrows (10 points)
        for (let i = 0; i < 5; i++) {
            landmarks.push({ x: centerX - 15 + i * 3, y: centerY - 12 });
            landmarks.push({ x: centerX + 5 + i * 3, y: centerY - 12 });
        }
        // Eyes (12 points)
        for (let i = 0; i < 6; i++) {
            const angle = (i / 6) * Math.PI * 2;
            landmarks.push({ x: centerX - 10 + Math.cos(angle) * 4, y: centerY - 5 + Math.sin(angle) * 3 });
            landmarks.push({ x: centerX + 10 + Math.cos(angle) * 4, y: centerY - 5 + Math.sin(angle) * 3 });
        }
        // Nose (9 points)
        for (let i = 0; i < 9; i++) {
            landmarks.push({ x: centerX - 2 + (i % 3) * 2, y: centerY + Math.floor(i / 3) * 4 });
        }
        // Mouth (15 points)
        for (let i = 0; i < 15; i++) {
            const angle = (i / 14) * Math.PI;
            landmarks.push({ x: centerX + Math.cos(angle) * 12 - 12, y: centerY + 15 + Math.sin(angle) * 6 });
        }

        // Render SVG Lines and Circles
        let svgContent = '';

        // Lines (Triangulation sim)
        landmarks.slice(0, 60).forEach((p, i) => {
            if (i < landmarks.length - 1) {
                const next = landmarks[(i + 1) % landmarks.length];
                svgContent += `<line class="landmark-line" x1="${p.x}%" y1="${p.y}%" x2="${next.x}%" y2="${next.y}%" />`;
            }
        });

        // Dots
        landmarks.forEach(p => {
            svgContent += `<circle class="landmark-point" cx="${p.x}%" cy="${p.y}%" r="1" />`;
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

    // --- 8. APPLY TRANSFORMATION & MOCK API ---

    applyBtn.addEventListener('click', async () => {
        if (!currentOriginalImage) return;

        loadingOverlay.style.display = 'flex';

        try {
            // MOCK DELAY (1.5s)
            await new Promise(res => setTimeout(res, 1500));

            // Simulating processed image (for demo, just keeping the same base64 but slightly tinting or applying CSS later? No, we just set the same image and update metric to fake it)
            currentProcessedImage = currentOriginalImage;
            afterImg.src = currentProcessedImage;

            // Update Metrics Mock
            const baseMSE = (Math.random() * 10 + 5).toFixed(2);
            const basePSNR = (Math.random() * 10 + 25).toFixed(2);
            const baseSSIM = (0.85 + Math.random() * 0.1).toFixed(3);

            mseValue.textContent = baseMSE;
            updateBadge(mseChange, Math.round((Math.random() - 0.5) * 20), true);

            psnrValue.textContent = basePSNR;
            updateBadge(psnrChange, Math.round((Math.random() - 0.3) * 15), false);

            ssimValue.textContent = baseSSIM;
            updateBadge(ssimChange, Math.round((Math.random() - 0.3) * 10), false);

            const opTitle = i18n[currentLang][document.querySelector(`.op-btn[data-op="${selectedOperation}"]`).dataset.i18n] || selectedOperation;

            analysisSummary.innerHTML = `<strong>Status: Success</strong><br/>Applied ${opTitle} with ${intensitySlider.value}% intensity.`;

            // Push History
            addHistory(opTitle);

            // Trigger Split Animation subtly
            if (isSplitMode) {
                sliderPos = 25;
                updateSplitSlider();
            }

        } catch (e) {
            console.error('Error applying transofrmation', e);
        } finally {
            loadingOverlay.style.display = 'none';
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

    // --- 9.5. NEW UI INTEGRATIONS ---

    // Emoji Buttons
    const emojiButtons = document.querySelectorAll('.emoji-btn');
    emojiButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const op = btn.dataset.op;
            const intensity = btn.dataset.intensity;

            // Set operation
            opButtons.forEach(b => b.classList.remove('active'));
            const targetOpBtn = document.querySelector(`.op-btn[data-op="${op}"]`);
            if (targetOpBtn) targetOpBtn.classList.add('active');
            selectedOperation = op;

            // Set intensity
            intensitySlider.value = intensity;
            intensityValue.textContent = intensity + '%';

            // Trigger apply
            if (!applyBtn.disabled) {
                applyBtn.click();
            }
        });
    });

    // Makeup
    const applyMakeupBtn = document.getElementById('applyMakeupBtn');
    const makeupRegion = document.getElementById('makeupRegion');
    const makeupColor = document.getElementById('makeupColor');
    const makeupOpacity = document.getElementById('makeupOpacity');

    if (applyMakeupBtn) {
        applyMakeupBtn.addEventListener('click', async () => {
            if (!uploadedFile || !currentOriginalImage) return;

            const formData = new FormData();
            formData.append('image', uploadedFile);
            formData.append('region', makeupRegion.value);
            formData.append('color', makeupColor.value);
            formData.append('opacity', makeupOpacity.value / 100.0);

            loadingOverlay.style.display = 'flex';
            try {
                const response = await fetch(`${API_BASE}/process/makeup`, { method: 'POST', body: formData });
                const payload = await response.json();
                if (!response.ok) throw new Error(payload?.detail || 'Makeup failed.');

                currentProcessedImage = payload.image_b64;
                afterImg.src = currentProcessedImage;
                landmarksOnlyImg.src = currentProcessedImage;

                addHistory(`Makeup: ${makeupRegion.value}`);
                analysisSummary.innerHTML = `<strong>Status: Success</strong><br/>Applied makeup to ${makeupRegion.value}.`;

                if (isSplitMode) { sliderPos = 25; updateSplitSlider(); }
            } catch (e) {
                console.error('[Makeup] Error:', e);
                analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>${e.message || 'Makeup failed.'}`;
            } finally {
                loadingOverlay.style.display = 'none';
            }
        });
    }

    // Hair Color
    const hairSwatches = document.querySelectorAll('.hair-swatch');
    hairSwatches.forEach(swatch => {
        swatch.addEventListener('click', async () => {
            if (!uploadedFile || !currentOriginalImage) return;
            const color = swatch.dataset.color;

            const formData = new FormData();
            formData.append('image', uploadedFile);
            formData.append('color', color);

            loadingOverlay.style.display = 'flex';
            try {
                const response = await fetch(`${API_BASE}/process/hair-color`, { method: 'POST', body: formData });
                const payload = await response.json();
                if (!response.ok) throw new Error(payload?.detail || 'Hair color failed.');

                currentProcessedImage = payload.image_b64;
                afterImg.src = currentProcessedImage;
                landmarksOnlyImg.src = currentProcessedImage;

                addHistory(`Hair Color: ${color}`);
                analysisSummary.innerHTML = `<strong>Status: Success</strong><br/>Applied hair color ${color}.`;

                if (isSplitMode) { sliderPos = 25; updateSplitSlider(); }
            } catch (e) {
                console.error('[Hair Color] Error:', e);
                analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>${e.message || 'Hair color failed.'}`;
            } finally {
                loadingOverlay.style.display = 'none';
            }
        });
    });

    // Cartoon Filter
    const cartoonBtn = document.getElementById('cartoonBtn');
    if (cartoonBtn) {
        cartoonBtn.addEventListener('click', async () => {
            if (!uploadedFile || !currentOriginalImage) return;

            const formData = new FormData();
            formData.append('image', uploadedFile);

            loadingOverlay.style.display = 'flex';
            try {
                const response = await fetch(`${API_BASE}/process/cartoon`, { method: 'POST', body: formData });
                const payload = await response.json();
                if (!response.ok) throw new Error(payload?.detail || 'Cartoon filter failed.');

                currentProcessedImage = payload.image_b64;
                afterImg.src = currentProcessedImage;
                landmarksOnlyImg.src = currentProcessedImage;

                addHistory('Cartoon Filter');
                analysisSummary.innerHTML = `<strong>Status: Success</strong><br/>Applied Cartoon Filter.`;

                if (isSplitMode) { sliderPos = 25; updateSplitSlider(); }
            } catch (e) {
                console.error('[Cartoon Filter] Error:', e);
                analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>${e.message || 'Cartoon filter failed.'}`;
            } finally {
                loadingOverlay.style.display = 'none';
            }
        });
    }

    // Camera Capture
    const cameraVideo = document.getElementById('cameraVideo');
    const startCameraBtn = document.getElementById('startCameraBtn');
    const captureBtn = document.getElementById('captureBtn');
    const cameraCanvas = document.getElementById('cameraCanvas');
    let mediaStream = null;

    if (startCameraBtn) {
        startCameraBtn.addEventListener('click', async () => {
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
                cameraVideo.style.display = 'none';
                captureBtn.style.display = 'none';
                startCameraBtn.textContent = 'Start Camera';
            } else {
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
                    cameraVideo.srcObject = mediaStream;
                    cameraVideo.style.display = 'block';
                    captureBtn.style.display = 'block';
                    startCameraBtn.textContent = 'Stop Camera';
                } catch (err) {
                    console.error('Error accessing camera:', err);
                    alert('Could not access camera. Please allow permissions.');
                }
            }
        });
    }

    if (captureBtn) {
        captureBtn.addEventListener('click', () => {
            if (!mediaStream) return;
            cameraCanvas.width = cameraVideo.videoWidth;
            cameraCanvas.height = cameraVideo.videoHeight;
            cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
            cameraCanvas.toBlob((blob) => {
                if (blob) {
                    const file = new File([blob], 'camera-capture.png', { type: 'image/png' });
                    handleFile(file);
                }
            }, 'image/png');
        });
    }

});
