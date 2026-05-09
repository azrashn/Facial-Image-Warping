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
            waitingSummary: "Özet analiz için görüntü işlemenin tamamlanması bekleniyor.",
            // New Face Features
            newFaceFeatures: "Yeni Yüz Özellikleri",
            eyeScale: "Göz Ölçeği",
            applyEyeScale: "Göz Ölçeği Uygula",
            facialHairStyle: "Yüz Kılı Stili",
            fullBeard: "Tam Sakal",
            mustache: "Bıyık",
            beardDarkness: "Sakal Koyuluğu",
            addFacialHair: "Yüz Kılı Ekle",
            ageComparison: "Yaş Karşılaştırması (Önce ve Sonra)",
            compareAges: "Yaşları Karşılaştır",
            // Panels
            emojiPresets: "Emoji Kalıpları",
            makeupPanel: "Makyaj",
            hairColor: "Saç Rengi",
            filters: "Filtreler",
            cameraCapture: "Kamera",
            uploadPrompt: "Başlamak için resim yükleyin",
            downloadPng: "Sonucu PNG Olarak İndir",
            // Makeup
            makeupLips: "Dudaklar",
            makeupEyeshadow: "Göz Farı",
            makeupBlush: "Allık",
            applyMakeup: "Uygula",
            // Filters & Camera
            cartoonFilter: "Karikatür Filtresi",
            cameraOffline: "Kamera Kapalı",
            startCamera: "Kamerayı Başlat",
            capturePhoto: "Fotoğraf Çek",
            // Eyewear
            eyewearPanel: "Aksesuar",
            glassesType: "Gözlük",
            metalAviator: "Klasik Damla (Aviator)",
            acetateWayfarer: "Kemik Çerçeve (Modern)",
            minimalistRound: "İnce Yuvarlak (Retro)",
            applyGlasses: "Gözlük Uygula",
            // Emoji Presets
            presetAlien: "Uzaylı",
            presetRobot: "Robot",
            presetClown: "Joker / Palyaço",
            presetStarEyes: "Yıldızlı Bakış",
            presetHeartEyes: "Aşık",
            presetCrying: "Ağlayan",
            // FFT Lab
            tabFFTLab: "FFT Laboratuvarı",
            fftMagnitude: "Büyüklük (Magnitude)",
            fftPhase: "Faz (Phase)",
            interactiveSelection: "Etkileşimli Bölge Seçimi: Bir alan seçmek için tıklayıp sürükleyin.",
            selectionOutput: "Seçim Çıktısı",
            noDataFft: "Veri yok (FFT Filtresi Çalıştırın)",
            selectRegionOutput: "Çıktı oluşturmak için bir bölge seçin",
            procFFTPhase: "İşlenmiş - FFT Fazı",
            origFFTPhase: "Orijinal - FFT Fazı"
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
            waitingSummary: "Waiting for image processing to generate summary analysis.",
            // New Face Features
            newFaceFeatures: "New Face Features",
            eyeScale: "Eye Scale",
            applyEyeScale: "Apply Eye Scale",
            facialHairStyle: "Facial Hair Style",
            fullBeard: "Full Beard",
            mustache: "Mustache",
            beardDarkness: "Beard Darkness",
            addFacialHair: "Add Facial Hair",
            ageComparison: "Age Comparison (Before vs After)",
            compareAges: "Compare Ages",
            // Panels
            emojiPresets: "Emoji Presets",
            makeupPanel: "Makeup",
            hairColor: "Hair Color",
            filters: "Filters",
            cameraCapture: "Camera",
            uploadPrompt: "Upload an image to begin",
            downloadPng: "Download Result as PNG",
            // Makeup
            makeupLips: "Lips",
            makeupEyeshadow: "Eyeshadow",
            makeupBlush: "Blush",
            applyMakeup: "Apply",
            // Filters & Camera
            cartoonFilter: "Cartoon Filter",
            cameraOffline: "Camera Offline",
            startCamera: "Start Camera",
            capturePhoto: "Capture",
            // Eyewear
            eyewearPanel: "Accessory",
            glassesType: "Glasses",
            metalAviator: "Classic Teardrop (Aviator)",
            acetateWayfarer: "Thick Frame (Modern)",
            minimalistRound: "Thin Round (Retro)",
            applyGlasses: "Apply Glasses",
            // Emoji Presets
            presetAlien: "Alien",
            presetRobot: "Robot",
            presetClown: "Joker",
            presetStarEyes: "Starry Gaze",
            presetHeartEyes: "Heart-Eyes",
            presetCrying: "Crying",
            // FFT Lab
            tabFFTLab: "FFT Laboratory",
            fftMagnitude: "Magnitude",
            fftPhase: "Phase",
            interactiveSelection: "Interactive Region Selection: Click and drag to select a patch.",
            selectionOutput: "Selection Output",
            noDataFft: "No data (Run FFT Filter)",
            selectRegionOutput: "Select a region to generate output",
            procFFTPhase: "Processed - FFT Phase",
            origFFTPhase: "Original - FFT Phase"
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
    const downloadPngBtn = document.getElementById('downloadPngBtn');

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
    const origPhaseImg = document.getElementById('origPhaseImg');
    const procPhaseImg = document.getElementById('procPhaseImg');
    
    // FFT Lab Elements
    const fftLabProcSpectrumImg = document.getElementById('fftLabProcSpectrumImg');
    const fftLabProcPhaseImg = document.getElementById('fftLabProcPhaseImg');
    const fftLabProcSpectrumPlaceholder = document.getElementById('fftLabProcSpectrumPlaceholder');
    const fftLabProcPhasePlaceholder = document.getElementById('fftLabProcPhasePlaceholder');
    const fftOutputImg = document.getElementById('fftOutputImg');
    const fftOutputPlaceholder = document.getElementById('fftOutputPlaceholder');
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

        // Restore the split-layout placeholder (flex, not block!)
        previewPlaceholder.style.display = 'flex';
        imageWrapper.style.display = 'none';
        viewModeControls.style.display = 'none';

        applyBtn.disabled = true;
        downloadBtn.disabled = true;
        if (downloadPngBtn) downloadPngBtn.disabled = true;

        sliderPos = 50;
        updateSplitSlider();

        // Clear SVG landmarks
        if (landmarksSvg) landmarksSvg.innerHTML = '';
        if (landmarksOnlySvg) landmarksOnlySvg.innerHTML = '';

        // Clear Landmarks only view
        landmarksOnlyImg.src = '';
        landmarksOnlyImg.style.display = 'none';
        landmarksOnlySvg.style.display = 'none';
        landmarksPlaceholder.style.display = 'block';
        setSpectrumImages(null, null, null, null);

        // Reset analysis summary
        analysisSummary.innerHTML = '';

        // Reset metrics display
        if (mseValue) mseValue.textContent = '—';
        if (psnrValue) psnrValue.textContent = '—';
        if (ssimValue) ssimValue.textContent = '—';
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

        // Hide the split-layout placeholder, show the image workspace
        previewPlaceholder.style.display = 'none';

        // Always clean-apply the current view mode
        imageWrapper.classList.remove('split-mode', 'side-mode');
        if (isSplitMode) {
            imageWrapper.classList.add('split-mode');
            imageWrapper.style.display = 'block';
            sliderPos = 50;
            updateSplitSlider();
        } else {
            imageWrapper.classList.add('side-mode');
            imageWrapper.style.display = 'flex';
            afterContainer.style.clipPath = 'none';
        }
        viewModeControls.style.display = 'flex';

        applyBtn.disabled = false;
        downloadBtn.disabled = false;
        if (downloadPngBtn) downloadPngBtn.disabled = false;

        // Auto draw landmarks if toggled
        if (toggleLandmarks.checked) generateLandmarks();
        setSpectrumImages(null, null, null, null);
    }

    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = (e) => setImage(e.target.result, file);
        reader.readAsDataURL(file);
    }

    imageUpload.addEventListener('click', function () {
        this.value = "";
    });

    imageUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    removeImgBtn.addEventListener('click', () => {
        clearImage();
        imageUpload.value = "";   // Reset file input so re-uploading the same file triggers 'change'
    });

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
        svgEl.style.left = rect.x + 'px';
        svgEl.style.top = rect.y + 'px';
        svgEl.style.width = rect.w + 'px';
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

    function setSpectrumImages(origSpectrumB64, procSpectrumB64, origPhaseB64, procPhaseB64) {
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
            if (fftLabProcSpectrumImg) {
                fftLabProcSpectrumImg.src = procSpectrumB64;
                fftLabProcSpectrumImg.style.display = 'block';
            }
            if (fftLabProcSpectrumPlaceholder) fftLabProcSpectrumPlaceholder.style.display = 'none';
        } else if (procSpectrumImg) {
            procSpectrumImg.src = '';
            procSpectrumImg.style.display = 'none';
            if (fftLabProcSpectrumImg) {
                fftLabProcSpectrumImg.src = '';
                fftLabProcSpectrumImg.style.display = 'none';
            }
            if (fftLabProcSpectrumPlaceholder) fftLabProcSpectrumPlaceholder.style.display = 'block';
        }

        if (origPhaseB64 && origPhaseImg) {
            origPhaseImg.src = origPhaseB64;
            origPhaseImg.style.display = 'block';
        } else if (origPhaseImg) {
            origPhaseImg.src = '';
            origPhaseImg.style.display = 'none';
        }

        if (procPhaseB64 && procPhaseImg) {
            procPhaseImg.src = procPhaseB64;
            procPhaseImg.style.display = 'block';
            if (fftLabProcPhaseImg) {
                fftLabProcPhaseImg.src = procPhaseB64;
                fftLabProcPhaseImg.style.display = 'block';
            }
            if (fftLabProcPhasePlaceholder) fftLabProcPhasePlaceholder.style.display = 'none';
        } else if (procPhaseImg) {
            procPhaseImg.src = '';
            procPhaseImg.style.display = 'none';
            if (fftLabProcPhaseImg) {
                fftLabProcPhaseImg.src = '';
                fftLabProcPhaseImg.style.display = 'none';
            }
            if (fftLabProcPhasePlaceholder) fftLabProcPhasePlaceholder.style.display = 'block';
        }
    }

    function updateMetricsFromApi(metrics) {
        const parsed = {
            mse: Number(metrics?.mse ?? 0),
            psnr: Number(metrics?.psnr ?? 0),
            ssim: Number(metrics?.ssim ?? 0),
        };
        mseValue.textContent = parsed.mse < 0.01 ? parsed.mse.toExponential(2) : parsed.mse.toFixed(2);
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
            ? `${API_BASE}/process/estimate-age`
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
            setSpectrumImages(
                payload.orig_spectrum_b64 || null,
                payload.proc_spectrum_b64 || null,
                payload.orig_phase_b64 || null,
                payload.proc_phase_b64 || null
            );

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

    // --- 9.5. NEW UI INTEGRATIONS ---

    // --- Emoji Preset Buttons (6 specific presets → /process/emoji-preset) ---
    const emojiPresetNames = [
        { id: 'btnPresetAlien', preset: 'alien', labelKey: 'presetAlien' },
        { id: 'btnPresetRobot', preset: 'robot', labelKey: 'presetRobot' },
        { id: 'btnPresetClown', preset: 'clown', labelKey: 'presetClown' },
        { id: 'btnPresetStarEyes', preset: 'star_eyes', labelKey: 'presetStarEyes' },
        { id: 'btnPresetHeartEyes', preset: 'heart_eyes', labelKey: 'presetHeartEyes' },
        { id: 'btnPresetCrying', preset: 'crying', labelKey: 'presetCrying' },
    ];

    async function applyEmojiPreset(presetName, labelKey) {
        if (!currentOriginalImage) return;

        // Highlight the clicked button
        document.querySelectorAll('.emoji-btn').forEach(b => b.classList.remove('active'));
        const clickedBtn = document.querySelector(`[data-preset="${presetName}"]`);
        if (clickedBtn) clickedBtn.classList.add('active');

        loadingOverlay.style.display = 'flex';

        try {
            // Build payload – alien gets a detailed description for the backend
            const payloadBody = {
                image_b64: currentOriginalImage,
                preset_name: presetName,
            };
            if (presetName === 'alien') {
                payloadBody.description = 'Highly refined alien transformation. Procedurally refine face selection. Perform eye scaling (positive, extreme). Sculpt chin into a distinct triangular (alien-like) shape. Apply a smooth, realistic bright green color overlay to the refined face mask.';
            }

            const response = await fetch(`${API_BASE}/process/emoji-preset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payloadBody),
            });

            const payload = await response.json();
            if (!response.ok) throw new Error(payload?.detail || 'Emoji preset failed.');

            if (!payload?.image_b64) throw new Error('Missing image_b64 in response.');

            currentProcessedImage = payload.image_b64;
            afterImg.src = currentProcessedImage;
            landmarksOnlyImg.src = currentProcessedImage;

            updateMetricsFromApi(payload.metrics || { mse: 0, psnr: 0, ssim: 0 });
            setSpectrumImages(
                payload.orig_spectrum_b64 || null,
                payload.proc_spectrum_b64 || null,
                payload.orig_phase_b64 || null,
                payload.proc_phase_b64 || null
            );

            const label = i18n[currentLang]?.[labelKey] || presetName;
            analysisSummary.innerHTML = `<strong>Status: Success</strong><br/>Applied Emoji Preset: ${label}.`;
            addHistory(`Emoji: ${label}`);

            if (isSplitMode) { sliderPos = 25; updateSplitSlider(); }
        } catch (e) {
            console.error(`[Emoji Preset: ${presetName}] Error:`, e);
            analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>${e.message || 'Emoji preset failed.'}`;
        } finally {
            loadingOverlay.style.display = 'none';
        }
    }

    emojiPresetNames.forEach(({ id, preset, labelKey }) => {
        const btn = document.getElementById(id);
        if (btn) {
            btn.addEventListener('click', () => applyEmojiPreset(preset, labelKey));
        }
    });

    // --- CLOWN: dedicated high-quality endpoint override ---
    const clownBtn = document.getElementById('btnPresetClown');
    if (clownBtn) {
        // Remove the generic emoji-preset listener by replacing with a dedicated handler
        clownBtn.replaceWith(clownBtn.cloneNode(true));   // strip old listeners
        const freshClownBtn = document.getElementById('btnPresetClown');

        freshClownBtn.addEventListener('click', async () => {
            if (!uploadedFile || !currentOriginalImage) return;

            document.querySelectorAll('.emoji-btn').forEach(b => b.classList.remove('active'));
            freshClownBtn.classList.add('active');
            loadingOverlay.style.display = 'flex';

            try {
                const formData = new FormData();
                formData.append('image', uploadedFile);

                console.log('[Clown] → POST /process/clown_transformation');

                const response = await fetch(`${API_BASE}/process/clown_transformation`, {
                    method: 'POST',
                    body: formData,
                });

                const payload = await response.json();
                console.log('[Clown] response:', payload);

                if (!response.ok) throw new Error(payload?.detail || 'Clown transformation failed.');

                const base64 = payload.proc_image_b64 || payload.image_b64 || payload.image || null;
                if (!base64) throw new Error('No image data in clown response.');

                const src = base64.startsWith('data:') ? base64 : `data:image/jpeg;base64,${base64}`;
                currentProcessedImage = src;
                afterImg.src = src;
                landmarksOnlyImg.src = src;

                if (payload.metrics) updateMetricsFromApi(payload.metrics);

                analysisSummary.innerHTML = `<strong>Status: Success</strong><br/>Clown Transformation applied.`;
                addHistory('Emoji: Clown');

                if (isSplitMode) { sliderPos = 25; updateSplitSlider(); }

            } catch (err) {
                console.error('[Clown] Error:', err);
                analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>${err.message}`;
            } finally {
                loadingOverlay.style.display = 'none';
            }
        });
    }

    // Makeup
    const applyMakeupBtn = document.getElementById('applyMakeupBtn');
    const makeupRegion = document.getElementById('makeupRegion');
    const makeupColor = document.getElementById('makeupColor');
    const makeupOpacity = document.getElementById('makeupOpacity');

    function hexToOpenCVHue(hex) {
        hex = hex.replace('#', '');
        let r = parseInt(hex.substring(0, 2), 16) / 255;
        let g = parseInt(hex.substring(2, 4), 16) / 255;
        let b = parseInt(hex.substring(4, 6), 16) / 255;
        let max = Math.max(r, g, b), min = Math.min(r, g, b);
        let h = 0, d = max - min;
        if (max !== min) {
            switch (max) {
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2; break;
                case b: h = (r - g) / d + 4; break;
            }
            h /= 6;
        }
        return Math.round(h * 179);
    }

    if (applyMakeupBtn) {
        applyMakeupBtn.addEventListener('click', async () => {
            if (!uploadedFile || !currentOriginalImage) return;

            const hueValue = hexToOpenCVHue(makeupColor.value);

            const formData = new FormData();
            formData.append('image', uploadedFile);
            formData.append('region', makeupRegion.value);
            formData.append('opacity', makeupOpacity.value / 100.0);

            // Convert Hex to Hue (0-179 for OpenCV)
            const hex = makeupColor.value;
            let r = parseInt(hex.slice(1, 3), 16) / 255;
            let g = parseInt(hex.slice(3, 5), 16) / 255;
            let b = parseInt(hex.slice(5, 7), 16) / 255;
            let max = Math.max(r, g, b), min = Math.min(r, g, b);
            let h = 0;
            if (max != min) {
                let d = max - min;
                if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
                else if (max === g) h = (b - r) / d + 2;
                else if (max === b) h = (r - g) / d + 4;
                h /= 6;
            }
            formData.append('hue', Math.round(h * 179));

            loadingOverlay.style.display = 'flex';
            try {
                const response = await fetch(`${API_BASE}/process/makeup`, { method: 'POST', body: formData });
                const payload = await response.json();
                if (!response.ok) throw new Error(payload?.detail || 'Makeup failed.');

                currentProcessedImage = payload.image_b64;
                afterImg.src = currentProcessedImage;
                landmarksOnlyImg.src = currentProcessedImage;

                updateMetricsFromApi(payload.metrics || { mse: 0, psnr: 0, ssim: 0 });
                setSpectrumImages(payload.orig_spectrum_b64 || null, payload.proc_spectrum_b64 || null);

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
            formData.append('target_color', color);
            formData.append('blend_strength', intensitySlider.value / 100.0 || 0.6);

            loadingOverlay.style.display = 'flex';
            try {
                const response = await fetch(`${API_BASE}/process/hair-color`, { method: 'POST', body: formData });
                const payload = await response.json();
                if (!response.ok) throw new Error(payload?.detail || 'Hair color failed.');

                currentProcessedImage = payload.image_b64;
                afterImg.src = currentProcessedImage;
                landmarksOnlyImg.src = currentProcessedImage;

                updateMetricsFromApi(payload.metrics || { mse: 0, psnr: 0, ssim: 0 });
                setSpectrumImages(payload.orig_spectrum_b64 || null, payload.proc_spectrum_b64 || null);

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

                updateMetricsFromApi(payload.metrics || { mse: 0, psnr: 0, ssim: 0 });
                setSpectrumImages(payload.orig_spectrum_b64 || null, payload.proc_spectrum_b64 || null);

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
    const cameraPlaceholder = document.getElementById('cameraPlaceholder');
    const startCameraBtn = document.getElementById('startCameraBtn');
    const stopCameraBtn = document.getElementById('stopCameraBtn');
    const captureBtn = document.getElementById('captureBtn');
    const cameraCanvas = document.getElementById('cameraCanvas');
    let mediaStream = null;

    function stopCamera() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        if (cameraVideo) cameraVideo.style.display = 'none';
        if (cameraPlaceholder) cameraPlaceholder.style.display = 'block';
        if (startCameraBtn) startCameraBtn.style.display = 'flex';
        if (captureBtn) captureBtn.style.display = 'none';
        if (stopCameraBtn) stopCameraBtn.style.display = 'none';
    }

    if (startCameraBtn) {
        startCameraBtn.addEventListener('click', async () => {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
                if (cameraVideo) {
                    cameraVideo.srcObject = mediaStream;
                    cameraVideo.style.display = 'block';
                }
                if (cameraPlaceholder) cameraPlaceholder.style.display = 'none';
                if (startCameraBtn) startCameraBtn.style.display = 'none';
                if (captureBtn) captureBtn.style.display = 'flex';
                if (stopCameraBtn) stopCameraBtn.style.display = 'flex';
            } catch (err) {
                console.error('Error accessing camera:', err);
                alert('Could not access camera. Please allow permissions.');
            }
        });
    }

    if (stopCameraBtn) {
        stopCameraBtn.addEventListener('click', stopCamera);
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
                    stopCamera(); // Auto-stop camera after capture
                }
            }, 'image/png');
        });
    }

    // --- NEW FACE FEATURES (Göz, Sakal, Yaş Karşılaştırma) ---
    const eyeSizeSlider = document.getElementById('eyeSizeSlider');
    const eyeSizeVal = document.getElementById('eyeSizeVal');
    const eyeSizeBtn = document.getElementById('eyeSizeBtn');

    const beardSelect = document.getElementById('beardSelect');
    const beardDarknessSlider = document.getElementById('beardDarknessSlider');
    const beardDarknessVal = document.getElementById('beardDarknessVal');
    const beardBtn = document.getElementById('beardBtn');

    const ageCompareBtn = document.getElementById('ageCompareBtn');

    // Update slider text values
    if (eyeSizeSlider) {
        eyeSizeSlider.addEventListener('input', (e) => {
            eyeSizeVal.textContent = e.target.value + '%';
        });
    }

    if (beardDarknessSlider) {
        beardDarknessSlider.addEventListener('input', (e) => {
            beardDarknessVal.textContent = e.target.value + '%';
        });
    }

    // Göz Büyütme/Küçültme Event
    if (eyeSizeBtn) {
        eyeSizeBtn.addEventListener('click', async () => {
            if (!uploadedFile || !currentOriginalImage) return;
            const formData = new FormData();
            formData.append('image', uploadedFile);
            formData.append('scale', eyeSizeSlider.value);

            loadingOverlay.style.display = 'flex';
            try {
                // Backend endpoint
                const response = await fetch(`${API_BASE}/process/eye-size`, { method: 'POST', body: formData });
                const payload = await response.json();
                if (!response.ok) throw new Error(payload?.detail || 'Eye scale processing failed.');

                currentProcessedImage = payload.image_b64;
                afterImg.src = currentProcessedImage;
                landmarksOnlyImg.src = currentProcessedImage;

                updateMetricsFromApi(payload.metrics || { mse: 0, psnr: 0, ssim: 0 });
                setSpectrumImages(payload.orig_spectrum_b64 || null, payload.proc_spectrum_b64 || null);

                addHistory(`Eye Scale: ${eyeSizeSlider.value}%`);
                analysisSummary.innerHTML = `<strong>Status: Success</strong><br/>Applied Eye Scale.`;

                if (isSplitMode) { sliderPos = 25; updateSplitSlider(); }
            } catch (e) {
                console.error('[Eye Scale] Error:', e);
                analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>${e.message || 'Eye scale failed.'}`;
            } finally {
                loadingOverlay.style.display = 'none';
            }
        });
    }

    // Sakal/Bıyık Event
    if (beardBtn) {
        beardBtn.addEventListener('click', async () => {
            if (!uploadedFile || !currentOriginalImage) return;
            const formData = new FormData();
            formData.append('image', uploadedFile);
            formData.append('style', beardSelect.value);
            formData.append('intensity', intensitySlider.value); // Use global intensity
            formData.append('darkness', beardDarknessSlider.value / 100.0);

            loadingOverlay.style.display = 'flex';
            try {
                // Backend endpoint
                const response = await fetch(`${API_BASE}/process/beard`, { method: 'POST', body: formData });
                const payload = await response.json();
                if (!response.ok) throw new Error(payload?.detail || 'Facial hair processing failed.');

                currentProcessedImage = payload.image_b64;
                afterImg.src = currentProcessedImage;
                landmarksOnlyImg.src = currentProcessedImage;

                updateMetricsFromApi(payload.metrics || { mse: 0, psnr: 0, ssim: 0 });
                setSpectrumImages(payload.orig_spectrum_b64 || null, payload.proc_spectrum_b64 || null);

                addHistory(`Facial Hair: ${beardSelect.value}`);
                analysisSummary.innerHTML = `<strong>Status: Success</strong><br/>Applied Facial Hair (${beardSelect.value}).`;

                if (isSplitMode) { sliderPos = 25; updateSplitSlider(); }
            } catch (e) {
                console.error('[Facial Hair] Error:', e);
                analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>${e.message || 'Facial hair failed.'}`;
            } finally {
                loadingOverlay.style.display = 'none';
            }
        });
    }

    // Yaş Karşılaştırma Event
    if (ageCompareBtn) {
        ageCompareBtn.addEventListener('click', async () => {
            if (!uploadedFile || !currentOriginalImage || !currentProcessedImage) return;

            loadingOverlay.style.display = 'flex';
            try {
                // Convert currentProcessedImage (base64) to Blob
                const res = await fetch(currentProcessedImage);
                const afterBlob = await res.blob();
                const afterFile = new File([afterBlob], 'after.png', { type: 'image/png' });

                const formData = new FormData();
                formData.append('before_image', uploadedFile);
                formData.append('after_image', afterFile);

                const response = await fetch(`${API_BASE}/process/estimate-age-compare`, { method: 'POST', body: formData });
                const payload = await response.json();
                if (!response.ok) throw new Error(payload?.detail || 'Age comparison failed.');

                const beforeAge = payload.before?.estimated_age || '—';
                const afterAge = payload.after?.estimated_age || '—';
                const diff = payload.age_difference || 0;
                const diffStr = diff > 0 ? `+${diff}` : `${diff}`;

                analysisSummary.innerHTML = `
                    <div style="display:flex; justify-content:space-evenly; align-items:center; text-align:center; padding:10px 0;">
                        <div>
                            <div style="font-size:0.85rem;opacity:0.8;">Before</div>
                            <div style="font-size:1.8rem;font-weight:800;color:#00e5ff;">${beforeAge}</div>
                        </div>
                        <div style="font-size:1.5rem;opacity:0.5;">→</div>
                        <div>
                            <div style="font-size:0.85rem;opacity:0.8;">After</div>
                            <div style="font-size:1.8rem;font-weight:800;color:#00e5ff;">${afterAge}</div>
                        </div>
                        <div style="width:1px; height:40px; background:rgba(255,255,255,0.2);"></div>
                        <div>
                            <div style="font-size:0.85rem;opacity:0.8;">Diff</div>
                            <div style="font-size:1.5rem;font-weight:700;color:#ffeb3b;">${diffStr}</div>
                        </div>
                    </div>
                `;
                addHistory('Age Comparison');
            } catch (e) {
                console.error('[Age Compare] Error:', e);
                analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>${e.message || 'Age comparison failed.'}`;
            } finally {
                loadingOverlay.style.display = 'none';
            }
        });
    }

    // Glasses / Gözlük Event
    const applyGlassesBtn = document.getElementById('applyGlassesBtn');
    const glassesSelect = document.getElementById('glassesSelect');

    if (applyGlassesBtn) {
        applyGlassesBtn.addEventListener('click', async () => {
            if (!uploadedFile || !currentOriginalImage) return;

            const formData = new FormData();
            formData.append('image', uploadedFile);
            formData.append('glasses_type', glassesSelect.value);

            loadingOverlay.style.display = 'flex';
            try {
                const response = await fetch(`${API_BASE}/process/glasses`, { method: 'POST', body: formData });
                const payload = await response.json();
                if (!response.ok) throw new Error(payload?.detail || 'Glasses processing failed.');

                currentProcessedImage = payload.image_b64;
                afterImg.src = currentProcessedImage;
                landmarksOnlyImg.src = currentProcessedImage;

                updateMetricsFromApi(payload.metrics || { mse: 0, psnr: 0, ssim: 0 });
                setSpectrumImages(
                    payload.orig_spectrum_b64 || null,
                    payload.proc_spectrum_b64 || null,
                    payload.orig_phase_b64 || null,
                    payload.proc_phase_b64 || null
                );

                const glassesLabelMap = {
                    aviator: i18n[currentLang]?.metalAviator || 'Classic Teardrop (Aviator)',
                    wayfarer: i18n[currentLang]?.acetateWayfarer || 'Thick Frame (Modern)',
                    round: i18n[currentLang]?.minimalistRound || 'Thin Round (Retro)',
                };
                const glassesLabel = glassesLabelMap[glassesSelect.value] || glassesSelect.value;
                addHistory(`Glasses: ${glassesLabel}`);
                const appliedPrefix = currentLang === 'tr' ? 'Uygulanan Gözlük:' : 'Applied Glasses:';
                analysisSummary.innerHTML = `<strong>Status: Success</strong><br/>${appliedPrefix} ${glassesLabel}.`;

                if (isSplitMode) { sliderPos = 25; updateSplitSlider(); }
            } catch (e) {
                console.error('[Glasses] Error:', e);
                analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>${e.message || 'Glasses processing failed.'}`;
            } finally {
                loadingOverlay.style.display = 'none';
            }
        });
    }

    // PNG Download Event
    if (downloadPngBtn) {
        downloadPngBtn.addEventListener('click', () => {
            if (!currentProcessedImage) return;
            const a = document.createElement('a');
            a.href = currentProcessedImage;
            a.download = 'facewarp_result.png';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        });
    }

    // --- PDF REPORT EXPORT ---

    /** Convert a base-64 data-URL to the raw base-64 string and its MIME type. */
    function splitDataUrl(dataUrl) {
        if (!dataUrl || !dataUrl.startsWith('data:')) return null;
        const [header, data] = dataUrl.split(',');
        const mime = header.match(/data:(.*?);/)?.[1] || 'image/png';
        const format = mime.includes('png') ? 'PNG' : 'JPEG';
        return { data, format };
    }

    downloadBtn.addEventListener('click', async () => {
        if (!currentOriginalImage) return;

        downloadBtn.disabled = true;
        downloadBtn.textContent = currentLang === 'TR' ? 'PDF hazırlanıyor…' : 'Generating PDF…';

        try {
            // jsPDF is loaded globally via <script> tag in index.html
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
            const pageW = doc.internal.pageSize.getWidth();
            let y = 15; // vertical cursor

            // ── Title ──
            doc.setFontSize(22);
            doc.setFont('helvetica', 'bold');
            doc.text('FaceDSP — Analysis Report', pageW / 2, y, { align: 'center' });
            y += 10;

            doc.setFontSize(10);
            doc.setFont('helvetica', 'normal');
            doc.setTextColor(120);
            doc.text(new Date().toLocaleString(), pageW / 2, y, { align: 'center' });
            doc.setTextColor(0);
            y += 10;

            // ── Before / After Images ──
            const imgW = 80;
            const imgH = 80;

            const origInfo = splitDataUrl(currentOriginalImage);
            const procInfo = splitDataUrl(currentProcessedImage);

            doc.setFontSize(12);
            doc.setFont('helvetica', 'bold');

            if (origInfo) {
                doc.text(currentLang === 'TR' ? 'Öncesi' : 'Before', 15, y);
                y += 3;
                doc.addImage(origInfo.data, origInfo.format, 15, y, imgW, imgH);
            }
            if (procInfo) {
                doc.text(currentLang === 'TR' ? 'Sonrası' : 'After', 110, y - 3);
                doc.addImage(procInfo.data, procInfo.format, 110, y, imgW, imgH);
            }
            y += imgH + 10;

            // ── Quality Metrics ──
            doc.setFontSize(14);
            doc.setFont('helvetica', 'bold');
            doc.text(i18n[currentLang]?.qualityMetrics || 'Quality Metrics', 15, y);
            y += 7;

            doc.setFontSize(11);
            doc.setFont('helvetica', 'normal');
            const metrics = [
                { label: 'MSE', value: mseValue.textContent },
                { label: 'PSNR', value: psnrValue.textContent + ' dB' },
                { label: 'SSIM', value: ssimValue.textContent },
            ];
            metrics.forEach(m => {
                doc.text(`${m.label}: ${m.value}`, 20, y);
                y += 6;
            });
            y += 4;

            // ── Analysis Summary ──
            const summaryText = analysisSummary.innerText || analysisSummary.textContent || '';
            if (summaryText) {
                doc.setFontSize(14);
                doc.setFont('helvetica', 'bold');
                doc.text(i18n[currentLang]?.analysisSummary || 'Analysis Summary', 15, y);
                y += 7;

                // Extract and display the applied operation explicitly
                const opMatch = summaryText.match(/Applied (?:Emoji Preset|Glasses|Makeup|Hair|Beard|Filter):\s*(.+)/i)
                    || summaryText.match(/Uygulanan .+?:\s*(.+)/i);
                if (opMatch) {
                    doc.setFontSize(11);
                    doc.setFont('helvetica', 'bold');
                    doc.setTextColor(30, 100, 180);
                    let opName = opMatch[1].replace(/\.$/, '').trim();
                    let finalStr = (currentLang === 'tr' ? 'Uygulanan İşlem: ' : 'Operation: ') + opName;
                    if (currentLang === 'tr' && opName === 'Joker / Palyaço') {
                        finalStr = 'Joker Estetiği Uygulandı';
                    }
                    doc.text(finalStr, 15, y);
                    doc.setTextColor(0);
                    y += 7;
                }

                doc.setFontSize(10);
                doc.setFont('helvetica', 'normal');
                const lines = doc.splitTextToSize(summaryText, pageW - 30);
                doc.text(lines, 15, y);
                y += lines.length * 5 + 6;
            }

            // ── Operation History ──
            if (operationHistory.length > 0) {
                if (y > 260) { doc.addPage(); y = 15; }
                doc.setFontSize(14);
                doc.setFont('helvetica', 'bold');
                doc.text(i18n[currentLang]?.opHistory || 'Operation History', 15, y);
                y += 7;
                doc.setFontSize(10);
                doc.setFont('helvetica', 'normal');
                operationHistory.forEach((op, i) => {
                    doc.text(`${i + 1}. ${op}`, 20, y);
                    y += 5;
                });
                y += 6;
            }

            // ── Spectrum Images (if available) ──
            const origSpec = splitDataUrl(origSpectrumImg?.src);
            const procSpec = splitDataUrl(procSpectrumImg?.src);
            if (origSpec || procSpec) {
                if (y > 200) { doc.addPage(); y = 15; }
                doc.setFontSize(14);
                doc.setFont('helvetica', 'bold');
                doc.text('FFT Magnitude Spectrum', 15, y);
                y += 5;
                const specW = 80, specH = 80;
                if (origSpec) {
                    doc.addImage(origSpec.data, origSpec.format, 15, y, specW, specH);
                }
                if (procSpec) {
                    doc.addImage(procSpec.data, procSpec.format, 110, y, specW, specH);
                }
                y += specH + 10;
            }

            // ── Phase Images (if available) ──
            const origPhase = splitDataUrl(origPhaseImg?.src);
            const procPhase = splitDataUrl(procPhaseImg?.src);
            if (origPhase || procPhase) {
                if (y > 200) { doc.addPage(); y = 15; }
                doc.setFontSize(14);
                doc.setFont('helvetica', 'bold');
                doc.text('FFT Phase Spectrum', 15, y);
                y += 5;
                const specW = 80, specH = 80;
                if (origPhase) {
                    doc.addImage(origPhase.data, origPhase.format, 15, y, specW, specH);
                }
                if (procPhase) {
                    doc.addImage(procPhase.data, procPhase.format, 110, y, specW, specH);
                }
            }

            doc.save('FaceDSP_Report.pdf');
        } catch (err) {
            console.error('[PDF Export] Error:', err);
            alert((currentLang === 'TR' ? 'PDF oluşturulamadı: ' : 'PDF generation failed: ') + err.message);
        } finally {
            downloadBtn.disabled = false;
            downloadBtn.textContent = i18n[currentLang]?.downloadPDF || 'Download Results as PDF';
        }
    });
    // --- HAIR COLOR LOGIC ---
    function hexToRgb(hex) {
        let h = hex.replace('#', '');
        if (h.length === 3) h = [...h].map(x => x + x).join('');
        return `${parseInt(h.substring(0,2), 16)},${parseInt(h.substring(2,4), 16)},${parseInt(h.substring(4,6), 16)}`;
    }

    const applyHairColorBtn = document.getElementById('applyHairColorBtn');
    const hairColorPicker  = document.getElementById('hairColorPicker');
    const hairOpacity      = document.getElementById('hairOpacity');

    if (applyHairColorBtn) {
        applyHairColorBtn.addEventListener('click', async () => {
            if (!uploadedFile || !currentOriginalImage) {
                alert(i18n[currentLang]?.uploadWait || 'Please upload an image first.');
                return;
            }

            const rgbColor      = hexToRgb(hairColorPicker.value);
            const intensityFloat = parseInt(hairOpacity.value) / 100.0;

            // Debug — verify values before request
            console.log('[Hair Color] target_color (RGB):', rgbColor);
            console.log('[Hair Color] intensity:', intensityFloat);
            console.log('[Hair Color] endpoint:', `${API_BASE}/process/hair-color`);

            const formData = new FormData();
            formData.append('image',        uploadedFile);
            formData.append('target_color', rgbColor);
            formData.append('intensity',    intensityFloat.toString());

            loadingOverlay.style.display = 'flex';

            try {
                const response = await fetch(`${API_BASE}/process/hair-color`, {
                    method: 'POST',
                    body: formData
                });

                const payload = await response.json();

                // 1. Tam payload'ı konsolda göster
                console.log('[Hair Color] Backend tam yanıtı (payload):', payload);

                if (!response.ok) {
                    throw new Error(payload?.detail || 'Hair color processing failed.');
                }

                // 2. Esnek key yakalama — tüm olası isimleri dene
                const base64Data = payload.proc_image_b64
                    || payload.image_b64
                    || payload.image
                    || payload.result
                    || null;

                if (base64Data) {
                    // 3. Görseli güncelle ve başarı logu
                    const resultSrc = base64Data.startsWith('data:')
                        ? base64Data
                        : `data:image/jpeg;base64,${base64Data}`;
                    currentProcessedImage = resultSrc;
                    afterImg.src = resultSrc;
                    console.log('[Hair Color] Görsel başarıyla ekrana basıldı.');

                    if (payload.metrics) {
                        updateMetricsFromApi(payload.metrics);
                    }

                    analysisSummary.innerHTML = `<strong>Status: Success</strong><br/>Hair color applied.`;

                    if (isSplitMode) {
                        sliderPos = 25;
                        updateSplitSlider();
                    }
                } else {
                    // 4. 200 OK ama görsel verisi yok
                    console.warn('[Hair Color] Uyarı: Backend yanıt verdi ancak içinde base64 görsel verisi bulunamadı. Mevcut anahtarlar:', Object.keys(payload));
                    analysisSummary.innerHTML = `<strong>Status: Warning</strong><br/>Backend yanıt verdi ancak görsel verisi bulunamadı.`;
                }

            } catch (err) {
                console.error('[Hair Color] Error:', err);
                analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>${err.message}`;
            } finally {
                loadingOverlay.style.display = 'none';
            }
        });
    }

    // --- FFT LAB INTERACTIVE SELECTION ---
    const fftSubBtns = document.querySelectorAll('.fft-sub-btn');
    const fftMagView = document.getElementById('fft-mag-view');
    const fftPhaseView = document.getElementById('fft-phase-view');

    fftSubBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            fftSubBtns.forEach(b => {
                b.classList.remove('active');
                b.style.background = 'transparent';
            });
            btn.classList.add('active');
            btn.style.background = 'var(--surface-color)';

            if (btn.dataset.sub === 'magnitude') {
                fftMagView.style.display = 'flex';
                fftPhaseView.style.display = 'none';
            } else {
                fftMagView.style.display = 'none';
                fftPhaseView.style.display = 'flex';
            }
        });
    });

    function setupInteractiveCanvas(canvasId, name) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        let startX = 0;
        let startY = 0;
        let currentX = 0;
        let currentY = 0;

        function resizeCanvas() {
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
        }

        window.addEventListener('resize', resizeCanvas);

        canvas.addEventListener('mousedown', (e) => {
            resizeCanvas();
            const rect = canvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
            isDrawing = true;
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            currentX = e.clientX - rect.left;
            currentY = e.clientY - rect.top;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            const x = Math.min(startX, currentX);
            const y = Math.min(startY, currentY);
            const w = Math.abs(currentX - startX);
            const h = Math.abs(currentY - startY);

            ctx.clearRect(x, y, w, h);

            ctx.strokeStyle = '#00ffcc';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(x, y, w, h);
        });

        const finishSelection = (e) => {
            if (!isDrawing) return;
            isDrawing = false;
            
            const rect = canvas.getBoundingClientRect();
            currentX = e.clientX - rect.left;
            currentY = e.clientY - rect.top;

            const x = Math.min(startX, currentX);
            const y = Math.min(startY, currentY);
            const w = Math.abs(currentX - startX);
            const h = Math.abs(currentY - startY);

            if (w > 5 && h > 5) {
                console.log(`[FFT Lab - ${name}] Region Selected - X: ${Math.round(x)}, Y: ${Math.round(y)}, Width: ${Math.round(w)}, Height: ${Math.round(h)}`);
                // Backend'den veri alinacak
            } else {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        };

        canvas.addEventListener('mouseup', finishSelection);
        canvas.addEventListener('mouseleave', finishSelection);
    }

    setupInteractiveCanvas('fftSelectionCanvas', 'Magnitude');
    setupInteractiveCanvas('fftPhaseCanvas', 'Phase');

});
