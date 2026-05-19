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

    // --- LIVE MODE STATE (outer scope so ALL handlers can check) ---
    let isLiveMode = false;           // true when WebSocket live stream is active
    let _liveWsRef = null;            // reference to the live WebSocket
    let liveActiveStates = {};        // single active live effect sent to backend
    let currentLivePreset = null;     // legacy preset tracker
    const LIVE_FEATURE_KEYS = [
        'smile', 'eyebrow', 'lip', 'slim', 'eye_scale', 'beard',
        'alien', 'robot', 'clown', 'star_eyes', 'heart_eyes', 'crying',
        'makeup_lips', 'makeup_eyeshadow', 'makeup_blush', 'makeup_eyeliner', 'makeup_mascara',
        'glasses', 'hair_color', 'aging', 'deaging', 'cartoon', 'fft',
        'eyebrow_raise', 'lip_widen', 'face_slim', 'eye_scaling',
    ];

    /**
     * Send a live state update to the WebSocket backend.
     * value=null removes the feature; otherwise it becomes the only active effect.
     */
    function sendLiveStateUpdate(feature, value) {
        const payload = {};

        if (value === null || value === undefined) {
            delete liveActiveStates[feature];
            payload[feature] = null;
        } else {
            LIVE_FEATURE_KEYS.forEach(key => { payload[key] = null; });
            liveActiveStates = {};
            liveActiveStates[feature] = value;
            payload[feature] = value;
        }
        if (_liveWsRef && _liveWsRef.readyState === WebSocket.OPEN) {
            _liveWsRef.send(JSON.stringify({
                action: 'update_live_state',
                active_states: payload,
            }));
            console.log('[Live] State update sent:', Object.keys(liveActiveStates));
        }
        // Update the active states chip bar
        renderLiveStatesBar();
    }

    /** Render active live effects as removable chips */
    function renderLiveStatesBar() {
        let bar = document.getElementById('liveStatesBar');
        if (!bar) {
            bar = document.createElement('div');
            bar.id = 'liveStatesBar';
            bar.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;padding:8px 12px;min-height:0;';
            const cameraColumn = document.getElementById('cameraColumn');
            if (cameraColumn) cameraColumn.appendChild(bar);
        }
        const keys = Object.keys(liveActiveStates);
        if (!isLiveMode || keys.length === 0) {
            bar.innerHTML = '';
            bar.style.display = 'none';
            return;
        }
        bar.style.display = 'flex';
        bar.innerHTML = keys.map(k => `<span style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border-radius:20px;background:rgba(102,126,234,0.25);border:1px solid rgba(102,126,234,0.5);font-size:0.75rem;color:#c5ceff;cursor:default;">${k}<button onclick="window._removeLiveState('${k}')" style="background:none;border:none;color:#ff6b6b;cursor:pointer;font-size:0.85rem;padding:0 2px;line-height:1;">✕</button></span>`).join('');
    }
    // Global hook for chip removal
    window._removeLiveState = function(feature) {
        sendLiveStateUpdate(feature, null);
    };

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
            makeupEyeliner: "Eyeliner",
            makeupMascara: "Rimel",
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
            futuristicStyle: "Fütüristik (Kalkan)",
            squareStyle: "Kare Çerçeve",
            retroStyle: "Retro Browline",
            sportStyle: "Spor (Sarmal)",
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
            origFFTPhase: "Orijinal - FFT Fazı",
            // Camera Live Mode
            liveMode: "🔴 Canlı",
            stopLive: "⏹ Canlıyı Durdur",
            // Face Swap
            faceSwap: "Yüz Değiştirme",
            realtimeFaceSwap: "Canlı Yüz Değiştirme",
            uploadSourceFace: "Kaynak Yüz Yükle",
            sourceFaceLoaded: "Kaynak yüklendi",
            startWebcam: "Kamerayı Başlat",
            stopWebcam: "Kamerayı Durdur",
            enableFaceSwap: "Yüz Değiştirmeyi Etkinleştir",
            blendStrength: "Harmanlama Gücü",
            stability: "Kararlılık",
            maskSoftness: "Maske Yumuşaklığı",
            captureScreenshot: "Ekran Görüntüsü Al"
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
            makeupEyeliner: "Eyeliner",
            makeupMascara: "Mascara",
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
            futuristicStyle: "Futuristic (Shield)",
            squareStyle: "Square Frame",
            retroStyle: "Retro Browline",
            sportStyle: "Sport Wraparound",
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
            origFFTPhase: "Original - FFT Phase",
            // Camera Live Mode
            liveMode: "🔴 Live",
            stopLive: "⏹ Stop Live",
            // Face Swap
            faceSwap: "Face Swap",
            realtimeFaceSwap: "Realtime Face Swap",
            uploadSourceFace: "Upload Source Face",
            sourceFaceLoaded: "Source loaded",
            startWebcam: "Start Webcam",
            stopWebcam: "Stop Webcam",
            enableFaceSwap: "Enable Face Swap",
            blendStrength: "Blend Strength",
            stability: "Stability",
            maskSoftness: "Mask Softness",
            captureScreenshot: "Capture Screenshot"
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
    const fftPhaseOutputImg = document.getElementById('fftPhaseOutputImg');
    const fftPhaseOutputPlaceholder = document.getElementById('fftPhaseOutputPlaceholder');
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

            const faceSwapSection = document.getElementById('faceSwapSection');
            if (faceSwapSection) {
                if (selectedOperation === 'face_swap') {
                    faceSwapSection.style.display = 'block';
                } else {
                    faceSwapSection.style.display = 'none';
                }
            }
        });
    });

    // Toggles
    toggleLandmarks.addEventListener('change', (e) => {
        // ── LIVE MODE: send show_landmarks as independent toggle ──
        if (isLiveMode && _liveWsRef && _liveWsRef.readyState === WebSocket.OPEN) {
            _liveWsRef.send(JSON.stringify({
                action: 'update_live_state',
                active_states: { show_landmarks: e.target.checked ? { enabled: true } : null },
            }));
            console.log('[Live] show_landmarks:', e.target.checked);
        }
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
    async function generateLandmarks(targetFile = uploadedFile, retries = 2) {
        if (!targetFile) return;

        const CANVAS = 512; // matches SVG viewBox and backend preprocess size

        try {
            const formData = new FormData();
            formData.append('image', targetFile);

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
            if (retries > 0) {
                await new Promise(resolve => setTimeout(resolve, 140));
                return generateLandmarks(targetFile, retries - 1);
            }
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
        // ── LIVE MODE: route geometric warps to WebSocket ──
        if (isLiveMode) {
            const warpOps = new Set(['smile', 'eyebrow', 'lip', 'slim']);
            if (warpOps.has(selectedOperation)) {
                sendLiveStateUpdate(selectedOperation, { intensity: Number(intensitySlider.value) });
                return;
            }
        }

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
        // ── LIVE MODE: route to WebSocket ──
        if (isLiveMode) {
            sendLiveStateUpdate(presetName, { intensity: 50 });
            // Also set visual active state
            document.querySelectorAll('.emoji-btn').forEach(b => b.classList.remove('active'));
            const clickedBtn = document.querySelector(`[data-preset="${presetName}"]`);
            if (clickedBtn) clickedBtn.classList.add('active');
            return;
        }

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
            // ── LIVE MODE: route to WebSocket ──
            if (isLiveMode) {
                sendLiveStateUpdate('clown', { intensity: 50 });
                document.querySelectorAll('.emoji-btn').forEach(b => b.classList.remove('active'));
                freshClownBtn.classList.add('active');
                return;
            }

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
            const region = makeupRegion.value || 'lips';
            const featureKey = `makeup_${region}`;
            currentLivePreset = featureKey;
            selectedOperation = featureKey;

            // ── LIVE MODE: route to WebSocket ──
            if (isLiveMode) {
                const hue = hexToOpenCVHue(makeupColor.value);
                const opacity = Number(makeupOpacity.value) / 100.0;
                sendLiveStateUpdate(featureKey, { makeup_hue: hue, makeup_opacity: opacity, makeup_color: makeupColor.value, intensity: 50 });
                return;
            }

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
            selectedOperation = 'cartoon';
            currentLivePreset = 'cartoon';

            // ── LIVE MODE: route to WebSocket ──
            if (isLiveMode) {
                sendLiveStateUpdate('cartoon', { intensity: 50 });
                return;
            }

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

    // Camera Capture — handled by the canonical initCamera() IIFE module below.
    // The duplicate camera system (mediaStream, stopCamera, etc.) was removed
    // to prevent state lifecycle conflicts (two competing systems on the same elements).

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
            selectedOperation = 'eye_scale';
            currentLivePreset = 'eye_scale';

            // ── LIVE MODE: route to WebSocket ──
            if (isLiveMode) {
                sendLiveStateUpdate('eye_scale', { intensity: Number(eyeSizeSlider.value) });
                return;
            }

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
            // ── LIVE MODE: route to WebSocket ──
            if (isLiveMode) {
                sendLiveStateUpdate('beard', {
                    beard_type: beardSelect.value,
                    beard_darkness: Number(beardDarknessSlider.value),
                    intensity: Number(beardDarknessSlider.value),
                });
                return;
            }

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
            // ── LIVE MODE: route to WebSocket ──
            if (isLiveMode) {
                sendLiveStateUpdate('glasses', { glasses_type: glassesSelect.value });
                selectedOperation = 'glasses';
                return;
            }

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
                    futuristic: i18n[currentLang]?.futuristicStyle || 'Futuristic (Shield)',
                    square: i18n[currentLang]?.squareStyle || 'Square Frame',
                    retro: i18n[currentLang]?.retroStyle || 'Retro Browline',
                    sport: i18n[currentLang]?.sportStyle || 'Sport Wraparound',
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
            // ── LIVE MODE: route to WebSocket ──
            if (isLiveMode) {
                const hex = (hairColorPicker?.value || '#ff0000').replace('#', '');
                const r = parseInt(hex.substring(0, 2), 16);
                const g = parseInt(hex.substring(2, 4), 16);
                const b = parseInt(hex.substring(4, 6), 16);
                const intensity = Number(hairOpacity?.value || 60) / 100.0;
                sendLiveStateUpdate('hair_color', { hair_color: `${r},${g},${b}`, hair_intensity: intensity });
                selectedOperation = 'hair_color';
                return;
            }

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

    function getFftSelectionCoords(canvas, imageElement, selection) {
        if (!imageElement || !imageElement.src || imageElement.style.display === 'none') {
            return null;
        }

        const canvasW = canvas.width || canvas.getBoundingClientRect().width;
        const canvasH = canvas.height || canvas.getBoundingClientRect().height;
        const naturalW = imageElement.naturalWidth || 1;
        const naturalH = imageElement.naturalHeight || 1;
        const imageRatio = naturalW / naturalH;
        const canvasRatio = canvasW / Math.max(canvasH, 1);

        let drawW = canvasW;
        let drawH = canvasH;
        let offsetX = 0;
        let offsetY = 0;

        if (canvasRatio > imageRatio) {
            drawH = canvasH;
            drawW = drawH * imageRatio;
            offsetX = (canvasW - drawW) / 2;
        } else {
            drawW = canvasW;
            drawH = drawW / imageRatio;
            offsetY = (canvasH - drawH) / 2;
        }

        const x0 = Math.max(selection.x, offsetX);
        const y0 = Math.max(selection.y, offsetY);
        const x1 = Math.min(selection.x + selection.w, offsetX + drawW);
        const y1 = Math.min(selection.y + selection.h, offsetY + drawH);

        if (x1 - x0 < 5 || y1 - y0 < 5) return null;

        return {
            x: (x0 - offsetX) / drawW,
            y: (y0 - offsetY) / drawH,
            w: (x1 - x0) / drawW,
            h: (y1 - y0) / drawH,
        };
    }

    async function applyFftRegionArtifact(coords, outputImg, outputPlaceholder) {
        if (!uploadedFile || !currentOriginalImage || !coords) {
            analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>Run FFT Filter first, then select a visible spectrum region.`;
            return;
        }

        const idleText = i18n[currentLang]?.selectRegionOutput || 'Select a region to generate output';
        const formData = new FormData();
        formData.append('image', uploadedFile);
        formData.append('intensity', intensitySlider.value);
        formData.append('mask_coords', JSON.stringify(coords));

        if (outputPlaceholder) {
            outputPlaceholder.style.display = 'block';
            outputPlaceholder.textContent = currentLang === 'TR' ? 'İşleniyor...' : 'Processing...';
        }
        if (outputImg) outputImg.style.display = 'none';
        loadingOverlay.style.display = 'flex';

        try {
            const response = await fetch(`${API_BASE}/process/fft`, {
                method: 'POST',
                body: formData,
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload?.detail || 'FFT region processing failed.');
            if (!payload?.image_b64) throw new Error('Missing image_b64 in response.');

            currentProcessedImage = payload.image_b64;
            afterImg.src = currentProcessedImage;
            landmarksOnlyImg.src = currentProcessedImage;

            if (outputImg) {
                outputImg.src = currentProcessedImage;
                outputImg.style.display = 'block';
            }
            if (outputPlaceholder) outputPlaceholder.style.display = 'none';

            updateMetricsFromApi(payload.metrics || { mse: 0, psnr: 0, ssim: 0 });
            setSpectrumImages(
                payload.orig_spectrum_b64 || null,
                payload.proc_spectrum_b64 || null,
                payload.orig_phase_b64 || null,
                payload.proc_phase_b64 || null
            );

            analysisSummary.innerHTML = `<strong>Status: Success</strong><br/>FFT partial-region artifact applied.`;
            addHistory('FFT Partial Region Artifact');
            if (isSplitMode) { sliderPos = 25; updateSplitSlider(); }
        } catch (err) {
            console.error('[FFT Region] Error:', err);
            if (outputPlaceholder) {
                outputPlaceholder.style.display = 'block';
                outputPlaceholder.textContent = err.message || idleText;
            }
            analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>${err.message || 'FFT region processing failed.'}`;
        } finally {
            loadingOverlay.style.display = 'none';
        }
    }

    function setupInteractiveCanvas(canvasId, name, imageElement, outputImg, outputPlaceholder) {
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
                const coords = getFftSelectionCoords(canvas, imageElement, { x, y, w, h });
                if (coords) {
                    applyFftRegionArtifact(coords, outputImg, outputPlaceholder);
                } else {
                    analysisSummary.innerHTML = `<strong>Status: Failed</strong><br/>Run FFT Filter first, then select inside the visible spectrum image.`;
                }
            } else {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        };

        canvas.addEventListener('mouseup', finishSelection);
        canvas.addEventListener('mouseleave', finishSelection);
    }

    setupInteractiveCanvas('fftSelectionCanvas', 'Magnitude', fftLabProcSpectrumImg, fftOutputImg, fftOutputPlaceholder);
    setupInteractiveCanvas('fftPhaseCanvas', 'Phase', fftLabProcPhaseImg, fftPhaseOutputImg, fftPhaseOutputPlaceholder);

    // =========================================================================
    // CAMERA MODULE — Browser-side webcam + Live processing via REST API
    // =========================================================================
    // Rewritten: fixes latency, response keys, FPS meter, memory leaks,
    // capture race conditions, temporal smoothing, coordinate transforms,
    // error handling, and state lifecycle.
    // =========================================================================
    (function initCamera() {
        const startCameraBtn = document.getElementById('startCameraBtn');
        const captureBtn     = document.getElementById('captureBtn');
        const liveBtn        = document.getElementById('liveBtn');
        const stopCameraBtn  = document.getElementById('stopCameraBtn');
        const cameraVideo    = document.getElementById('cameraVideo');
        const cameraCanvas   = document.getElementById('cameraCanvas');
        const cameraPlaceholder = document.getElementById('cameraPlaceholder');
        const cameraColumn   = document.getElementById('cameraColumn');
        const liveProcessedImg = document.getElementById('liveProcessedImg');
        const liveFpsBadge   = document.getElementById('liveFpsBadge');
        const liveIndicator  = document.getElementById('liveIndicator');
        const uploadCol      = document.querySelector('#previewPlaceholder > .split-col:first-child');

        if (!startCameraBtn) return;

        const liveStreamState = {
            cameraStream: null,
            isLiveMode: false,
            liveRafId: null,
            liveProcessing: false,
            liveFrameCount: 0,
            liveFpsTimer: null,
            currentLivePreset: null,
            frameSeq: 0,
            lastRenderLatencyMs: 0,
            lastProfile: null,
        };
        const captureState = { lastCapturedAtMs: 0 };
        const analysisState = { lastError: null };
        const uiState = { isLiveIndicatorError: false };
        let cameraStream = null;
        // NOTE: isLiveMode and currentLivePreset are declared in outer scope
        let liveRafId = null;
        let liveProcessing = false;
        let liveFrameCount = 0;
        let liveFpsTimer = null;

        let _lastBlobUrl = null;      // for memory management (revoke old blob URLs)
        let _liveErrorCount = 0;      // consecutive error counter
        const _MAX_LIVE_ERRORS = 10;  // pause live after this many consecutive errors
        let _lastFrameTime = 0;       // for rAF throttling
        const _TARGET_INTERVAL = 33;  // ~30 FPS target (ms between frames)
        let _liveHasPendingTick = false;
        let liveWorker = null;
        let liveWorkerReady = false;
        let _liveWs = null;           // WebSocket connection for live mode
        let _wsReady = false;         // WebSocket open state
        let _wsPendingFrame = false;  // throttle: waiting for server response
        const _LIVE_JPEG_QUALITY = 0.94;

        // Sync the WS ref to outer scope so sendLiveStateUpdate() can use it
        function _syncWsRef() { _liveWsRef = _liveWs; }

        // ── Profiling helper ────────────────────────────────────────────
        function _profileLog(label, startMs) {
            const elapsed = performance.now() - startMs;
            if (elapsed > 50) { // only log slow stages
                console.debug(`[Live Profile] ${label}: ${elapsed.toFixed(1)}ms`);
            }
        }

        function ensureLiveWorker() {
            if (liveWorker) return;
            try {
                liveWorker = new Worker('live-worker.js');
                liveWorkerReady = true;
            } catch (err) {
                console.warn('[Live] Worker unavailable, fallback to main thread encoding:', err?.message);
                liveWorker = null;
                liveWorkerReady = false;
            }
        }

        function encodeFrameMainThread() {
            return Promise.resolve(cameraCanvas.toDataURL('image/jpeg', _LIVE_JPEG_QUALITY));
        }

        function encodeFrameOffThread() {
            if (!liveWorkerReady || !liveWorker) return encodeFrameMainThread();
            return new Promise(async (resolve, reject) => {
                try {
                    const bitmap = await createImageBitmap(cameraCanvas);
                    const onMessage = (ev) => {
                        const msg = ev.data || {};
                        if (msg.type === 'encodedFrame') {
                            liveWorker.removeEventListener('message', onMessage);
                            resolve(msg.dataUrl);
                        } else if (msg.type === 'workerError') {
                            liveWorker.removeEventListener('message', onMessage);
                            reject(new Error(msg.error || 'Worker encoding failed'));
                        }
                    };
                    liveWorker.addEventListener('message', onMessage);
                    liveWorker.postMessage({
                        type: 'encodeFrame',
                        bitmap,
                        width: cameraCanvas.width,
                        height: cameraCanvas.height,
                        quality: _LIVE_JPEG_QUALITY,
                    }, [bitmap]);
                } catch (err) {
                    reject(err);
                }
            });
        }

        // ── START CAMERA ────────────────────────────────────────────────
        startCameraBtn.addEventListener('click', async () => {
            try {
                cameraStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
                });
                cameraVideo.srcObject = cameraStream;
                cameraVideo.style.display = 'block';
                cameraPlaceholder.style.display = 'none';

                startCameraBtn.style.display = 'none';
                captureBtn.style.display = 'inline-flex';
                liveBtn.style.display = 'inline-flex';
                stopCameraBtn.style.display = 'inline-flex';

                console.log('[Camera] Started');
            } catch (err) {
                console.error('[Camera] getUserMedia failed:', err);
                analysisSummary.innerHTML = `<strong>Camera Error:</strong> ${err.message}`;
            }
        });

        // ── STOP CAMERA ─────────────────────────────────────────────────
        stopCameraBtn.addEventListener('click', () => {
            stopLiveMode();
            if (cameraStream) {
                cameraStream.getTracks().forEach(t => t.stop());
                cameraStream = null;
            }
            cameraVideo.srcObject = null;
            cameraVideo.style.display = 'none';
            cameraPlaceholder.style.display = 'block';
            liveProcessedImg.style.display = 'none';

            startCameraBtn.style.display = 'inline-flex';
            captureBtn.style.display = 'none';
            liveBtn.style.display = 'none';
            stopCameraBtn.style.display = 'none';

            // Restore camera column — but do NOT clear captured comparison state
            restoreCameraColumn();

            // Revoke last blob URL to free memory
            if (_lastBlobUrl) { URL.revokeObjectURL(_lastBlobUrl); _lastBlobUrl = null; }
            if (liveWorker) {
                liveWorker.terminate();
                liveWorker = null;
                liveWorkerReady = false;
            }
            console.log('[Camera] Stopped');
        });

        // ── CAPTURE (snapshot → normal before/after workflow) ────────────
        // FIX: async to prevent race condition — wait for blob before calling setImage
        captureBtn.addEventListener('click', async () => {
            if (!cameraStream) return;
            const ctx = cameraCanvas.getContext('2d');
            cameraCanvas.width = cameraVideo.videoWidth;
            cameraCanvas.height = cameraVideo.videoHeight;
            ctx.drawImage(cameraVideo, 0, 0);

            // Convert canvas to blob (await to prevent race)
            const blob = await new Promise(resolve =>
                cameraCanvas.toBlob(resolve, 'image/jpeg', 0.92)
            );
            if (!blob) return;

            const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
            const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.92);
            // setImage is synchronous and will trigger generateLandmarks
            // only AFTER uploadedFile is set (no race)
            setImage(dataUrl, file);
            if (toggleLandmarks.checked) {
                await generateLandmarks(file, 2);
            }
            console.log('[Camera] Frame captured');
        });

        // ── LIVE MODE ───────────────────────────────────────────────────
        liveBtn.addEventListener('click', () => {
            if (isLiveMode) {
                stopLiveMode();
                restoreCameraColumn();
                liveBtn.textContent = '🔴 Live';
                liveBtn.style.background = 'linear-gradient(135deg, #f85032, #e73827)';
            } else {
                startLiveMode();
                expandCameraColumn();
                liveBtn.textContent = '⏹ Stop Live';
                liveBtn.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
            }
        });

        function expandCameraColumn() {
            if (uploadCol) uploadCol.style.display = 'none';
            cameraColumn.classList.add('camera-live-expanded');
        }

        function restoreCameraColumn() {
            if (uploadCol) uploadCol.style.display = '';
            cameraColumn.classList.remove('camera-live-expanded');
        }

        function startLiveMode() {
            ensureLiveWorker();
            isLiveMode = true;
            liveProcessedImg.style.display = 'none';
            liveFpsBadge.style.display = 'inline';
            liveIndicator.style.display = 'inline';
            liveFrameCount = 0;
            _liveErrorCount = 0;
            _lastFrameTime = 0;
            _liveHasPendingTick = false;
            _wsPendingFrame = false;

            // Open WebSocket connection to backend
            const wsUrl = API_BASE.replace('http', 'ws') + '/live/ws';
            _liveWs = new WebSocket(wsUrl);
            _wsReady = false;
            _syncWsRef();

            _liveWs.onopen = () => {
                _wsReady = true;
                console.log('[Live] WebSocket connected');
                // Send initial config
                const preset = getCurrentLivePreset();
                _liveWs.send(JSON.stringify({
                    type: 'config',
                    filter: _mapPresetToFilter(preset),
                    intensity: Number(intensitySlider?.value || 50),
                }));
            };

            _liveWs.onmessage = (ev) => {
                try {
                    const msg = JSON.parse(ev.data);
                    if (msg.type === 'frame' && isLiveMode) {
                        _wsPendingFrame = false;
                        if (_lastBlobUrl) { URL.revokeObjectURL(_lastBlobUrl); _lastBlobUrl = null; }
                        liveProcessedImg.src = msg.data;
                        liveProcessedImg.style.display = 'block';
                        liveFrameCount++;
                        _liveErrorCount = 0;
                        liveStreamState.lastRenderLatencyMs = 0;
                        if (liveIndicator.textContent !== '● LIVE') {
                            liveIndicator.textContent = '● LIVE';
                            liveIndicator.style.background = 'rgba(255,40,40,0.85)';
                        }
                        // Update FPS from server
                        if (msg.fps) {
                            liveFpsBadge.textContent = `${msg.fps} FPS`;
                        }
                        // Face detection indicator
                        if (msg.face_detected === false) {
                            liveFpsBadge.textContent += ' | No Face';
                        }
                    }
                } catch (e) { console.debug('[Live] WS parse error:', e); }
            };

            _liveWs.onerror = (err) => {
                console.error('[Live] WebSocket error:', err);
                _wsReady = false;
            };

            _liveWs.onclose = () => {
                console.log('[Live] WebSocket closed');
                _wsReady = false;
                _liveWs = null;
                _syncWsRef();
            };

            // True end-to-end FPS counter (counts only rendered frames)
            if (liveFpsTimer) clearInterval(liveFpsTimer);
            liveFpsTimer = setInterval(() => {
                const latency = Math.round(liveStreamState.lastRenderLatencyMs || 0);
                liveFpsBadge.textContent = `${liveFrameCount} FPS`;
                liveFrameCount = 0;
            }, 1000);

            // Start with requestAnimationFrame
            liveRafLoop(performance.now());
            console.log('[Live] Started (WebSocket mode)');
        }

        // Helper: map sidebar preset names to backend filter names
        function _mapPresetToFilter(preset) {
            if (!preset) return 'none';
            const map = {
                // Geometric warps
                'smile': 'smile', 'eyebrow': 'eyebrow_raise', 'lip': 'lip_widen',
                'slim': 'face_slim', 'eye_scale': 'eye_scaling',
                // Emoji presets (direct names from process.py _EMOJI_PRESETS_MAP)
                'alien': 'alien', 'robot': 'robot', 'clown': 'clown',
                'star_eyes': 'star_eyes', 'heart_eyes': 'heart_eyes', 'crying': 'crying',
                // Makeup
                'makeup_lips': 'makeup_lips', 'makeup_eyeshadow': 'makeup_eyeshadow',
                'makeup_blush': 'makeup_blush', 'makeup_eyeliner': 'makeup_eyeliner',
                'makeup_mascara': 'makeup_mascara',
                // Glasses
                'glasses': 'glasses',
                // Hair
                'hair_color': 'hair_color',
                // Aging
                'aging': 'aging', 'deaging': 'deaging',
                // Cartoon
                'cartoon': 'cartoon',
                // Beard
                'beard': 'beard',
            };
            return map[preset] || preset;
        }

        // Build full config payload with all extra parameters
        function _buildLiveConfig(preset) {
            const filterName = _mapPresetToFilter(preset);
            const config = {
                type: 'config',
                filter: filterName,
                intensity: Number(intensitySlider?.value || 50),
            };

            // Makeup params
            const makeupColor = document.getElementById('makeupColor');
            const makeupOpacity = document.getElementById('makeupOpacity');
            if (filterName.startsWith('makeup_')) {
                // Convert hex color to OpenCV hue
                const hex = (makeupColor?.value || '#ff0000').replace('#', '');
                const r = parseInt(hex.substring(0, 2), 16);
                const g = parseInt(hex.substring(2, 4), 16);
                const b = parseInt(hex.substring(4, 6), 16);
                // Approximate HSV hue from RGB
                const maxC = Math.max(r, g, b), minC = Math.min(r, g, b);
                let h = 0;
                if (maxC !== minC) {
                    if (maxC === r) h = 60 * (((g - b) / (maxC - minC)) % 6);
                    else if (maxC === g) h = 60 * ((b - r) / (maxC - minC) + 2);
                    else h = 60 * ((r - g) / (maxC - minC) + 4);
                }
                if (h < 0) h += 360;
                config.makeup_hue = Math.round(h / 2); // OpenCV hue range is 0-179
                config.makeup_opacity = Number(makeupOpacity?.value || 50) / 100.0;
            }

            // Glasses type
            if (filterName === 'glasses') {
                const glassesSelect = document.getElementById('glassesSelect');
                config.glasses_type = glassesSelect?.value || 'aviator';
            }

            // Hair color
            if (filterName === 'hair_color') {
                const hairPicker = document.getElementById('hairColorPicker');
                const hairOpacity = document.getElementById('hairOpacity');
                const hex = (hairPicker?.value || '#ff0000').replace('#', '');
                const r = parseInt(hex.substring(0, 2), 16);
                const g = parseInt(hex.substring(2, 4), 16);
                const b = parseInt(hex.substring(4, 6), 16);
                config.hair_color = `${r},${g},${b}`;
                config.hair_intensity = Number(hairOpacity?.value || 60) / 100.0;
            }

            // Beard type & darkness
            if (filterName === 'beard') {
                config.beard_type = document.getElementById('beardSelect')?.value || 'beard';
                config.beard_darkness = Number(document.getElementById('beardDarknessSlider')?.value || 60);
                config.intensity = config.beard_darkness;
            }

            // Eye scale uses raw slider value (-100 to 100)
            if (filterName === 'eye_scaling') {
                const eyeSlider = document.getElementById('eyeSizeSlider');
                config.intensity = Number(eyeSlider?.value || 0);
            }

            return config;
        }

        function stopLiveMode() {
            isLiveMode = false;
            if (liveRafId) { cancelAnimationFrame(liveRafId); liveRafId = null; }
            if (liveFpsTimer) { clearInterval(liveFpsTimer); liveFpsTimer = null; }
            // Close WebSocket
            if (_liveWs) {
                try { _liveWs.close(); } catch (_) {}
                _liveWs = null;
                _wsReady = false;
            }
            _wsPendingFrame = false;
            liveFpsBadge.style.display = 'none';
            liveIndicator.style.display = 'none';
            liveFpsBadge.textContent = '0 FPS';
            if (_lastBlobUrl) { URL.revokeObjectURL(_lastBlobUrl); _lastBlobUrl = null; }
            // Clear stacked states on stop
            liveActiveStates = {};
            renderLiveStatesBar();
            _syncWsRef();
            console.log('[Live] Stopped');
        }

        // ── rAF-based processing loop (throttled to target FPS) ────────
        function liveRafLoop(timestamp) {
            if (!isLiveMode) return;

            // Throttle: skip frame if too soon
            if (timestamp - _lastFrameTime >= _TARGET_INTERVAL) {
                _lastFrameTime = timestamp;

                if (!liveProcessing) {
                    liveProcessing = true;
                    processLiveFrame().finally(() => {
                        liveProcessing = false;
                        if (isLiveMode && _liveHasPendingTick) {
                            _liveHasPendingTick = false;
                            liveProcessing = true;
                            processLiveFrame().finally(() => { liveProcessing = false; });
                        }
                    });
                } else {
                    _liveHasPendingTick = true;
                }
            }

            liveRafId = requestAnimationFrame(liveRafLoop);
        }

        async function processLiveFrame() {
            if (!cameraStream || !isLiveMode) return;
            if (!_wsReady || !_liveWs) return;
            if (_wsPendingFrame) return; // throttle: wait for server to respond

            // If stacked states are active, they're already sent via sendLiveStateUpdate().
            // Only use legacy config path when no stacked states exist.
            const hasStackedStates = Object.keys(liveActiveStates).length > 0;

            if (!hasStackedStates) {
                const preset = getCurrentLivePreset();
                const liveConfig = _buildLiveConfig(preset);
                // Send legacy config update
                _liveWs.send(JSON.stringify(liveConfig));

                if (!liveConfig.filter || liveConfig.filter === 'none') {
                    liveProcessedImg.style.display = 'none';
                    liveFrameCount++;
                    return;
                }
            }

            try {
                // Stage 1: Capture frame to canvas
                const ctx = cameraCanvas.getContext('2d');
                const vw = cameraVideo.videoWidth;
                const vh = cameraVideo.videoHeight;
                if (!vw || !vh) return;

                cameraCanvas.width = vw;
                cameraCanvas.height = vh;
                ctx.drawImage(cameraVideo, 0, 0);

                // Stage 2: Encode to base64
                const b64 = await encodeFrameOffThread();
                if (!isLiveMode || !_wsReady) return;

                // Stage 3: Send via WebSocket
                _wsPendingFrame = true;
                _liveWs.send(JSON.stringify({
                    type: 'frame',
                    data: b64,
                }));

                _liveErrorCount = 0;
            } catch (err) {
                _liveErrorCount++;
                _wsPendingFrame = false;
                if (_liveErrorCount <= 3) {
                    console.warn('[Live] Frame error:', err.message);
                }
                if (_liveErrorCount >= _MAX_LIVE_ERRORS) {
                    liveIndicator.textContent = '● ERROR';
                    liveIndicator.style.background = 'rgba(255,165,0,0.85)';
                }
            }
        }

        // ── Determine which preset the sidebar has selected ─────────────
        function getCurrentLivePreset() {
            // Check emoji buttons first
            const activeEmoji = document.querySelector('.emoji-btn.emoji-active');
            if (activeEmoji) return activeEmoji.dataset.preset;

            // Check if a sidebar operation is selected
            if (selectedOperation) {
                const allowedLiveOps = new Set([
                    'smile', 'eyebrow', 'lip', 'slim',
                    'aging', 'deaging', 'fft', 'cartoon',
                    'eye_scale', 'beard',
                    'makeup_lips', 'makeup_eyeshadow', 'makeup_blush',
                    'makeup_eyeliner', 'makeup_mascara',
                    'glasses', 'hair_color',
                ]);
                if (allowedLiveOps.has(selectedOperation)) return selectedOperation;
            }
            return currentLivePreset;
        }

        // ── Wire sidebar emoji buttons to live mode ─────────────────────
        document.querySelectorAll('.emoji-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const wasActive = btn.classList.contains('emoji-active');
                document.querySelectorAll('.emoji-btn').forEach(b => b.classList.remove('emoji-active'));

                if (wasActive) {
                    currentLivePreset = null;
                    btn.style.outline = '';
                    // ── LIVE MODE: remove emoji from stacked state ──
                    if (isLiveMode && btn.dataset.preset) {
                        sendLiveStateUpdate(btn.dataset.preset, null);
                    }
                } else {
                    btn.classList.add('emoji-active');
                    currentLivePreset = btn.dataset.preset;
                    btn.style.outline = '2px solid var(--primary-color)';
                    // ── LIVE MODE: add emoji to stacked state ──
                    if (isLiveMode && btn.dataset.preset) {
                        sendLiveStateUpdate(btn.dataset.preset, { intensity: 50 });
                    }
                }

                document.querySelectorAll('.emoji-btn:not(.emoji-active)').forEach(b => {
                    b.style.outline = '';
                });
            });
        });

        // ── Wire sidebar ACTION buttons to live mode ────────────────────
        // When live is active, clicking Apply buttons on sidebar panels
        // switches the live filter instead of running a one-shot API call.

        const livePanelButtons = {
            'applyMakeupBtn': () => {
                const region = document.getElementById('makeupRegion')?.value || 'lips';
                selectedOperation = `makeup_${region}`;
            },
            'applyGlassesBtn': () => { selectedOperation = 'glasses'; },
            'applyHairColorBtn': () => { selectedOperation = 'hair_color'; },
            'cartoonBtn': () => { selectedOperation = 'cartoon'; },
            'beardBtn': () => { selectedOperation = 'beard'; },
            'eyeSizeBtn': () => { selectedOperation = 'eye_scale'; },
        };

        Object.entries(livePanelButtons).forEach(([btnId, setOp]) => {
            const btn = document.getElementById(btnId);
            if (!btn) return;
            // Add a capture-phase listener that fires BEFORE the normal handler
            btn.addEventListener('click', () => {
                if (isLiveMode) {
                    setOp();
                    console.log('[Live] Sidebar switched to:', selectedOperation);
                }
            }, true); // capture phase
        });

        // Also wire makeup region dropdown to update immediately in live mode
        const makeupRegionSelect = document.getElementById('makeupRegion');
        if (makeupRegionSelect) {
            makeupRegionSelect.addEventListener('change', () => {
                if (isLiveMode && selectedOperation?.startsWith('makeup_')) {
                    const previousMakeupOp = selectedOperation;
                    const region = makeupRegionSelect.value || 'lips';
                    selectedOperation = `makeup_${region}`;
                    if (previousMakeupOp && previousMakeupOp !== selectedOperation) {
                        sendLiveStateUpdate(previousMakeupOp, null);
                    }
                    const hexVal = document.getElementById('makeupColor')?.value || '#ff0000';
                    const hue = hexToOpenCVHue(hexVal);
                    const opacity = Number(document.getElementById('makeupOpacity')?.value || 50) / 100.0;
                    sendLiveStateUpdate(selectedOperation, { makeup_hue: hue, makeup_opacity: opacity, makeup_color: hexVal, intensity: 50 });
                    console.log('[Live] Makeup region changed to:', selectedOperation);
                }
            });
        }

        [document.getElementById('makeupColor'), document.getElementById('makeupOpacity')].forEach(control => {
            if (!control) return;
            control.addEventListener('input', () => {
                if (isLiveMode && selectedOperation?.startsWith('makeup_')) {
                    const hexVal = document.getElementById('makeupColor')?.value || '#ff0000';
                    const hue = hexToOpenCVHue(hexVal);
                    const opacity = Number(document.getElementById('makeupOpacity')?.value || 50) / 100.0;
                    sendLiveStateUpdate(selectedOperation, { makeup_hue: hue, makeup_opacity: opacity, makeup_color: hexVal, intensity: 50 });
                }
            });
        });

        // Wire glasses dropdown to push config immediately in live mode
        const liveGlassesSelect = document.getElementById('glassesSelect');
        if (liveGlassesSelect) {
            liveGlassesSelect.addEventListener('change', () => {
                if (isLiveMode) {
                    selectedOperation = 'glasses';
                    console.log('[Live] Glasses type changed to:', liveGlassesSelect.value);
                }
            });
        }

        window.addEventListener('beforeunload', () => {
            stopLiveMode();
            if (cameraStream) {
                cameraStream.getTracks().forEach(t => t.stop());
                cameraStream = null;
            }
            if (liveWorker) {
                liveWorker.terminate();
                liveWorker = null;
                liveWorkerReady = false;
            }
        });

    })();

    // =========================================================================
    // FACE SWAP UI & API INTEGRATION
    // =========================================================================
    (function initFaceSwap() {
        const fsUploadInput = document.getElementById('faceSwapUpload');
        const fsMiniDropZone = document.getElementById('fsMiniDropZone');
        const fsUploadState = document.getElementById('fsUploadState');
        const fsPreviewState = document.getElementById('fsPreviewState');
        const fsPreviewImg = document.getElementById('faceSwapPreviewImg');
        const removeFsImgBtn = document.getElementById('removeFaceSwapImgBtn');
        const fsLoadedStatus = document.getElementById('faceSwapLoadedStatus');

        const fsEnableToggle = document.getElementById('enableFaceSwapToggle');
        
        const fsBlendSlider = document.getElementById('fsBlendSlider');
        const fsStabilitySlider = document.getElementById('fsStabilitySlider');
        const fsSoftnessSlider = document.getElementById('fsSoftnessSlider');

        let sourceFaceLoaded = false;
        let fsSourceFile = null;

        // Sliders value updates
        if (fsBlendSlider) fsBlendSlider.addEventListener('input', (e) => document.getElementById('fsBlendVal').textContent = e.target.value + '%');
        if (fsStabilitySlider) fsStabilitySlider.addEventListener('input', (e) => document.getElementById('fsStabilityVal').textContent = e.target.value + '%');
        if (fsSoftnessSlider) fsSoftnessSlider.addEventListener('input', (e) => document.getElementById('fsSoftnessVal').textContent = e.target.value + '%');

        // Drop Zone UI Handlers
        if (fsMiniDropZone && fsUploadInput) {
            fsMiniDropZone.addEventListener('click', (e) => {
                if (e.target !== removeFsImgBtn) {
                    fsUploadInput.click();
                }
            });

            fsMiniDropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                fsMiniDropZone.style.backgroundColor = 'rgba(102, 126, 234, 0.1)';
                fsMiniDropZone.style.borderColor = 'var(--primary-color)';
            });

            fsMiniDropZone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                fsMiniDropZone.style.backgroundColor = 'rgba(0,0,0,0.2)';
                fsMiniDropZone.style.borderColor = 'var(--border-color)';
            });

            fsMiniDropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                fsMiniDropZone.style.backgroundColor = 'rgba(0,0,0,0.2)';
                fsMiniDropZone.style.borderColor = 'var(--border-color)';
                if (e.dataTransfer.files.length > 0) {
                    handleSourceUpload(e.dataTransfer.files[0]);
                }
            });

            fsUploadInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleSourceUpload(e.target.files[0]);
                }
            });
        }

        async function handleSourceUpload(file) {
            if (!file.type.startsWith('image/')) return;
            
            fsSourceFile = file;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (ev) => {
                fsPreviewImg.src = ev.target.result;
                fsUploadState.style.display = 'none';
                fsPreviewState.style.display = 'block';
                fsLoadedStatus.textContent = "Uploading...";
                fsLoadedStatus.style.color = "var(--text-color)";
            };
            reader.readAsDataURL(file);

            // API: POST /api/face-swap/upload-source
            const formData = new FormData();
            formData.append('source', file);

            try {
                const response = await fetch(`${API_BASE}/face-swap/upload-source`, {
                    method: 'POST',
                    body: formData
                });
                if (!response.ok) {
                    const errPayload = await response.json().catch(() => ({}));
                    throw new Error(errPayload.detail || 'Upload failed');
                }
                
                sourceFaceLoaded = true;
                fsLoadedStatus.textContent = "Ready!";
                fsLoadedStatus.style.color = "var(--success-color)";
            } catch (err) {
                console.error('[Face Swap] Source upload error:', err);
                fsLoadedStatus.textContent = `Error`;
                fsLoadedStatus.style.color = "var(--error-color)";
                
                // Handle backend unavailable by showing warning but keep preview
                if (err.message.includes('fetch') || err.message.includes('Network')) {
                    fsLoadedStatus.textContent = "Backend offline";
                }
            }
        }

        if (removeFsImgBtn) {
            removeFsImgBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                fsSourceFile = null;
                sourceFaceLoaded = false;
                fsUploadInput.value = "";
                fsPreviewState.style.display = 'none';
                fsUploadState.style.display = 'flex';
                if (fsEnableToggle) fsEnableToggle.checked = false;
                // If live face swap was active, we should also stop it.
                fetch(`${API_BASE}/face-swap/stop`, { method: 'POST' }).catch(() => {});
            });
        }

        // Enable Face Swap Toggle
        if (fsEnableToggle) {
            fsEnableToggle.addEventListener('change', async (e) => {
                if (!sourceFaceLoaded) {
                    alert("Please upload a source face first.");
                    e.target.checked = false;
                    return;
                }

                if (e.target.checked) {
                    // Start Face Swap API
                    try {
                        const payload = {
                            blend_strength: parseInt(fsBlendSlider.value),
                            stability: parseInt(fsStabilitySlider.value),
                            mask_softness: parseInt(fsSoftnessSlider.value)
                        };
                        const response = await fetch(`${API_BASE}/face-swap/start`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                        if (!response.ok) {
                            throw new Error('Failed to start face swap');
                        }
                        console.log('[Face Swap] Live mode started');
                        
                        // Set selected operation to face_swap to trigger existing live loops if needed
                        selectedOperation = 'face_swap';
                    } catch (err) {
                        console.error('[Face Swap] Start error:', err);
                        // Backend unavailable handling
                        alert("Could not start face swap: " + (err.message || 'Backend unavailable'));
                        e.target.checked = false;
                    }
                } else {
                    // Stop Face Swap API
                    try {
                        await fetch(`${API_BASE}/face-swap/stop`, { method: 'POST' });
                        console.log('[Face Swap] Live mode stopped');
                    } catch (err) {
                        console.error('[Face Swap] Stop error:', err);
                    }
                }
            });
        }
        
        // Static Face Swap override for the Apply Button
        const mainApplyBtn = document.getElementById('applyBtn');
        if (mainApplyBtn) {
            mainApplyBtn.addEventListener('click', async (e) => {
                if (selectedOperation === 'face_swap') {
                    e.stopPropagation(); // prevent default apply action
                    
                    if (!uploadedFile) {
                        alert("Please upload a target image in the main view.");
                        return;
                    }
                    if (!fsSourceFile) {
                        alert("Please upload a source face in the Face Swap panel.");
                        return;
                    }

                    const loadingOverlay = document.getElementById('loadingOverlay');
                    if (loadingOverlay) loadingOverlay.style.display = 'flex';

                    const formData = new FormData();
                    formData.append('source', fsSourceFile);
                    formData.append('target', uploadedFile);

                    try {
                        const response = await fetch(`${API_BASE}/face-swap`, {
                            method: 'POST',
                            body: formData
                        });

                        const payload = await response.json();
                        if (!response.ok) {
                            throw new Error(payload?.detail || 'Face swap failed.');
                        }

                        // Display result
                        const afterImg = document.getElementById('afterImg');
                        if (afterImg) afterImg.src = payload.swapped_image;
                        
                        const visualPreviewArea = document.getElementById('visualPreviewArea');
                        const imageWrapper = document.getElementById('imageWrapper');
                        if (visualPreviewArea) visualPreviewArea.style.display = 'flex';
                        if (imageWrapper) imageWrapper.style.display = 'block';

                        document.getElementById('analysisSummary').innerHTML = `<strong>Status: Success</strong><br/>Face Swap completed in ${payload.processing_time_ms} ms.`;
                        
                    } catch (err) {
                        console.error('[Face Swap] Static apply error:', err);
                        document.getElementById('analysisSummary').innerHTML = `<strong>Status: Failed</strong><br/>${err.message || 'Face swap processing failed.'}`;
                        alert("Face Swap Error: " + (err.message || 'Backend unavailable'));
                    } finally {
                        if (loadingOverlay) loadingOverlay.style.display = 'none';
                    }
                }
            }, true); // Use capture phase so it runs first
        }


        // Live Mode specific slider listeners
        [fsBlendSlider, fsStabilitySlider, fsSoftnessSlider].forEach(slider => {
            if (slider) {
                slider.addEventListener('change', async () => {
                    if (fsEnableToggle && fsEnableToggle.checked) {
                        // Resend start command to update params
                        try {
                            const payload = {
                                blend_strength: parseInt(fsBlendSlider.value),
                                stability: parseInt(fsStabilitySlider.value),
                                mask_softness: parseInt(fsSoftnessSlider.value)
                            };
                            await fetch(`${API_BASE}/face-swap/start`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(payload)
                            });
                        } catch (err) {
                            console.error('[Face Swap] Update params error:', err);
                        }
                    }
                });
            }
        });

    })();

});

