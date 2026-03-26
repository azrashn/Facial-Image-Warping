// HTML elemanlarını seçelim
const fileInput = document.getElementById('imageInput');
const intensitySlider = document.getElementById('intensitySlider');
const applyBtn = document.getElementById('applyBtn');
const beforeImg = document.getElementById('beforeImg');
const afterImg = document.getElementById('afterImg');
const loading = document.getElementById('loading');

// Ön izleme (Kullanıcı resim seçtiğinde Before alanında göster)
fileInput.addEventListener('change', function (e) {
    if (e.target.files[0]) {
        beforeImg.src = URL.createObjectURL(e.target.files[0]);
        beforeImg.style.display = 'block';
    }
});

// Slider değerini canlı güncelle
intensitySlider.addEventListener('input', function (e) {
    document.getElementById('intensityVal').innerText = e.target.value;
});

// BUTONA BASILDIĞINDA (API'YE İSTEK ATMA)
applyBtn.addEventListener('click', async function () {
    const file = fileInput.files[0];
    if (!file) {
        alert("Lütfen önce bir fotoğraf yükleyin!");
        return;
    }

    // Yükleniyor yazısını göster
    loading.style.display = 'inline';

    // Verileri paketle (Resim ve Slider değeri)
    const formData = new FormData();
    formData.append("file", file);
    formData.append("operation", "Smile"); // Şimdilik sabit
    formData.append("intensity", intensitySlider.value);

    try {
        // FastAPI Sunucusuna İsteği Gönder!
        const response = await fetch("http://127.0.0.1:8000/apply_transformation", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        // 1. İşlenmiş Resmi Ekrana Bas
        afterImg.src = data.processed_image;
        afterImg.style.display = 'block';

        // 2. Metrikleri (Rol 6'nın değerlerini) HTML içine yaz
        document.getElementById('mse_value').innerText = data.metrics.mse;
        document.getElementById('psnr_value').innerText = data.metrics.psnr;
        document.getElementById('ssim_value').innerText = data.metrics.ssim;

    } catch (error) {
        console.error("API Hatası:", error);
        alert("Sunucuya bağlanılamadı. FastAPI çalışıyor mu?");
    } finally {
        // İşlem bitince yükleniyor yazısını gizle
        loading.style.display = 'none';
    }
});