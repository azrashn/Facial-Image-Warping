f"""
mask_stabilizer.py — Live kamera maske overlay stabilizasyonu.

Rol 4 — Görev Dağılımı v2:
  - MaskStabilizer: uzaylı, robot, sakal gibi yüz maskelerini
    live kamerada titremeden sabit tutar
  - BlushEffect: cilt tonuyla uyumlu yanak makyaj efekti
  - apply_live_alien: live kamera için uzaylı filtresi
  - apply_live_clown: live kamera için palyaço filtresi
  - apply_live_beard: live kamera için sakal/bıyık filtresi
  - apply_live_blush: live kamera için blush/makyaj efekti

Stabilizasyon yöntemi:
  EMA (Exponential Moving Average) ile her maske elemanının
  pozisyonu ve boyutu frame'ler arası yumuşatılır.
  Bu titremeleri ve ani zıplamaları engeller.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict

import cv2
import numpy as np

logger = logging.getLogger("facial_pipeline.mask_stabilizer")


# ---------------------------------------------------------------------------
# Yardımcı: EMA ile tek değer yumuşatma
# ---------------------------------------------------------------------------
class _EMAValue:
    """Tek bir sayısal değeri EMA ile yumuşatır."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, new_val: float) -> float:
        if self._value is None:
            self._value = new_val
        else:
            self._value = self.alpha * new_val + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None


# ---------------------------------------------------------------------------
# Yardımcı: EMA ile 2D nokta yumuşatma
# ---------------------------------------------------------------------------
class _EMAPoint:
    """Bir (x, y) noktasını EMA ile yumuşatır."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._pt: Optional[np.ndarray] = None

    def update(self, new_pt: np.ndarray) -> np.ndarray:
        if self._pt is None:
            self._pt = new_pt.astype(np.float32).copy()
        else:
            self._pt = self.alpha * new_pt + (1.0 - self.alpha) * self._pt
        return self._pt.copy()

    def reset(self) -> None:
        self._pt = None


# ---------------------------------------------------------------------------
# Ana Stabilizatör Sınıfı
# ---------------------------------------------------------------------------
class MaskStabilizer:
    """
    Live kamerada yüz maskesi overlay'lerini stabilize eder.

    Her maske elemanı (burun, göz merkezi, ağız köşesi vb.) için
    ayrı bir EMA tracker tutar. Bu sayede her eleman bağımsız
    yumuşatılır ve genel maske stabilitesi artar.

    Kullanım:
        stabilizer = MaskStabilizer(alpha=0.35)
        smooth_lm = stabilizer.smooth_landmarks(raw_landmarks)
        # smooth_lm ile maske çiz
    """

    def __init__(self, alpha: float = 0.35):
        """
        Parameters
        ----------
        alpha : float
            EMA faktörü. Düşük = daha smooth (daha az reaktif).
            Yüksek = daha reaktif (daha az smooth).
            Önerilen: 0.25-0.45 arası.
        """
        self.alpha = alpha
        self._smoothers: Dict[int, _EMAPoint] = {}
        self._face_lost_frames: int = 0
        self._MAX_LOST = 15  # Bu kadar frame sonra reset

    def smooth_landmarks(
        self, landmarks: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Ham landmark'ları EMA ile yumuşat.

        Parameters
        ----------
        landmarks : np.ndarray | None
            (N, 2) ham landmark dizisi veya yüz yoksa None.

        Returns
        -------
        np.ndarray | None
            Yumuşatılmış landmark dizisi.
        """
        if landmarks is None:
            self._face_lost_frames += 1
            if self._face_lost_frames > self._MAX_LOST:
                self._reset_all()
            return None

        self._face_lost_frames = 0

        n = len(landmarks)
        smoothed = np.zeros_like(landmarks)

        for i in range(n):
            if i not in self._smoothers:
                self._smoothers[i] = _EMAPoint(alpha=self.alpha)
            smoothed[i] = self._smoothers[i].update(landmarks[i])

        return smoothed

    def _reset_all(self) -> None:
        for s in self._smoothers.values():
            s.reset()
        self._smoothers.clear()
        self._face_lost_frames = 0

    def reset(self) -> None:
        """Tüm state'i sıfırla."""
        self._reset_all()


# ---------------------------------------------------------------------------
# Cilt Tonu Tespiti
# ---------------------------------------------------------------------------
def detect_skin_tone(
    image_bgr: np.ndarray, landmarks: np.ndarray
) -> np.ndarray:
    """
    Yanak landmark'larından cilt tonunu tespit et.

    Parameters
    ----------
    landmarks : (N, 2) float32

    Returns
    -------
    np.ndarray
        BGR cilt rengi (3,)
    """
    h, w = image_bgr.shape[:2]
    cheek_indices = [234, 454, 205, 425]
    colors = []

    for idx in cheek_indices:
        if idx >= len(landmarks):
            continue
        x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
        x = max(5, min(w - 6, x))
        y = max(5, min(h - 6, y))
        patch = image_bgr[y - 5:y + 5, x - 5:x + 5]
        if patch.size > 0:
            colors.append(patch.mean(axis=(0, 1)))

    if not colors:
        return np.array([150, 120, 100], dtype=np.float32)

    return np.mean(colors, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Live Uzaylı Filtresi
# ---------------------------------------------------------------------------
# Global stabilizer instance'ları (her filtre için ayrı)
_alien_stabilizer = MaskStabilizer(alpha=0.35)
_alien_eye_smoother_left  = _EMAPoint(alpha=0.15)   # gözler için daha yüksek smoothing
_alien_eye_smoother_right = _EMAPoint(alpha=0.15)
_clown_stabilizer = MaskStabilizer(alpha=0.35)
_beard_stabilizer = MaskStabilizer(alpha=0.30)
_blush_stabilizer = MaskStabilizer(alpha=0.30)


def apply_live_alien(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
) -> np.ndarray:
    """
    Live kamera için uzaylı filtresi.
    Landmarks zaten TemporalSmoother'dan geçmiş olmalı.
    Ek stabilizasyon için MaskStabilizer kullanır.

    Parameters
    ----------
    image_bgr  : BGR frame
    landmarks  : (N, 2) smoothed landmarks
    """
    try:
        h, w = image_bgr.shape[:2]

        # Ek stabilizasyon katmanı
        stable_lm = _alien_stabilizer.smooth_landmarks(landmarks)
        if stable_lm is None:
            return image_bgr

        # Yüz ölçeği
        face_sz = float(np.linalg.norm(stable_lm[133] - stable_lm[362]))

        # Yüz oval maskesi
        jaw_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109
        ]
        jaw_pts = np.array(
            [[int(stable_lm[i][0]), int(stable_lm[i][1])] for i in jaw_indices],
            dtype=np.int32
        )
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, cv2.convexHull(jaw_pts), 255)
        face_mask_blur = cv2.GaussianBlur(face_mask, (31, 31), 0).astype(np.float32) / 255.0
        face_mask_3ch = np.stack([face_mask_blur] * 3, axis=-1)

        # Yeşil tonu yüze uygula
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_green = hsv.copy()
        hsv_green[:, :, 0] = 75.0
        hsv_green[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2 + 25, 0, 255)
        hsv_green[:, :, 2] = np.clip(hsv[:, :, 2] * 0.88, 0, 255)
        green_img = cv2.cvtColor(hsv_green.astype(np.uint8), cv2.COLOR_HSV2BGR)

        result = (
            green_img.astype(np.float32) * face_mask_3ch * 0.60
            + image_bgr.astype(np.float32) * (1.0 - face_mask_3ch * 0.60)
        ).astype(np.uint8)

        # Büyük siyah oval gözler
        left_eye_pts  = [33, 133, 160, 159, 158, 157, 163, 144, 145, 153, 154, 155]
        right_eye_pts = [362, 263, 387, 386, 385, 384, 390, 373, 374, 380, 381, 382]

        c_left  = _alien_eye_smoother_left.update(stable_lm[left_eye_pts].mean(axis=0))
        c_right = _alien_eye_smoother_right.update(stable_lm[right_eye_pts].mean(axis=0))

        eye_rx = int(face_sz * 0.28)
        eye_ry = int(face_sz * 0.22)

        eye_layer = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.ellipse(eye_layer, (int(c_left[0]),  int(c_left[1])),
                    (eye_rx, eye_ry), 0, 0, 360, (12, 12, 12), -1)
        cv2.ellipse(eye_layer, (int(c_right[0]), int(c_right[1])),
                    (eye_rx, eye_ry), 0, 0, 360, (12, 12, 12), -1)

        # Parlaklık noktası
        ho, vo = int(eye_rx * 0.28), int(eye_ry * 0.28)
        cv2.circle(eye_layer, (int(c_left[0])  - ho, int(c_left[1])  - vo),
                   int(eye_rx * 0.12), (70, 70, 70), -1)
        cv2.circle(eye_layer, (int(c_right[0]) - ho, int(c_right[1]) - vo),
                   int(eye_rx * 0.12), (70, 70, 70), -1)

        eye_alpha = (eye_layer.sum(axis=2) > 0).astype(np.float32)
        eye_alpha = cv2.GaussianBlur(eye_alpha, (5, 5), 0)
        eye_alpha_3ch = np.stack([eye_alpha] * 3, axis=-1)

        result = (
            eye_layer.astype(np.float32) * eye_alpha_3ch
            + result.astype(np.float32) * (1.0 - eye_alpha_3ch)
        ).astype(np.uint8)

        return result

    except Exception as e:
        logger.error("apply_live_alien failed: %s", e)
        return image_bgr


# ---------------------------------------------------------------------------
# Live Palyaço Filtresi
# ---------------------------------------------------------------------------
def apply_live_clown(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
) -> np.ndarray:
    """
    Live kamera için Joker tarzı palyaço filtresi.
    Warp YOK — sadece makyaj overlay.
    """
    try:
        h, w = image_bgr.shape[:2]

        stable_lm = _clown_stabilizer.smooth_landmarks(landmarks)
        if stable_lm is None:
            return image_bgr

        face_sz = float(np.linalg.norm(stable_lm[133] - stable_lm[362]))
        result = image_bgr.copy()

        # Yüz maskesi
        jaw_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
            162, 21, 54, 103, 67, 109
        ]
        jaw_pts = np.array(
            [[int(stable_lm[i][0]), int(stable_lm[i][1])] for i in jaw_indices],
            dtype=np.int32
        )
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, cv2.convexHull(jaw_pts), 255)
        face_mask_blur = cv2.GaussianBlur(face_mask, (25, 25), 0).astype(np.float32) / 255.0
        face_mask_3ch = np.stack([face_mask_blur] * 3, axis=-1)

        # Beyaz yüz boyası %55
        white = np.ones_like(result, dtype=np.float32) * 255
        result = (
            white * face_mask_3ch * 0.55
            + result.astype(np.float32) * (1.0 - face_mask_3ch * 0.55)
        ).astype(np.uint8)

        paint = np.zeros((h, w, 3), dtype=np.float32)

        # Göz merkezleri
        le_cx = int((stable_lm[33][0] + stable_lm[133][0]) / 2)
        le_cy = int((stable_lm[33][1] + stable_lm[133][1]) / 2)
        re_cx = int((stable_lm[362][0] + stable_lm[263][0]) / 2)
        re_cy = int((stable_lm[362][1] + stable_lm[263][1]) / 2)

        # Mavi eşkenar dörtgen göz makyajı
        e_r = int(face_sz * 0.22)
        def rhombus(cx, cy, r):
            return np.array([
                [cx - r, cy], [cx, cy - r],
                [cx + r, cy], [cx, cy + r]
            ], dtype=np.int32)

        cv2.fillPoly(paint, [rhombus(le_cx, le_cy, e_r)], (210, 90, 10))
        cv2.fillPoly(paint, [rhombus(re_cx, re_cy, e_r)], (210, 90, 10))

        # Kırmızı kaşlar
        lb = np.array([[int(stable_lm[i][0]), int(stable_lm[i][1])]
                       for i in [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]], dtype=np.int32)
        rb = np.array([[int(stable_lm[i][0]), int(stable_lm[i][1])]
                       for i in [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]], dtype=np.int32)
        brow_thick = max(int(face_sz * 0.06), 2)
        cv2.polylines(paint, [lb], False, (0, 0, 220), brow_thick)
        cv2.polylines(paint, [rb], False, (0, 0, 220), brow_thick)

        # Büyük kırmızı burun
        nose_pt = (int(stable_lm[4][0]), int(stable_lm[4][1]))
        nose_r  = int(face_sz * 0.20)
        cv2.circle(paint, nose_pt, nose_r, (0, 0, 240), -1)
        cv2.circle(paint,
                   (nose_pt[0] - int(nose_r * 0.3), nose_pt[1] - int(nose_r * 0.35)),
                   int(nose_r * 0.22), (100, 100, 255), -1)

        # Kırmızı dudak boyası
        mouth_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                     291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        om_pts = np.array([[int(stable_lm[i][0]), int(stable_lm[i][1])]
                           for i in mouth_idx], dtype=np.int32)
        om_c   = om_pts.mean(axis=0).astype(int)
        om_big = ((om_pts - om_c) * 1.35 + om_c).astype(np.int32)
        cv2.fillPoly(paint, [om_big], (0, 0, 225))

        # Gülüş çizgileri
        lc = (int(stable_lm[61][0]),  int(stable_lm[61][1]))
        rc = (int(stable_lm[291][0]), int(stable_lm[291][1]))
        lch = (int(stable_lm[205][0] - face_sz * 0.20), int(stable_lm[205][1] + face_sz * 0.05))
        rch = (int(stable_lm[425][0] + face_sz * 0.20), int(stable_lm[425][1] + face_sz * 0.05))
        line_w = max(int(face_sz * 0.08), 3)
        cv2.line(paint, lc, lch, (0, 0, 225), line_w)
        cv2.line(paint, rc, rch, (0, 0, 225), line_w)

        # Blend
        paint_blur  = cv2.GaussianBlur(paint, (9, 9), 0)
        paint_alpha = np.clip(paint_blur.sum(axis=2, keepdims=True) / 280.0, 0, 1)
        paint_alpha = np.repeat(paint_alpha, 3, axis=2)

        final = (
            paint_blur * paint_alpha * 0.85
            + result.astype(np.float32) * (1.0 - paint_alpha * 0.85)
        ).astype(np.uint8)

        return final

    except Exception as e:
        logger.error("apply_live_clown failed: %s", e)
        return image_bgr


# ---------------------------------------------------------------------------
# Live Sakal/Bıyık Filtresi (Düzeltilmiş)
# ---------------------------------------------------------------------------
def apply_live_beard(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    intensity: int = 70,
    style: str = "beard",
) -> np.ndarray:
    """
    Live kamera için sakal/bıyık filtresi.

    Görev Dağılımı v2 — Rol 4:
      Sakal & Bıyık Filtresi Onarımı:
      - Landmark hizalama düzeltildi
      - Cilt tonuyla uyumlu renk
      - Stabilize overlay

    Parameters
    ----------
    style : "beard" | "mustache" | "full"
    """
    try:
        h, w = image_bgr.shape[:2]

        stable_lm = _beard_stabilizer.smooth_landmarks(landmarks)
        if stable_lm is None:
            return image_bgr

        alpha = max(0.0, min(1.0, intensity / 100.0))
        face_sz = float(np.linalg.norm(stable_lm[133] - stable_lm[362]))

        # Cilt tonu tespiti
        skin_color = detect_skin_tone(image_bgr, stable_lm)
        beard_color = skin_color * 0.25  # Cilt tonunun %25'i — koyu ama uyumlu

        result = image_bgr.copy()
        paint = np.zeros((h, w, 3), dtype=np.float32)

        if style in ("beard", "full"):
            # Sakal bölgesi: alt dudak altından çene hattına
            beard_idx = [
                17, 18, 200, 199, 175, 152, 377, 400, 378,
                379, 365, 397, 288, 361, 323, 454, 356, 389,
                251, 284, 332, 297, 338, 10, 109, 67, 103,
                54, 21, 162, 127, 234, 93, 132, 58, 172,
                136, 150, 149, 176, 148, 152
            ]
            b_pts = np.array(
                [[int(stable_lm[i][0]), int(stable_lm[i][1])]
                 for i in beard_idx if i < len(stable_lm)],
                dtype=np.int32
            )
            if len(b_pts) >= 3:
                beard_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(beard_mask, cv2.convexHull(b_pts), 255)

                # Üst dudak üstünü çıkar
                upper_lip_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                                  308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
                ul_pts = np.array(
                    [[int(stable_lm[i][0]), int(stable_lm[i][1])]
                     for i in upper_lip_idx if i < len(stable_lm)],
                    dtype=np.int32
                )
                if len(ul_pts) >= 3:
                    cv2.fillConvexPoly(beard_mask, cv2.convexHull(ul_pts), 0)

                # Noise-based doku
                noise = np.random.randint(0, 60, (h, w), dtype=np.uint8)
                _, hair_tex = cv2.threshold(noise, 150, 255, cv2.THRESH_BINARY)
                hair_tex = cv2.bitwise_and(hair_tex, beard_mask)
                hair_tex = cv2.GaussianBlur(hair_tex, (3, 3), 0)

                soft_mask = cv2.GaussianBlur(
                    (hair_tex > 0).astype(np.float32), (15, 15), 0
                )
                for c in range(3):
                    paint[:, :, c] += soft_mask * beard_color[c]

        if style in ("mustache", "full"):
            # Bıyık bölgesi: burun altı ile üst dudak üstü arası
            mus_idx = [2, 326, 327, 4, 97, 98, 60, 75, 290, 305,
                       0, 267, 269, 270, 37, 39, 40]
            m_pts = np.array(
                [[int(stable_lm[i][0]), int(stable_lm[i][1])]
                 for i in mus_idx if i < len(stable_lm)],
                dtype=np.int32
            )
            if len(m_pts) >= 3:
                mus_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mus_mask, cv2.convexHull(m_pts), 255)

                noise2 = np.random.randint(0, 60, (h, w), dtype=np.uint8)
                _, hair2 = cv2.threshold(noise2, 140, 255, cv2.THRESH_BINARY)
                hair2 = cv2.bitwise_and(hair2, mus_mask)

                soft_mus = cv2.GaussianBlur(
                    (hair2 > 0).astype(np.float32), (11, 11), 0
                )
                for c in range(3):
                    paint[:, :, c] += soft_mus * beard_color[c]

        if cv2.countNonZero((paint.sum(axis=2) > 0).astype(np.uint8)) == 0:
            return result

        paint_blur  = cv2.GaussianBlur(paint, (7, 7), 0)
        paint_alpha = np.clip(paint_blur.sum(axis=2, keepdims=True) / 200.0, 0, 1)
        paint_alpha = np.repeat(paint_alpha, 3, axis=2) * alpha

        final = (
            paint_blur * paint_alpha
            + result.astype(np.float32) * (1.0 - paint_alpha)
        ).astype(np.uint8)

        return final

    except Exception as e:
        logger.error("apply_live_beard failed: %s", e)
        return image_bgr


# ---------------------------------------------------------------------------
# Live Blush / Makyaj Efekti
# ---------------------------------------------------------------------------
def apply_live_blush(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    intensity: int = 60,
    color: str = "pink",
) -> np.ndarray:
    """
    Live kamera için blush/yanak makyaj efekti.

    Görev Dağılımı v2 — Rol 4:
      Blush / Makyaj Efekti İyileştirmesi:
      - Cilt tonuyla uyumlu renk karışımı
      - Yüz hareket ettiğinde doğal takip
      - Stabilize overlay

    Parameters
    ----------
    color : "pink" | "coral" | "peach" | "rose"
    """
    try:
        h, w = image_bgr.shape[:2]

        stable_lm = _blush_stabilizer.smooth_landmarks(landmarks)
        if stable_lm is None:
            return image_bgr

        face_sz = float(np.linalg.norm(stable_lm[133] - stable_lm[362]))
        alpha = max(0.0, min(1.0, intensity / 100.0))

        # Cilt tonunu tespit et ve blush rengini cilt tonuyla uyumlu yap
        skin_color = detect_skin_tone(image_bgr, stable_lm)

        # Blush renk presetleri — cilt tonuyla blend edilecek
        blush_presets = {
            "pink":  np.array([120, 80, 210], dtype=np.float32),   # BGR
            "coral": np.array([80, 100, 220], dtype=np.float32),
            "peach": np.array([130, 150, 220], dtype=np.float32),
            "rose":  np.array([100, 70, 180], dtype=np.float32),
        }
        base_color = blush_presets.get(color, blush_presets["pink"])

        # Cilt tonuyla %40 blend — daha doğal görünüm
        blush_color = base_color * 0.60 + skin_color * 0.40

        # Yanak merkezleri
        left_cheek_c  = (int(stable_lm[205][0]), int(stable_lm[205][1]))
        right_cheek_c = (int(stable_lm[425][0]), int(stable_lm[425][1]))
        blush_r = int(face_sz * 0.22)

        paint = np.zeros((h, w, 3), dtype=np.float32)
        cv2.circle(paint, left_cheek_c,  blush_r, blush_color.tolist(), -1)
        cv2.circle(paint, right_cheek_c, blush_r, blush_color.tolist(), -1)

        # Yumuşat (Gaussian blur = spatial low-pass filter)
        paint_blur = cv2.GaussianBlur(paint, (int(blush_r * 1.5) | 1, int(blush_r * 1.5) | 1), 0)

        # Alpha mask
        paint_alpha = np.clip(
            paint_blur.sum(axis=2, keepdims=True) / (blush_color.sum() + 1e-6),
            0, 1
        )
        paint_alpha = np.repeat(paint_alpha, 3, axis=2) * alpha * 0.55

        result = (
            paint_blur * paint_alpha
            + image_bgr.astype(np.float32) * (1.0 - paint_alpha)
        ).astype(np.uint8)

        return result

    except Exception as e:
        logger.error("apply_live_blush failed: %s", e)
        return image_bgr
