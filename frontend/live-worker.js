let workerCanvas = null;
let workerCtx = null;

self.onmessage = async (event) => {
  const { type, bitmap, width, height, quality } = event.data || {};
  if (type !== "encodeFrame" || !bitmap) return;

  try {
    if (!workerCanvas) {
      workerCanvas = new OffscreenCanvas(width, height);
      workerCtx = workerCanvas.getContext("2d", { alpha: false, desynchronized: true });
    } else if (workerCanvas.width !== width || workerCanvas.height !== height) {
      workerCanvas.width = width;
      workerCanvas.height = height;
    }

    workerCtx.clearRect(0, 0, width, height);
    workerCtx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();

    const blob = await workerCanvas.convertToBlob({
      type: "image/jpeg",
      quality: typeof quality === "number" ? quality : 0.7,
    });
    const reader = new FileReader();
    reader.onloadend = () => {
      const dataUrl = typeof reader.result === "string" ? reader.result : "";
      self.postMessage({ type: "encodedFrame", dataUrl });
    };
    reader.readAsDataURL(blob);
  } catch (err) {
    self.postMessage({
      type: "workerError",
      error: err?.message || "Worker frame encoding failed",
    });
  }
};
