/**
 * Background service worker.
 *
 * Responsibilities:
 * 1. Register right-click context menu "Detect Font"
 * 2. Receive rectangle coordinates from content.js
 * 3. Screenshot the visible tab and crop to the selected rectangle
 * 4. Send cropped PNG to the backend API
 * 5. Store results for the popup to read
 */

const DEFAULT_API_URL = "https://guanc27-check-fonts.hf.space";

// -------------------------------------------------------------------
// Context menu
// -------------------------------------------------------------------
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "detect-font",
    title: "Detect Font",
    contexts: ["all"],
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "detect-font" && tab?.id) {
    chrome.tabs.sendMessage(tab.id, { action: "startSelection" });
  }
});

// -------------------------------------------------------------------
// Message handler from content.js
// -------------------------------------------------------------------
chrome.runtime.onMessage.addListener((msg, sender) => {
  if (msg.action === "captureRegion" && sender.tab?.id) {
    handleCapture(sender.tab.id, msg.rect, msg.devicePixelRatio);
  }
});

// -------------------------------------------------------------------
// Core pipeline: screenshot -> crop -> API -> store
// -------------------------------------------------------------------
async function handleCapture(tabId, rect, dpr) {
  try {
    // Save "loading" state so popup can show a spinner immediately
    await chrome.storage.local.set({
      fontResults: null,
      fontError: null,
      fontLoading: true,
    });

    // 1. Screenshot the visible tab
    const dataUrl = await chrome.tabs.captureVisibleTab(null, {
      format: "png",
    });

    // 2. Crop using OffscreenCanvas (scaled by devicePixelRatio)
    const croppedBlob = await cropImage(dataUrl, rect, dpr);

    // 3. Send to backend
    const apiUrl = await getApiUrl();
    const formData = new FormData();
    formData.append("file", croppedBlob, "selection.png");

    const response = await fetch(`${apiUrl}/detect`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`API error ${response.status}: ${text}`);
    }

    const results = await response.json();

    // 4. Store for popup
    await chrome.storage.local.set({
      fontResults: results,
      fontError: null,
      fontLoading: false,
    });
  } catch (err) {
    console.error("Font detection failed:", err);
    await chrome.storage.local.set({
      fontResults: null,
      fontError: err.message || "Unknown error",
      fontLoading: false,
    });
  }
}

// -------------------------------------------------------------------
// Image cropping via OffscreenCanvas
// -------------------------------------------------------------------
async function cropImage(dataUrl, rect, dpr) {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);

  const sx = Math.round(rect.x * dpr);
  const sy = Math.round(rect.y * dpr);
  const sw = Math.round(rect.w * dpr);
  const sh = Math.round(rect.h * dpr);

  const canvas = new OffscreenCanvas(sw, sh);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, sw, sh);

  return canvas.convertToBlob({ type: "image/png" });
}

// -------------------------------------------------------------------
// Settings helpers
// -------------------------------------------------------------------
async function getApiUrl() {
  const data = await chrome.storage.local.get("apiUrl");
  return data.apiUrl || DEFAULT_API_URL;
}
