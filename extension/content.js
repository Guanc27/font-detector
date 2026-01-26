/**
 * Content script – injected into every page.
 *
 * When activated, shows a full-page transparent overlay so the user can
 * drag-select a rectangle.  On mouseup the rectangle coordinates (plus
 * devicePixelRatio) are sent to the background service worker which
 * handles screenshot + crop + API call.
 *
 * Escape cancels selection.
 */

(() => {
  let overlay = null;
  let selectionBox = null;
  let startX = 0;
  let startY = 0;
  let isSelecting = false;

  /** Remove the overlay and all listeners. */
  function cleanup() {
    if (overlay) {
      overlay.remove();
      overlay = null;
    }
    selectionBox = null;
    isSelecting = false;
    document.removeEventListener("keydown", onKeyDown);
  }

  function onKeyDown(e) {
    if (e.key === "Escape") {
      cleanup();
    }
  }

  function onMouseDown(e) {
    isSelecting = true;
    startX = e.clientX;
    startY = e.clientY;

    selectionBox = document.createElement("div");
    Object.assign(selectionBox.style, {
      position: "fixed",
      border: "2px dashed #4285f4",
      backgroundColor: "rgba(66, 133, 244, 0.15)",
      zIndex: "2147483647",
      pointerEvents: "none",
      left: `${startX}px`,
      top: `${startY}px`,
      width: "0px",
      height: "0px",
    });
    overlay.appendChild(selectionBox);
  }

  function onMouseMove(e) {
    if (!isSelecting || !selectionBox) return;

    const x = Math.min(e.clientX, startX);
    const y = Math.min(e.clientY, startY);
    const w = Math.abs(e.clientX - startX);
    const h = Math.abs(e.clientY - startY);

    Object.assign(selectionBox.style, {
      left: `${x}px`,
      top: `${y}px`,
      width: `${w}px`,
      height: `${h}px`,
    });
  }

  function onMouseUp(e) {
    if (!isSelecting) return;
    isSelecting = false;

    const x = Math.min(e.clientX, startX);
    const y = Math.min(e.clientY, startY);
    const w = Math.abs(e.clientX - startX);
    const h = Math.abs(e.clientY - startY);

    cleanup();

    // Ignore tiny accidental clicks
    if (w < 10 || h < 10) return;

    chrome.runtime.sendMessage({
      action: "captureRegion",
      rect: { x, y, w, h },
      devicePixelRatio: window.devicePixelRatio || 1,
    });
  }

  /** Activate selection mode – called from background.js or popup. */
  function activate() {
    if (overlay) return; // already active

    overlay = document.createElement("div");
    Object.assign(overlay.style, {
      position: "fixed",
      top: "0",
      left: "0",
      width: "100vw",
      height: "100vh",
      cursor: "crosshair",
      zIndex: "2147483646",
      backgroundColor: "rgba(0, 0, 0, 0.05)",
    });

    overlay.addEventListener("mousedown", onMouseDown);
    overlay.addEventListener("mousemove", onMouseMove);
    overlay.addEventListener("mouseup", onMouseUp);
    document.addEventListener("keydown", onKeyDown);

    document.body.appendChild(overlay);
  }

  // Listen for activation messages from popup / background
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === "startSelection") {
      activate();
    }
  });
})();
