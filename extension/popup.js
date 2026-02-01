/**
 * Popup script.
 *
 * - "Select Text Area" button tells content.js to start selection, then
 *   closes the popup so the user can interact with the page.
 * - Reads results / loading / error state from chrome.storage.local and
 *   renders accordingly.
 * - Settings section lets users configure the API URL.
 * - Loads Google Fonts CSS on the fly so each result is previewed in its
 *   own typeface.
 */

const selectBtn = document.getElementById("select-btn");
const loadingEl = document.getElementById("loading");
const errorEl = document.getElementById("error");
const resultsEl = document.getElementById("results");
const resultsList = document.getElementById("results-list");
const metaEl = document.getElementById("meta");
const apiUrlInput = document.getElementById("api-url");
const saveBtn = document.getElementById("save-settings");
const saveStatus = document.getElementById("save-status");

// -------------------------------------------------------------------
// Selection activation
// -------------------------------------------------------------------
selectBtn.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) {
    chrome.tabs.sendMessage(tab.id, { action: "startSelection" });
    window.close(); // close popup so user can draw
  }
});

// -------------------------------------------------------------------
// Render state from storage
// -------------------------------------------------------------------
function render({ fontResults, fontError, fontLoading }) {
  // Reset
  loadingEl.classList.add("hidden");
  errorEl.classList.add("hidden");
  resultsEl.classList.add("hidden");

  if (fontLoading) {
    loadingEl.classList.remove("hidden");
    return;
  }

  if (fontError) {
    errorEl.textContent = fontError;
    errorEl.classList.remove("hidden");
    return;
  }

  if (fontResults && fontResults.matches && fontResults.matches.length) {
    renderResults(fontResults);
    resultsEl.classList.remove("hidden");
  }
}

function renderResults(data) {
  resultsList.innerHTML = "";

  // Collect font names to load via Google Fonts
  const fontNames = data.matches.map((m) => m.font_name);
  loadGoogleFonts(fontNames);

  for (const match of data.matches) {
    const li = document.createElement("li");
    li.className = "result-item";

    li.innerHTML = `
      <span class="result-rank">${match.rank}.</span>
      <span class="result-name" style="font-family: '${match.font_name}', sans-serif">
        ${match.font_name}
      </span>
      <span class="result-score">${(match.score * 100).toFixed(1)}%</span>
      <a class="result-link" href="${match.google_fonts_url}" target="_blank">View</a>
    `;
    resultsList.appendChild(li);
  }

  const parts = [];
  if (data.processing_time_ms != null) {
    parts.push(`${data.processing_time_ms.toFixed(0)} ms`);
  }
  if (data.regions_detected != null) {
    parts.push(
      `${data.regions_detected} region${data.regions_detected !== 1 ? "s" : ""}`
    );
  }
  metaEl.textContent = parts.join(" · ");
}

/** Inject a <link> for Google Fonts so results preview in the real typeface. */
function loadGoogleFonts(names) {
  const families = names.map((n) => `family=${n.replace(/ /g, "+")}`).join("&");
  const href = `https://fonts.googleapis.com/css2?${families}&display=swap`;

  // Avoid duplicate link tags
  if (document.querySelector(`link[href="${href}"]`)) return;

  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  document.head.appendChild(link);
}

// -------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------
async function loadSettings() {
  const data = await chrome.storage.local.get("apiUrl");
  if (data.apiUrl) {
    apiUrlInput.value = data.apiUrl;
  }
}

saveBtn.addEventListener("click", async () => {
  const url = apiUrlInput.value.trim().replace(/\/+$/, "");
  await chrome.storage.local.set({ apiUrl: url || null });
  saveStatus.classList.remove("hidden");
  setTimeout(() => saveStatus.classList.add("hidden"), 1500);
});

// -------------------------------------------------------------------
// Init
// -------------------------------------------------------------------
(async () => {
  loadSettings();

  // Render current state
  const data = await chrome.storage.local.get([
    "fontResults",
    "fontError",
    "fontLoading",
  ]);
  render(data);

  // Live-update while popup is open
  chrome.storage.onChanged.addListener((changes) => {
    const updated = {};
    for (const key of ["fontResults", "fontError", "fontLoading"]) {
      if (changes[key]) updated[key] = changes[key].newValue;
    }
    if (Object.keys(updated).length) {
      chrome.storage.local
        .get(["fontResults", "fontError", "fontLoading"])
        .then(render);
    }
  });
})();
