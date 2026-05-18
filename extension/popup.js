const API_BASES = ["http://127.0.0.1:8765", "http://localhost:8765"];
const CONTENT_SCRIPT_VERSION = "2026-05-15-targeted-delete-v2";
let apiBase = API_BASES[0];

let currentCase = null;
let busy = false;
let addedCodes = [];
let removedEntryIds = new Set();

const caseTitleEl = document.getElementById("case-title");
const caseDateEl = document.getElementById("case-date");
const caseSubEl = document.getElementById("case-sub");
const caseYearEl = document.getElementById("case-year");
const caseRoleEl = document.getElementById("case-role");
const casePatientEl = document.getElementById("case-patient");
const caseCodeCountEl = document.getElementById("case-code-count");
const caseSourceEl = document.getElementById("case-source");
const codeListEl = document.getElementById("code-list");
const popupMainEl = document.getElementById("popup-main");
const startViewEl = document.getElementById("start-view");
const startStatusEl = document.getElementById("start-status");
const guiViewEl = document.getElementById("gui-view");
const activeActionsEl = document.getElementById("active-actions");
const statusEl = document.getElementById("status");
const editModalEl = document.getElementById("edit-modal");
const currentCodesEl = document.getElementById("current-codes");
const addedCodesEl = document.getElementById("added-codes");
const searchResultsEl = document.getElementById("search-results");
const codeSearchEl = document.getElementById("code-search");
const uploadModeEl = document.getElementById("upload-mode");
const delayMinEl = document.getElementById("delay-min");
const delayMaxEl = document.getElementById("delay-max");

function setStatus(text) {
  statusEl.textContent = text || "";
}

function setStartStatus(text) {
  startStatusEl.textContent = text || "";
}

function showView(view) {
  const showStart = view === "start";
  startViewEl.hidden = !showStart;
  guiViewEl.hidden = showStart;
  popupMainEl.classList.toggle("start-only", showStart);
}

function codeLabel(code) {
  return [code.area, code.type, code.acgme_description].filter(Boolean).join(" / ");
}

function setBusy(nextBusy) {
  busy = nextBusy;
  document.querySelectorAll("button, input, select").forEach(el => {
    if (el.id !== "code-search") el.disabled = busy;
  });
}

function renderCase(queueCase) {
  currentCase = queueCase;
  const active = Boolean(queueCase);
  activeActionsEl.hidden = !active;

  if (!queueCase) {
    caseTitleEl.textContent = "Queue not started";
    caseDateEl.textContent = "";
    caseSubEl.textContent = "Open the ACGME Add Cases page, then start the queue.";
    caseYearEl.textContent = "-";
    caseRoleEl.textContent = "-";
    casePatientEl.textContent = "-";
    caseCodeCountEl.textContent = "0";
    caseSourceEl.textContent = "";
    codeListEl.textContent = "";
    return;
  }

  caseTitleEl.textContent = `#${queueCase.source_case_id} ${queueCase.case_id}`;
  caseDateEl.textContent = queueCase.case_date;
  caseSubEl.textContent = `${queueCase.codes.length} selected code${queueCase.codes.length === 1 ? "" : "s"}`;
  caseYearEl.textContent = queueCase.case_year;
  caseRoleEl.textContent = queueCase.role;
  casePatientEl.textContent = queueCase.patient_type;
  caseCodeCountEl.textContent = String(queueCase.codes.length);
  caseSourceEl.textContent = `${queueCase.exam_code || ""}${queueCase.procedure_text ? " | " + queueCase.procedure_text : ""}`;
  codeListEl.replaceChildren(
    ...queueCase.codes.map(code => {
      const item = document.createElement("div");
      item.className = "code-pill";
      item.title = codeLabel(code);
      item.textContent = codeLabel(code);
      return item;
    })
  );
}

function apiUnavailableError(cause) {
  const message = cause?.message ? ` (${cause.message})` : "";
  return new Error(`Local API unavailable. Start it with: uvicorn app.api:api --host 127.0.0.1 --port 8765${message}`);
}

async function fetchApi(base, path, options) {
  return fetch(`${base}${path}`, {
    method: options.method || "GET",
    headers: {"Content-Type": "application/json"},
    body: options.body ? JSON.stringify(options.body) : undefined
  });
}

async function api(path, options = {}) {
  const bases = [apiBase, ...API_BASES.filter(base => base !== apiBase)];
  let lastNetworkError = null;

  for (const base of bases) {
    let res;
    try {
      res = await fetchApi(base, path, options);
    } catch (err) {
      lastNetworkError = err;
      continue;
    }

    apiBase = base;
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return res.json();
  }

  throw apiUnavailableError(lastNetworkError);
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
  return tab;
}

function isInjectableTab(tab) {
  return tab?.id && /^https?:\/\//.test(tab.url || "");
}

async function ensureContentScript(tab) {
  if (!isInjectableTab(tab)) {
    throw new Error("Open a normal http/https page, such as the ACGME Add Cases page, before starting.");
  }
  await chrome.scripting.executeScript({
    target: {tabId: tab.id},
    files: ["content.js"]
  });
  const response = await callContent(tab.id, {action: "ping"});
  if (!response?.ok || response.version !== CONTENT_SCRIPT_VERSION) {
    throw new Error("Content script did not initialize. Reload the ACGME tab and try again.");
  }
}

async function callContent(tabId, message) {
  const [result] = await chrome.scripting.executeScript({
    target: {tabId},
    args: [message],
    func: async payload => {
      const api = window.__ACGME_IR_AUTOFILL__;
      if (!api?.handleMessage) {
        return {ok: false, error: "Content script API unavailable."};
      }
      return api.handleMessage(payload);
    }
  });
  return result?.result;
}

async function sendToContent(action, queueCase) {
  const tab = await activeTab();
  await ensureContentScript(tab);
  const response = await callContent(tab.id, {action, entry: queueCase, case: queueCase});
  if (!response || !response.ok) {
    throw new Error(response?.error || "No response from content script");
  }
  return response;
}

async function fillAndMark(queueCase, actionStatus = "Filling case...") {
  setStatus(actionStatus);
  await sendToContent("fillGroup", queueCase);
  const data = await api(`/queue/group/${queueCase.source_case_id}/autofilled`, {method: "POST"});
  renderCase(data.case || queueCase);
  setStatus("Prepared for review.");
}

async function prepareNext(status = "Preparing next case...") {
  setStatus(status);
  const data = await api("/queue/group/claim_next", {method: "POST"});
  showView("gui");
  if (!data.case) {
    renderCase(null);
    setStatus("No approved cases available.");
    return;
  }
  renderCase(data.case);
  try {
    await fillAndMark(data.case);
  } catch (err) {
    await api(`/queue/group/${data.case.source_case_id}/failed`, {
      method: "POST",
      body: {failure_reason: err.message}
    }).catch(() => {});
    renderCase(null);
    throw err;
  }
}

function jitterDelayMs() {
  const min = Math.max(0, Number(delayMinEl.value || 500));
  const max = Math.max(min, Number(delayMaxEl.value || 1500));
  return Math.round(min + Math.random() * (max - min));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForPortalReady() {
  const tab = await activeTab();
  await ensureContentScript(tab);
  for (let attempt = 0; attempt < 30; attempt += 1) {
    const response = await callContent(tab.id, {action: "ping"});
    if (response?.ok) return;
    await sleep(100);
  }
}

async function autoUploadLoop() {
  showView("gui");
  let submittedCount = 0;
  while (true) {
    setStatus(submittedCount ? `Submitted ${submittedCount}; claiming next...` : "Claiming next accepted case...");
    const data = await api("/queue/group/claim_next", {method: "POST"});
    if (!data.case) {
      renderCase(null);
      setStatus(submittedCount ? `Auto upload complete. Submitted ${submittedCount}.` : "No accepted cases available.");
      return;
    }
    renderCase(data.case);
    try {
      await fillAndMark(data.case, "Auto filling case...");
      setStatus("Auto submitting in ACGME...");
      await sendToContent("submitPage", data.case);
      await api(`/queue/group/${data.case.source_case_id}/submitted`, {method: "POST"});
      submittedCount += 1;
      const delay = jitterDelayMs();
      setStatus(`Submitted #${data.case.source_case_id}; waiting ${delay} ms...`);
      await sleep(delay);
      await waitForPortalReady();
    } catch (err) {
      await api(`/queue/group/${data.case.source_case_id}/failed`, {
        method: "POST",
        body: {failure_reason: err.message}
      }).catch(() => {});
      renderCase(null);
      throw err;
    }
  }
}

function showStartGate() {
  renderCase(null);
  setStartStatus("");
  setStatus("");
  showView("start");
}

function openEditor() {
  if (!currentCase) return;
  addedCodes = [];
  removedEntryIds = new Set();
  codeSearchEl.value = "";
  searchResultsEl.textContent = "";
  renderEditor();
  editModalEl.hidden = false;
  codeSearchEl.focus();
}

function closeEditor() {
  editModalEl.hidden = true;
}

function renderEditor() {
  currentCodesEl.replaceChildren(
    ...currentCase.codes.map(code => {
      const label = document.createElement("label");
      label.className = "check-row";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = !removedEntryIds.has(Number(code.local_entry_id));
      checkbox.dataset.entryId = String(code.local_entry_id);
      const text = document.createElement("span");
      text.textContent = codeLabel(code);
      label.append(checkbox, text);
      return label;
    })
  );
  renderAddedCodes();
}

function renderAddedCodes() {
  if (!addedCodes.length) {
    addedCodesEl.textContent = "No added codes.";
    return;
  }
  addedCodesEl.replaceChildren(
    ...addedCodes.map((code, index) => {
      const label = document.createElement("label");
      label.className = "check-row";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = true;
      checkbox.dataset.addedIndex = String(index);
      const text = document.createElement("span");
      text.textContent = codeLabel(code);
      label.append(checkbox, text);
      return label;
    })
  );
}

function addedCodeIndex(option) {
  const label = codeLabel(option);
  return addedCodes.findIndex(code => codeLabel(code) === label);
}

function currentCodeForOption(option) {
  const label = codeLabel(option);
  return currentCase.codes.find(code => codeLabel(code) === label);
}

function currentCodeCheckboxForOption(option) {
  const code = currentCodeForOption(option);
  if (!code?.local_entry_id) return null;
  return currentCodesEl.querySelector(`input[data-entry-id="${CSS.escape(String(code.local_entry_id))}"]`);
}

function addedCodeCheckboxForOption(option) {
  const index = addedCodeIndex(option);
  if (index === -1) return null;
  return addedCodesEl.querySelector(`input[data-added-index="${CSS.escape(String(index))}"]`);
}

function isEditorCodeSelected(option) {
  const checkbox = currentCodeCheckboxForOption(option);
  if (checkbox) return checkbox.checked;
  const addedCheckbox = addedCodeCheckboxForOption(option);
  if (addedCheckbox) return addedCheckbox.checked;
  return false;
}

function addEditorCode(option) {
  const checkbox = currentCodeCheckboxForOption(option);
  if (checkbox) {
    removedEntryIds.delete(Number(checkbox.dataset.entryId));
    checkbox.checked = true;
    setStatus("Code selected.");
    return;
  }
  const addedCheckbox = addedCodeCheckboxForOption(option);
  if (addedCheckbox) {
    addedCheckbox.checked = true;
    setStatus("Code selected.");
    return;
  }
  if (addedCodeIndex(option) === -1) {
    addedCodes.push(option);
    renderAddedCodes();
  }
  setStatus("Code added.");
}

function removeEditorCode(option) {
  const checkbox = currentCodeCheckboxForOption(option);
  if (checkbox) {
    removedEntryIds.add(Number(checkbox.dataset.entryId));
    checkbox.checked = false;
    setStatus("Code removed.");
    return;
  }
  const index = addedCodeIndex(option);
  if (index !== -1) {
    addedCodes.splice(index, 1);
    renderAddedCodes();
    setStatus("Code removed.");
  }
}

function updateSearchResultButton(button, option) {
  const selected = isEditorCodeSelected(option);
  button.textContent = selected ? "Remove" : "Add";
  button.className = selected ? "tiny remove" : "tiny";
}

let latestSearchOptions = [];

function renderSearchResults(options) {
  latestSearchOptions = options;
  searchResultsEl.replaceChildren(
    ...options.map(option => {
      const row = document.createElement("div");
      row.className = "result-row";
      const text = document.createElement("span");
      text.textContent = codeLabel(option);
      const button = document.createElement("button");
      button.type = "button";
      updateSearchResultButton(button, option);
      button.addEventListener("click", () => {
        if (isEditorCodeSelected(option)) {
          removeEditorCode(option);
        } else {
          addEditorCode(option);
        }
        renderSearchResults(latestSearchOptions);
      });
      row.append(text, button);
      return row;
    })
  );
}

function refreshSearchResults() {
  if (latestSearchOptions.length) renderSearchResults(latestSearchOptions);
}

let searchTimer = null;
codeSearchEl.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    const q = codeSearchEl.value.trim();
    if (q.length < 2) {
      latestSearchOptions = [];
      searchResultsEl.textContent = "";
      return;
    }
    try {
      const data = await api(`/acgme/options?q=${encodeURIComponent(q)}&limit=20`);
      renderSearchResults(data.options);
    } catch (err) {
      searchResultsEl.textContent = err.message;
    }
  }, 180);
});

currentCodesEl.addEventListener("change", event => {
  const checkbox = event.target;
  if (checkbox?.matches?.("input[type='checkbox'][data-entry-id]")) {
    const entryId = Number(checkbox.dataset.entryId);
    const code = currentCase.codes.find(item => Number(item.local_entry_id) === entryId);
    if (!code) return;
    (checkbox.checked ? addEditorCode : removeEditorCode)(code);
  }
  refreshSearchResults();
});
addedCodesEl.addEventListener("change", event => {
  const checkbox = event.target;
  if (checkbox?.matches?.("input[type='checkbox'][data-added-index]")) {
    const code = addedCodes[Number(checkbox.dataset.addedIndex)];
    if (!code) return;
    (checkbox.checked ? addEditorCode : removeEditorCode)(code);
  }
  refreshSearchResults();
});

function selectedEditorCodes() {
  const selected = [];
  currentCodesEl.querySelectorAll("input[type='checkbox']").forEach(checkbox => {
    if (!checkbox.checked) return;
    const entryId = Number(checkbox.dataset.entryId);
    const found = currentCase.codes.find(code => Number(code.local_entry_id) === entryId);
    if (found) selected.push(found);
  });
  addedCodesEl.querySelectorAll("input[type='checkbox']").forEach(checkbox => {
    if (!checkbox.checked) return;
    const found = addedCodes[Number(checkbox.dataset.addedIndex)];
    if (found) selected.push(found);
  });
  return selected;
}

function removedEditorCodes() {
  currentCodesEl.querySelectorAll("input[type='checkbox'][data-entry-id]").forEach(checkbox => {
    const entryId = Number(checkbox.dataset.entryId);
    if (checkbox.checked) {
      removedEntryIds.delete(entryId);
    } else {
      removedEntryIds.add(entryId);
    }
  });
  return currentCase.codes.filter(code => removedEntryIds.has(Number(code.local_entry_id)));
}

document.getElementById("start-queue").addEventListener("click", async () => {
  try {
    setBusy(true);
    setStartStatus("");
    if (uploadModeEl.value === "auto") {
      await autoUploadLoop();
    } else {
      await prepareNext("Starting queue...");
    }
  } catch (err) {
    if (guiViewEl.hidden) {
      setStartStatus(err.message);
    } else {
      setStatus(err.message);
    }
  } finally {
    setBusy(false);
  }
});

document.getElementById("submit").addEventListener("click", async () => {
  try {
    if (!currentCase) return;
    setBusy(true);
    const submittedSourceId = currentCase.source_case_id;
    setStatus("Submitting in ACGME...");
    await sendToContent("submitPage", currentCase);
    await api(`/queue/group/${submittedSourceId}/submitted`, {method: "POST"});
    await prepareNext(`Submitted #${submittedSourceId}; preparing next...`);
  } catch (err) {
    setStatus(err.message);
  } finally {
    setBusy(false);
  }
});

document.getElementById("skip").addEventListener("click", async () => {
  try {
    if (!currentCase) return;
    setBusy(true);
    const skippedSourceId = currentCase.source_case_id;
    await api(`/queue/group/${skippedSourceId}/skip_upload`, {method: "POST"});
    await prepareNext(`Skipped #${skippedSourceId}; preparing next...`);
  } catch (err) {
    setStatus(err.message);
  } finally {
    setBusy(false);
  }
});

document.getElementById("edit").addEventListener("click", openEditor);
document.getElementById("edit-close").addEventListener("click", closeEditor);
document.getElementById("edit-cancel").addEventListener("click", closeEditor);

document.getElementById("edit-save").addEventListener("click", async () => {
  try {
    const codes = selectedEditorCodes();
    const removedCodes = removedEditorCodes();
    setBusy(true);
    if (!codes.length) {
      closeEditor();
      setStatus("Clearing selected codes...");
      await sendToContent("fillGroup", {...currentCase, codes: [], removedCodes});
      renderCase({...currentCase, codes: []});
      setStatus("Cleared selected codes. Add at least one code before submitting.");
      return;
    }
    const data = await api(`/queue/group/${currentCase.source_case_id}/edit`, {
      method: "POST",
      body: {codes}
    });
    closeEditor();
    const editedCase = {...data.case, removedCodes};
    await fillAndMark(editedCase, "Saving edit and applying...");
  } catch (err) {
    setStatus(err.message);
  } finally {
    setBusy(false);
  }
});

showStartGate();
