const API = "http://localhost:8765";
let currentEntry = null;

const entryEl = document.getElementById("entry");
const entryTitleEl = document.getElementById("entry-title");
const entryDateEl = document.getElementById("entry-date");
const entrySubEl = document.getElementById("entry-sub");
const entryYearEl = document.getElementById("entry-year");
const entryRoleEl = document.getElementById("entry-role");
const entryPatientEl = document.getElementById("entry-patient");
const entryConfidenceEl = document.getElementById("entry-confidence");
const entrySourceEl = document.getElementById("entry-source");
const entryDetailsEl = document.getElementById("entry-details");
const statusEl = document.getElementById("status");
const detailsButton = document.getElementById("details");

function setStatus(text) {
  statusEl.textContent = text || "";
}

function renderEntry(entry) {
  currentEntry = entry;
  if (!entry) {
    entryEl.classList.remove("expanded");
    entryTitleEl.textContent = "No case";
    entryDateEl.textContent = "";
    entrySubEl.textContent = "No case claimed.";
    entryYearEl.textContent = "";
    entryRoleEl.textContent = "";
    entryPatientEl.textContent = "";
    entryConfidenceEl.textContent = "";
    entrySourceEl.textContent = "";
    entryDetailsEl.textContent = "";
    detailsButton.hidden = true;
    return;
  }
  entryEl.classList.remove("expanded");
  entryTitleEl.textContent = `#${entry.local_entry_id} ${entry.case_id}`;
  entryDateEl.textContent = entry.case_date;
  entrySubEl.textContent = `${entry.area} -> ${entry.type}`;
  entryYearEl.textContent = entry.case_year;
  entryRoleEl.textContent = entry.role;
  entryPatientEl.textContent = entry.patient_type;
  entryConfidenceEl.textContent = entry.mapping_confidence || "";
  entrySourceEl.textContent = `${entry.exam_code || ""}${entry.procedure_text ? " | " + entry.procedure_text : ""}`;
  entryDetailsEl.textContent = [
    `Map: ${entry.mapping_rule_name || ""}`,
    `Class: ${entry.case_class}`,
    entry.acgme_description ? `ACGME row: ${entry.acgme_description}` : "",
    entry.acgme_def_category ? `Def Cat: ${entry.acgme_def_category}` : "",
    entry.study_description ? `Study: ${entry.study_description}` : "",
    entry.procedure_text ? `Procedure: ${entry.procedure_text}` : "",
    entry.component_label ? `Component: ${entry.component_label}` : "",
    entry.comments ? `Comments: ${entry.comments}` : ""
  ].filter(Boolean).join("\n");
  detailsButton.hidden = false;
  detailsButton.textContent = "Details";
}

async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    method: options.method || "GET",
    headers: {"Content-Type": "application/json"},
    body: options.body ? JSON.stringify(options.body) : undefined
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
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
    throw new Error("Open a normal http/https page, such as the ACGME Add Cases page, before fill. Extensions cannot run on this page.");
  }
  try {
    const response = await chrome.tabs.sendMessage(tab.id, {action: "ping"});
    if (response?.ok) return;
  } catch (_err) {
    // Expected when the content script is not already present.
  }
  await chrome.scripting.executeScript({
    target: {tabId: tab.id},
    files: ["content.js"]
  });
}

async function sendToContent(action, entry) {
  const tab = await activeTab();
  await ensureContentScript(tab);
  const response = await chrome.tabs.sendMessage(tab.id, {action, entry});
  if (!response || !response.ok) {
    throw new Error(response?.error || "No response from content script");
  }
  return response;
}

async function claimNext() {
  setStatus("Claiming next case...");
  const data = await api("/queue/claim_next", {method: "POST"});
  renderEntry(data.entry);
  setStatus(data.entry ? "Case claimed." : "No approved cases available.");
}

async function loadCurrent() {
  try {
    const data = await api("/queue/current");
    renderEntry(data.entry);
  } catch (err) {
    setStatus(`Local API unavailable: ${err.message}`);
  }
}

document.getElementById("claim").addEventListener("click", () => claimNext().catch(err => setStatus(err.message)));

detailsButton.addEventListener("click", () => {
  entryEl.classList.toggle("expanded");
  detailsButton.textContent = entryEl.classList.contains("expanded") ? "Hide details" : "Details";
});

document.getElementById("fill").addEventListener("click", async () => {
  try {
    if (!currentEntry) await claimNext();
    if (!currentEntry) return;
    await sendToContent("fill", currentEntry);
    await api(`/entries/${currentEntry.local_entry_id}/autofilled`, {method: "POST"});
    setStatus("Filled.");
  } catch (err) {
    if (currentEntry) {
      await api(`/entries/${currentEntry.local_entry_id}/failed`, {method: "POST", body: {failure_reason: err.message}}).catch(() => {});
    }
    setStatus(err.message);
  }
});

document.getElementById("submitted").addEventListener("click", async () => {
  try {
    if (!currentEntry) return setStatus("No current case.");
    const submittedEntryId = currentEntry.local_entry_id;
    setStatus("Submitting in ACGME...");
    await sendToContent("submitPage", currentEntry);
    await api(`/entries/${submittedEntryId}/submitted`, {method: "POST"});
    setStatus(`Submitted #${submittedEntryId}; claiming next...`);
    const next = await api("/queue/claim_next", {method: "POST"});
    renderEntry(next.entry);
    setStatus(next.entry ? `Submitted #${submittedEntryId}. Next case ready.` : `Submitted #${submittedEntryId}. No approved cases available.`);
  } catch (err) {
    setStatus(err.message);
  }
});

document.getElementById("back").addEventListener("click", async () => {
  const data = await api("/session/back", {method: "POST"});
  renderEntry(data.entry);
  setStatus(data.entry ? "Returned to previous case." : "No previous case.");
});

loadCurrent();
