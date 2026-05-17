(() => {
const CONTENT_SCRIPT_VERSION = "2026-05-15-targeted-delete-v2";

const FIELD_LABELS = {
  case_id: ["Case ID", "Case Id", "CaseID", "Case Number", "Accession", "Accession Number"],
  case_date: ["Case Date", "CaseDate", "Date", "Date of Case", "Procedure Date", "Case/Procedure Date"],
  keyword: ["Keyword"],
  comments: ["Comments"]
};

const ADDED_ENTRY_KEYS = new Set();

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function nextFrame() {
  return new Promise(resolve => requestAnimationFrame(() => resolve()));
}

async function waitUntil(predicate, timeoutMs = 1500, intervalMs = 40) {
  const start = performance.now();
  let lastError;
  while (performance.now() - start < timeoutMs) {
    try {
      const value = predicate();
      if (value) return value;
    } catch (err) {
      lastError = err;
    }
    await sleep(intervalMs);
  }
  if (lastError) throw lastError;
  return null;
}

function normalize(text) {
  return String(text || "").replace(/\s+/g, " ").trim().toLowerCase();
}

function loose(text) {
  return normalize(text).replace(/[^a-z0-9]/g, "");
}

function labelsFor(el) {
  const labels = [];
  if (el.id) {
    const explicit = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
    if (explicit) labels.push(explicit.textContent);
    labels.push(el.id);
  }
  const parentLabel = el.closest("label");
  if (parentLabel) labels.push(parentLabel.textContent);
  const container = el.closest(".form-group, .field, .row, div");
  if (container) {
    const label = container.querySelector("label");
    if (label) labels.push(label.textContent);
  }
  if (el.getAttribute("aria-label")) labels.push(el.getAttribute("aria-label"));
  if (el.placeholder) labels.push(el.placeholder);
  if (el.name) labels.push(el.name);
  if (el.getAttribute("data-field")) labels.push(el.getAttribute("data-field"));
  if (el.getAttribute("data-testid")) labels.push(el.getAttribute("data-testid"));
  return labels.map(normalize).filter(Boolean);
}

function directText(el) {
  return Array.from(el.childNodes)
    .filter(node => node.nodeType === Node.TEXT_NODE)
    .map(node => node.textContent || "")
    .join(" ");
}

function labelMatches(text, wanted, wantedLoose) {
  const normal = normalize(text).replace(/\*/g, "");
  const compact = loose(normal);
  return wanted.some(w => normal.includes(w.replace(/\*/g, ""))) ||
    wantedLoose.some(w => compact.includes(w));
}

function controlsInside(el) {
  return Array.from(el.querySelectorAll("input, select, textarea, button.dropdown-toggle, [role='combobox'], .select2-selection, .k-dropdown, .k-picker"))
    .filter(control => !control.disabled && control.offsetParent !== null);
}

function allControls() {
  return Array.from(document.querySelectorAll("input, select, textarea, button.dropdown-toggle, [role='combobox'], .select2-selection, .k-dropdown, .k-picker"))
    .filter(control => !control.disabled && control.offsetParent !== null && isVisible(control));
}

function nearestControlForLabel(labelEl) {
  const labelRect = labelEl.getBoundingClientRect();
  const controls = allControls();
  const ranked = controls
    .map(control => {
      const rect = control.getBoundingClientRect();
      const vertical = rect.top - labelRect.bottom;
      const horizontal = Math.abs(rect.left - labelRect.left);
      const centerHorizontal = Math.abs((rect.left + rect.right) / 2 - (labelRect.left + labelRect.right) / 2);
      const plausible = vertical >= -12 && vertical <= 130 && horizontal <= 80;
      return {control, plausible, score: Math.max(0, vertical) * 4 + horizontal + centerHorizontal * 0.25};
    })
    .filter(item => item.plausible)
    .sort((a, b) => a.score - b.score);
  return ranked[0]?.control || null;
}

function findByNearbyVisibleLabel(fieldKey) {
  const wanted = FIELD_LABELS[fieldKey].map(normalize);
  const wantedLoose = FIELD_LABELS[fieldKey].map(loose);
  const candidates = Array.from(document.querySelectorAll("label, div, span, p, h1, h2, h3, h4, strong"));

  for (const labelEl of candidates) {
    const text = directText(labelEl) || labelEl.textContent || "";
    if (!labelMatches(text, wanted, wantedLoose)) continue;

    const nearest = nearestControlForLabel(labelEl);
    if (nearest) return nearest;

    const sameContainer = labelEl.closest(".form-group, .field, .row, [class*='form'], [class*='field'], [class*='row'], [class*='col']");
    if (sameContainer) {
      const found = controlsInside(sameContainer)[0];
      if (found) return found;
    }

    let sibling = labelEl.nextElementSibling;
    for (let i = 0; sibling && i < 4; i += 1, sibling = sibling.nextElementSibling) {
      if (sibling.matches?.("input, select, textarea, [role='combobox']")) return sibling;
      const found = controlsInside(sibling)[0];
      if (found) return found;
    }

    let parent = labelEl.parentElement;
    for (let depth = 0; parent && depth < 4; depth += 1, parent = parent.parentElement) {
      const found = controlsInside(parent).find(control => {
        const rect = control.getBoundingClientRect();
        const labelRect = labelEl.getBoundingClientRect();
        return rect.top >= labelRect.top - 8 && rect.top <= labelRect.bottom + 120;
      });
      if (found) return found;
    }
  }
  return null;
}

function findControl(fieldKey) {
  const wanted = FIELD_LABELS[fieldKey].map(normalize);
  const wantedLoose = FIELD_LABELS[fieldKey].map(loose);
  const controls = allControls();
  for (const el of controls) {
    const labels = labelsFor(el);
    if (labels.some(label => wanted.some(w => label.includes(w)))) return el;
    if (labels.some(label => wantedLoose.some(w => loose(label).includes(w)))) return el;
  }
  const nearby = findByNearbyVisibleLabel(fieldKey);
  if (nearby) return nearby;
  const page = location.hostname.includes("127.0.0.1") || location.hostname.includes("localhost")
    ? " This looks like a local app page; open the ACGME Add Cases page in Chrome before preview/fill."
    : "";
  throw new Error(`Field not found: ${fieldKey}.${page}`);
}

function highlight(el, ok = true) {
  el.style.outline = ok ? "3px solid #155eef" : "3px solid #d92d20";
  el.style.outlineOffset = "2px";
}

function showToast(message, ok = true) {
  let toast = document.getElementById("acgme-ir-autofill-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "acgme-ir-autofill-toast";
    toast.style.position = "fixed";
    toast.style.right = "18px";
    toast.style.bottom = "18px";
    toast.style.zIndex = "2147483647";
    toast.style.maxWidth = "420px";
    toast.style.padding = "12px 14px";
    toast.style.borderRadius = "8px";
    toast.style.font = "13px/1.4 system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    toast.style.boxShadow = "0 10px 30px rgba(0,0,0,.2)";
    document.body.appendChild(toast);
  }
  toast.style.background = ok ? "#155eef" : "#b42318";
  toast.style.color = "#fff";
  toast.textContent = message;
  clearTimeout(window.__ACGME_IR_TOAST_TIMER__);
  window.__ACGME_IR_TOAST_TIMER__ = setTimeout(() => toast.remove(), 2500);
}

function setNativeValue(el, value) {
  el.focus();
  el.value = value ?? "";
  el.dispatchEvent(new Event("input", {bubbles: true}));
  el.dispatchEvent(new Event("change", {bubbles: true}));
  if (window.jQuery) {
    window.jQuery(el).trigger("input").trigger("change");
  }
}

function isVisible(el) {
  if (!el) return false;
  const style = getComputedStyle(el);
  const rect = el.getBoundingClientRect();
  return style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
}

function displayText(el) {
  if (!el) return "";
  if ("value" in el && el.value) return String(el.value);
  return el.textContent || "";
}

function dropdownClickable(el) {
  if (el.matches?.("button, [role='combobox'], .select2-selection, .k-dropdown, .k-picker")) return el;
  return el.querySelector?.("button.dropdown-toggle, button, [role='combobox'], .select2-selection, .k-dropdown, .k-picker") || el;
}

function clickLikeUser(el) {
  el.scrollIntoView({block: "center", inline: "nearest"});
  el.focus?.();
  const opts = {bubbles: true, cancelable: true, view: window};
  el.dispatchEvent(new MouseEvent("mousedown", opts));
  el.dispatchEvent(new MouseEvent("mouseup", opts));
  el.click();
}

function optionCandidates() {
  const selectors = [
    "[role='option']",
    ".dropdown-menu li",
    ".dropdown-menu a",
    ".dropdown-menu button",
    ".dropdown-menu span",
    ".bootstrap-select .dropdown-menu li a",
    ".bootstrap-select .dropdown-menu .text",
    ".dropdown-item",
    ".open li",
    ".show li",
    ".open a",
    ".show a",
    ".select2-results__option",
    ".k-list-item",
    ".k-item",
    "[class*='option']",
    "mat-option",
    "li",
    "a",
    "button"
  ];
  return Array.from(document.querySelectorAll(selectors.join(","))).filter(isVisible);
}

async function selectCustomByText(el, visibleText) {
  if (normalize(displayText(el)).includes(normalize(visibleText))) return;
  const clickable = dropdownClickable(el);
  clickLikeUser(clickable);
  await waitUntil(() => optionCandidates().length > 0, 900, 25);
  let options = optionCandidates();
  let target = options.find(o => normalize(o.textContent) === normalize(visibleText));
  if (!target) {
    target = options.find(o => normalize(o.textContent).includes(normalize(visibleText)));
  }
  if (!target) {
    highlight(clickable, false);
    const sample = options.slice(0, 8).map(o => normalize(o.textContent)).filter(Boolean).join(" | ");
    throw new Error(`Option not found: ${visibleText}${sample ? `. Visible options: ${sample}` : ""}`);
  }
  clickLikeUser(target);
  await nextFrame();
  const after = normalize(displayText(el) + " " + displayText(clickable));
  if (!after.includes(normalize(visibleText))) {
    // Some widgets render selected text elsewhere; do not fail if the menu closed, but make this visible.
    showToast(`Selected ${visibleText}; verify the field changed.`, true);
  }
}

function setSelectByTextFast(el, visibleText) {
  const target = findSelectOption(el, visibleText);
  if (!target) {
    highlight(el, false);
    throw new Error(`Option not found: ${visibleText}. Available: ${selectOptionSample(el)}`);
  }
  if (el.value !== target.value) {
    el.value = target.value;
    el.dispatchEvent(new Event("input", {bubbles: true}));
    el.dispatchEvent(new Event("change", {bubbles: true}));
    if (window.jQuery) {
      window.jQuery(el).trigger("input").trigger("change");
    }
  }
  return target;
}

async function selectByText(el, visibleText) {
  if (el.tagName.toLowerCase() === "select") {
    setSelectByTextFast(el, visibleText);
    await nextFrame();
    return;
  }
  await selectCustomByText(el, visibleText);
}

function byId(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`ACGME field not found: #${id}`);
  return el;
}

function findSelectOption(selectEl, visibleText) {
  const wanted = normalize(visibleText);
  const wantedLoose = loose(visibleText);
  const options = Array.from(selectEl.options);
  return options.find(o => normalize(o.textContent) === wanted) ||
    options.find(o => loose(o.textContent) === wantedLoose) ||
    options.find(o => normalize(o.textContent).includes(wanted)) ||
    options.find(o => wanted.includes(normalize(o.textContent)) && normalize(o.textContent) !== "all");
}

function selectOptionSample(selectEl) {
  return Array.from(selectEl.options).slice(0, 12).map(o => o.textContent.trim()).join(" | ");
}

function topFields() {
  return {
    case_id: findControl("case_id"),
    case_date: findControl("case_date"),
    case_year: byId("ProcedureYear"),
    role: byId("ResidentRoles"),
    site: byId("Institutions"),
    patient_type: byId("PatientTypes"),
  };
}

function categoryFields() {
  return {
    case_class: byId("RRCClass"),
    area: byId("Areas"),
    type: byId("Types"),
    code_description: document.getElementById("CodeDescription"),
    search: byId("searchByAreaTypeButton"),
    table: byId("codes-by-areatype-table"),
  };
}

function clickTabByText(tabText) {
  const tab = Array.from(document.querySelectorAll("a, button, [role='tab']"))
    .find(el => isVisible(el) && normalize(el.textContent) === normalize(tabText));
  if (!tab) throw new Error(`Tab not found: ${tabText}`);
  clickLikeUser(tab);
}

async function fillMetadata(entry) {
  const fields = topFields();
  showToast(`Filling case ${entry.case_id}...`);
  setNativeValue(fields.case_id, entry.case_id);
  setNativeValue(fields.case_date, entry.case_date);
  setSelectByTextFast(fields.case_year, String(entry.case_year));
  setSelectByTextFast(fields.role, entry.role);
  setSelectByTextFast(fields.site, entry.site);
  setSelectByTextFast(fields.patient_type, entry.patient_type);
  await nextFrame();
}

function rowParts(row) {
  const cells = Array.from(row.querySelectorAll("td")).map(cell => normalize(cell.textContent));
  return {
    cells,
    code: cells[0] || "",
    description: cells[1] || "",
    area: cells[2] || "",
    type: cells[3] || "",
    text: normalize(row.textContent)
  };
}

function rowScore(row, entry) {
  const parts = rowParts(row);
  const text = parts.text;
  const description = normalize(entry.acgme_description);
  const defCategory = normalize(entry.acgme_def_category);
  let score = 0;
  if (parts.area === normalize(entry.area) || text.includes(normalize(entry.area))) score += 4;
  if (parts.type === normalize(entry.type) || text.includes(normalize(entry.type))) score += 6;
  if (description && parts.description === description) score += 20;
  else if (description && text.includes(description)) score += 12;
  if (defCategory && text.includes(defCategory)) score += 3;
  if (entry.keyword && text.includes(normalize(entry.keyword))) score += 2;
  return score;
}

function addButtonForBestCategoryRow(entry) {
  const row = bestCategoryRow(entry);
  if (!row) return null;
  const cells = Array.from(row.querySelectorAll("td"));
  const exactAdd = el => isVisible(el) && normalize(el.textContent) === "add";
  const isFavoriteControl = el => {
    const cell = el.closest("td");
    const cellIndex = cells.indexOf(cell);
    const text = normalize(el.textContent);
    const cls = String(el.className || "").toLowerCase();
    const title = normalize(el.getAttribute("title") || el.getAttribute("aria-label") || "");
    return cellIndex === cells.length - 2 ||
      text.includes("fav") ||
      text.includes("favorite") ||
      title.includes("fav") ||
      title.includes("favorite") ||
      cls.includes("fav") ||
      cls.includes("star");
  };

  const lastCell = cells[cells.length - 1];
  const lastCellAdd = lastCell
    ? Array.from(lastCell.querySelectorAll("button, a")).find(el => exactAdd(el) && !isFavoriteControl(el))
    : null;
  if (lastCellAdd) return lastCellAdd;

  const rowAdd = Array.from(row.querySelectorAll("button, a"))
    .find(el => exactAdd(el) && !isFavoriteControl(el));
  if (rowAdd) return rowAdd;

  const cellTexts = cells.map(cell => normalize(cell.textContent)).join(" | ");
  const buttonTexts = Array.from(row.querySelectorAll("button, a"))
    .filter(isVisible)
    .map(el => `[text="${normalize(el.textContent)}" class="${String(el.className || "")}"]`)
    .join(" ");
  throw new Error(`Matched row but no safe Add button found. Cells: ${cellTexts}. Buttons: ${buttonTexts}`);
}

function bestCategoryRow(entry) {
  const rows = Array.from(document.querySelectorAll("#codes-by-areatype-table tbody tr"))
    .filter(row => isVisible(row) && normalize(row.textContent));
  const description = normalize(entry.acgme_description);
  const defCategory = normalize(entry.acgme_def_category);
  const area = normalize(entry.area);
  const type = normalize(entry.type);
  if (description) {
    const exact = rows.filter(row => {
      const parts = rowParts(row);
      const descriptionMatches = parts.description === description || parts.text.includes(description);
      const areaMatches = parts.area === area || parts.text.includes(area);
      const typeMatches = parts.type === type || parts.text.includes(type);
      return descriptionMatches && areaMatches && typeMatches;
    });
    if (!exact.length) {
      const samples = rows.slice(0, 8).map(row => rowParts(row).cells.join(" | ")).join(" || ");
      throw new Error(
        `No ACGME row found for description "${entry.acgme_description}" ` +
        `with ${entry.area} / ${entry.type}${entry.acgme_def_category ? ` / ${entry.acgme_def_category}` : ""}.` +
        `${samples ? ` Visible rows: ${samples}` : ""}`
      );
    }
    return exact.sort((a, b) => rowScore(b, entry) - rowScore(a, entry))[0];
  }
  const ranked = rows
    .map(row => ({row, score: rowScore(row, entry)}))
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score);
  return ranked[0]?.row || null;
}

async function waitForMatchingResultRow(entry) {
  return waitUntil(() => bestCategoryRow(entry), 1800, 50);
}

function selectedCount() {
  const panels = Array.from(document.querySelectorAll(".selectedCodes"));
  const match = panels
    .map(panel => panel?.textContent?.match(/\bSelected\s+(\d+)/i))
    .find(Boolean);
  return match ? Number(match[1]) : null;
}

function selectedEntryKey(entry) {
  return [
    entry.local_entry_id || entry.id || "",
    entry.case_id || "",
    entry.area || "",
    entry.type || "",
    entry.acgme_description || "",
    entry.component_label || ""
  ].map(loose).join("|");
}

function groupPrimaryEntry(group) {
  const code = group?.codes?.[0] || group;
  return {...group, ...code};
}

function selectedRegions() {
  const headings = Array.from(document.querySelectorAll(".selectedCodes"));
  const regions = new Set();
  const addRegion = el => {
    if (!el || el === document.body || el.querySelector?.("#codes-by-areatype-table")) return;
    regions.add(el);
  };
  for (const heading of headings) {
    [
      heading.closest(".panel"),
      heading.closest(".card"),
      heading.closest("[class*='panel']"),
      heading.parentElement,
      heading.nextElementSibling,
      heading
    ].filter(Boolean).forEach(addRegion);

    let ancestor = heading.parentElement;
    for (let depth = 0; ancestor && depth < 4; depth += 1, ancestor = ancestor.parentElement) {
      addRegion(ancestor);
    }

    let sibling = heading.nextElementSibling;
    for (let i = 0; sibling && i < 4; i += 1, sibling = sibling.nextElementSibling) {
      addRegion(sibling);
    }
  }
  return Array.from(regions).filter(isVisible);
}

function selectedPanelText() {
  return normalize(selectedRegions()
    .map(el => el.textContent || "")
    .join(" "));
}

function selectedCodeCards(entry) {
  const wantedType = normalize(entry.type);
  const wantedDescription = normalize(entry.acgme_description);
  const wantedArea = normalize(entry.area);
  const regions = selectedRegions();
  const cards = [];
  for (const region of regions) {
    const candidates = Array.from(region.querySelectorAll("div, li, tr, td, section, article, label, dl, dt, dd"))
      .filter(el => isVisible(el) && !el.closest("#codes-by-areatype-table"));
    for (const el of candidates) {
      const text = normalize(el.textContent || "");
      if (!text) continue;
      if (
        (wantedDescription && text.includes(wantedDescription)) ||
        (wantedType && text.includes(wantedType)) ||
        (wantedArea && text.includes(wantedArea) && text.includes("type:"))
      ) {
        cards.push(el);
      }
    }
  }
  return cards;
}

function selectedCodeCardContainers(entry) {
  return selectedCodeCards(entry)
    .map(card => (
      card.closest("[class*='card']") ||
      card.closest("[class*='item']") ||
      card.closest("[class*='code']") ||
      card.closest("li") ||
      card.closest("tr") ||
      card
    ))
    .filter(Boolean);
}

function selectedPanelHasEntry(entry) {
  if (ADDED_ENTRY_KEYS.has(selectedEntryKey(entry))) return true;
  const wantedType = normalize(entry.type);
  const wantedArea = normalize(entry.area);
  const wantedDescription = normalize(entry.acgme_description);
  if (!wantedType) return false;
  const cardTexts = selectedCodeCards(entry).map(el => normalize(el.textContent || ""));
  if (wantedDescription) {
    if (cardTexts.some(text => text.includes(wantedDescription) && text.includes(wantedType))) return true;
    const panel = selectedPanelText();
    return panel.includes(wantedDescription) && panel.includes(wantedType);
  }
  if (cardTexts.some(text => text.includes(wantedType))) return true;
  if (cardTexts.some(text => text.includes(wantedArea) && text.includes(wantedType))) return true;
  const text = selectedPanelText();
  return text.includes(wantedType) && (!wantedArea || text.includes(wantedArea));
}

function removeButtonForSelectedCard(card) {
  const containers = [
    card,
    card.parentElement,
    card.closest("tr"),
    card.closest("li"),
    card.closest("[class*='card']"),
    card.closest(".row"),
    card.closest("[class*='row']"),
    card.closest("[class*='item']"),
    card.closest("[class*='code']")
  ].filter(Boolean);

  const controls = containers.flatMap(container =>
    Array.from(container.querySelectorAll(
      "button, a, i.removeCode, .removeCode, .fa-trash-can, input[type='button'], input[type='submit'], input[type='image'], input[type='checkbox'], [role='button']"
    ))
  )
    .filter(isVisible);
  return controls.find(el => {
    const text = normalize(el.textContent || el.value || "");
    const title = normalize(el.getAttribute("title") || el.getAttribute("aria-label") || "");
    const cls = String(el.className || "").toLowerCase();
    return text === "x" ||
      text === "×" ||
      text === "-" ||
      text === "remove" ||
      text.includes("remove") ||
      text === "delete" ||
      text.includes("delete") ||
      title.includes("remove") ||
      title.includes("delete") ||
      (el.type === "checkbox" && el.checked) ||
      cls.includes("remove") ||
      cls.includes("delete") ||
      cls.includes("close") ||
      cls.includes("trash");
  }) || null;
}

function isRemovalControl(el) {
  const text = normalize(el.textContent || el.value || "");
  const title = normalize(el.getAttribute("title") || el.getAttribute("aria-label") || "");
  const cls = String(el.className || "").toLowerCase();
  return text === "x" ||
    text === "×" ||
    text === "-" ||
    text === "remove" ||
    text.includes("remove") ||
    text === "delete" ||
    text.includes("delete") ||
    title.includes("remove") ||
    title.includes("delete") ||
    (el.type === "checkbox" && el.checked) ||
    cls.includes("removecode") ||
    cls.includes("remove") ||
    cls.includes("fa-trash-can") ||
    cls.includes("delete") ||
    cls.includes("close") ||
    cls.includes("trash");
}

function selectedRemovalControls() {
  const seen = new Set();
  const controls = [];
  for (const region of selectedRegions()) {
    const found = Array.from(region.querySelectorAll(
      "button, a, i.removeCode, .removeCode, .fa-trash-can, input[type='button'], input[type='submit'], input[type='image'], input[type='checkbox'], [role='button']"
    ))
      .filter(el => isVisible(el) && !el.closest("#codes-by-areatype-table") && isRemovalControl(el));
    for (const el of found) {
      if (seen.has(el)) continue;
      seen.add(el);
      controls.push(el);
    }
  }
  return controls;
}

function trashButtonForSelectedCard(entry) {
  const ranked = selectedCodeCardContainers(entry)
    .map(card => ({card, score: removalScore(card, entry)}))
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score);
  for (const item of ranked) {
    const button = removeButtonForSelectedCard(item.card);
    if (button) return button;
  }
  return null;
}

async function clearSelectedCodes() {
  clickTabByText("Area/Type/Code");
  await nextFrame();
  for (let attempt = 0; attempt < 25; attempt += 1) {
    const count = selectedCount();
    const controls = selectedRemovalControls();
    if (count === 0) {
      ADDED_ENTRY_KEYS.clear();
      return;
    }
    if (controls.length === 0) {
      ADDED_ENTRY_KEYS.clear();
      if (count && count > 0) {
        throw new Error(`Could not find remove controls for ${count} selected ACGME code${count === 1 ? "" : "s"}.`);
      }
      return;
    }
    clickLikeUser(controls[0]);
    ADDED_ENTRY_KEYS.clear();
    await sleep(180);
    await nextFrame();
  }
  const remaining = selectedCount();
  if (remaining && remaining > 0) {
    throw new Error(`Could not clear selected ACGME codes; ${remaining} still selected.`);
  }
}

function removalScore(card, entry) {
  const text = normalize(card.textContent || "");
  const wantedType = normalize(entry.type);
  const wantedDescription = normalize(entry.acgme_description);
  const wantedArea = normalize(entry.area);
  let score = 0;
  if (wantedDescription && text.includes(wantedDescription)) score += 12;
  if (wantedType && text.includes(wantedType)) score += 8;
  if (wantedArea && text.includes(wantedArea)) score += 3;
  return score;
}

async function removeSelectedCode(entry) {
  clickTabByText("Area/Type/Code");
  await nextFrame();
  if (!selectedPanelHasEntry(entry)) {
    return;
  }
  const button = trashButtonForSelectedCard(entry);
  if (!button) {
    throw new Error(`Could not safely remove deselected code: ${entry.area} / ${entry.type}`);
  }
  clickLikeUser(button);
  ADDED_ENTRY_KEYS.delete(selectedEntryKey(entry));
  const removed = await waitUntil(() => !selectedPanelHasEntry(entry), 1200, 50);
  if (!removed && selectedPanelHasEntry(entry)) {
    throw new Error(`Deselected code did not disappear after remove: ${entry.area} / ${entry.type}`);
  }
}

async function waitForSelectedCodeAdded(beforeCount, entry) {
  return waitUntil(() => {
    const after = selectedCount();
    return selectedPanelHasEntry(entry) ||
      (beforeCount !== null && after !== null && after > beforeCount);
  }, 1200, 50);
}

async function selectCategory(entry) {
  showToast("Selecting Area/Type/Code...");
  if (selectedPanelHasEntry(entry)) {
    ADDED_ENTRY_KEYS.add(selectedEntryKey(entry));
    showToast("Procedure already selected; skipping Add.");
    return;
  }
  clickTabByText("Area/Type/Code");
  await nextFrame();
  const fields = categoryFields();
  highlight(fields.case_class, true);
  highlight(fields.area, true);
  highlight(fields.type, true);
  setSelectByTextFast(fields.case_class, entry.case_class || "Interventional Procedures");
  await waitUntil(() => findSelectOption(fields.area, entry.area), 1200, 40);
  setSelectByTextFast(fields.area, entry.area);
  await waitUntil(() => findSelectOption(fields.type, entry.type), 1200, 40);
  setSelectByTextFast(fields.type, entry.type);
  if (fields.code_description) setNativeValue(fields.code_description, "");
  const before = selectedCount();
  if (selectedPanelHasEntry(entry)) {
    ADDED_ENTRY_KEYS.add(selectedEntryKey(entry));
    showToast("Procedure already selected; skipping Add.");
    return;
  }
  clickLikeUser(fields.search);
  await waitForMatchingResultRow(entry);
  const add = addButtonForBestCategoryRow(entry);
  if (!add) {
    const rows = Array.from(document.querySelectorAll("#codes-by-areatype-table tbody tr"))
      .slice(0, 5)
      .map(row => normalize(row.textContent))
      .filter(Boolean)
      .join(" || ");
    throw new Error(`No Add button found for ${entry.area} / ${entry.type}${rows ? `. Rows: ${rows}` : ""}`);
  }
  if (selectedPanelHasEntry(entry)) {
    ADDED_ENTRY_KEYS.add(selectedEntryKey(entry));
    showToast("Procedure already selected; skipping Add.");
    return;
  }
  clickLikeUser(add);
  const added = await waitForSelectedCodeAdded(before, entry);
  if (!added) {
    showToast("Clicked category Add; verify Selected count changed.", true);
  } else {
    ADDED_ENTRY_KEYS.add(selectedEntryKey(entry));
  }
}

async function validate(entry, doHighlight = true) {
  const fields = topFields();
  for (const key of Object.keys(fields)) {
    try {
      if (doHighlight) highlight(fields[key], true);
    } catch (err) {
      if (["keyword", "comments"].includes(key)) continue;
      throw err;
    }
  }
  const dropdowns = [["case_year", String(entry.case_year)], ["role", entry.role], ["site", entry.site], ["patient_type", entry.patient_type]];
  for (const [key, label] of dropdowns) {
    const el = fields[key];
    if (el.tagName.toLowerCase() === "select") {
      const found = findSelectOption(el, label);
      if (!found) {
        highlight(el, false);
        throw new Error(`Missing dropdown option for ${key}: ${label}. Available: ${selectOptionSample(el)}`);
      }
    }
  }
  const category = categoryFields();
  if (doHighlight) {
    highlight(category.case_class, true);
    highlight(category.area, true);
    highlight(category.type, true);
    highlight(category.search, true);
  }
  for (const [key, label] of [["case_class", entry.case_class], ["area", entry.area]]) {
    const el = category[key];
    const found = findSelectOption(el, label);
    if (!found) {
      highlight(el, false);
      throw new Error(`Missing category option for ${key}: ${label}. Available: ${selectOptionSample(el)}`);
    }
  }
  const currentType = findSelectOption(category.type, entry.type);
  if (doHighlight && !currentType) {
    showToast("Preview found fields. Type will be checked after Area updates during Fill.", true);
    return fields;
  }
  if (doHighlight) {
    showToast(`Preview found fields and Area/Type controls for case ${entry.case_id}.`);
  }
  return fields;
}

async function fill(entry) {
  try {
    await validate(entry, false);
    await fillMetadata(entry);
    await selectCategory(entry);
    const fields = {keyword: document.getElementById("Keyword") || null, comments: document.getElementById("Comments") || null};
    if (entry.keyword && fields.keyword) setNativeValue(fields.keyword, entry.keyword);
    if (entry.comments && fields.comments) setNativeValue(fields.comments, entry.comments);
    showToast(`Filled case ${entry.case_id}. Review before submitting.`);
  } finally {
    window.scrollTo({top: 0, behavior: "smooth"});
  }
}

async function fillGroup(group) {
  try {
    const codes = group?.codes || [];
    const primary = groupPrimaryEntry(group);
    if (codes.length) await validate(primary, false);
    await fillMetadata(primary);
    for (const removed of group.removedCodes || []) {
      await removeSelectedCode({...primary, ...removed});
    }
    if (!codes.length) {
      await clearSelectedCodes();
      showToast(`Cleared selected codes for case ${primary.case_id}.`);
      return;
    }
    for (const code of codes) {
      await selectCategory({...primary, ...code});
    }
    const fields = {keyword: document.getElementById("Keyword") || null, comments: document.getElementById("Comments") || null};
    const keyword = codes.map(code => code.keyword).find(Boolean);
    const comments = codes.map(code => code.comments).filter(Boolean).join("; ");
    if (keyword && fields.keyword) setNativeValue(fields.keyword, keyword);
    if (comments && fields.comments) setNativeValue(fields.comments, comments);
    showToast(`Filled case ${primary.case_id}. Review before submitting.`);
  } finally {
    window.scrollTo({top: 0, behavior: "smooth"});
  }
}

async function submitPage() {
  const button = document.getElementById("submitButton");
  if (!button) throw new Error("ACGME submit button not found: #submitButton");
  if (button.disabled || button.getAttribute("aria-disabled") === "true") {
    throw new Error("ACGME submit button is disabled.");
  }
  showToast("Submitting ACGME case...");
  clickLikeUser(button);
  await nextFrame();
}

async function handleMessage(message) {
  if (message.action === "ping") {
    return {ok: true, message: "Content script ready.", version: CONTENT_SCRIPT_VERSION};
  }
  if (message.action === "preview") {
    await validate(message.entry, true);
    return {ok: true, message: "Fields highlighted and options validated.", version: CONTENT_SCRIPT_VERSION};
  }
  if (message.action === "fill") {
    await fill(message.entry);
    return {ok: true, message: "Form filled.", version: CONTENT_SCRIPT_VERSION};
  }
  if (message.action === "fillGroup") {
    await fillGroup(message.case || message.entry);
    return {ok: true, message: "Form filled.", version: CONTENT_SCRIPT_VERSION};
  }
  if (message.action === "submitPage") {
    await submitPage();
    return {ok: true, message: "ACGME submit clicked.", version: CONTENT_SCRIPT_VERSION};
  }
  return {ok: false, error: "Unknown action", version: CONTENT_SCRIPT_VERSION};
}

if (window.__ACGME_IR_AUTOFILL__?.listener) {
  chrome.runtime.onMessage.removeListener(window.__ACGME_IR_AUTOFILL__.listener);
}

const listener = (message, _sender, sendResponse) => {
  (async () => {
    sendResponse(await handleMessage(message));
  })().catch(err => sendResponse({ok: false, error: err.message}));
  return true;
};

window.__ACGME_IR_AUTOFILL__ = {
  version: CONTENT_SCRIPT_VERSION,
  listener,
  handleMessage
};

chrome.runtime.onMessage.addListener(listener);
})();
