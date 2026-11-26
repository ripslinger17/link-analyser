const MODELS = [
  { id: "gemini-3.5-flash-lite", label: "Gemini 3.5 Flash Lite" },
  { id: "gemini-3.6-flash", label: "Gemini 3.6 Flash" },
  { id: "gemini-3.8-flash", label: "Gemini 3.8 Flash" },
];
const DEFAULT_MODEL = MODELS[0].id;
const MODEL_STORAGE_KEY = "link-analyser-gemini-model";

const MESSAGES = [
  "Where should we point our compass today?",
  "Paste a link. We'll do the digging.",
  "Curiosity looks good on you.",
  "Let's see what this corner of the web is hiding.",
  "One link is all we need.",
  "The internet is big. Start somewhere.",
  "Got a URL? We're listening.",
  "Whisper a website and we'll check it out.",
];

// The single FastAPI endpoint. The UI opts into SSE progress; plain POSTs return JSON.
const API_URL =
  window.location.protocol === "file:"
    ? "http://localhost:4000/link-analysis?stream=true"
    : "/link-analysis?stream=true";

const stage = document.getElementById("stage");
const thread = document.getElementById("thread");
const template = document.getElementById("analysis-template");
const form = document.getElementById("search-form");
const input = document.getElementById("link-input");
const tagline = document.getElementById("tagline");
const screenReaderStatus = document.getElementById("screen-reader-status");
const addonWrap = document.getElementById("addon-wrap");
const addButton = document.getElementById("add-button");
const sendButton = document.getElementById("send-button");
const modelOptions = document.querySelectorAll(".model-option");

let messageTimer;
let isSubmitting = false;
let selectedModel = DEFAULT_MODEL;

function refreshIcons(root = document) {
  if (window.lucide) window.lucide.createIcons({ root });
}

function randomMessage() {
  return MESSAGES[Math.floor(Math.random() * MESSAGES.length)];
}

function showTagline(message) {
  clearTimeout(messageTimer);
  tagline.textContent = message;
}

function closeMenu() {
  addonWrap.classList.remove("open");
  addButton.setAttribute("aria-expanded", "false");
}

function modelLabel(id) {
  return MODELS.find((model) => model.id === id)?.label || MODELS[0].label;
}

function applyModelSelection(id) {
  selectedModel = MODELS.some((model) => model.id === id) ? id : DEFAULT_MODEL;
  modelOptions.forEach((option) => {
    const isSelected = option.dataset.model === selectedModel;
    option.classList.toggle("is-selected", isSelected);
    option.setAttribute("aria-checked", String(isSelected));
  });
  const label = `Choose Gemini model — current: ${modelLabel(selectedModel)}`;
  addButton.setAttribute("aria-label", label);
  addButton.title = label;
}

function readStoredModel() {
  try {
    const saved = localStorage.getItem(MODEL_STORAGE_KEY);
    return MODELS.some((model) => model.id === saved) ? saved : DEFAULT_MODEL;
  } catch {
    return DEFAULT_MODEL;
  }
}

function announce(message) {
  screenReaderStatus.textContent = message;
}

function scrollToLatest(entry) {
  // Scroll the new request into view without moving the composer.
  requestAnimationFrame(() => {
    thread.scrollTo({ top: entry.offsetTop - thread.offsetTop, behavior: "smooth" });
  });
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    thread.scrollTo({ top: thread.scrollHeight, behavior: "smooth" });
  });
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function shortHost(url) {
  if (typeof url !== "string" || !url.trim()) return "";
  try {
    return new URL(url).hostname.replace(/^www\./i, "");
  } catch {
    return "";
  }
}

function parseWebUrl(value) {
  const raw = value.trim();
  if (!raw) throw new Error("Paste a URL to get started.");

  // Accept example.com as well as https://example.com.
  const hasScheme = /^[a-z][a-z\d+.-]*:/i.test(raw);
  const url = new URL(hasScheme ? raw : `https://${raw}`);
  if (!["http:", "https:"].includes(url.protocol) || !url.hostname) {
    throw new Error("Please enter a valid HTTP or HTTPS link.");
  }
  return url;
}

function setStep(entry, index, state, description) {
  const step = entry.querySelectorAll(".timeline-step")[index];
  if (!step) return;
  step.classList.remove("is-pending", "is-active", "is-complete", "is-error");
  step.classList.add(`is-${state}`);
  if (description) step.querySelector(".step-description").textContent = description;
}

function addSectionProgress(entry, index, title) {
  const timeline = entry.querySelector(".timeline");
  if (!timeline) return;

  const step = element("li", "timeline-step is-active");
  step.dataset.sectionProgress = String(index);
  step.append(element("span", "step-marker"));

  const copy = element("div", "step-copy");
  copy.append(element("p", "step-title", title || `Generating section ${index}`));
  copy.append(element("p", "step-description", "Generating answer..."));
  step.append(copy);
  timeline.append(step);
  scrollToBottom();
}

function completeSectionProgress(entry, index) {
  const step = entry.querySelector(`[data-section-progress="${index}"]`);
  if (!step) return;
  step.classList.remove("is-active");
  step.classList.add("is-complete");
  step.querySelector(".step-description").textContent = "Answer ready";
}

function failActiveSectionProgress(entry, message) {
  const step = entry.querySelector('.timeline-step.is-active[data-section-progress]');
  if (!step) return false;
  step.classList.remove("is-active");
  step.classList.add("is-error");
  step.querySelector(".step-description").textContent = message;
  return true;
}

function parseSSEFrame(frame) {
  let event = "message";
  const dataLines = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

function showSiteIcon(entry, faviconUrl) {
  const image = entry.querySelector(".site-favicon");
  const fallback = entry.querySelector(".site-fallback");
  image.hidden = true;
  fallback.hidden = false;

  if (!faviconUrl) return;
  image.onload = () => {
    image.hidden = false;
    fallback.hidden = true;
  };
  image.onerror = () => {
    image.hidden = true;
    fallback.hidden = false;
  };
  image.src = faviconUrl;
}

function safeImageUrl(value, baseUrl) {
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const parsed = new URL(value.trim(), baseUrl);
    return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : null;
  } catch {
    return null;
  }
}

function getWebsiteName(data) {
  if (!data || typeof data !== "object") return null;
  const candidates = [
    data.site_name,
    data.website_name,
    data.title,
    data.metadata?.site_name,
    data.metadata?.website_name,
    data.metadata?.["og:site_name"],
    data.metadata?.og_site_name,
    data.open_graph?.site_name,
  ];
  return candidates.find((name) => typeof name === "string" && name.trim())?.trim();
}

function getErrorMessage(data, response) {
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
  if (typeof detail?.error === "string") return detail.error;
  if (typeof detail?.message === "string") return detail.message;
  if (Array.isArray(detail)) {
    return detail.map((issue) => issue.msg).filter(Boolean).join("; ") || "The link could not be analysed.";
  }
  if (typeof data?.message === "string") return data.message;
  return `The server returned HTTP ${response.status}.`;
}

function getResultMessage(data) {
  if (!data || typeof data !== "object") return "The report was received successfully.";

  for (const value of [data.message, data.summary, data.result]) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "The report was received successfully.";
}

function revealResult(entry, title, message, isError = false, statusCode) {
  const result = entry.querySelector(".final-result");
  const code = entry.querySelector(".result-code");
  result.hidden = false;
  result.classList.toggle("is-error", isError);
  result.querySelector(".result-title").textContent = title;
  result.querySelector(".result-message").textContent = message;
  const icon = result.querySelector(".result-symbol i");
  if (icon) icon.setAttribute("data-lucide", isError ? "alert-circle" : "check");

  if (statusCode !== undefined && statusCode !== null) {
    code.textContent = `HTTP ${statusCode}`;
    code.hidden = false;
  } else {
    code.hidden = true;
  }
  refreshIcons();
}

function createEntry(url) {
  const entry = template.content.firstElementChild.cloneNode(true);
  const domain = url.hostname.replace(/^www\./i, "");
  const request = entry.querySelector(".request-bubble");
  request.href = url.href;
  entry.querySelector(".request-link").textContent = url.href;
  entry.querySelector(".site-name").textContent = domain;
  entry.querySelector(".site-domain").textContent = domain;
  entry.querySelector(".site-fallback").textContent = domain.charAt(0).toUpperCase();

  // The hostname appears immediately. A backend-provided site_name replaces it later.
  showSiteIcon(entry, safeImageUrl("/favicon.ico", url.origin));
  thread.appendChild(entry);
  stage.classList.add("has-conversation");
  refreshIcons();
  scrollToLatest(entry);
  return entry;
}

function renderTableSection(section) {
  const card = element("section", "report-section");
  const heading = element("div", "section-heading");

  heading.append(element("span", "section-number", section.index ?? ""));
  const headingCopy = element("div", "section-copy");
  headingCopy.append(element("p", "section-kicker", `Section ${section.index ?? ""}`));
  headingCopy.append(element("h3", "section-title", section.title || "Report"));
  heading.append(headingCopy);

  const body = element("div", "section-body");
  if (section.note) body.append(element("p", "section-note", section.note));
  const table = element("table", "comparison-table");
  const head = element("thead");
  const headRow = element("tr");
  const showRowHeader = section.data?.rowHeader !== false;

  if (showRowHeader) headRow.append(element("th", "row-label-head"));

  const columns = Array.isArray(section.data?.columns) ? section.data.columns : [];
  table.style.minWidth = `${Math.max(620, columns.length * 190 + (showRowHeader ? 150 : 0))}px`;
  columns.forEach((column) => {
    const cell = element("th", "company-head");
    cell.append(element("span", "company-name", column.name || "Company"));
    const host = shortHost(column.url);
    if (host) cell.append(element("span", "company-url", host));
    if (column.role === "main") cell.append(element("span", "you-badge", "Your site"));
    headRow.append(cell);
  });
  head.append(headRow);
  table.append(head);

  const bodyRows = element("tbody");
  const rows = Array.isArray(section.data?.rows) ? section.data.rows : [];
  rows.forEach((row) => {
    const tr = element("tr");
    if (showRowHeader) tr.append(element("th", "row-label", row.label || ""));
    const values = Array.isArray(row.values) ? row.values : [];
    columns.forEach((_, index) => {
      tr.append(element("td", "row-value", values[index] || "—"));
    });
    bodyRows.append(tr);
  });

  table.append(bodyRows);
  body.append(table);
  card.append(heading, body);
  return card;
}

function renderSuggestionsSection(section) {
  const card = element("section", "report-section");
  const heading = element("div", "section-heading");

  heading.append(element("span", "section-number", section.index ?? ""));
  const headingCopy = element("div", "section-copy");
  headingCopy.append(element("p", "section-kicker", `Section ${section.index ?? ""}`));
  headingCopy.append(element("h3", "section-title", section.title || "Report"));
  heading.append(headingCopy);

  const body = element("div", "section-body suggestion-body");
  const groups = Array.isArray(section.data?.groups) ? section.data.groups : [];

  groups.forEach((group) => {
    const block = element("div", "suggestion-group");
    block.append(element("h4", "suggestion-group-title", group.title || ""));

    const list = element("ul", "suggestion-list");
    const items = Array.isArray(group.items) ? group.items : [];
    items.forEach((item) => {
      const li = element("li", "suggestion-item");
      li.append(element("strong", "suggestion-label", item.label || ""));
      li.append(element("span", "suggestion-text", item.text || "—"));
      list.append(li);
    });

    block.append(list);
    body.append(block);
  });

  card.append(heading, body);
  return card;
}

function asText(value) {
  if (Array.isArray(value)) return value.join(", ");
  return value ? String(value) : "—";
}

function buildSearchTable(queries) {
  const columns = [
    "Search Query",
    "Relevant Platform",
    "Search Intent",
    "Business Relevance",
    "Target audience for our website",
  ].map((name) => ({ name }));
  const rows = (queries || []).map((query) => ({
    values: [
      query.query,
      asText(query.platforms),
      query.intent,
      query.business_relevance,
      query.audience,
    ],
  }));
  return { rowHeader: false, columns, rows };
}

function buildSeoSuggestions(seo) {
  const support = seo?.support_content_details || {};
  const supportItems = Object.entries(support).map(([label, text]) => ({ label, text }));
  if (!supportItems.length && (seo?.supporting_content || []).length) {
    supportItems.push({
      label: "Recommended content",
      text: seo.supporting_content.join(", "),
    });
  }

  return {
    groups: [
      {
        title: "Keyword Recommendations",
        items: [
          { label: "Primary Keyword", text: seo?.primary_keyword },
          { label: "Secondary Keyword", text: asText(seo?.secondary_keywords) },
          { label: "Long-tail keywords", text: asText(seo?.long_tail_keywords) },
          { label: "Relevant business keywords", text: asText(seo?.business_keywords) },
        ],
      },
      {
        title: "SEO Title",
        items: [
          { label: "Recommended SEO Title", text: seo?.recommended_title },
          { label: "Relevant primary keyword", text: seo?.relevant_primary_keyword },
          { label: "Improved title structure", text: seo?.title_structure },
        ],
      },
      {
        title: "Metadata Description",
        items: [
          { label: "Recommended meta description", text: seo?.meta_description },
          { label: "Short human-friendly business summary", text: seo?.short_summary },
          { label: "Relevant keywords", text: asText(seo?.relevant_keywords) },
        ],
      },
      {
        title: "Additional Support Content",
        items: supportItems,
      },
    ],
  };
}

function buildServicesTable(services) {
  const columns = [
    "Service Name",
    "Description",
    "Target Industry",
    "Target Customer",
    "Main Benefit",
  ].map((name) => ({ name }));
  const rows = (services || []).map((service) => ({
    values: [
      service.name,
      service.description,
      service.target_industry,
      service.target_customer,
      service.main_benefit,
    ],
  }));
  return { rowHeader: false, columns, rows };
}

function buildAudienceSuggestions(audience) {
  const items = [
    ["Primary Target Audience", audience?.primary_audience],
    ["Secondary Target Audience", audience?.secondary_audience],
    ["Industry Segments", audience?.industries],
    ["Business Types", audience?.business_types],
    ["Customer Profiles", audience?.customer_profiles],
    ["Location or Geographical Focus", audience?.location_focus],
    ["Customer Needs and Challenges", audience?.customer_needs],
    ["Potential Customer Interests", audience?.customer_interests],
    ["Recommended Audience Segments for Marketing", audience?.recommended_segments],
  ].map(([label, value]) => ({ label, text: asText(value) }));

  return { groups: [{ title: "Target Audience", items }] };
}

function renderStructuredReport(entry, payload) {
  const analysis = payload?.analysis || {};
  const report = [
    {
      index: 1,
      title: "Competitive overview",
      type: "table",
      note: analysis.competitor_analysis?.source_note,
      data: analysis.competitor_analysis?.table,
    },
    {
      index: 2,
      title: "Search opportunity map",
      type: "table",
      data: buildSearchTable(analysis.target_search_queries),
    },
    {
      index: 3,
      title: "SEO recommendations",
      type: "suggestions",
      data: buildSeoSuggestions(analysis.seo_improvements),
    },
    {
      index: 4,
      title: "Core services",
      type: "table",
      note: analysis.top_services_note,
      data: buildServicesTable(analysis.top_services),
    },
    {
      index: 5,
      title: "Target audience",
      type: "suggestions",
      data: buildAudienceSuggestions(analysis.target_audience),
    },
  ];

  const container = entry.querySelector(".report-sections");
  if (!container) return;
  container.replaceChildren(
    ...report.map((section) =>
      section.type === "table"
        ? renderTableSection(section)
        : renderSuggestionsSection(section),
    ),
  );
  scrollToBottom();
}

function applyResult(entry, payload, url, domain, stateLabel) {
  const siteName = getWebsiteName(payload);
  if (siteName) entry.querySelector(".site-name").textContent = siteName;

  const faviconValue =
    payload.favicon && typeof payload.favicon === "object"
      ? payload.favicon.url
      : payload.favicon;
  const iconValue =
    payload.favicon_url || faviconValue || payload.icon_url ||
    payload.metadata?.favicon_url || payload.metadata?.favicon;
  if (iconValue) showSiteIcon(entry, safeImageUrl(iconValue, url.origin));

  renderStructuredReport(entry, payload);
  stateLabel.textContent = "Completed";
  stateLabel.classList.add("is-done");
  announce(`Full report for ${siteName || domain} is ready.`);
}

async function analyse(url, entry) {
  const domain = url.hostname.replace(/^www\./i, "");
  const stateLabel = entry.querySelector(".analysis-state");
  let activeStep = 0;

  setStep(entry, 0, "active", "Checking the link format");
  announce(`Valid URL. Analysing ${domain}.`);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url.href, model: selectedModel }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `The server returned HTTP ${response.status}.`);
    }
    if (!response.body) throw new Error("This browser cannot read streamed responses.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalPayload = null;

    function handleFrame(frame) {
      const parsed = parseSSEFrame(frame);
      if (!parsed) return;

      if (parsed.event === "step") {
        const { step, state, description } = parsed.data;
        if (!Number.isInteger(step)) return;
        activeStep = step;
        setStep(entry, step, state, description);
        if (state === "active") announce(description);
      } else if (parsed.event === "section_start") {
        addSectionProgress(entry, parsed.data?.index, parsed.data?.title);
      } else if (parsed.event === "section_done") {
        completeSectionProgress(entry, parsed.data?.index);
      } else if (parsed.event === "error") {
        throw new Error(parsed.data?.message || "Analysis failed.");
      } else if (parsed.event === "result") {
        finalPayload = parsed.data;
      }
    }

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";
      frames.forEach(handleFrame);
    }
    if (buffer.trim()) handleFrame(buffer);

    if (!finalPayload) throw new Error("The server finished without a result.");
    applyResult(entry, finalPayload, url, domain, stateLabel);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not reach the server.";
    if (!failActiveSectionProgress(entry, message)) {
      setStep(entry, activeStep, "error", message);
    }
    stateLabel.textContent = "Failed";
    stateLabel.classList.add("is-error");
    revealResult(entry, "Analysis failed", message, true);
    announce(`Analysis failed: ${message}`);
  }
}

showTagline(randomMessage());
applyModelSelection(readStoredModel());
refreshIcons();

addButton.addEventListener("click", (event) => {
  event.stopPropagation();
  const willOpen = !addonWrap.classList.contains("open");
  addonWrap.classList.toggle("open", willOpen);
  addButton.setAttribute("aria-expanded", String(willOpen));
});

document.addEventListener("click", (event) => {
  if (!addonWrap.contains(event.target)) closeMenu();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeMenu();
});

modelOptions.forEach((option) => {
  option.addEventListener("click", () => {
    const id = option.dataset.model;
    applyModelSelection(id);
    try {
      localStorage.setItem(MODEL_STORAGE_KEY, id);
    } catch {
      // Storage can be unavailable in private browsing; the selection still applies.
    }
    showTagline(`${modelLabel(id)} selected.`);
    closeMenu();
    messageTimer = setTimeout(() => showTagline(randomMessage()), 2200);
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (isSubmitting) return;

  let url;
  try {
    url = parseWebUrl(input.value);
  } catch (error) {
    showTagline(error.message || "Please enter a valid URL.");
    input.focus();
    return;
  }

  isSubmitting = true;
  sendButton.disabled = true;
  closeMenu();
  input.value = "";
  const entry = createEntry(url);

  try {
    await analyse(url, entry);
  } finally {
    isSubmitting = false;
    sendButton.disabled = false;
    input.focus();
  }
});
