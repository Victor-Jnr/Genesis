const page = document.body.dataset.page;
const ACTIVE = new Set(["pending", "planning", "executing", "waiting", "paused"]);

function clear(node) {
  node.replaceChildren();
}

function el(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function text(tag, value, className) {
  const node = el(tag, className);
  node.textContent = value == null || value === "" ? "—" : String(value);
  return node;
}

function append(parent, ...nodes) {
  parent.append(...nodes);
  return parent;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const raw = await response.text();
  let body = null;
  if (raw) {
    try {
      body = JSON.parse(raw);
    } catch {
      body = raw;
    }
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

function post(path, payload) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload == null ? "{}" : JSON.stringify(payload),
  });
}

function showError(error) {
  const slot = document.querySelector("[data-error]");
  if (!slot) return;
  clear(slot);
  slot.append(text("p", error.message, "error"));
}

function when(value) {
  if (!value) return "—";
  return String(value).replace("T", " ").slice(0, 19);
}

function chip(status) {
  const name = status || "unknown";
  const tone = name === "completed" || name === "passed" || name === "retained" || name === "active" || name === "baseline"
    ? "ok"
    : name === "failed" || name === "rejected"
      ? "bad"
      : ACTIVE.has(name) || name === "open"
        ? "warn"
        : "";
  return text("span", name, tone ? `chip ${tone}` : "chip");
}

function table(headers, rows) {
  const node = document.createElement("table");
  const head = document.createElement("tr");
  headers.forEach((header) => head.append(text("th", header)));
  node.append(head);
  rows.forEach((row) => node.append(row));
  return node;
}

function empty(message) {
  return text("p", message, "empty");
}

function link(href, label) {
  const node = document.createElement("a");
  node.href = href;
  node.textContent = label;
  return node;
}

async function loadDashboard() {
  const [health, model, objectives, archive, capabilities, modules] = await Promise.all([
    api("/health"),
    api("/model"),
    api("/objectives"),
    api("/archive"),
    api("/capabilities"),
    api("/modules"),
  ]);
  const status = document.querySelector("#status");
  clear(status);
  status.append(
    stat("Status", health.status),
    stat("Phase", health.phase),
    stat("Model", model.loaded ? model.filename : "Not loaded"),
    stat("Context", model.context_length),
    stat("GPU layers", model.n_gpu_layers),
    stat("Capabilities", capabilities.length),
    stat("Modules", modules.length),
  );
  if (model.detail) status.append(text("p", model.detail, "muted"));

  const active = objectives.filter((item) => ACTIVE.has(item.status));
  renderObjectives(document.querySelector("#objectives"), active.length ? active : objectives.slice(0, 6), !active.length);
  renderGeneration(document.querySelector("#generation"), archive);
  const activity = document.querySelector("#activity");
  clear(activity);
  renderEvents(activity, await recentEvents(objectives.slice(0, 4)));
}

function stat(label, value) {
  const node = el("div", "stat");
  node.append(text("span", label), text("strong", value));
  return node;
}

function renderObjectives(slot, items, note) {
  clear(slot);
  if (!items.length) {
    slot.append(empty("No objectives yet."));
    return;
  }
  if (note) slot.append(text("p", "Nothing is running. Showing the latest objectives.", "muted"));
  const rows = items.map((item) => {
    const row = document.createElement("tr");
    const request = document.createElement("td");
    request.append(link(`/workstation/objectives/${item.id}`, item.request));
    row.append(request, cell(chip(item.status)), text("td", when(item.updated_at)));
    return row;
  });
  slot.append(table(["Request", "Status", "Updated"], rows));
}

function cell(node) {
  const column = document.createElement("td");
  column.append(node);
  return column;
}

function renderGeneration(slot, archive) {
  clear(slot);
  if (!archive.length) {
    slot.append(empty("No generations yet. The live tree is the current source."));
    return;
  }
  const selected = archive.find((item) => item.selected) || archive.find((item) => item.status === "baseline") || archive[0];
  slot.append(
    stat("Generation", selected.label),
    stat("Status", selected.status),
    stat("Success", selected.success == null ? "not measured" : selected.success),
    stat("Elapsed ms", selected.elapsed_ms == null ? "not measured" : selected.elapsed_ms),
  );
}

async function recentEvents(objectives) {
  const batches = await Promise.all(objectives.map((item) => api(`/objectives/${item.id}`)));
  return batches
    .flatMap((item) => (item.events || []).map((event) => ({ ...event, request: item.request })))
    .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
    .slice(0, 12);
}

function renderEvents(slot, events, emptyMessage) {
  if (!events.length) {
    slot.append(empty(emptyMessage || "No activity yet."));
    return;
  }
  const feed = el("div", "feed");
  events.forEach((event) => {
    const row = el("div", "event");
    row.append(text("time", when(event.created_at)), text("strong", event.kind), text("span", event.message));
    feed.append(row);
  });
  slot.append(feed);
}

function bindChat() {
  const form = document.querySelector("#chat-form");
  if (!form) return;
  let conversationId = null;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = new FormData(form).get("message");
    try {
      const result = await post("/chat", { message, conversation_id: conversationId });
      conversationId = result.conversation_id;
      const conversation = await api(`/conversations/${conversationId}`);
      const slot = document.querySelector("#chat-log");
      clear(slot);
      (conversation.messages || []).forEach((item) => {
        slot.append(text("h3", item.role), text("pre", item.content));
      });
      form.reset();
    } catch (error) {
      showError(error);
    }
  });
}

function bindObjectiveForm() {
  const form = document.querySelector("#objective-form");
  if (!form) return;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const created = await post("/objectives", { request: new FormData(form).get("request") });
      location.assign(`/workstation/objectives/${created.id}`);
    } catch (error) {
      showError(error);
    }
  });
}

let monitorTimer = 0;
let monitorRun = 0;

async function loadMonitor() {
  window.clearTimeout(monitorTimer);
  const run = ++monitorRun;
  const objectives = await api("/objectives");
  if (run !== monitorRun) return;
  const params = new URLSearchParams(location.search);
  const requested = params.get("id");
  const current = objectives.find((item) => item.id === requested)
    || objectives.find((item) => ACTIVE.has(item.status))
    || objectives[0];
  const picker = document.querySelector("#monitor-list");
  clear(picker);
  if (!objectives.length) {
    picker.append(empty("No objectives yet."));
    return;
  }
  const rows = objectives.map((item) => {
    const row = document.createElement("tr");
    const request = document.createElement("td");
    request.append(link(`/workstation/monitor?id=${item.id}`, item.request));
    row.append(request, cell(chip(item.status)));
    return row;
  });
  picker.append(table(["Objective", "Status"], rows));
  if (current) await renderMonitor(current.id);
  if (run !== monitorRun) return;
  monitorTimer = window.setTimeout(() => {
    loadMonitor().catch(showError);
  }, 2000);
}

async function renderMonitor(id) {
  const objective = await api(`/objectives/${id}`);
  const slot = document.querySelector("#monitor-detail");
  clear(slot);
  slot.append(text("h2", objective.request), chip(objective.status));
  const controls = el("div", "row");
  if (ACTIVE.has(objective.status)) controls.append(actionButton("Pause", () => post(`/objectives/${id}/pause`)));
  if (objective.status === "paused" || objective.status === "waiting" || objective.status === "failed") {
    controls.append(actionButton("Resume", () => post(`/objectives/${id}/resume`)));
  }
  slot.append(controls);
  renderEvents(slot, objective.events || [], "No events yet.");
  if (objective.result) slot.append(text("p", objective.result));
  if (objective.error) slot.append(text("p", objective.error, "error"));
}

function actionButton(label, action) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.addEventListener("click", async () => {
    try {
      await action();
      await loadMonitor();
    } catch (error) {
      showError(error);
    }
  });
  return button;
}

async function loadObjectiveList() {
  renderObjectives(document.querySelector("#objective-list"), await api("/objectives"), false);
}

async function loadObjective() {
  const id = location.pathname.split("/").filter(Boolean).pop();
  const [objective, experiments, resources] = await Promise.all([
    api(`/objectives/${id}`),
    api(`/objectives/${id}/experiments`),
    api(`/objectives/${id}/resources`),
  ]);
  const slot = document.querySelector("#objective");
  clear(slot);
  slot.append(text("p", objective.request), chip(objective.status), text("p", when(objective.updated_at), "muted"));
  if (objective.result) slot.append(text("h2", "Result"), text("pre", objective.result));
  if (objective.error) slot.append(text("p", objective.error, "error"));
  slot.append(text("h2", "Steps"));
  if (!objective.steps.length) slot.append(empty("No steps yet."));
  objective.steps.forEach((step, index) => {
    const block = el("article", "panel");
    block.append(text("h3", `${index + 1}. ${step.title}`), text("p", step.instruction), text("pre", step.result || "Not finished."));
    slot.append(block);
  });
  slot.append(text("h2", "Events"));
  renderEvents(slot, objective.events || [], "No events yet.");
  slot.append(text("h2", "Experiments"));
  if (!experiments.length) slot.append(empty("No experiments."));
  else {
    slot.append(table(["Name", "Result", "Outcome"], experiments.map((item) => {
      const row = document.createElement("tr");
      row.append(text("td", item.name), cell(chip(item.success ? "completed" : "failed")), text("td", item.outcome));
      return row;
    })));
  }
  slot.append(text("h2", "Sources"));
  if (!resources.length) slot.append(empty("No sources."));
  else {
    slot.append(table(["Kind", "Title"], resources.map((item) => {
      const row = document.createElement("tr");
      const title = document.createElement("td");
      title.textContent = item.title;
      if (item.url) title.append(document.createTextNode(" "), link(item.url, item.url));
      row.append(text("td", item.kind), title);
      return row;
    })));
  }
}

async function loadDevelopment() {
  const params = new URLSearchParams(location.search);
  const path = params.get("path") || "genesis";
  const generation = params.get("generation") || "";
  document.querySelector("#path").value = path;
  const archive = await api("/archive");
  const slot = document.querySelector("#archive");
  clear(slot);
  if (!archive.length) slot.append(empty("No generations yet."));
  else {
    slot.append(table(["Generation", "Status", "Parent", "Success", "Elapsed"], archive.map((item) => {
      const row = document.createElement("tr");
      const label = document.createElement("td");
      label.append(link(developmentHref(path, item.id), item.selected ? `${item.label} selected` : item.label));
      row.append(label, cell(chip(item.status)), text("td", item.parent_id || "—"), text("td", item.success), text("td", item.elapsed_ms));
      return row;
    })));
  }
  const source = document.querySelector("#source");
  clear(source);
  try {
    const body = generation
      ? await api(`/archive/${generation}/source?path=${encodeURIComponent(path)}`)
      : await api(`/source?path=${encodeURIComponent(path)}`);
    renderSource(source, path, generation, body.text || "");
  } catch (error) {
    source.append(text("p", error.message, "error"));
  }
}

function developmentHref(path, generation) {
  const params = new URLSearchParams({ path });
  if (generation) params.set("generation", generation);
  return `/workstation/development?${params.toString()}`;
}

function renderSource(slot, path, generation, body) {
  const last = path.split("/").filter(Boolean).pop() || path;
  const listing = !last.includes(".") && body.includes("\n") || (!last.includes(".") && body.startsWith(`${path}/`));
  slot.append(text("h2", generation ? `${path} in ${generation}` : path));
  if (listing) {
    const lines = body.split("\n").map((line) => line.trim()).filter(Boolean);
    if (!lines.length) {
      slot.append(empty("That directory is empty."));
      return;
    }
    const list = document.createElement("ul");
    lines.forEach((line) => {
      const item = document.createElement("li");
      item.append(link(developmentHref(line, generation), line));
      list.append(item);
    });
    slot.append(list);
    return;
  }
  slot.append(text("pre", body));
}

function bindDevelopment() {
  const form = document.querySelector("#source-form");
  if (!form) return;
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const params = new URLSearchParams(location.search);
    params.set("path", new FormData(form).get("path"));
    location.assign(`/workstation/development?${params.toString()}`);
  });
}

async function loadCapabilities() {
  const [capabilities, modules, skills, config, model] = await Promise.all([
    api("/capabilities"),
    api("/modules"),
    api("/skills"),
    api("/config"),
    api("/model"),
  ]);
  const caps = document.querySelector("#capabilities");
  clear(caps);
  caps.append(table(["Name", "Kind", "Status", "Description"], capabilities.map((item) => {
    const row = document.createElement("tr");
    row.append(text("td", item.name), text("td", item.kind), cell(chip(item.status)), text("td", item.description));
    return row;
  })));
  const moduleSlot = document.querySelector("#modules");
  clear(moduleSlot);
  if (!modules.length) moduleSlot.append(empty("No modules installed."));
  else {
    moduleSlot.append(table(["Module", "Version", "Status", "Path"], modules.map((item) => {
      const row = document.createElement("tr");
      row.append(text("td", item.name), text("td", item.version), cell(chip(item.status)), text("td", item.path));
      return row;
    })));
  }
  const skillSlot = document.querySelector("#skills");
  clear(skillSlot);
  if (!skills.length) skillSlot.append(empty("No saved skills."));
  skills.forEach((item) => {
    const block = el("article", "panel");
    block.append(text("h3", item.name), text("p", when(item.updated_at), "muted"), text("pre", item.body));
    skillSlot.append(block);
  });
  const configSlot = document.querySelector("#config");
  clear(configSlot);
  const list = document.createElement("dl");
  Object.keys(config).forEach((key) => {
    list.append(text("dt", key.replaceAll("_", " ")), text("dd", config[key]));
  });
  configSlot.append(list);
  const models = document.querySelector("#models");
  clear(models);
  models.append(text("p", model.loaded ? `Loaded ${model.filename}` : "No model is loaded.", "muted"));
  (model.discovered || []).forEach((filename) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = filename === model.filename ? "" : "secondary";
    button.textContent = filename === model.filename ? `${filename} loaded` : `Select ${filename}`;
    button.disabled = filename === model.filename;
    button.addEventListener("click", async () => {
      try {
        await post("/model/select", { filename });
        await loadCapabilities();
      } catch (error) {
        showError(error);
      }
    });
    models.append(button);
  });
  const reload = document.createElement("button");
  reload.type = "button";
  reload.className = "secondary";
  reload.textContent = "Reload";
  reload.addEventListener("click", async () => {
    try {
      await post("/model/reload");
      await loadCapabilities();
    } catch (error) {
      showError(error);
    }
  });
  models.append(reload);
}

const loaders = {
  dashboard: async () => {
    bindChat();
    bindObjectiveForm();
    await loadDashboard();
  },
  monitor: loadMonitor,
  objectives: loadObjectiveList,
  objective: loadObjective,
  development: async () => {
    bindDevelopment();
    await loadDevelopment();
  },
  capabilities: loadCapabilities,
};

if (loaders[page]) {
  loaders[page]().catch(showError);
}
