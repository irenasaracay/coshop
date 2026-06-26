// Dataset explorer: renders SEC splits and preference stacks from data/samples.json.
(function () {
  const state = { data: null, domainIdx: 0, specIdx: 0 };

  const el = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const SEC_META = {
    search: { title: "Search", sub: "unlockable by any action, including a clarifying question" },
    experience: { title: "Experience", sub: "unlockable by example options" },
    credence: { title: "Credence", sub: "unlockable by a technical explanation" },
  };

  function tags(features, cls) {
    if (!features || !features.length) return '<span class="note">—</span>';
    return features
      .map((f) => {
        const attrs = f.label && f.label !== f.name
          ? ` class="tag ${cls} has-tip" title="${esc(f.label)}"`
          : ` class="tag ${cls}"`;
        const val = f.value
          ? `<span class="tval">${esc(f.value)}</span>`
          : "";
        return `<span${attrs}><span class="tname">${esc(f.name)}</span>${val}</span>`;
      })
      .join("");
  }

  function renderDomainTabs() {
    el("domain-tabs").innerHTML = state.data.domains
      .map(
        (d, i) =>
          `<button class="domain-tab ${i === state.domainIdx ? "active" : ""}" data-i="${i}">${esc(d.label)}</button>`
      )
      .join("");
    el("domain-tabs")
      .querySelectorAll(".domain-tab")
      .forEach((b) =>
        b.addEventListener("click", () => {
          state.domainIdx = +b.dataset.i;
          state.specIdx = 0;
          render();
        })
      );
  }

  function renderStats(domain) {
    el("domain-stats").innerHTML = Object.entries(domain.stats)
      .map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>`)
      .join("");
  }

  function renderSecDist(domain) {
    const dist = (domain.sec_distribution || [])
      .slice()
      .sort((a, b) => a.name.localeCompare(b.name));
    el("sec-dist").innerHTML = dist
      .map((f) => {
        const seg = (cls, v) =>
          v > 0 ? `<span class="seg ${cls}" style="width:${(v * 100).toFixed(1)}%"></span>` : "";
        const label = f.label && f.label !== f.name ? ` title="${esc(f.label)}"` : "";
        return (
          `<div class="dist-row">` +
          `<span class="dist-name has-tip"${label}>${esc(f.name)}</span>` +
          `<span class="dist-bar">${seg("search", f.search)}${seg("experience", f.experience)}${seg("credence", f.credence)}</span>` +
          `</div>`
        );
      })
      .join("");
  }

  function renderTarget(spec) {
    const t = spec.target || {};
    if (!t.name && !t.id) {
      el("target-note").innerHTML = "";
      return;
    }
    el("target-note").innerHTML =
      `<span class="tgt-label">Target item</span> ` +
      `<span class="tgt-name">${esc(t.name || t.id)}</span>` +
      (t.name && t.id ? ` <span class="tgt-id">#${esc(t.id)}</span>` : "");
  }

  function renderSpecPicker(domain) {
    el("spec-picker").innerHTML = domain.specs
      .map(
        (s, i) =>
          `<button class="spec-btn ${i === state.specIdx ? "active" : ""}" data-i="${i}">user&nbsp;#${esc(s.spec_index)}</button>`
      )
      .join("");
    el("spec-picker")
      .querySelectorAll(".spec-btn")
      .forEach((b) =>
        b.addEventListener("click", () => {
          state.specIdx = +b.dataset.i;
          render();
        })
      );
  }

  function layer({ cls, badge, name, note, content, initial }) {
    return (
      `<div class="layer ${initial ? "initial" : ""}">` +
      `<div class="layer-head"><span class="badge ${cls}">${badge}</span>` +
      `<span class="layer-name">${name}</span>` +
      (note ? `<span class="layer-note">${note}</span>` : "") +
      `</div><div class="feature-tags">${content}</div></div>`
    );
  }

  function renderStack(spec) {
    const initTags = spec.initial_state && spec.initial_state.length
      ? tags(spec.initial_state, "search")
      : '<span class="note">sparse</span>';

    let html = layer({
      cls: "search",
      badge: "S₁",
      name: "Initial state",
      note: "what the user already knows",
      content: initTags,
      initial: true,
    });
    html += layer({
      cls: "search",
      badge: SEC_META.search.title,
      name: SEC_META.search.sub,
      content: tags(spec.sec_split.search, "search"),
    });
    html += layer({
      cls: "experience",
      badge: SEC_META.experience.title,
      name: SEC_META.experience.sub,
      content: tags(spec.sec_split.experience, "experience"),
    });
    html += layer({
      cls: "credence",
      badge: SEC_META.credence.title,
      name: SEC_META.credence.sub,
      content: tags(spec.sec_split.credence, "credence"),
    });
    el("stack").innerHTML = html;
  }

  function render() {
    const domain = state.data.domains[state.domainIdx];
    const spec = domain.specs[state.specIdx];
    renderDomainTabs();
    renderStats(domain);
    renderSecDist(domain);
    renderSpecPicker(domain);
    renderTarget(spec);
    renderStack(spec);
  }

  fetch("data/samples.json")
    .then((r) => r.json())
    .then((data) => {
      state.data = data;
      render();
    })
    .catch((err) => {
      const c = el("explorer");
      if (c) c.innerHTML = `<div class="note">Could not load dataset samples (${esc(err.message)}). Run <code>python website/tools/export_website_data.py</code> first.</div>`;
    });
})();
