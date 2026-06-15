/* Dashboard app: builds the ticker tree, renders one Plotly card per figure,
   wires visibility toggles + drag-reorder, and fills the manifest/config panes.
   All data arrives via the inlined globals DASHBOARD_DATA / MANIFEST_TEXT /
   CONFIG_TEXT. No network, no build step. */
(function () {
  "use strict";

  var DATA = window.DASHBOARD_DATA || { episode: {}, groups: [] };
  var cards = {}; // figKey -> { card, graph, figCb }

  function el(tag, cls, htmlStr) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (htmlStr != null) e.innerHTML = htmlStr;
    return e;
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function figKey(group, idx) { return group + "::" + idx; }

  // ---- top bar -----------------------------------------------------------
  function buildTopbar() {
    var ep = DATA.episode || {};
    var bits = ["<b>Episode " + esc(ep.id) + "</b>"];
    if (ep.date) bits.push("date: " + esc(ep.date));
    if (ep.achieved_reward != null)
      bits.push("<span class='metric'>achieved_reward " + Number(ep.achieved_reward).toFixed(2) + "</span>");
    if (ep.cum_E_kWh != null)
      bits.push("<span class='metric'>cum_E_kWh " + Number(ep.cum_E_kWh).toFixed(2) + "</span>");
    if (ep.cum_price_EUR != null)
      bits.push("<span class='metric'>cum_EUR " + Number(ep.cum_price_EUR).toFixed(2) + "</span>");
    if (ep.seed != null) bits.push("seed " + esc(ep.seed));
    if (ep.length != null) bits.push("len " + esc(ep.length));
    document.getElementById("topbar").innerHTML =
      bits.join(" <span class='sep'>|</span> ");
  }

  // ---- plot cards --------------------------------------------------------
  function renderCard(groupName, idx, fig) {
    var key = figKey(groupName, idx);
    var card = el("div", "plot-card" + (fig.wide ? " wide" : ""));
    card.style.height = (fig.height + 34) + "px";
    card.dataset.key = key;

    var header = el("div", "plot-card-header");
    header.innerHTML =
      "<span class='grip'>&#x283F;</span>" +
      "<span class='plot-card-title' title='" + esc(fig.title) + "'>" + esc(fig.label) + "</span>" +
      "<span class='card-x' title='hide'>&times;</span>";

    var body = el("div", "plot-body");
    var graph = el("div");
    graph.id = "plot-" + groupName + "-" + idx;
    body.appendChild(graph);
    card.appendChild(header);
    card.appendChild(body);
    document.getElementById("plotFlex").appendChild(card);

    // Strip fixed size so the card drives layout; keep everything else.
    var layout = Object.assign({}, fig.spec.layout);
    delete layout.width;
    delete layout.height;
    layout.autosize = true;
    Plotly.newPlot(graph, fig.spec.data, layout, { responsive: true, displaylogo: false });

    var ro = new ResizeObserver(function () {
      if (window.Plotly && Plotly.Plots) Plotly.Plots.resize(graph);
    });
    ro.observe(card);

    header.querySelector(".card-x").addEventListener("click", function () {
      var cb = cards[key].figCb;
      if (cb) { cb.checked = false; cb.dispatchEvent(new Event("change")); }
    });

    cards[key] = { card: card, graph: graph, figCb: null };
  }

  function setCardVisible(key, visible) {
    var c = cards[key];
    if (!c) return;
    c.card.style.display = visible ? "" : "none";
    if (visible && window.Plotly && Plotly.Plots) Plotly.Plots.resize(c.graph);
  }

  function syncGroupCheckbox(groupCb, children) {
    var cbs = children.querySelectorAll("input.fig-cb");
    var all = true, none = true;
    cbs.forEach(function (cb) { if (cb.checked) none = false; else all = false; });
    groupCb.checked = all;
    groupCb.indeterminate = !all && !none;
  }

  // ---- left ticker tree (2 levels) + plot cards --------------------------
  function buildTreeAndPlots() {
    var tree = document.getElementById("tree");
    var footnotes = document.getElementById("footnotes");

    DATA.groups.forEach(function (group) {
      var wrap = el("div", "group");

      // 1st level: group name (the per-group .html plot name).
      var groupLabel = el("label");
      var groupCb = el("input");
      groupCb.type = "checkbox";
      groupCb.checked = true;
      groupCb.className = "group-cb";
      groupLabel.appendChild(groupCb);
      groupLabel.appendChild(el("b", null, esc(group.name) + " (" + group.figures.length + ")"));
      wrap.appendChild(groupLabel);

      // 2nd level: one checkbox per subplot, labelled by its key name(s).
      var children = el("div", "children");
      group.figures.forEach(function (fig, idx) {
        var key = figKey(group.name, idx);
        renderCard(group.name, idx, fig);

        var label = el("label");
        label.title = fig.title;
        var cb = el("input");
        cb.type = "checkbox";
        cb.checked = true;
        cb.className = "fig-cb";
        cb.addEventListener("change", function () {
          setCardVisible(key, cb.checked);
          syncGroupCheckbox(groupCb, children);
        });
        cards[key].figCb = cb;
        label.appendChild(cb);
        label.appendChild(el("span", "name", esc(fig.label)));
        children.appendChild(label);
      });

      groupCb.addEventListener("change", function () {
        children.querySelectorAll("input.fig-cb").forEach(function (cb) {
          if (cb.checked !== groupCb.checked) {
            cb.checked = groupCb.checked;
            cb.dispatchEvent(new Event("change"));
          }
        });
      });

      wrap.appendChild(children);
      tree.appendChild(wrap);

      if (group.footnote) footnotes.appendChild(el("div", "fn", group.footnote));
    });
  }

  // ---- right panel: manifest + config ------------------------------------
  function fmtDate(s) {
    if (!s) return "";
    return String(s).replace("T", " ").replace(/[+-]\d\d:\d\d$/, "");
  }
  function buildManifest() {
    var pane = document.getElementById("manifestPane");
    var text = window.MANIFEST_TEXT || "";
    if (!text) { pane.style.display = "none"; return; }

    document.getElementById("manifestCode").textContent = text;
    var summary = document.getElementById("manifestSummary");
    try {
      var m = JSON.parse(text);
      var rows = [];
      // note + date first and most prominently.
      if (m.note) rows.push("<div class='note'>“" + esc(m.note) + "”</div>");
      if (m.created_at) rows.push("<div class='date'>" + esc(fmtDate(m.created_at)) + "</div>");
      var meta = [];
      if (m.trial_name) meta.push("trial: " + esc(m.trial_name));
      if (m.git_branch) meta.push("branch: " + esc(m.git_branch));
      if (m.git_sha) meta.push("sha: " + esc(String(m.git_sha).slice(0, 10)));
      if (meta.length) rows.push("<div class='row'>" + meta.join(" · ") + "</div>");
      if (m.git_dirty) rows.push("<div class='row dirty'>git dirty (uncommitted changes)</div>");
      summary.innerHTML = rows.join("");
    } catch (e) {
      summary.innerHTML = "";
    }
  }
  function buildConfig() {
    var text = window.CONFIG_TEXT || "";
    if (!text) { document.getElementById("configPane").style.display = "none"; return; }
    document.getElementById("configCode").textContent = text;
  }

  // ---- resizable panels --------------------------------------------------
  function resizeAllPlots() {
    if (!window.Plotly || !Plotly.Plots) return;
    Object.keys(cards).forEach(function (k) {
      if (cards[k].card.style.display !== "none") Plotly.Plots.resize(cards[k].graph);
    });
  }

  // Vertical splitter: drag resizes the adjacent column (`panel`); the middle
  // column (flex: 1) absorbs the rest. `side` says where `panel` sits.
  function makeVSplit(splitter, panel, side) {
    if (!splitter || !panel) return;
    splitter.addEventListener("mousedown", function (e) {
      e.preventDefault();
      var startX = e.clientX;
      var startW = panel.getBoundingClientRect().width;
      splitter.classList.add("dragging");
      document.body.style.userSelect = "none";
      function move(ev) {
        var dx = ev.clientX - startX;
        var w = side === "left" ? startW + dx : startW - dx;
        w = Math.max(120, Math.min(window.innerWidth * 0.75, w));
        panel.style.flex = "0 0 " + w + "px";
        resizeAllPlots();
      }
      function up() {
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        document.body.style.userSelect = "";
        splitter.classList.remove("dragging");
        resizeAllPlots();
      }
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });
  }

  // Horizontal splitter: drag resizes the top pane (`topPane`); the bottom pane
  // (flex: 1) absorbs the rest.
  function makeHSplit(splitter, topPane) {
    if (!splitter || !topPane) return;
    splitter.addEventListener("mousedown", function (e) {
      e.preventDefault();
      var startY = e.clientY;
      var startH = topPane.getBoundingClientRect().height;
      splitter.classList.add("dragging");
      document.body.style.userSelect = "none";
      function move(ev) {
        var h = Math.max(60, startH + (ev.clientY - startY));
        topPane.style.flex = "0 0 " + h + "px";
      }
      function up() {
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        document.body.style.userSelect = "";
        splitter.classList.remove("dragging");
      }
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });
  }

  function wireSplitters() {
    makeVSplit(document.getElementById("splitLeft"), document.getElementById("leftPanel"), "left");
    makeVSplit(document.getElementById("splitRight"), document.getElementById("rightPanel"), "right");
    makeHSplit(document.getElementById("splitPanes"), document.getElementById("manifestPane"));
    // Hide the inter-pane splitter if either right-side pane is absent.
    var splitPanes = document.getElementById("splitPanes");
    var manifestHidden = document.getElementById("manifestPane").style.display === "none";
    var configHidden = document.getElementById("configPane").style.display === "none";
    if (splitPanes && (manifestHidden || configHidden)) splitPanes.style.display = "none";
    // Hide the right splitter entirely if the whole right panel is empty.
    if (manifestHidden && configHidden) {
      var sr = document.getElementById("splitRight");
      var rp = document.getElementById("rightPanel");
      if (sr) sr.style.display = "none";
      if (rp) rp.style.display = "none";
    }
  }

  // ---- init --------------------------------------------------------------
  function init() {
    buildTopbar();
    buildTreeAndPlots();
    buildManifest();
    buildConfig();
    wireSplitters();
    if (window.Prism) Prism.highlightAll();
    if (window.Sortable) {
      new Sortable(document.getElementById("plotFlex"), {
        handle: ".plot-card-header",
        draggable: ".plot-card",
        animation: 150,
        ghostClass: "sortable-ghost",
      });
    }
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
