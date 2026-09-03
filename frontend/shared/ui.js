/**
 * Small DOM and formatting helpers shared by both frontends.
 *
 * One job: the handful of primitives every screen needs, so no screen
 * re-implements a meter, a badge or a sparkline slightly differently.
 *
 * Deliberately dependency-free. No build step, no framework, no CDN -- the
 * build environment has no package-registry access, so both apps are plain ES
 * modules the browser loads directly. For a demo that has to run offline on
 * someone else's laptop this is arguably the right answer anyway.
 */

/**
 * Create an element with attributes and children.
 * @param {string} tag Tag name.
 * @param {Object} [attrs] Attributes; `class`, `text`, `html` and `on*` handled specially.
 * @param {Array} [children] Child nodes or strings.
 * @returns {HTMLElement} The created element.
 */
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

/** Remove all children from a node. @param {HTMLElement} node Target. */
export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/**
 * Map a risk level to its CSS class.
 * @param {string} level "Normal", "Moderate" or "High".
 * @returns {string} CSS modifier class.
 */
export function levelClass(level) {
  return String(level || "").toLowerCase() || "neutral";
}

/**
 * Build a risk-level badge.
 * @param {string} level Risk level.
 * @param {boolean} [small] Use the compact size.
 * @returns {HTMLElement} The badge.
 */
export function badge(level, small) {
  return el("span", {
    class: `badge ${levelClass(level)}${small ? " badge-sm" : ""}`,
    text: level || "-",
  });
}

/**
 * Build a labelled 0-100 meter.
 * @param {string} label Row label.
 * @param {number} value Value on the 0-100 scale.
 * @param {string} [tone] Optional fill modifier ("normal"/"moderate"/"high").
 * @returns {HTMLElement} The meter.
 */
export function meter(label, value, tone) {
  const pct = Math.max(0, Math.min(100, Number(value) || 0));
  const fillTone = tone || (pct >= 65 ? "high" : pct >= 40 ? "moderate" : "normal");
  return el("div", { class: "meter" }, [
    el("div", { class: "meter-head" }, [
      el("span", { class: "label", text: label }),
      el("span", { class: "value", text: pct.toFixed(0) }),
    ]),
    el("div", { class: "meter-track" }, [
      el("div", { class: `meter-fill ${fillTone}`, style: `width:${pct}%` }),
    ]),
  ]);
}

/**
 * Build a contributing-factor bar from a SHAP attribution.
 * @param {Object} factor `{label, contribution, signal_value}`.
 * @param {number} maxAbs Largest absolute contribution in the set, for scaling.
 * @returns {HTMLElement} The factor row.
 */
export function factorBar(factor, maxAbs) {
  const contribution = Number(factor.contribution) || 0;
  const negative = contribution < 0;
  const width = maxAbs > 0 ? (Math.abs(contribution) / maxAbs) * 100 : 0;
  return el("div", { class: "factor" }, [
    el("div", { class: "factor-head" }, [
      el("span", { text: factor.label }),
      el("span", {
        class: `factor-contrib${negative ? " negative" : ""}`,
        text: `${contribution >= 0 ? "+" : ""}${contribution.toFixed(1)} pts`,
      }),
    ]),
    el("div", { class: "factor-track" }, [
      el("div", { class: `factor-fill${negative ? " negative" : ""}`, style: `width:${width}%` }),
    ]),
    el("div", {
      class: "factor-note",
      text: `indicator level ${Number(factor.signal_value).toFixed(0)} / 100`,
    }),
  ]);
}

/**
 * Draw a small inline sparkline of score history.
 * @param {Array<Object>} points `[{snapshot_date, score}]`, oldest first.
 * @param {Object} [thresholds] `{risk_moderate_min, risk_high_min}` guide lines.
 * @param {number} [width] Pixel width.
 * @param {number} [height] Pixel height.
 * @returns {SVGElement} The sparkline.
 */
export function sparkline(points, thresholds = {}, width = 520, height = 150) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "spark");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", String(height));

  if (!points || points.length === 0) return svg;

  const pad = { top: 10, right: 10, bottom: 22, left: 30 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const y = (score) => pad.top + innerH * (1 - Math.max(0, Math.min(100, score)) / 100);
  const x = (i) => pad.left + (points.length === 1 ? innerW / 2 : (innerW * i) / (points.length - 1));

  const guide = (value, colour, label) => {
    if (value === undefined || value === null) return;
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", pad.left);
    line.setAttribute("x2", width - pad.right);
    line.setAttribute("y1", y(value));
    line.setAttribute("y2", y(value));
    line.setAttribute("stroke", colour);
    line.setAttribute("stroke-dasharray", "3 4");
    line.setAttribute("stroke-width", "1");
    line.setAttribute("opacity", "0.55");
    svg.appendChild(line);

    const text = document.createElementNS(ns, "text");
    text.setAttribute("x", 4);
    text.setAttribute("y", y(value) + 3.5);
    text.setAttribute("fill", colour);
    text.setAttribute("font-size", "9.5");
    text.textContent = label;
    svg.appendChild(text);
  };
  guide(thresholds.risk_moderate_min, "#d9a441", "Mod");
  guide(thresholds.risk_high_min, "#cf7860", "High");

  const path = document.createElementNS(ns, "path");
  path.setAttribute(
    "d",
    points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.score).toFixed(1)}`).join(" ")
  );
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "#5aa9e6");
  path.setAttribute("stroke-width", "2");
  path.setAttribute("stroke-linejoin", "round");
  svg.appendChild(path);

  points.forEach((p, i) => {
    const dot = document.createElementNS(ns, "circle");
    dot.setAttribute("cx", x(i));
    dot.setAttribute("cy", y(p.score));
    dot.setAttribute("r", i === points.length - 1 ? 4 : 2.6);
    dot.setAttribute("fill", "#5aa9e6");
    svg.appendChild(dot);

    if (i === 0 || i === points.length - 1) {
      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", x(i));
      label.setAttribute("y", height - 6);
      label.setAttribute("fill", "#6f8098");
      label.setAttribute("font-size", "10");
      label.setAttribute("text-anchor", i === 0 ? "start" : "end");
      label.textContent = String(p.snapshot_date || "").slice(0, 10);
      svg.appendChild(label);
    }
  });

  return svg;
}

/**
 * Describe a calibrated score range in one short phrase.
 * @param {Object} risk A case's `risk` block (`interval`, `band_certainty`, `bands_plausible`).
 * @returns {string} e.g. "calibrated range 57–75 (90%) · borderline: Moderate / High", or "" when uncalibrated.
 */
export function intervalText(risk) {
  if (!risk || !risk.interval) return "";
  const { low, high, coverage } = risk.interval;
  const pct = coverage ? ` (${Math.round(coverage * 100)}%)` : "";
  const bands = (risk.bands_plausible || []).join(" / ");
  const tail = risk.band_certainty === "borderline" ? ` · borderline: ${bands}` : " · band certain";
  return `calibrated range ${Number(low).toFixed(0)}–${Number(high).toFixed(0)}${pct}${tail}`;
}

/**
 * Build a small badge saying whether a band is certain or borderline.
 * @param {string} certainty "certain", "borderline", or null.
 * @returns {HTMLElement|null} The badge, or null when uncalibrated.
 */
export function certaintyBadge(certainty) {
  if (!certainty) return null;
  return el("span", {
    class: `badge badge-sm ${certainty === "borderline" ? "moderate" : "neutral"}`,
    text: certainty,
  });
}

/**
 * Format a trend direction with an arrow glyph.
 * @param {string} direction Trend direction.
 * @returns {string} Display text.
 */
export function trendText(direction) {
  const arrows = { Rising: "↑", Improving: "↓", Stable: "→" };
  return `${arrows[direction] || "·"} ${direction || "Unknown"}`;
}

/**
 * Render an error into a container.
 * @param {HTMLElement} container Target node.
 * @param {Error} error The error to display.
 */
export function showError(container, error) {
  clear(container);
  container.appendChild(
    el("div", { class: "empty" }, [
      el("div", { text: error.message || "Something went wrong." }),
      el("div", {
        class: "small muted",
        text: "If the pipeline has not been run yet: python scripts/train_models.py, then python scripts/run_pipeline.py",
      }),
    ])
  );
}
