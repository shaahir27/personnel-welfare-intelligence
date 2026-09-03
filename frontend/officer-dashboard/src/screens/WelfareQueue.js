/**
 * WelfareQueue — the prioritised list of cases visible at officer level.
 * One job: show which cases have met the escalation threshold, most urgent first.
 */
import { api } from "../../../shared/api.js";
import { badge, certaintyBadge, el, trendText } from "../../../shared/ui.js";

/**
 * Render the welfare queue.
 * @param {Function} go Navigation callback (screenId, caseId).
 * @returns {Promise<HTMLElement>} The screen node.
 */
export async function renderWelfareQueue(go) {
  const data = await api.officer.queue();
  const root = el("div");
  root.appendChild(el("h1", { class: "page-title", text: "Welfare queue" }));
  root.appendChild(el("p", { class: "page-sub", text:
    `${data.visible_count} of ${data.population} personnel are visible at officer level ` +
    `as at ${data.snapshot_date}.` }));
  root.appendChild(el("div", { class: "note accent", text: data.visibility_rule }));
  if (data.band_certainty_note) {
    root.appendChild(el("div", { class: "note", text:
      `${data.band_certainty_note} ${data.borderline_count} of ${data.visible_count} visible cases are borderline.` }));
  }

  const table = el("table", {}, [el("thead", {}, [el("tr", {}, [
    el("th", { text: "Case" }), el("th", { text: "Unit" }), el("th", { text: "Level" }),
    el("th", { text: "Score" }), el("th", { text: "Band" }), el("th", { text: "Trend" }), el("th", { text: "Attribution" }),
    el("th", { text: "Confidence" }), el("th", { text: "Unit near-miss" }),
  ])])]);
  const tbody = el("tbody");
  data.cases.forEach((c) => tbody.appendChild(el("tr", {
    class: "clickable", onclick: () => go("case", c.pseudonym_id),
  }, [
    el("td", { class: "mono", text: c.pseudonym_id.slice(0, 12) + "…" }),
    el("td", { text: c.unit_id }),
    el("td", {}, [badge(c.risk_level, true)]),
    el("td", { class: "num", text: c.interval ? `${c.score} (${c.interval.low}–${c.interval.high})` : String(c.score) }),
    el("td", {}, [certaintyBadge(c.band_certainty) || el("span", { class: "small muted", text: "—" })]),
    el("td", { class: "small", text: trendText(c.trend_direction) }),
    el("td", { class: "small", text: c.attribution }),
    el("td", { class: "small", text: c.confidence_level }),
    el("td", { class: "small", text: c.unit_near_miss ? "yes" : "—" }),
  ])));
  table.appendChild(tbody);
  root.appendChild(el("div", { class: "card" }, [table]));
  return root;
}
