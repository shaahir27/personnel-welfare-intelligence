/**
 * WelfareQueue — the prioritised list of cases visible at officer level.
 * One job: show which cases have met the escalation threshold, most urgent first.
 *
 * Two counts, and the screen must not conflate them. `total_eligible` is how
 * many people the escalation rule admits — a statement about those people.
 * `visible_count` is how many this view shows first, capped at the officer's
 * working capacity — a statement about the officer. Reporting only the second
 * against the population would tell a reader the rule admits far fewer people
 * than it does, which is the one thing this header must not do.
 *
 * The held-back cases are one click away, never hidden: "show all eligible"
 * re-requests with ?all=1.
 */
import { api } from "../../../shared/api.js";
import { badge, certaintyBadge, el, trendText } from "../../../shared/ui.js";

/**
 * Render the welfare queue.
 * @param {Function} go Navigation callback (screenId, caseId).
 * @param {boolean} [showAll] Lift the capacity cap and list every eligible case.
 * @returns {Promise<HTMLElement>} The screen node.
 */
export async function renderWelfareQueue(go, showAll) {
  const data = await api.officer.queue(showAll);
  const root = el("div");
  root.appendChild(el("h1", { class: "page-title", text: "Welfare queue" }));
  root.appendChild(el("p", { class: "page-sub", text:
    `${data.total_eligible} of ${data.population} personnel meet the escalation rule ` +
    `as at ${data.snapshot_date}. Showing ${data.visible_count}.` }));
  root.appendChild(el("div", { class: "note accent", text: data.visibility_rule }));
  if (data.capacity_rule) {
    root.appendChild(el("div", { class: "note", text: data.capacity_rule }));
  }
  if (data.held_back_count > 0) {
    const link = el("button", {
      class: "linklike",
      text: `Show all ${data.total_eligible} eligible cases`,
      onclick: async () => {
        const full = await renderWelfareQueue(go, true);
        root.replaceWith(full);
      },
    });
    root.appendChild(el("p", {}, [link]));
  } else if (showAll) {
    const link = el("button", {
      class: "linklike",
      text: "Back to the prioritised view",
      onclick: async () => {
        const capped = await renderWelfareQueue(go, false);
        root.replaceWith(capped);
      },
    });
    root.appendChild(el("p", {}, [link]));
  }
  if (data.band_certainty_note) {
    root.appendChild(el("div", { class: "note", text:
      `${data.band_certainty_note} ${data.borderline_count} of ${data.visible_count} shown cases are borderline.` }));
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
