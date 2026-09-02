/**
 * CommanderUnitView — unit-level aggregates only.
 *
 * One job: show a commander the condition of their units without ever showing
 * them an individual.
 *
 * This is a STRUCTURALLY SEPARATE module from CaseDetail, calling separate API
 * routes (`/api/commander/*`), not the officer view with fields hidden. There is
 * no code path from this file to an individual score, because the endpoints it
 * calls do not return one and the server refuses to build a commander payload
 * containing any individual field.
 */
import { api } from "../../../shared/api.js";
import { el } from "../../../shared/ui.js";

/**
 * Render the commander unit view.
 * @returns {Promise<HTMLElement>} The screen node.
 */
export async function renderCommanderUnitView() {
  const [units, nearMisses] = await Promise.all([
    api.commander.units(), api.commander.nearMisses(),
  ]);
  const root = el("div");
  root.appendChild(el("h1", { class: "page-title", text: "Commander view — unit level" }));
  root.appendChild(el("p", { class: "page-sub", text: units.scope_note }));
  root.appendChild(el("div", { class: "note accent", text: units.purpose_note }));

  const dist = units.band_distribution || {};
  root.appendChild(el("div", { class: "card" }, [
    el("h2", { text: "Force-wide distribution" }),
    el("dl", { class: "kv" }, Object.entries(dist).flatMap(([k, v]) => [
      el("dt", { text: k }), el("dd", { text: `${v} personnel` }),
    ])),
  ]));

  if (nearMisses.findings.length) {
    const card = el("div", { class: "card" }, [el("h2", { text: "Welfare near-misses" })]);
    nearMisses.findings.forEach((f) => card.appendChild(el("div", { style: "margin-bottom:12px" }, [
      el("strong", { text: `Unit ${f.unit_id}` }),
      el("p", { class: "small", text: f.summary }),
    ])));
    card.appendChild(el("div", { class: "note", text: nearMisses.criteria_note }));
    root.appendChild(card);
  }

  const table = el("table", {}, [el("thead", {}, [el("tr", {}, [
    el("th", { text: "Unit" }), el("th", { text: "Personnel" }), el("th", { text: "Mean" }),
    el("th", { text: "Median" }), el("th", { text: "Elevated" }), el("th", { text: "High" }),
    el("th", { text: "Strained" }), el("th", { text: "Near-miss" }),
  ])])]);
  const tbody = el("tbody");
  units.units.forEach((u) => {
    if (u.is_suppressed) {
      tbody.appendChild(el("tr", {}, [
        el("td", { text: u.unit_id }),
        el("td", { colspan: "7", class: "small muted", text: u.suppression_reason }),
      ]));
      return;
    }
    tbody.appendChild(el("tr", {}, [
      el("td", { text: u.unit_id }),
      el("td", { class: "num", text: String(u.personnel_count) }),
      el("td", { class: "num", text: String(u.mean_risk) }),
      el("td", { class: "num", text: String(u.median_risk) }),
      el("td", { class: "num", text: `${(u.elevated_share * 100).toFixed(0)}%` }),
      el("td", { class: "num", text: `${(u.high_share * 100).toFixed(0)}%` }),
      el("td", { class: "small", text: u.is_systemically_strained ? "yes" : "—" }),
      el("td", { class: "small", text: u.is_near_miss ? "yes" : "—" }),
    ]));
  });
  table.appendChild(tbody);
  root.appendChild(el("div", { class: "card" }, [el("h2", { text: "Units" }), table]));

  root.appendChild(el("div", { class: "note", text:
    "No individual identity, score or contributing factor appears anywhere on this " +
    "screen, because none is present in the responses that feed it." }));
  return root;
}
