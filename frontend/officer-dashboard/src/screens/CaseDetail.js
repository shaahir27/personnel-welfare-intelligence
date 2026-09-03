/**
 * CaseDetail — one case in full for a welfare officer.
 * One job: score, trajectory, contributing factors, confidence, attribution and
 * any unit near-miss, with the handling constraint stated on the screen.
 */
import { api } from "../../../shared/api.js";
import { badge, certaintyBadge, el, factorBar, intervalText, meter, sparkline, trendText } from "../../../shared/ui.js";

/**
 * Render the case detail screen.
 * @param {string} caseId Pseudonym of the case.
 * @param {Function} go Navigation callback.
 * @param {Object} [meta] API metadata; band thresholds are read from it rather than hard-coded.
 * @returns {Promise<HTMLElement>} The screen node.
 */
export async function renderCaseDetail(caseId, go, meta) {
  if (!caseId) {
    return el("div", { class: "empty", text: "Select a case from the welfare queue." });
  }
  const data = await api.officer.case(caseId);
  const root = el("div");
  root.appendChild(el("h1", { class: "page-title", text: "Case detail" }));
  root.appendChild(el("p", { class: "page-sub mono", text: `${data.pseudonym_id} · unit ${data.unit_id} · ${data.posting_type} · ${data.snapshot_date}` }));
  root.appendChild(el("div", { class: "note caution", text: data.handling_note }));

  root.appendChild(el("div", { class: "grid two" }, [
    el("div", { class: "card" }, [
      el("h2", { text: "Level" }),
      el("div", { class: "score-row" }, [
        el("span", { class: "score-value", text: String(data.risk.score) }),
        el("span", { class: "score-max", text: "/ 100" }),
        badge(data.risk.level),
        certaintyBadge(data.risk.band_certainty),
      ]),
      data.risk.interval ? el("div", { class: "small muted", style: "margin-top:6px", text: intervalText(data.risk) }) : null,
      data.risk.is_borderline ? el("div", { class: "note caution", text: data.risk.borderline_note }) : null,
      data.trend ? el("div", { class: "small muted", style: "margin-top:8px", text:
        `${trendText(data.trend.direction)} · ${data.trend.slope_per_30_days} pts / 30 days · ` +
        `elevated ${data.trend.persistence_snapshots} snapshot(s)` }) : null,
    ]),
    el("div", { class: "card" }, [
      el("h2", { text: "Individual or systemic" }),
      el("div", {}, [badge(data.attribution.classification === "Systemic" ? "Moderate" : "Normal", true),
        el("span", { class: "small", text: ` ${data.attribution.classification}` })]),
      el("p", { class: "small", style: "margin-top:8px", text: data.attribution.explanation }),
    ]),
  ]));

  root.appendChild(el("div", { class: "card" }, [
    el("h2", { text: "Trajectory" }),
    sparkline(data.history, data.thresholds || (meta && meta.thresholds) || {}),
  ]));

  const factorCard = el("div", { class: "card" }, [el("h2", { text: "Contributing factors (exact Shapley)" })]);
  if (data.contributing_factors && data.contributing_factors.length) {
    const maxAbs = Math.max(...data.contributing_factors.map((f) => Math.abs(f.contribution)));
    data.contributing_factors.forEach((f) => factorCard.appendChild(factorBar(f, maxAbs)));
  } else {
    factorCard.appendChild(el("div", { class: "small muted", text: "No precomputed explanation for this case." }));
  }
  root.appendChild(factorCard);

  const recCard = el("div", { class: "card" }, [
    el("h2", { text: "Recommended interventions" }),
  ]);
  if (data.recommendations && data.recommendations.length) {
    data.recommendations.forEach((r) => {
      recCard.appendChild(el("div", { class: "rec" }, [
        el("div", { class: "rec-head" }, [
          el("strong", { text: r.title }),
          el("span", { class: "rec-owner", text: r.action_owner.replace(/_/g, " ") }),
        ]),
        el("p", { class: "small", text: r.description }),
        r.low_confidence
          ? el("div", { class: "small muted", text:
              "This case rests on thin data. The action still stands; the picture behind it is less complete." })
          : null,
      ]));
    });
    recCard.appendChild(el("div", { class: "note", text: data.recommendation_note }));
  } else {
    recCard.appendChild(el("div", { class: "small muted", text:
      "No pre-approved intervention matched this case's risk level, contributing signals and attribution." }));
  }
  root.appendChild(recCard);

  if (data.alerts && data.alerts.length) {
    const alertCard = el("div", { class: "card" }, [el("h2", { text: "Alerts raised" })]);
    data.alerts.forEach((a) => {
      alertCard.appendChild(el("div", { class: "rec" }, [
        el("div", { class: "rec-head" }, [
          el("strong", { text: a.title }),
          el("span", { class: "rec-owner", text: a.priority }),
        ]),
        el("p", { class: "small", text: a.body }),
      ]));
    });
    alertCard.appendChild(el("div", { class: "note", text:
      "The individual is not told that an officer was notified. That is deliberate: " +
      "a person who knows an alert fired has a reason to manage their indicators." }));
    root.appendChild(alertCard);
  }

  const labels = data.signal_labels || {};
  const indicators = el("div", { class: "card" }, [el("h2", { text: "All indicators" })]);
  Object.entries(data.signals).forEach(([name, value]) => {
    if (name === "voice_signal_present") return;
    if (name === "voice_stress_signal" && !data.has_voice_signal) return;
    indicators.appendChild(meter(labels[name] || name, value));
  });
  root.appendChild(indicators);

  root.appendChild(el("div", { class: "card" }, [
    el("h2", { text: "Data confidence" }),
    el("div", { class: "small", text: `${data.confidence.level} (${(data.confidence.score * 100).toFixed(0)}% completeness)` }),
    el("div", { class: "note caution", text: data.confidence.disclaimer }),
    data.risk.interval ? el("div", { class: "small muted", style: "margin-top:8px", text:
      "The calibrated range above is the separate, statistical statement: a split-conformal " +
      "interval whose coverage is guaranteed on data the model never saw. Completeness says how " +
      "much data the score rests on; the range says how far the model is typically wrong." }) : null,
  ]));

  if (data.unit_near_miss) {
    root.appendChild(el("div", { class: "card" }, [
      el("h2", { text: "Unit near-miss" }),
      el("p", { class: "small", text: data.unit_near_miss.summary }),
    ]));
  }

  if (data.access_note) root.appendChild(el("div", { class: "small muted", style: "margin:8px 0", text: data.access_note }));
  root.appendChild(el("button", { class: "ghost", text: "Open in what-if simulator", onclick: () => go("whatif", caseId) }));
  return root;
}
