/**
 * WellbeingHome — your own score, level, trend and contributing factors.
 * One job: show a person everything the system currently holds about them.
 */
import { api } from "../../../shared/api.js";
import { badge, el, factorBar, intervalText, meter, trendText } from "../../../shared/ui.js";

/**
 * Render the wellbeing home screen.
 * @param {string} pseudonymId Whose record to show.
 * @param {Object} meta API metadata (signal labels, thresholds).
 * @returns {Promise<HTMLElement>} The screen node.
 */
export async function renderWellbeingHome(pseudonymId, meta) {
  const [data, notifications] = await Promise.all([
    api.personal.summary(pseudonymId),
    api.personal.notifications(pseudonymId),
  ]);
  const root = el("div");

  root.appendChild(el("h1", { class: "page-title", text: "My wellbeing" }));
  root.appendChild(el("p", { class: "page-sub", text:
    `Based on your duty, leave, deployment and posting records as at ${data.snapshot_date}. ` +
    `This describes your working conditions, not you.` }));

  const scoreCard = el("div", { class: "card" }, [
    el("h2", { text: "Current indicator level" }),
    el("div", { class: "score-row" }, [
      el("span", { class: "score-value", text: String(data.risk.score) }),
      el("span", { class: "score-max", text: "/ 100" }),
      badge(data.risk.level),
      data.trend ? el("span", { class: "muted small", text: trendText(data.trend.direction) }) : null,
    ]),
    el("p", { class: "small", style: "margin-top:10px", text: data.risk.description }),
    data.risk.interval ? el("div", { class: "small muted", text: intervalText(data.risk) }) : null,
    data.risk.is_borderline ? el("div", { class: "note", text:
      "Your score sits close to a band boundary. The calibrated range around it crosses into the " +
      "neighbouring band, so the band shown is provisional rather than settled." }) : null,
    el("div", { class: "small muted", style: "margin-top:8px", text:
      data.is_officer_visible
        ? "Your case is currently visible to a welfare officer so that support can be offered."
        : "Your case is not visible to a welfare officer. " + (data.visibility_rule || "") }),
  ]);
  root.appendChild(scoreCard);

  if (notifications && notifications.count) {
    const noticeCard = el("div", { class: "card" }, [
      el("h2", { text: "Notifications" }),
    ]);
    notifications.notifications.forEach((n) => {
      noticeCard.appendChild(el("div", { class: "notice" }, [
        el("strong", { class: "notice-title", text: n.title }),
        el("div", { class: "small", text: n.body }),
      ]));
    });
    noticeCard.appendChild(el("div", { class: "note", text: notifications.note }));
    root.appendChild(noticeCard);
  }

  const factors = data.contributing_factors;
  const factorCard = el("div", { class: "card" }, [ el("h2", { text: "What is contributing" }) ]);
  if (factors && factors.length) {
    const maxAbs = Math.max(...factors.map((f) => Math.abs(f.contribution)));
    factors.forEach((f) => factorCard.appendChild(factorBar(f, maxAbs)));
    factorCard.appendChild(el("div", { class: "note", text:
      "These are the largest contributors to your score, computed exactly (Shapley values) " +
      "from the trained model. Each one is an organisational circumstance, not a judgement about you." }));
  } else {
    factorCard.appendChild(el("div", { class: "small muted", text:
      data.explanation_note || "No factor breakdown available for this record." }));
  }
  root.appendChild(factorCard);

  const labels = data.signal_labels || (meta && meta.signal_labels) || {};
  const indicatorCard = el("div", { class: "card" }, [ el("h2", { text: "All indicators" }) ]);
  Object.entries(data.signals).forEach(([name, value]) => {
    if (name === "voice_signal_present") return;
    if (name === "voice_stress_signal" && !data.has_voice_signal) return;
    indicatorCard.appendChild(meter(labels[name] || name, value));
  });
  if (!data.has_voice_signal) {
    indicatorCard.appendChild(el("div", { class: "small muted", text:
      "You have not opted into voice check-in. That does not affect your score." }));
  }
  root.appendChild(indicatorCard);

  root.appendChild(el("div", { class: "card" }, [
    el("h2", { text: "How much data this rests on" }),
    el("div", { class: "score-row" }, [
      badge(data.confidence.level === "High" ? "Normal" : data.confidence.level === "Medium" ? "Moderate" : "High"),
      el("span", { class: "small", text: `${data.confidence.level} data completeness (${(data.confidence.score * 100).toFixed(0)}%)` }),
    ]),
    el("div", { class: "note caution", text: data.confidence.disclaimer }),
  ]));

  return root;
}
