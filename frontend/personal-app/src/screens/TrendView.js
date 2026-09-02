/**
 * TrendView — how the score has moved over time.
 * One job: plot the history and state the trajectory honestly.
 */
import { api } from "../../../shared/api.js";
import { badge, el, sparkline, trendText } from "../../../shared/ui.js";

/**
 * Render the trend screen.
 * @param {string} pseudonymId Whose history to show.
 * @returns {Promise<HTMLElement>} The screen node.
 */
export async function renderTrendView(pseudonymId) {
  const [history, summary] = await Promise.all([
    api.personal.history(pseudonymId),
    api.personal.summary(pseudonymId),
  ]);
  const root = el("div");
  root.appendChild(el("h1", { class: "page-title", text: "Over time" }));
  root.appendChild(el("p", { class: "page-sub", text:
    "Where your indicators are going matters more than where they are. A rising line " +
    "is the thing this system exists to notice early." }));

  const chart = el("div", { class: "card" }, [ el("h2", { text: "Indicator level by snapshot" }) ]);
  chart.appendChild(sparkline(history.points, history.thresholds));
  root.appendChild(chart);

  if (summary.trend) {
    const t = summary.trend;
    root.appendChild(el("div", { class: "card" }, [
      el("h2", { text: "Trajectory" }),
      el("dl", { class: "kv" }, [
        el("dt", { text: "Direction" }), el("dd", { text: trendText(t.direction) }),
        el("dt", { text: "Change" }), el("dd", { text: `${t.slope_per_30_days > 0 ? "+" : ""}${t.slope_per_30_days} points per 30 days` }),
        el("dt", { text: "Snapshots used" }), el("dd", { text: String(t.points_used) }),
        el("dt", { text: "Elevated for" }), el("dd", { text: `${t.persistence_snapshots} consecutive snapshot(s)` }),
      ]),
      el("div", { class: "note", text:
        "This is a description of what has already happened. It is not a forecast." }),
    ]));
  }

  const table = el("table", {}, [
    el("thead", {}, [ el("tr", {}, [
      el("th", { text: "Snapshot" }), el("th", { text: "Score" }), el("th", { text: "Level" }),
    ])]),
  ]);
  const tbody = el("tbody");
  history.points.forEach((p) => tbody.appendChild(el("tr", {}, [
    el("td", { text: p.snapshot_date }),
    el("td", { class: "num", text: String(p.score) }),
    el("td", {}, [badge(p.level, true)]),
  ])));
  table.appendChild(tbody);
  root.appendChild(el("div", { class: "card" }, [el("h2", { text: "History" }), table]));
  return root;
}
