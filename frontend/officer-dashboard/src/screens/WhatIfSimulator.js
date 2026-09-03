/**
 * WhatIfSimulator — illustrative scenario projection.
 * One job: let an officer move indicator values and see how the model responds.
 *
 * Labelled illustrative on the screen and in the API response. It shows how the
 * MODEL responds to different inputs. It is not a forecast and has not been
 * validated against outcomes.
 */
import { api } from "../../../shared/api.js";
import { clear, el } from "../../../shared/ui.js";

/**
 * Render the what-if simulator.
 * @param {string} caseId Pseudonym of the case.
 * @param {Object} meta API metadata for signal labels.
 * @returns {Promise<HTMLElement>} The screen node.
 */
export async function renderWhatIfSimulator(caseId, meta) {
  if (!caseId) {
    return el("div", { class: "empty", text: "Open a case first, then return here." });
  }
  const data = await api.officer.case(caseId);
  const labels = data.signal_labels || (meta && meta.signal_labels) || {};
  const adjustments = {};

  const root = el("div");
  root.appendChild(el("h1", { class: "page-title", text: "What-if simulator" }));
  root.appendChild(el("div", { class: "note caution", text:
    "Illustrative only. This shows how the model responds to different indicator " +
    "values. It is not a forecast, has not been validated against outcomes, and " +
    "cannot account for anything the model was not given." }));

  const output = el("div", { class: "card" }, [el("div", { class: "small muted", text: "Move a slider to project." })]);

  /** Recompute the projection from the current slider values. */
  async function recompute() {
    const result = await api.officer.whatIf(caseId, adjustments);
    clear(output);
    output.appendChild(el("h2", { text: "Projection" }));
    output.appendChild(el("div", { class: "score-row" }, [
      el("span", { class: "score-value", text: String(result.projected_score) }),
      el("span", { class: "score-max", text: `was ${result.current_score}` }),
      el("span", { class: "small", text: `${result.change > 0 ? "+" : ""}${result.change} points` }),
    ]));
    if (result.projected_interval) {
      const p = result.projected_interval;
      const c = result.current_interval;
      output.appendChild(el("div", { class: "small muted", text:
        `calibrated range ${p.low}–${p.high} (was ${c.low}–${c.high}). The range moves with the ` +
        `score; it does not narrow, because the model's typical error does not depend on the scenario.` }));
    }
    output.appendChild(el("div", { class: "small muted", text: result.disclaimer }));
  }

  const controls = el("div", { class: "card" }, [el("h2", { text: "Adjust indicators" })]);
  Object.entries(data.signals).forEach(([name, value]) => {
    if (name === "voice_signal_present" || name === "voice_stress_signal") return;
    const readout = el("span", { class: "small mono", text: String(value) });
    const slider = el("input", {
      type: "range", min: "0", max: "100", value: String(value),
      oninput: (e) => {
        adjustments[name] = Number(e.target.value);
        readout.textContent = e.target.value;
      },
      onchange: () => recompute(),
    });
    controls.appendChild(el("div", { style: "margin-bottom:14px" }, [
      el("div", { class: "meter-head" }, [
        el("span", { class: "label", text: labels[name] || name }), readout,
      ]),
      slider,
    ]));
  });

  root.appendChild(output);
  root.appendChild(controls);
  return root;
}
