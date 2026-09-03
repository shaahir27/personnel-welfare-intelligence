/**
 * Officer / Commander dashboard — shell and routing.
 *
 * One job: switch between the welfare-officer screens and the commander screen.
 *
 * Note that the commander screen is a separate module calling separate API
 * routes, not the officer view with fields hidden. That separation exists on
 * both sides of the wire deliberately.
 */
import { api } from "../../shared/api.js";
import { signInForDashboard } from "../../shared/demo-login.js";
import { clear, el, showError } from "../../shared/ui.js";
import { renderWelfareQueue } from "./screens/WelfareQueue.js";
import { renderCaseDetail } from "./screens/CaseDetail.js";
import { renderCommanderUnitView } from "./screens/CommanderUnitView.js";
import { renderWhatIfSimulator } from "./screens/WhatIfSimulator.js";

const state = { screen: "queue", caseId: null, meta: null };

const SCREENS = [
  { id: "queue", label: "Welfare queue", role: "welfare_officer" },
  { id: "case", label: "Case detail", role: "welfare_officer" },
  { id: "whatif", label: "What-if", role: "welfare_officer" },
  { id: "commander", label: "Commander view", role: "commander" },
];

/** Navigate to a screen and re-render. @param {string} id Screen id. @param {string} [caseId] Case. */
export function go(id, caseId) {
  state.screen = id;
  if (caseId) state.caseId = caseId;
  renderNav();
  render();
}

/** Render the nav bar. */
function renderNav() {
  const nav = document.getElementById("nav");
  clear(nav);
  SCREENS.forEach((s) => nav.appendChild(el("button", {
    class: state.screen === s.id ? "active" : "",
    text: s.label,
    onclick: () => go(s.id),
  })));
}

/** Render the active screen. */
async function render() {
  const container = document.getElementById("screen");
  clear(container);
  container.appendChild(el("div", { class: "empty", text: "Loading…" }));
  try {
    let node;
    if (state.screen === "queue") node = await renderWelfareQueue(go);
    else if (state.screen === "case") node = await renderCaseDetail(state.caseId, go, state.meta);
    else if (state.screen === "whatif") node = await renderWhatIfSimulator(state.caseId, state.meta);
    else node = await renderCommanderUnitView();
    clear(container);
    container.appendChild(node);
  } catch (error) {
    showError(container, error);
  }
}

/** Boot: sign in for both dashboard roles, then render. */
async function main() {
  try {
    await signInForDashboard(api);
    state.meta = await api.meta();
    renderNav();
    await render();
  } catch (error) {
    showError(document.getElementById("screen"), error);
  }
}
main();
