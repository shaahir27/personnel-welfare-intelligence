/**
 * Personal Wellness App — shell and routing.
 *
 * One job: pick the acting identity, hold the four screens, and render one of
 * them.
 *
 * On the identity selector: this build has no authentication, so the app needs
 * *some* way to know whose record to show. It offers a short list of demo
 * pseudonyms from `/api/demo/identities`. In a deployment this is replaced by
 * the signed-in user, and the selector disappears — the screens themselves take
 * a pseudonym and do not care where it came from.
 */

import { api } from "../../shared/api.js";
import { clear, el, showError } from "../../shared/ui.js";
import { renderWellbeingHome } from "./screens/WellbeingHome.js";
import { renderTrendView } from "./screens/TrendView.js";
import { renderVoiceCheckIn } from "./screens/VoiceCheckIn.js";
import { renderPrivacyCenter } from "./screens/PrivacyCenter.js";

const SCREENS = [
  { id: "home", label: "My wellbeing", render: renderWellbeingHome },
  { id: "trend", label: "Over time", render: renderTrendView },
  { id: "checkin", label: "Check in", render: renderVoiceCheckIn },
  { id: "privacy", label: "My data", render: renderPrivacyCenter },
];

const state = { pseudonymId: null, screen: "home", meta: null };

/** Render the navigation bar, marking the active screen. */
function renderNav() {
  const nav = document.getElementById("nav");
  clear(nav);
  for (const screen of SCREENS) {
    nav.appendChild(
      el("button", {
        class: state.screen === screen.id ? "active" : "",
        text: screen.label,
        onclick: () => {
          state.screen = screen.id;
          renderNav();
          renderScreen();
        },
      })
    );
  }
}

/** Render the current screen into the main container. */
async function renderScreen() {
  const container = document.getElementById("screen");
  clear(container);
  container.appendChild(el("div", { class: "empty", text: "Loading…" }));
  const screen = SCREENS.find((s) => s.id === state.screen) || SCREENS[0];
  try {
    const content = await screen.render(state.pseudonymId, state.meta);
    clear(container);
    container.appendChild(content);
  } catch (error) {
    showError(container, error);
  }
}

/** Populate the identity selector from the demo endpoint. */
async function setUpIdentities() {
  const select = document.getElementById("identity");
  const { identities } = await api.demoIdentities();
  clear(select);
  for (const identity of identities) {
    select.appendChild(
      el("option", {
        value: identity.pseudonym_id,
        text: `${identity.pseudonym_id.slice(0, 11)}… (${identity.risk_level})`,
      })
    );
  }
  state.pseudonymId = identities.length ? identities[0].pseudonym_id : null;
  select.addEventListener("change", () => {
    state.pseudonymId = select.value;
    renderScreen();
  });
}

/** Boot the application. */
async function main() {
  const container = document.getElementById("screen");
  try {
    state.meta = await api.meta();
    await setUpIdentities();
    renderNav();
    await renderScreen();
  } catch (error) {
    showError(container, error);
  }
}

main();
