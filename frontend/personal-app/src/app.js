/**
 * Personal Wellness App — shell and routing.
 *
 * One job: pick the acting identity, hold the four screens, and render one of
 * them.
 *
 * On the identity selector: this build has no real sign-ups, so the app needs
 * *some* way to choose which of the 800 synthetic people to look at. It offers a
 * short list from `/api/demo/identities` and signs in as whoever is picked. The
 * token that comes back is scoped to that one pseudonym and the server enforces
 * that scope, so picking somebody is signing in as them — not a way to read
 * somebody else's record while signed in as yourself. In a deployment the
 * selector disappears and the person signs in as themselves; the screens
 * themselves take a pseudonym and do not care where it came from.
 */

import { api } from "../../shared/api.js";
import { signInAsPersonnel } from "../../shared/demo-login.js";
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
  // Switching identity means signing in again: the token is scoped to one
  // pseudonym, and the server checks that scope on every personal route. That
  // is the whole point — the selector picks who to sign in as, it does not let
  // one signed-in person read another's record.
  select.addEventListener("change", async () => {
    state.pseudonymId = select.value;
    const container = document.getElementById("screen");
    try {
      await signInAsPersonnel(api, state.pseudonymId);
      await renderScreen();
    } catch (error) {
      showError(container, error);
    }
  });
}

/** Boot the application. */
async function main() {
  const container = document.getElementById("screen");
  try {
    state.meta = await api.meta();
    await setUpIdentities();
    if (state.pseudonymId) await signInAsPersonnel(api, state.pseudonymId);
    renderNav();
    await renderScreen();
  } catch (error) {
    showError(container, error);
  }
}

main();
