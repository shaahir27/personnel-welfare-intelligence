/**
 * PrivacyCenter — what the system holds, why, who can see it, and for how long.
 * One job: make the privacy position legible rather than legally complete.
 */
import { api } from "../../../shared/api.js";
import { el } from "../../../shared/ui.js";

/**
 * Render the privacy centre.
 * @param {string} pseudonymId Whose record.
 * @returns {Promise<HTMLElement>} The screen node.
 */
export async function renderPrivacyCenter(pseudonymId) {
  const data = await api.personal.privacy(pseudonymId);
  const root = el("div");
  root.appendChild(el("h1", { class: "page-title", text: "My data" }));
  root.appendChild(el("p", { class: "page-sub", text: data.identity_note }));

  const dataCard = el("div", { class: "card" }, [el("h2", { text: "What is held" })]);
  data.data_categories.forEach((c) => {
    dataCard.appendChild(el("div", { style: "margin-bottom:16px" }, [
      el("div", {}, [
        el("strong", { text: c.category }),
        c.voluntary ? el("span", { class: "optional-tag", text: "voluntary" }) : null,
      ]),
      el("div", { class: "small muted", text: c.examples }),
      el("dl", { class: "kv small", style: "margin-top:6px" }, [
        el("dt", { text: "Used for" }), el("dd", { text: c.used_for }),
        el("dt", { text: "Visible to" }), el("dd", { text: c.visible_to }),
        el("dt", { text: "Kept for" }), el("dd", { text: `${c.retention_days} days` }),
      ]),
      c.note ? el("div", { class: "note", text: c.note }) : null,
    ]));
  });
  root.appendChild(dataCard);

  const seesCard = el("div", { class: "card" }, [el("h2", { text: "Who sees what" })]);
  data.who_sees_what.forEach((r) => seesCard.appendChild(el("div", { style: "margin-bottom:12px" }, [
    el("strong", { text: r.role }), el("div", { class: "small", text: r.sees }),
  ])));
  root.appendChild(seesCard);

  const choiceCard = el("div", { class: "card" }, [el("h2", { text: "Your choices" })]);
  data.your_choices.forEach((c) => choiceCard.appendChild(el("div", { style: "margin-bottom:12px" }, [
    el("div", {}, [el("strong", { text: c.choice }), el("span", { class: "small muted", text: ` — ${c.state}` })]),
    el("div", { class: "small", text: `If you decline: ${c.effect_if_declined}` }),
  ])));
  root.appendChild(choiceCard);

  root.appendChild(el("div", { class: "card" }, [
    el("h2", { text: "Never used for" }),
    el("ul", {}, data.not_used_for.map((t) => el("li", { class: "small", text: t }))),
  ]));
  return root;
}
