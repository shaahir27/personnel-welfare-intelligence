/**
 * API client shared by both frontends.
 *
 * One job: hold signed tokens for the acting principals and attach the right
 * one to every request.
 *
 * On authentication: the app calls `POST /api/auth/login` at boot and keeps the
 * returned JWT in memory for the life of the page. Every subsequent request
 * carries it as `Authorization: Bearer <token>`. Nothing is written to
 * localStorage — a token that outlives the tab is a token somebody else can
 * find on a shared terminal, and these screens are meant to be usable on one.
 *
 * On holding more than one token: the officer dashboard demonstrates two roles,
 * so it signs in as both and each call uses the token for the role that route
 * belongs to. A real deployment issues one token to one signed-in person and
 * the commander screen simply is not reachable for an officer — the separation
 * that matters is enforced on the server either way, which is the next point.
 *
 * The point worth understanding about roles: a token proves which role the
 * caller holds, but it does not *grant* any of the scope rules. The server
 * enforces those itself. A commander with a perfectly valid commander token
 * still gets the commander view, which contains no individual data, because the
 * server will not build a commander payload that has any. That is the
 * difference between hiding fields in a UI and enforcing them on a server.
 */

const API_BASE = "";

/** In-memory tokens, keyed by role. Cleared when the page is closed or reloaded. */
const sessions = {};

/**
 * Exchange credentials for a token and hold it for this page.
 * @param {string} username Account name.
 * @param {string} password Account password.
 * @param {string} [subject] Pseudonym to act as; only the personnel account may send one.
 * @returns {Promise<Object>} The login response, including the role it granted.
 */
export async function login(username, password, subject) {
  const payload = { username, password };
  if (subject) payload.subject = subject;
  const response = await fetch(API_BASE + "/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.detail || `Sign-in failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  sessions[body.role] = { token: body.token, subject: body.subject };
  return body;
}

/**
 * Whether a token is held for a role.
 * @param {string} role Role to check.
 * @returns {boolean} True when signed in as that role.
 */
export function isSignedIn(role) {
  return Boolean(sessions[role] && sessions[role].token);
}

/**
 * Build request headers for a role's token.
 * @param {string} [role] Which role's token to attach.
 * @returns {Object} Header map.
 */
function headers(role) {
  const h = { "Content-Type": "application/json" };
  const held = role && sessions[role];
  if (held && held.token) h["Authorization"] = `Bearer ${held.token}`;
  return h;
}

/**
 * Perform a GET and parse JSON, surfacing server messages as thrown errors.
 * @param {string} path Request path.
 * @param {string} [role] Which role's token to send.
 * @returns {Promise<Object>} Parsed response body.
 */
export async function get(path, role) {
  const response = await fetch(API_BASE + path, { headers: headers(role) });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.detail || `Request failed (${response.status})`);
    error.status = response.status;
    error.body = body;
    throw error;
  }
  return body;
}

/**
 * Sign out of a role, ending that session on the server.
 *
 * The token is revoked server-side, not merely dropped here — until
 * `POST /api/auth/logout` existed a token stayed valid for its full hour
 * whatever the holder did, which is the wrong default on a shared unit
 * terminal.
 * @param {string} role Which role's session to end.
 * @returns {Promise<Object>} The logout response.
 */
export async function logout(role) {
  const held = sessions[role];
  if (!held || !held.token) return { revoked: false, detail: "not signed in" };
  const response = await fetch(API_BASE + "/api/auth/logout", {
    method: "POST",
    headers: headers(role),
  });
  const body = await response.json().catch(() => ({}));
  delete sessions[role];
  return body;
}

/**
 * Perform a POST with a JSON body.
 * @param {string} path Request path.
 * @param {Object} payload Body to send.
 * @param {string} [role] Which role's token to send.
 * @returns {Promise<Object>} Parsed response body.
 */
export async function post(path, payload, role) {
  const response = await fetch(API_BASE + path, {
    method: "POST",
    headers: headers(role),
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.detail || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return body;
}

export const api = {
  login,
  logout,
  isSignedIn,
  meta: () => get("/api/meta"),
  health: () => get("/api/health"),
  demoIdentities: () => get("/api/demo/identities"),

  personal: {
    summary: (id) => get(`/api/personal/${id}/summary`, "personnel"),
    history: (id) => get(`/api/personal/${id}/history`, "personnel"),
    checkIn: (id) => get(`/api/personal/${id}/check-in`, "personnel"),
    submitCheckIn: (id, answers) =>
      post(`/api/personal/${id}/check-in`, { answers }, "personnel"),
    /** Past submissions, with the question text resolved, plus the comparison. */
    checkInHistory: (id) => get(`/api/personal/${id}/check-in/history`, "personnel"),
    privacy: (id) => get(`/api/personal/${id}/privacy`, "personnel"),
    notifications: (id) => get(`/api/personal/${id}/notifications`, "personnel"),
  },

  officer: {
    /**
     * The prioritised queue.
     * @param {boolean} [showAll] Pass true to lift the capacity cap and get
     *   every case the escalation rule admits. The default response reports
     *   `total_eligible` and `held_back_count` so the UI can offer this.
     */
    queue: (showAll) =>
      get(`/api/officer/queue${showAll ? "?all=1" : ""}`, "welfare_officer"),
    case: (id) => get(`/api/officer/case/${id}`, "welfare_officer"),
    /**
     * The automatic counterfactual sweep for one case.
     *
     * Answers what would *change* the score. The `contributing_factors` on the
     * case detail answer what *built* it. With a non-linear model those lists
     * can disagree — render them as two separate panels with their own
     * headings, never merged into one "top factors" list.
     */
    counterfactual: (id) =>
      get(`/api/officer/case/${id}/counterfactual`, "welfare_officer"),
    /**
     * Record a welfare action taken on a case.
     * @param {string} id Pseudonym.
     * @param {string} interventionId An id from the case's recommendations.
     * @param {string} status One of offered | arranged | completed | not_pursued.
     * @param {string} [note] Short context for the next officer.
     */
    recordIntervention: (id, interventionId, status, note) =>
      post(
        `/api/officer/case/${id}/intervention`,
        { intervention_id: interventionId, status, note },
        "welfare_officer"
      ),
    whatIf: (id, adjustments) =>
      post(
        "/api/officer/what-if",
        { pseudonym_id: id, adjustments },
        "welfare_officer"
      ),
  },

  commander: {
    units: () => get("/api/commander/units", "commander"),
    nearMisses: () => get("/api/commander/near-misses", "commander"),
  },

  /**
   * The medical booking domain.
   *
   * A separate trust boundary, and the client has to respect it too: sign in
   * for these calls with a **service identity** as the subject
   * (`login("personnel", pw, "P00123")`), not a welfare pseudonym. The server
   * refuses a pseudonym here by design — see backend/medical/README.md — so
   * passing the wrong one produces a 403 rather than an empty list.
   *
   * There is no welfare_officer or commander entry point in this domain. That
   * is not an omission.
   */
  medical: {
    doctors: () => get("/api/medical/doctors", "personnel"),
    slots: (doctorId) =>
      get(
        `/api/medical/slots${doctorId ? `?doctor_id=${encodeURIComponent(doctorId)}` : ""}`,
        "personnel"
      ),
    /**
     * Book a slot.
     * @param {string} slotId Which slot.
     * @param {Object} [options] `reason`, and the consent pair.
     * @param {boolean} [options.shareContext] Must be explicitly true to share
     *   anything with the doctor. Render it unchecked every time — it is a
     *   per-appointment decision, not a remembered preference.
     * @param {string} [options.contextNote] What to share. Discarded server-side
     *   unless shareContext is true.
     */
    book: (slotId, options = {}) =>
      post(
        "/api/medical/appointments",
        {
          slot_id: slotId,
          reason: options.reason || "",
          share_context: Boolean(options.shareContext),
          context_note: options.contextNote || "",
        },
        "personnel"
      ),
    appointments: () => get("/api/medical/appointments", "personnel"),
    cancel: (appointmentId) =>
      post(`/api/medical/appointments/${appointmentId}/cancel`, {}, "personnel"),
    prescription: (prescriptionId) =>
      get(`/api/medical/prescriptions/${prescriptionId}`, "personnel"),

    /** Medical officer only. */
    schedule: () => get("/api/medical/schedule", "medical_officer"),
    writeNote: (appointmentId, noteText) =>
      post(
        `/api/medical/appointments/${appointmentId}/prescription`,
        { note_text: noteText },
        "medical_officer"
      ),

    /** Establishment admin only. */
    upsertDoctor: (doctor) =>
      post("/api/medical/doctors", doctor, "establishment_admin"),
    publishSlot: (slot) => post("/api/medical/slots", slot, "establishment_admin"),
  },
};
