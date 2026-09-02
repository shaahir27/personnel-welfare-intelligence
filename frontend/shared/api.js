/**
 * API client shared by both frontends.
 *
 * One job: talk to the pwiews API and carry the acting role with every
 * request.
 *
 * Note on the role header: this build identifies the caller by an
 * `X-Pwiews-Role` header rather than a verified token. Authentication was
 * deferred in this pass, and this file is where the token would go when it is
 * added -- `headers()` is the only place a credential is attached.
 *
 * The point worth understanding: the header does not *grant* anything. The
 * server enforces every scope rule itself. Setting the header to "commander"
 * in a browser console gets you the commander view, which contains no
 * individual data because the server will not build a commander payload that
 * has any. That is the difference between hiding fields in a UI and enforcing
 * them on a server.
 */

const API_BASE = "";

/**
 * Build request headers for the acting principal.
 * @param {string} role One of "personnel", "welfare_officer", "commander".
 * @param {string} [subject] The pseudonym this principal is acting as.
 * @returns {Object} Header map.
 */
function headers(role, subject) {
  const h = { "Content-Type": "application/json", "X-Pwiews-Role": role };
  if (subject) h["X-Pwiews-Subject"] = subject;
  return h;
}

/**
 * Perform a GET and parse JSON, surfacing server messages as thrown errors.
 * @param {string} path Request path.
 * @param {string} role Acting role.
 * @param {string} [subject] Acting subject pseudonym.
 * @returns {Promise<Object>} Parsed response body.
 */
export async function get(path, role, subject) {
  const response = await fetch(API_BASE + path, { headers: headers(role, subject) });
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
 * Perform a POST with a JSON body.
 * @param {string} path Request path.
 * @param {Object} payload Body to send.
 * @param {string} role Acting role.
 * @param {string} [subject] Acting subject pseudonym.
 * @returns {Promise<Object>} Parsed response body.
 */
export async function post(path, payload, role, subject) {
  const response = await fetch(API_BASE + path, {
    method: "POST",
    headers: headers(role, subject),
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
  meta: () => get("/api/meta", "personnel"),
  health: () => get("/api/health", "personnel"),
  demoIdentities: () => get("/api/demo/identities", "personnel"),

  personal: {
    summary: (id) => get(`/api/personal/${id}/summary`, "personnel", id),
    history: (id) => get(`/api/personal/${id}/history`, "personnel", id),
    checkIn: (id) => get(`/api/personal/${id}/check-in`, "personnel", id),
    privacy: (id) => get(`/api/personal/${id}/privacy`, "personnel", id),
  },

  officer: {
    queue: () => get("/api/officer/queue", "welfare_officer"),
    case: (id) => get(`/api/officer/case/${id}`, "welfare_officer"),
    whatIf: (id, adjustments) =>
      post("/api/officer/what-if", { pseudonym_id: id, adjustments }, "welfare_officer"),
  },

  commander: {
    units: () => get("/api/commander/units", "commander"),
    nearMisses: () => get("/api/commander/near-misses", "commander"),
  },
};
