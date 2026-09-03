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
    privacy: (id) => get(`/api/personal/${id}/privacy`, "personnel"),
    notifications: (id) => get(`/api/personal/${id}/notifications`, "personnel"),
  },

  officer: {
    queue: () => get("/api/officer/queue", "welfare_officer"),
    case: (id) => get(`/api/officer/case/${id}`, "welfare_officer"),
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
};
