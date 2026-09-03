/**
 * Demo sign-in.
 *
 * One job: hold the demo account names in one place and sign the apps in.
 *
 * These credentials are published — they are also in README.md and their PBKDF2
 * hashes are in `backend/auth/demo_accounts.json`. That is deliberate and it is
 * the honest state of a demonstration build: the corpus is 800 synthetic people
 * and a reviewer has to be able to open both apps without being issued an
 * account. What matters is that the server no longer takes a client's word for
 * which role it holds — it checks a signature.
 *
 * A deployment deletes this file. The apps would then show a sign-in form and
 * call `api.login()` with what the person typed; nothing else about the
 * request path changes, because every screen already goes through the token in
 * `api.js`.
 */

/** Officer account: fixed subject, welfare_officer role. */
export const OFFICER = { username: "officer", password: "welfare-officer-demo" };

/** Commander account: fixed subject, commander role. */
export const COMMANDER = { username: "commander", password: "commander-demo" };

/** Personnel account: chooses which synthetic person it acts as at login. */
export const PERSONNEL = { username: "personnel", password: "personnel-demo" };

/**
 * Sign in as a given pseudonym for the personal app.
 * @param {Object} api The api module.
 * @param {string} pseudonymId Which synthetic person to act as.
 * @returns {Promise<Object>} The login response.
 */
export function signInAsPersonnel(api, pseudonymId) {
  return api.login(PERSONNEL.username, PERSONNEL.password, pseudonymId);
}

/**
 * Sign in as both dashboard roles.
 * @param {Object} api The api module.
 * @returns {Promise<Array>} Both login responses.
 */
export function signInForDashboard(api) {
  return Promise.all([
    api.login(OFFICER.username, OFFICER.password),
    api.login(COMMANDER.username, COMMANDER.password),
  ]);
}
