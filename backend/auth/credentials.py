"""Demo credential store for the login route.

One job: check a username and password against stored hashes, and say which
role the account holds.

Why this exists
---------------
``jwt_handler`` could always issue and verify tokens, but nothing could obtain
one: there was no login route, because there were no credentials to validate
against. So both frontends kept sending ``X-Pwiews-Role`` and the whole JWT
layer sat unused. This module is the missing half.

What it is and is not
---------------------
It is a real password check: PBKDF2-HMAC-SHA256, 200,000 iterations, a distinct
random salt per account, compared with ``hmac.compare_digest``. No password is
stored or recoverable, and an unknown username costs the same time as a known
one with a wrong password.

It is not a user management system. There are three accounts, they are fixed at
build time, and the passwords are published in the README because this is a
demonstration corpus of synthetic people. A deployment replaces this module
with the force's own directory service; the login route above it only needs
``authenticate()`` to keep returning an :class:`Account`, so that swap does not
reach any other file.

The personnel account and its subject
-------------------------------------
Officer and commander accounts are service accounts: their subject is fixed by
the account. The personnel account is different -- it may act as any pseudonym,
because the demo has 800 synthetic people and no real sign-ups, and the app
needs some way to look at more than one of them. That capability is carried on
the account as ``may_choose_subject`` rather than being implied anywhere in the
route, so it is greppable, it is off for every other account, and turning the
demo into a deployment means deleting one account rather than finding a special
case buried in a handler.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from backend.auth.rbac import AuthorisationError
from backend.config import settings

ACCOUNTS_PATH: Path = Path(__file__).resolve().parent / "demo_accounts.json"

# PBKDF2 work factor. SOURCE: OWASP Password Storage Cheat Sheet recommends at
# least 600,000 iterations for PBKDF2-HMAC-SHA256 as of its current revision;
# 200,000 is used here because these are published demo passwords guarding
# synthetic data and the login must stay responsive on a laptop during a
# demonstration. A deployment should raise it and is told so in the README.
PBKDF2_ITERATIONS = 200_000


@dataclass(frozen=True)
class Account:
    """One authenticated account.

    Attributes:
        username: The login name.
        role: One of ``settings.ROLES``.
        subject: The pseudonym or service identifier this account acts as.
            Empty for accounts that choose their subject at login.
        may_choose_subject: Whether login may set the subject. True only for
            the demo personnel account.
    """

    username: str
    role: str
    subject: str = ""
    may_choose_subject: bool = False


def _load_accounts() -> Dict[str, Dict]:
    """Load the account file.

    Returns:
        Mapping of username to the stored account record.

    Raises:
        FileNotFoundError: If the account file is missing.
    """
    if not ACCOUNTS_PATH.exists():
        raise FileNotFoundError(
            f"Demo account file not found at {ACCOUNTS_PATH}. "
            "It should be committed alongside this module."
        )
    return json.loads(ACCOUNTS_PATH.read_text(encoding="utf-8"))["accounts"]


# Loaded once at import: the file is static between deployments.
_ACCOUNTS: Dict[str, Dict] = _load_accounts()

# Hashed once at import so that authenticating an unknown username does the same
# PBKDF2 work as authenticating a known one. Without this, a caller can tell a
# real username from a fake one by timing the response.
_DUMMY_SALT = bytes.fromhex("00" * 16)


def _derive(password: str, salt: bytes) -> bytes:
    """Derive the PBKDF2 hash of a password.

    Args:
        password: The submitted password.
        salt: Per-account salt.

    Returns:
        The derived 32-byte key.
    """
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )


def authenticate(username: str, password: str) -> Account:
    """Check a username and password.

    Args:
        username: The submitted login name.
        password: The submitted password.

    Returns:
        The :class:`Account` when the credentials are correct.

    Raises:
        AuthorisationError: If the username is unknown or the password is
            wrong. The message is identical either way -- telling a caller
            which of the two failed tells them which usernames exist.
    """
    record: Optional[Dict] = _ACCOUNTS.get(str(username).strip())

    if record is None:
        # Do the work anyway, then fail. Constant-ish time by construction.
        _derive(str(password), _DUMMY_SALT)
        raise AuthorisationError("invalid username or password")

    expected = bytes.fromhex(record["password_hash"])
    derived = _derive(str(password), bytes.fromhex(record["salt"]))
    if not hmac.compare_digest(derived, expected):
        raise AuthorisationError("invalid username or password")

    role = record["role"]
    if role not in settings.ROLES:
        raise AuthorisationError(f"account '{username}' holds unknown role {role!r}")

    return Account(
        username=str(username).strip(),
        role=role,
        subject=str(record.get("subject", "")),
        may_choose_subject=bool(record.get("may_choose_subject", False)),
    )
