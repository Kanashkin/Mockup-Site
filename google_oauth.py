"""Minimal Google OAuth2 "Sign in with Google" flow (authorization-code
grant), implemented directly against Google's endpoints via httpx so we
don't need to add a whole OAuth client library for one provider.

Requires GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET env vars, created in the
Google Cloud Console (APIs & Services -> Credentials -> OAuth client ID,
type "Web application") with the site's own /api/auth/google/callback URL
added under "Authorized redirect URIs".
"""
import os
import secrets
from urllib.parse import urlencode

import httpx

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def configured() -> bool:
    return bool(os.environ.get("GOOGLE_CLIENT_ID") and os.environ.get("GOOGLE_CLIENT_SECRET"))


def new_state() -> str:
    return secrets.token_urlsafe(24)


def authorize_url(redirect_uri: str, state: str) -> str:
    params = {
        "client_id": os.environ["GOOGLE_CLIENT_ID"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def exchange_code(code: str, redirect_uri: str) -> dict:
    resp = httpx.post(TOKEN_URL, data={
        "client_id": os.environ["GOOGLE_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_userinfo(access_token: str) -> dict:
    resp = httpx.get(USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"}, timeout=15)
    resp.raise_for_status()
    return resp.json()
