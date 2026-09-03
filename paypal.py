"""Thin wrapper around PayPal's REST Subscriptions API (v1).

All the pieces that need real PayPal credentials (PAYPAL_CLIENT_ID,
PAYPAL_CLIENT_SECRET) and a pre-created billing plan (PAYPAL_PLAN_ID, made
once via setup_paypal.py) live behind the functions below, so server.py
never talks to PayPal's HTTP API directly.

PAYPAL_MODE controls which PayPal environment we hit: "sandbox" (default,
for testing with fake PayPal test accounts) or "live" (real money).
"""
import os
import httpx

PAYPAL_MODE = os.environ.get("PAYPAL_MODE", "sandbox")
PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
PAYPAL_CLIENT_SECRET = os.environ.get("PAYPAL_CLIENT_SECRET")
PAYPAL_WEBHOOK_ID = os.environ.get("PAYPAL_WEBHOOK_ID")

BASE_URL = "https://api-m.sandbox.paypal.com" if PAYPAL_MODE != "live" else "https://api-m.paypal.com"


def configured() -> bool:
    return bool(PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET)


def get_access_token() -> str:
    if not configured():
        raise RuntimeError("PAYPAL_CLIENT_ID/PAYPAL_CLIENT_SECRET are not set")
    resp = httpx.post(
        f"{BASE_URL}/v1/oauth2/token",
        auth=(PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET),
        data={"grant_type": "client_credentials"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {get_access_token()}", "Content-Type": "application/json"}


def create_product(name: str, description: str = "") -> str:
    """One-off: creates a PayPal "product" that plans are attached to. Used
    only by setup_paypal.py, not called at request time."""
    resp = httpx.post(
        f"{BASE_URL}/v1/catalogs/products",
        headers=_auth_headers(),
        json={"name": name, "description": description, "type": "SERVICE", "category": "SOFTWARE"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def create_plan(product_id: str, name: str, price_usd: str, interval_unit: str = "MONTH") -> str:
    """One-off: creates a billing plan (recurring price) under a product.
    Used only by setup_paypal.py — the resulting plan id is then stored as
    PAYPAL_PLAN_ID and reused for every subscriber."""
    resp = httpx.post(
        f"{BASE_URL}/v1/billing/plans",
        headers=_auth_headers(),
        json={
            "product_id": product_id,
            "name": name,
            "billing_cycles": [{
                "frequency": {"interval_unit": interval_unit, "interval_count": 1},
                "tenure_type": "REGULAR",
                "sequence": 1,
                "total_cycles": 0,  # 0 = infinite, until cancelled
                "pricing_scheme": {"fixed_price": {"value": price_usd, "currency_code": "USD"}},
            }],
            "payment_preferences": {
                "auto_bill_outstanding": True,
                "payment_failure_threshold": 2,
            },
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def create_subscription(plan_id: str, return_url: str, cancel_url: str, custom_id: str) -> dict:
    """Starts a subscription for one user. Returns the PayPal subscription id
    plus the "approve" URL the user must be redirected to so they can
    authorize the recurring payment on PayPal's own site."""
    resp = httpx.post(
        f"{BASE_URL}/v1/billing/subscriptions",
        headers=_auth_headers(),
        json={
            "plan_id": plan_id,
            "custom_id": custom_id,
            "application_context": {
                "return_url": return_url,
                "cancel_url": cancel_url,
                "user_action": "SUBSCRIBE_NOW",
                "brand_name": "T-Shirt Mockup",
            },
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    approve_url = next((l["href"] for l in data["links"] if l["rel"] == "approve"), None)
    return {"id": data["id"], "approve_url": approve_url, "status": data["status"]}


def get_subscription(subscription_id: str) -> dict:
    resp = httpx.get(
        f"{BASE_URL}/v1/billing/subscriptions/{subscription_id}",
        headers=_auth_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def cancel_subscription(subscription_id: str, reason: str = "User requested cancellation"):
    resp = httpx.post(
        f"{BASE_URL}/v1/billing/subscriptions/{subscription_id}/cancel",
        headers=_auth_headers(),
        json={"reason": reason},
        timeout=15,
    )
    # PayPal returns 204 No Content on success
    if resp.status_code not in (204, 200):
        resp.raise_for_status()


def verify_webhook_signature(headers: dict, body: bytes) -> bool:
    """Confirms a webhook actually came from PayPal (not a forged POST from
    anywhere on the internet) before we trust it to change a user's
    subscription status. Requires PAYPAL_WEBHOOK_ID (from the PayPal
    developer dashboard's webhook subscription page)."""
    if not PAYPAL_WEBHOOK_ID:
        return False
    import json
    payload = {
        "auth_algo": headers.get("paypal-auth-algo"),
        "cert_url": headers.get("paypal-cert-url"),
        "transmission_id": headers.get("paypal-transmission-id"),
        "transmission_sig": headers.get("paypal-transmission-sig"),
        "transmission_time": headers.get("paypal-transmission-time"),
        "webhook_id": PAYPAL_WEBHOOK_ID,
        "webhook_event": json.loads(body),
    }
    resp = httpx.post(
        f"{BASE_URL}/v1/notifications/verify-webhook-signature",
        headers=_auth_headers(),
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("verification_status") == "SUCCESS"
