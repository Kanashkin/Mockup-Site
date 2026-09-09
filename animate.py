"""Thin wrapper around PiAPI's Kling image-to-video endpoint ("Oживить" —
turns a rendered mockup photo into a short AI video).

Uses PiAPI (https://piapi.ai) rather than Kling's own API directly: Kling's
native API uses JWT-signed requests and is aimed at businesses registered in
China, which makes it impractical to get direct access to quickly. PiAPI
re-sells the same Kling models behind a plain API-key REST API at a modest
markup — much simpler to integrate, and swappable later if direct Kling
access becomes worthwhile.

Requires PIAPI_API_KEY (get one at https://piapi.ai — no waitlist as of
writing). Everything here is a no-op/raises clearly if it's unset, so the
rest of the app can check `configured()` and disable the "Oживить" button
instead of erroring.
"""
import os
import httpx

PIAPI_API_KEY = os.environ.get("PIAPI_API_KEY")
BASE_URL = "https://api.piapi.ai/api/v1"

# 5-second clips only for now — matches the $1.99 price point this was
# costed against (see server.py ANIMATE_PRICE_USD). Longer clips cost
# noticeably more on PiAPI's end and would need a different price.
CLIP_DURATION_SECONDS = 5


def configured() -> bool:
    return bool(PIAPI_API_KEY)


def _headers() -> dict:
    return {"X-API-Key": PIAPI_API_KEY, "Content-Type": "application/json"}


def submit_animation(image_url: str, prompt: str = "") -> str:
    """Starts a Kling image-to-video task for the image at image_url (must
    be a publicly-fetchable URL — PiAPI's servers download it, they don't
    accept raw image bytes). Returns PiAPI's task_id for polling."""
    if not configured():
        raise RuntimeError("PIAPI_API_KEY is not set")
    resp = httpx.post(
        f"{BASE_URL}/task",
        headers=_headers(),
        json={
            "model": "kling",
            "task_type": "video_generation",
            "input": {
                "version": "3.0",
                "mode": "std",
                "image_url": image_url,
                "prompt": prompt or "subtle natural movement, gentle breathing, fabric moving slightly in a light breeze, realistic, photorealistic",
                "negative_prompt": "distortion, warping, extra limbs, blurry, low quality",
                "duration": CLIP_DURATION_SECONDS,
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") not in (200, 0, None):
        raise RuntimeError(f"PiAPI task creation failed: {data}")
    return data["data"]["task_id"]


def get_task(task_id: str) -> dict:
    """Polls task status. Returns {"status": "pending"|"processing"|"completed"|"failed",
    "video_url": str|None, "error": str|None}."""
    resp = httpx.get(f"{BASE_URL}/task/{task_id}", headers=_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()["data"]
    status = data.get("status", "pending")
    video_url = None
    if status == "completed":
        video_url = (data.get("output") or {}).get("video_url")
    error = None
    if status == "failed":
        error = (data.get("error") or {}).get("message") or "Video generation failed"
    return {"status": status, "video_url": video_url, "error": error}
