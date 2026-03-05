import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from playwright.async_api import async_playwright, Page
from bedrock_agentcore.tools.browser_client import BrowserClient, DEFAULT_SESSION_TIMEOUT

logger = logging.getLogger(__name__)

_page_holder: dict[str, Page | None] = {"page": None}
_client_holder: dict[str, BrowserClient | None] = {"client": None}


def set_page(page: Page):
    _page_holder["page"] = page


def get_page() -> Page:
    if _page_holder["page"] is None:
        raise RuntimeError("Browser session not active. Call set_page first.")
    return _page_holder["page"]


def save_browser_profile(profile_id: str) -> None:                            
    bc = _client_holder["client"]
    if bc is None:
        raise RuntimeError("No active browser client. Must be called inside get_bedrock_browser context.")
    bc.data_plane_client.save_browser_session_profile(
        profileIdentifier=profile_id,
        browserIdentifier=bc.identifier,
        sessionId=bc.session_id,
    )
    logger.info("Browser session saved to profile: %s", profile_id)


@asynccontextmanager
async def get_bedrock_browser(aws_region: str, browser_id: str, profile_id: Optional[str] = None):
    logger.info("Connecting to Bedrock Browser")
    bc = BrowserClient(aws_region)

    if profile_id:
        logger.info("Starting session with profile: %s", profile_id)
        response = bc.data_plane_client.start_browser_session(
            browserIdentifier=browser_id,
            name=f"browser-session-{uuid.uuid4().hex[:8]}",
            sessionTimeoutSeconds=DEFAULT_SESSION_TIMEOUT,
            profileConfiguration={"profileIdentifier": profile_id},
        )
        bc.identifier = response["browserIdentifier"]
        bc.session_id = response["sessionId"]
        logger.info("Session started (with profile): %s", bc.session_id)
    else:
        bc.start(identifier=browser_id)

    _client_holder["client"] = bc
    ws_url, headers = bc.generate_ws_headers()

    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp(ws_url, headers=headers)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        await context.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "ngrok-skip-browser-warning": "true",
        })
        page = context.pages[0] if context.pages else await context.new_page()
        try:
            yield page
        finally:
            await page.close()
            await browser.close()
            bc.stop()
            _page_holder["page"] = None
            _client_holder["client"] = None
            logger.info("Browser session ended")
