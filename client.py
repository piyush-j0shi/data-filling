import logging
from contextlib import asynccontextmanager

from playwright.async_api import async_playwright, Page
from bedrock_agentcore.tools.browser_client import browser_session

logger = logging.getLogger(__name__)

_page_holder: dict[str, Page | None] = {"page": None}


def set_page(page: Page):
    _page_holder["page"] = page


def get_page() -> Page:
    if _page_holder["page"] is None:
        raise RuntimeError("Browser session not active. Call set_page first.")
    return _page_holder["page"]


@asynccontextmanager
async def get_bedrock_browser(aws_region: str, browser_id: str):
    logger.info("Connecting to Bedrock Browser")
    with browser_session(aws_region, identifier=browser_id) as client:
        ws_url, headers = client.generate_ws_headers()
        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(ws_url, headers=headers)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            await context.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "ngrok-skip-browser-warning": "true"
            })
            page = context.pages[0] if context.pages else await context.new_page()
            try:
                yield page
            finally:
                await page.close()
                await browser.close()
                logger.info("Browser session ended")
