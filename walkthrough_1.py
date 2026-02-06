import asyncio
import json
import os
import re

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import nest_asyncio
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from playwright.async_api import async_playwright, Page

from bedrock_agentcore.tools.browser_client import browser_session

load_dotenv()

FORM_DATA_FILE = "form_data.json"
APP_URL = os.environ.get("APP_URL", "https://your-public-ngrok-url.ngrok-free.app")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
BROWSER_ID = os.environ.get("BROWSER_ID", "your-bedrock-browser-id")

_page_holder: dict[str, Page | None] = {"page": None}

def set_page(page: Page):
    _page_holder["page"] = page

def get_page() -> Page:
    if _page_holder["page"] is None:
        raise RuntimeError("Browser session not active. Call set_page first.")
    return _page_holder["page"]

@asynccontextmanager
async def get_bedrock_browser():
    """Starts an AWS Bedrock browser session and connects via CDP with custom headers."""
    with browser_session(AWS_REGION, identifier=BROWSER_ID) as client:
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

@tool
async def navigate_to_url(url: str) -> str:
    """Navigates the browser to the specified URL."""
    page = get_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)
        return f"Navigated to {url}"
    except Exception as e:
        return f"Navigation failed: {e}"

@tool
async def get_page_html() -> str:
    """Returns cleaned HTML of the current page. Use this to discover form fields, buttons, and links."""
    page = get_page()
    try:
        html = await page.evaluate("""() => {
            const clone = document.body.cloneNode(true);
            clone.querySelectorAll('script, style, svg, noscript, link[rel="stylesheet"]').forEach(el => el.remove());
            return clone.innerHTML;
        }""")
        html = re.sub(r'\s+', ' ', html).strip()
        if len(html) > 15000:
            html = html[:15000] + "\n... (truncated)"
        return html
    except Exception as e:
        return f"Failed to get page HTML: {e}"

@tool
async def get_page_text() -> str:
    """Returns the visible text on the current page."""
    page = get_page()
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=5000)
        text = await page.inner_text("body", timeout=5000)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        if len(text) > 3000:
            text = text[:3000] + "\n... (truncated)"
        return text
    except Exception as e:
        return f"Failed to get page text: {e}"

@tool
async def fill_field(selector: str, value: str) -> str:
    """Fills a form field identified by a CSS selector with the given value."""
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=5000, state="visible")
        await page.fill(selector, value, timeout=5000)
        await page.wait_for_timeout(200)
        return f"Filled '{selector}' with '{value[:50]}{'...' if len(value) > 50 else ''}'"
    except Exception as e:
        return f"Failed to fill '{selector}': {e}"

@tool
async def click_element(selector: str) -> str:
    """Clicks an element identified by a CSS selector."""
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=5000, state="visible")
        await page.click(selector, timeout=5000)
        await page.wait_for_timeout(1000)
        return f"Clicked '{selector}'"
    except Exception as e:
        return f"Failed to click '{selector}': {e}"

@tool
async def select_option(selector: str, value: str) -> str:
    """Selects an option from a dropdown element identified by a CSS selector."""
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=5000, state="visible")
        try:
            await page.select_option(selector, value=value, timeout=3000)
        except Exception:
            await page.select_option(selector, label=value, timeout=3000)
        await page.wait_for_timeout(200)
        return f"Selected '{value}' in '{selector}'"
    except Exception as e:
        return f"Failed to select option in '{selector}': {e}"

@tool
async def press_key(key: str) -> str:
    """Presses a keyboard key such as Enter, Tab, or Escape."""
    page = get_page()
    try:
        await page.keyboard.press(key)
        await page.wait_for_timeout(500)
        return f"Pressed '{key}'"
    except Exception as e:
        return f"Failed to press '{key}': {e}"

@tool
async def wait_for_element(selector: str, timeout: int = 10000) -> str:
    """Waits for an element matching the CSS selector to appear on the page."""
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=timeout, state="visible")
        return f"Element '{selector}' is now visible"
    except Exception as e:
        return f"Timed out waiting for '{selector}': {e}"

SYSTEM_PROMPT = """You are a browser automation agent. You can work with any web application.

RULES:
1. ALWAYS call get_page_html first to discover the page structure before interacting with any elements.
2. NEVER guess CSS selectors. Only use selectors you found in the HTML.
3. Map data fields to form fields by meaning, not exact name match.
4. Submit form entries ONE AT A TIME. Wait for each to complete before the next.
5. After clicking submit or navigating, call get_page_html again to see the updated page.
6. For multi-step forms, fill visible fields, click Next or Submit, then read the new page."""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_automation_agent():
    llm = init_chat_model(
        model="llama-3.3-70b-versatile",
        model_provider="groq"
    )

    tools = [
        navigate_to_url,
        get_page_html,
        get_page_text,
        fill_field,
        click_element,
        select_option,
        press_key,
        wait_for_element,
    ]

    workflow = StateGraph(AgentState)

    async def call_model(state):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)

        if hasattr(response, 'tool_calls') and len(response.tool_calls) > 1:
            print(f"\n  LLM wanted {len(response.tool_calls)} tools, limiting to first:")
            first_tc = response.tool_calls[0]
            response = AIMessage(
                content=response.content,
                tool_calls=[first_tc],
            )

        if hasattr(response, 'tool_calls') and response.tool_calls:
            tc = response.tool_calls[0]
            args_str = ', '.join(
                f"{k}={v[:50] if isinstance(v, str) else v}"
                for k, v in tc.get('args', {}).items()
            )
            print(f"  Tool: {tc.get('name')}({args_str})")

        return {"messages": [response]}

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()

def load_form_data(filepath: str) -> list[dict]:
    with open(filepath, "r") as f:
        return json.load(f)

async def run_agent():
    nest_asyncio.apply()

    try:
        form_data = load_form_data(FORM_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {FORM_DATA_FILE} not found. Create it with a JSON list of entries.")
        return

    print(f"Loaded {len(form_data)} form entries to submit\n")
    graph = create_automation_agent()

    async with get_bedrock_browser() as page:
        set_page(page)
        print(f"Connected to Bedrock Browser\n")
        print("=" * 70)

        entries_desc = []
        for i, entry in enumerate(form_data, 1):
            fields = ', '.join(f'{k}="{v}"' for k, v in entry.items())
            entries_desc.append(f"  Entry {i}: {fields}")
            
        username = os.environ.get("LOGIN_USERNAME", "admin")
        password = os.environ.get("LOGIN_PASSWORD", "admin")

        user_message = f"""Please automate the following workflow on the web application:

Application URL: {APP_URL}
Login credentials: username="{username}", password="{password}"

Tasks:
1. Navigate to the application URL
2. Read the page to understand the login form, then log in with the given credentials
3. After logging in, read the page to understand the form structure
4. Submit these {len(form_data)} data entries ONE AT A TIME (wait for each to complete before starting the next):

{chr(10).join(entries_desc)}

5. After ALL entries are submitted, look for a way to view/verify the submissions (e.g., a "View Submissions" tab or page)
6. Read the page and report how many submissions are shown

IMPORTANT: Always read the page HTML first before interacting. Submit entries sequentially."""

        result = await graph.ainvoke(
            {"messages": [("user", user_message)]},
            {"recursion_limit": 100}
        )

        print("=" * 70)
        final_message = result["messages"][-1]
        print(final_message.content if hasattr(final_message, 'content') else str(final_message))
        print("=" * 70 + "\n")

        await page.wait_for_timeout(1000)
        await page.screenshot(path="final_result.png", full_page=True)
        print("Screenshot saved to final_result.png")

if __name__ == "__main__":
    asyncio.run(run_agent())
