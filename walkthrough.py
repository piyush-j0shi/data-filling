"""
AgentCore Browser Walkthrough — LangGraph + Playwright + Bedrock

This script uses a LangGraph ReAct agent (powered by Bedrock via init_chat_model)
to drive a Playwright browser. It navigates to a React app, logs in, fills out
forms from a JSON file, and verifies submissions.

Architecture follows the AWS Bedrock AgentCore Browser pattern:
- Local mode (default): launches local Chromium for localhost development
- AgentCore mode (USE_AGENTCORE=true): connects to AWS-managed browser via CDP

Usage:
    # Start the React app first: npm run dev
    # Then run:
    export AWS_REGION=us-west-2
    python walkthrough.py
"""

import asyncio
import json
import os
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from contextlib import asynccontextmanager

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from playwright.async_api import async_playwright, Page

# ---------------------------------------------------------------------------
# Configuration (all via environment variables with sensible defaults)
# ---------------------------------------------------------------------------
USE_AGENTCORE = os.environ.get("USE_AGENTCORE", "false").lower() == "true"
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
APP_URL = os.environ.get("APP_URL", "http://localhost:5173")
FORM_DATA_FILE = os.environ.get("FORM_DATA_FILE", "form_data.json")
HEADLESS = os.environ.get("HEADLESS", "true").lower() == "true"
MODEL_ID = os.environ.get(
    "MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
)


# ---------------------------------------------------------------------------
# Layer 1: Browser Provider — local Playwright vs AgentCore remote browser
# ---------------------------------------------------------------------------
_page_holder: dict[str, Page | None] = {"page": None}


def set_page(page: Page):
    _page_holder["page"] = page


def get_page() -> Page:
    page = _page_holder["page"]
    if page is None:
        raise RuntimeError("Browser page not initialized. Call set_page() first.")
    return page


@asynccontextmanager
async def get_browser_page():
    """Yield a Playwright Page from either a local browser or AgentCore."""
    if USE_AGENTCORE:
        from bedrock_agentcore.tools.browser_client import browser_session

        pw = await async_playwright().start()
        with browser_session(AWS_REGION) as client:
            ws_url, headers = client.generate_ws_headers()
            browser = await pw.chromium.connect_over_cdp(ws_url, headers=headers)
            context = await browser.new_context(
                extra_http_headers={"ngrok-skip-browser-warning": "any-value"}
            )
            page = await context.new_page()
            
            try:
                yield page
            finally:
                await page.close()
                await browser.close()
    else:
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            yield page
        finally:
            await page.close()
            await context.close()
            await browser.close()
            await pw.stop()


# ---------------------------------------------------------------------------
# Layer 2: Playwright Tools — exposed to the LLM via @tool
# ---------------------------------------------------------------------------
@tool
async def navigate_to(url: str) -> str:
    """Navigate the browser to a specific URL. Use this to open the login page
    or any other page. Returns the page title after navigation."""
    page = get_page()
    try:
        await page.goto(url, wait_until="networkidle", timeout=15000)
        title = await page.title()
        return f"Navigated to {url}. Page title: '{title}'"
    except Exception as e:
        return f"Error navigating to {url}: {e}"


@tool
async def fill_field(selector: str, value: str) -> str:
    """Fill an input field or textarea with a value. The selector should be a
    CSS selector like 'input[name=\"email\"]' or 'textarea[name=\"message\"]'.
    Clears the field first, then sets the new value."""
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=5000)
        await page.fill(selector, value)
        return f"Filled '{selector}' with '{value}'"
    except Exception as e:
        return f"Error filling '{selector}': {e}"


@tool
async def click_element(selector: str) -> str:
    """Click an element on the page. The selector can be a CSS selector like
    'button[type=\"submit\"]' or text-based like 'button:has-text(\"Login\")'."""
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=5000)
        await page.click(selector)
        await page.wait_for_timeout(500)  # let React re-render
        return f"Clicked element: '{selector}'"
    except Exception as e:
        return f"Error clicking '{selector}': {e}"


@tool
async def get_page_text() -> str:
    """Get all visible text on the current page. Use this to verify what page
    you are on, read labels, check error messages, or confirm form submissions
    appeared in the table."""
    page = get_page()
    try:
        text = await page.inner_text("body")
        if len(text) > 3000:
            text = text[:3000] + "\n... (truncated)"
        return f"Page text:\n{text}"
    except Exception as e:
        return f"Error getting page text: {e}"


@tool
async def take_screenshot(filename: str = "screenshot.png") -> str:
    """Take a screenshot of the current page for debugging. Provide a
    descriptive filename like 'after_login.png'."""
    page = get_page()
    try:
        filepath = os.path.join(os.getcwd(), filename)
        await page.screenshot(path=filepath, full_page=True)
        return f"Screenshot saved to {filepath}"
    except Exception as e:
        return f"Error taking screenshot: {e}"


# ---------------------------------------------------------------------------
# Layer 3: LangGraph Agent — ReAct loop with Bedrock LLM
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


SYSTEM_PROMPT = SystemMessage(content="""You are a browser automation agent. You interact with a web application
using the provided tools. You will be given tasks such as logging in and filling forms.

Important guidelines:
- Always call get_page_text() after navigating or clicking to understand the current page.
- For the LOGIN page:
  Username input selector: 'form input:first-of-type'
  Password input selector: 'input[type="password"]'
  Login button selector:   'button[type="submit"]'
- For the FORM page (after login), fields have name attributes:
  'input[name="name"]'
  'input[name="email"]'
  'input[name="phone"]'
  'input[name="address"]'
  'textarea[name="message"]'
  Submit button: 'button[type="submit"]'
- After submitting each form entry, call get_page_text() to verify the data appears in the table.
- Process all form entries one at a time.
- When all entries are submitted and verified, report completion.""")


def build_agent_graph():
    """Build and compile the LangGraph ReAct agent."""
    llm = init_chat_model(
        MODEL_ID,
        model_provider="bedrock_converse",
        region_name=AWS_REGION,
        temperature=0,
        max_tokens=4096,
    )

    tools = [navigate_to, fill_field, click_element, get_page_text, take_screenshot]
    llm_with_tools = llm.bind_tools(tools)

    async def chatbot_node(state: AgentState):
        messages = [SYSTEM_PROMPT] + list(state["messages"])
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", chatbot_node)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile()


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def load_form_data(filepath: str) -> list[dict]:
    """Load form entries from the JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {filepath}, got {type(data).__name__}")
    return data


async def run_agent():
    """Start the browser, build the agent, and execute the form-filling task."""
    form_data = load_form_data(FORM_DATA_FILE)
    print(f"Loaded {len(form_data)} form entries from {FORM_DATA_FILE}")

    graph = build_agent_graph()

    async with get_browser_page() as page:
        set_page(page)
        mode = "AgentCore" if USE_AGENTCORE else "local Playwright"
        print(f"Browser started ({mode})")

        user_message = f"""Please perform the following tasks on the web application at {APP_URL}:

1. Navigate to {APP_URL}
2. Log in with username "admin" and password "admin"
3. After logging in, you should see a form page titled "Submit Form"
4. For each of the following form entries, fill in all 5 fields and click Submit:

{json.dumps(form_data, indent=2)}

5. After submitting ALL entries, use get_page_text() to verify that all {len(form_data)} entries appear in the submissions table.
6. Report what you see in the table to confirm success."""

        print("Starting agent execution...\n")
        result = await graph.ainvoke(
            {"messages": [("user", user_message)]},
        )

        final_message = result["messages"][-1]
        print("\n" + "=" * 60)
        print("AGENT FINAL RESPONSE:")
        print("=" * 60)
        print(final_message.content)

        await page.screenshot(path="final_result.png", full_page=True)
        print("\nFinal screenshot saved to final_result.png")


if __name__ == "__main__":
    asyncio.run(run_agent())
