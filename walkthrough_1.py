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
    """Navigate the browser to a URL."""
    page = get_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)
        return "Done"
    except Exception as e:
        return f"Failed: {e}"

@tool
async def get_page_html() -> str:
    """Returns visible interactive elements on the current page."""
    page = get_page()
    try:
        elements = await page.evaluate("""() => {
            const results = [];
            const seen = new Set();

            // Standard interactive tags
            const tagSelector = 'input, button, a, select, textarea, label, form, option';
            // ARIA role-based elements (catches framework components like mat-select, ant-dropdown, etc.)
            const roleSelector = '[role="button"], [role="combobox"], [role="listbox"], [role="option"], [role="textbox"], [role="link"], [role="menuitem"], [role="tab"], [role="checkbox"], [role="radio"], [role="switch"], [role="slider"]';
            // Other interactive patterns
            const interactiveSelector = '[contenteditable="true"], [onclick], [data-testid], [tabindex]:not([tabindex="-1"])';

            const combined = `${tagSelector}, ${roleSelector}, ${interactiveSelector}`;
            const attrs = ['id', 'name', 'type', 'placeholder', 'value', 'href', 'for',
                           'role', 'aria-label', 'data-testid', 'data-field',
                           'contenteditable', 'action', 'method'];

            for (const el of document.querySelectorAll(combined)) {
                // Visibility check: skip elements hidden by CSS or zero-size
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
                if (el.offsetWidth === 0 && el.offsetHeight === 0 && el.tagName !== 'INPUT' && el.tagName !== 'OPTION') continue;

                // Dedupe by DOM node
                if (seen.has(el)) continue;
                seen.add(el);

                const tag = el.tagName.toLowerCase();
                const info = {tag: tag};

                for (const a of attrs) {
                    const v = el.getAttribute(a);
                    if (v && v.trim()) info[a] = v.trim().substring(0, 80);
                }

                // Mark disabled/readonly state
                if (el.disabled) info.disabled = 'true';
                if (el.readOnly) info.readonly = 'true';

                const text = el.textContent?.trim().substring(0, 60);
                if (text) info.text = text;

                results.push(info);
            }
            return results;
        }""")
        if not elements:
            return "No interactive elements found on the page."
        lines = []
        for el in elements:
            tag = el.pop('tag')
            parts = [f"<{tag}"]
            for k, v in el.items():
                if k != 'text':
                    parts.append(f'{k}="{v}"')
            text = el.get('text', '')
            tag_str = ' '.join(parts) + '>'
            if text:
                tag_str += text
            lines.append(tag_str)
        result = '\n'.join(lines)
        if len(result) > 4000:
            result = result[:4000] + '\n... (truncated)'
        return result
    except Exception as e:
        return f"Failed to get page HTML: {e}"

@tool
async def get_page_text() -> str:
    """Returns visible text on the page."""
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
    """Fill a form field by CSS selector with a value."""
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=5000, state="visible")
        await page.fill(selector, value, timeout=5000)
        await page.wait_for_timeout(200)
        return f"Filled {selector}"
    except Exception as e:
        return f"Failed {selector}: {e}"

@tool
async def click_element(selector: str) -> str:
    """Click an element by CSS selector."""
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=5000, state="visible")
        await page.click(selector, timeout=5000)
        await page.wait_for_timeout(1000)
        return f"Clicked {selector}"
    except Exception as e:
        return f"Failed {selector}: {e}"

@tool
async def select_option(selector: str, value: str) -> str:
    """Select a dropdown option by CSS selector."""
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=5000, state="visible")
        try:
            await page.select_option(selector, value=value, timeout=3000)
        except Exception:
            await page.select_option(selector, label=value, timeout=3000)
        await page.wait_for_timeout(200)
        return f"Selected {value} in {selector}"
    except Exception as e:
        return f"Failed {selector}: {e}"

@tool
async def press_key(key: str) -> str:
    """Press a keyboard key (Enter, Tab, Escape, etc)."""
    page = get_page()
    try:
        await page.keyboard.press(key)
        await page.wait_for_timeout(500)
        return f"Pressed {key}"
    except Exception as e:
        return f"Failed: {e}"

@tool
async def wait_for_element(selector: str, timeout: int = 10000) -> str:
    """Wait for an element by CSS selector to appear."""
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=timeout, state="visible")
        return f"Visible: {selector}"
    except Exception as e:
        return f"Timeout {selector}: {e}"

SYSTEM_PROMPT = """You are a browser automation agent. You can work with any web application.

RULES:
1. ALWAYS call get_page_html first to discover the page structure before interacting with any elements.
2. NEVER guess CSS selectors. Only use selectors you found from get_page_html.
3. Build selectors from id, name, data-testid, or role attributes. Prefer #id or [name="x"] over class-based selectors.
4. Map data fields to form fields by meaning, not exact name match.
5. Submit form entries ONE AT A TIME. Wait for each to complete before the next.
6. After clicking submit or navigating, call get_page_html again to see the updated page.
7. For multi-step forms, fill visible fields, click Next or Submit, then read the new page.
8. Elements marked disabled="true" or readonly="true" cannot be interacted with — skip them.
9. For custom dropdowns (role="combobox" or role="listbox"), click to open, then click the role="option" element."""

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
        all_msgs = list(state["messages"])
        if len(all_msgs) > 16:
            all_msgs = all_msgs[:1] + all_msgs[-14:]
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + all_msgs
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

ERROR_KEYWORDS = ["failed", "error", "could not", "unable", "not found", "timeout", "exception"]
MAX_RETRIES = 1

def _detect_failure(message) -> str | None:
    content = message.content if hasattr(message, 'content') else str(message)
    content_lower = content.lower()
    for keyword in ERROR_KEYWORDS:
        if keyword in content_lower:
            return content
    return None

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

        username = os.environ.get("LOGIN_USERNAME", "admin")
        password = os.environ.get("LOGIN_PASSWORD", "admin")

        login_message = f"""Please automate the following workflow on the web application:

Application URL: {APP_URL}
Login credentials: username="{username}", password="{password}"

Tasks:
1. Navigate to the application URL
2. Read the page to understand the login form, then log in with the given credentials
3. After logging in, read the page to understand the form structure and confirm you are logged in"""

        print("Phase 1: Logging in...")
        print("-" * 70)
        result = await graph.ainvoke(
            {"messages": [("user", login_message)]},
            {"recursion_limit": 30}
        )
        current_messages = list(result["messages"])
        print("-" * 70)

        results = []

        for i, entry in enumerate(form_data, 1):
            fields = ', '.join(f'{k}="{v}"' for k, v in entry.items())
            print(f"\nEntry {i}/{len(form_data)}: {fields}")
            print("-" * 70)

            entry_message = f"""Now submit the following data entry into the form:
{fields}

Steps:
1. First call get_page_html to see the current form
2. Fill in each field using the appropriate tool
3. Click the submit button
4. Confirm the submission was successful by reading the page after submit"""

            success = False
            failure_reason = ""

            for attempt in range(1 + MAX_RETRIES):
                if attempt > 0:
                    print(f"  Retrying entry {i} (attempt {attempt + 1})...")

                try:
                    entry_result = await graph.ainvoke(
                        {"messages": current_messages + [("user", entry_message)]},
                        {"recursion_limit": 30}
                    )
                    final_msg = entry_result["messages"][-1]
                    failure_reason = _detect_failure(final_msg)

                    if failure_reason is None:
                        success = True
                        current_messages = list(entry_result["messages"])
                        break
                    else:
                        print(f"  Entry {i} attempt {attempt + 1} detected issue in response")
                        if attempt < MAX_RETRIES:
                            current_messages = list(entry_result["messages"])

                except Exception as e:
                    failure_reason = str(e)
                    print(f"  Entry {i} attempt {attempt + 1} raised exception: {e}")

            if success:
                print(f"  Entry {i}: SUCCESS")
                results.append({"entry": i, "data": entry, "status": "success"})
            else:
                print(f"  Entry {i}: FAILED")
                results.append({"entry": i, "data": entry, "status": "failed", "reason": failure_reason})

            print("-" * 70)

        print("\n" + "=" * 70)
        print("SUBMISSION SUMMARY")
        print("=" * 70)

        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]

        print(f"Total entries : {len(results)}")
        print(f"Successful : {len(successful)}")
        print(f"Failed : {len(failed)}")

        if failed:
            print(f"\n{'─' * 70}")
            print("FAILED ENTRIES:")
            print(f"{'─' * 70}")
            for r in failed:
                fields = ', '.join(f'{k}="{v}"' for k, v in r["data"].items())
                print(f"\n  Entry {r['entry']}: {fields}")
                reason = r["reason"] or "Unknown error"
                
                if len(reason) > 300:
                    reason = reason[:300] + "..."
                print(f"  Reason: {reason}")

        print("=" * 70 + "\n")

        await page.wait_for_timeout(1000)
        await page.screenshot(path="final_result.png", full_page=True)
        print("Screenshot saved to final_result.png")

if __name__ == "__main__":
    asyncio.run(run_agent())
