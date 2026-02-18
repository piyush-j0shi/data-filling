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

# ── Only JS still needed: clicking the first visible suggestion ───────────────
_CLICK_SUGGESTION_JS = """() => {
    const sels = [
        '.ui-select-choices-row', '.ui-select-choices-row-inner',
        '.dropdown-menu li a', '.dropdown-menu li',
        '.autocomplete-suggestion', '.typeahead-result',
        'ul.ui-select-choices li', '[role="option"]',
        '.ui-menu-item', '.search-result-item', '.suggestion-item'
    ];
    for (const sel of sels) {
        for (const el of document.querySelectorAll(sel)) {
            const s = window.getComputedStyle(el);
            if (s.display === 'none' || s.visibility === 'hidden') continue;
            if (el.offsetWidth === 0 && el.offsetHeight === 0) continue;
            if (el.closest('nav, header, footer, .navbar')) continue;
            const t = el.textContent?.trim();
            if (t && t.length > 2 && t.length < 300) { el.click(); return t; }
        }
    }
    return null;
}"""


@asynccontextmanager
async def get_bedrock_browser():
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


# ── Tools for the LLM agent ──────────────────────────────────────────────────

@tool
async def fill_field(selector: str, value: str) -> str:
    """Fill a form field by CSS selector with a value."""
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=5000, state="visible")
        await page.fill(selector, value, timeout=5000)
        await page.wait_for_timeout(200)
        return f"Filled {selector} with '{value}'"
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
async def type_and_select(selector: str, text: str) -> str:
    """Type into an autocomplete/search field and select the first suggestion.
    Use this for fields that show a dropdown of suggestions as you type.
    Only type the search term — do NOT include extra info like [DOB:...] in the text.

    Args:
        selector: CSS selector for the input field
        text: Search text to type (name only, no extra annotations)
    """
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=5000, state="visible")
        await page.click(selector)
        await page.fill(selector, "")
        await page.type(selector, text, delay=50)
        await page.wait_for_timeout(2000)

        clicked = await page.evaluate(_CLICK_SUGGESTION_JS)

        if clicked:
            await page.wait_for_timeout(1000)
            return f"Selected: {clicked}"
        return f"No suggestion appeared after typing '{text}'"
    except Exception as e:
        return f"Failed: {e}"

@tool
async def select_option(selector: str, value: str) -> str:
    """Select an option from a <select> dropdown by value or visible text.

    Args:
        selector: CSS selector for the <select> element
        value: The option value or visible label to select
    """
    page = get_page()
    try:
        await page.wait_for_selector(selector, timeout=5000, state="visible")
        try:
            await page.select_option(selector, value=value, timeout=3000)
        except Exception:
            await page.select_option(selector, label=value, timeout=3000)
        await page.wait_for_timeout(200)
        return f"Selected '{value}' in {selector}"
    except Exception as e:
        return f"Failed {selector}: {e}"

@tool
async def fill_ui_select(container_selector: str, text: str) -> str:
    """Fill a ui-select dropdown widget. Clicks the toggle to open it, types to search,
    and selects the first match. Use for dropdown fields that show a search box when clicked.

    Args:
        container_selector: CSS selector for the ui-select container (e.g. #placeOfServiceSelect)
        text: Text to type to search and filter options
    """
    page = get_page()
    try:
        # Wait for the container itself first
        await page.wait_for_selector(container_selector, timeout=5000, state="visible")

        # Try clicking the toggle button; fall back to clicking the container directly
        toggle = f"{container_selector} .ui-select-toggle"
        try:
            await page.wait_for_selector(toggle, timeout=3000, state="visible")
            await page.click(toggle)
        except Exception:
            await page.click(container_selector)
        await page.wait_for_timeout(700)

        # The search input appears after clicking — try multiple known selectors
        search_selectors = [
            f"{container_selector} input.ui-select-search",
            f"{container_selector} input[type='search']",
            f"{container_selector} input",
            "div.ui-select-dropdown input.ui-select-search",  # sometimes renders outside container
        ]
        search_el = None
        for sel in search_selectors:
            try:
                await page.wait_for_selector(sel, timeout=2000, state="visible")
                search_el = sel
                break
            except Exception:
                continue

        if not search_el:
            return f"Failed {container_selector}: could not find search input after clicking"

        await page.fill(search_el, "")
        await page.type(search_el, text, delay=50)
        await page.wait_for_timeout(2000)

        clicked = await page.evaluate(_CLICK_SUGGESTION_JS)
        if clicked:
            await page.wait_for_timeout(1000)
            return f"Selected: {clicked}"
        return f"No suggestion appeared after typing '{text}' in {container_selector}"
    except Exception as e:
        return f"Failed {container_selector}: {e}"


# ── Hardcoded field definitions for Create Bill form ─────────────────────────

CREATE_BILL_FIELDS = """[Form: createBillForm]
[type_and_select] "Patient Name" → #emaPatientQuickSearch
[fill_ui_select] "Service Location" → #placeOfServiceSelect
[fill_ui_select] "Primary Biller" → #renderingProviderSelect
[fill_field] "Date of Service" → input#dateOfServiceInput
[fill_ui_select] "Primary Provider" → #primaryProviderSelect
[fill_ui_select] "Referring Provider" → #referringProviderSelect
[select_option] "Reportable Reason" → #reportableReasonSelect
[type_and_select] "Diagnoses" → #diagnosesInput .ui-select-search"""

# ── Hardcoded field definitions for Services Rendered form ────────────────────
# Update these selectors to match your actual services form DOM
SERVICES_FIELDS = """[Form: servicesRenderedForm]
[type_and_select] "Service Code / CPT" → #serviceCodeInput
[fill_field] "Units" → #serviceUnitsInput
[fill_field] "DX Pointers" → #dxPointersInput"""


# ── LLM Agent ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a browser automation agent. You fill web forms one field at a time.
Each field looks like: [tool_name] "Label" → selector

CRITICAL RULES:
1. Call ONLY ONE tool per turn. Never call multiple tools at once.
2. Wait for the result of each tool before calling the next.
3. If a tool returns "Failed", retry it ONCE with the same arguments.
4. Do NOT call get_page_html — field definitions are already provided.

HOW TO FILL EACH FIELD:
- Match each data key to a field by comparing meaning to the Label.
- Use the tool shown in [brackets] with the selector after →:
   [type_and_select] → type_and_select(selector=<selector>, text=<value>)
     • For patient name: strip any [DOB:...] annotation, type the name only
   [fill_ui_select]  → fill_ui_select(container_selector=<selector>, text=<value>)
   [fill_field]      → fill_field(selector=<selector>, value=<value>)
   [select_option]   → select_option(selector=<selector>, value=<value>)
   [click_element]   → click_element(selector=<selector>)
- Copy the selector EXACTLY as shown. Never modify or shorten it.
- Skip data keys with no matching field.
- After filling ALL fields, respond with: "All fields filled successfully." """

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_automation_agent():
    llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq")
    tools = [fill_field, click_element, press_key, type_and_select, select_option, fill_ui_select]

    workflow = StateGraph(AgentState)

    async def call_model(state):
        all_msgs = list(state["messages"])
        # Keep system + first user message + last 8 messages to avoid context overflow
        if len(all_msgs) > 10:
            all_msgs = all_msgs[:1] + all_msgs[-8:]

        # ── KEY FIX: parallel_tool_calls=False forces one tool at a time ──
        response = await llm.bind_tools(tools, parallel_tool_calls=False).ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + all_msgs
        )

        if hasattr(response, 'tool_calls') and response.tool_calls:
            tc = response.tool_calls[0]
            args_str = ', '.join(
                f"{k}={v[:60] if isinstance(v, str) else v}"
                for k, v in tc.get('args', {}).items()
            )
            print(f"  Tool: {tc.get('name')}({args_str})")

        return {"messages": [response]}

    tool_node = ToolNode(tools)

    async def call_tools(state):
        result = await tool_node.ainvoke(state)
        for msg in result.get("messages", []):
            content = msg.content if hasattr(msg, 'content') else str(msg)
            print(f"  → {content[:150]}")
        return result

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()


async def run_phase(graph, messages, phase_name, recursion_limit=40):
    print(f"\n{phase_name}")
    print(f"{'─' * 60}")
    result = await graph.ainvoke({"messages": messages}, {"recursion_limit": recursion_limit})
    return list(result["messages"])


# ── Helper: strip DOB annotation from patient name ───────────────────────────

def clean_patient_name(raw: str) -> str:
    """Remove [DOB:...] or (DOB: ...) annotations from patient name for typeahead search."""
    cleaned = re.sub(r'\s*[\[\(]DOB[:\s][^\]\)]*[\]\)]', '', raw, flags=re.IGNORECASE)
    return cleaned.strip()


# ── Hardcoded Navigation ─────────────────────────────────────────────────────

async def login(page: Page):
    username = os.environ.get("LOGIN_USERNAME", "admin")
    password = os.environ.get("LOGIN_PASSWORD", "admin")

    print("Navigating to app & selecting Practice Staff")
    await page.goto(APP_URL, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(2000)
    await page.click("input[name='redirectToNonPatientLoginPage']", timeout=10000)
    await page.wait_for_timeout(2000)

    print("Logging in")
    await page.fill("#username", username, timeout=5000)
    await page.fill("#password", password, timeout=5000)
    await page.click("button[name='login']", timeout=5000)
    await page.wait_for_timeout(3000)

    try:
        next_btn = await page.wait_for_selector("text=Next", timeout=3000)
        if next_btn:
            await next_btn.click()
            await page.wait_for_timeout(2000)
            print("Dismissed survey page")
    except Exception:
        pass

    print("Logged in successfully\n")


async def navigate_to_create_bill(page: Page, bill_type: str):
    financials_url = APP_URL.rstrip("/").rsplit("/ema", 1)[0] + "/ema/practice/financial/Financials.action#/home/bills"
    await page.goto(financials_url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(2000)
    await page.click("text=Create a Bill", timeout=10000)
    await page.wait_for_timeout(2000)
    print("  Create a Bill modal open")

    await page.click(f".modal-content label:has-text('{bill_type}')", timeout=10000)
    await page.wait_for_timeout(3000)
    print(f"  Selected bill type: {bill_type}")


async def click_create_bill(page: Page):
    try:
        await page.click(".modal-content button:has-text('Create Bill')", timeout=10000)
    except Exception:
        await page.click("button:has-text('Create Bill'):visible", timeout=10000)
    await page.wait_for_timeout(3000)
    print("  Clicked Create Bill")


async def save_and_exit(page: Page):
    await page.click("button:has-text('Save & Exit')", timeout=10000)
    await page.wait_for_timeout(3000)
    print("  Clicked Save & Exit")


# ── Main ──────────────────────────────────────────────────────────────────────

ERROR_KEYWORDS = ["failed", "error", "could not", "unable", "not found", "timeout", "exception"]
MAX_RETRIES = 1

def _detect_failure(message) -> str | None:
    content = message.content if hasattr(message, 'content') else str(message)
    for kw in ERROR_KEYWORDS:
        if kw in content.lower():
            return content
    return None


async def run_agent():
    nest_asyncio.apply()

    try:
        with open(FORM_DATA_FILE, "r") as f:
            form_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {FORM_DATA_FILE} not found.")
        return

    print(f"Loaded {len(form_data)} bill entries to submit\n")
    graph = create_automation_agent()

    async with get_bedrock_browser() as page:
        set_page(page)
        print("Connected to Bedrock Browser\n")

        await login(page)

        results = []

        for i, entry in enumerate(form_data, 1):
            fields_display = ', '.join(f'{k}="{v}"' for k, v in entry.items())
            print(f"\n{'=' * 70}")
            print(f"BILL ENTRY {i}/{len(form_data)}")
            print(f"{fields_display}")
            print(f"{'=' * 70}")

            success = False
            failure = ""

            for attempt in range(1 + MAX_RETRIES):
                try:
                    if attempt > 0:
                        print(f"  Retrying entry {i} (attempt {attempt + 1})...")

                    # Step 1: Navigate and open Create Bill modal
                    bill_type = entry.get("bill_type", "Patient")
                    await navigate_to_create_bill(page, bill_type)

                    # Step 2: Build fill data — clean patient name for typeahead
                    bill_fields = {k: v for k, v in entry.items()
                                   if k not in ("bill_type", "service_code", "service_units", "dx_ptrs")}

                    # Strip DOB annotation from patient name so typeahead works
                    if "patient_name" in bill_fields:
                        bill_fields["patient_name"] = clean_patient_name(bill_fields["patient_name"])

                    fields_str = '\n'.join(f'  - {k}: "{v}"' for k, v in bill_fields.items())

                    fill_msg = f"""Fill the form fields below ONE AT A TIME. \
Match each data key to a field by label meaning, use the indicated tool, \
and copy the selector exactly.

FIELDS:
{CREATE_BILL_FIELDS}

DATA TO FILL:
{fields_str}"""

                    fill_messages = await run_phase(
                        graph, [("user", fill_msg)],
                        f"Fill Bill Fields (Entry {i})"
                    )

                    fill_failure = _detect_failure(fill_messages[-1])
                    if fill_failure:
                        failure = fill_failure
                        print(f"  Fill attempt {attempt + 1} detected issue")
                        continue

                    # Check at least one fill tool was called
                    fill_tool_names = {
                        getattr(m, 'name', '') for m in fill_messages
                        if hasattr(m, 'name')
                    }
                    actually_filled = fill_tool_names & {"fill_field", "type_and_select", "select_option", "fill_ui_select", "click_element"}
                    if not actually_filled:
                        failure = "LLM did not fill any form fields"
                        print(f"  Fill attempt {attempt + 1}: no fields were filled, skipping")
                        continue

                    # Step 3: Click Create Bill button
                    await click_create_bill(page)

                    # Step 4: Fill Services Rendered
                    service_fields = {k: entry[k] for k in ("service_code", "service_units", "dx_ptrs") if k in entry}
                    svc_fields_str = '\n'.join(f'  - {k}: "{v}"' for k, v in service_fields.items())

                    svc_msg = f"""Fill the remaining form fields below ONE AT A TIME. \
Match each data key to a field by label meaning, use the indicated tool, \
and copy the selector exactly.

FIELDS:
{SERVICES_FIELDS}

DATA TO FILL:
{svc_fields_str}"""

                    svc_messages = await run_phase(
                        graph, [("user", svc_msg)],
                        f"Fill Services Rendered (Entry {i})"
                    )

                    svc_failure = _detect_failure(svc_messages[-1])
                    if svc_failure:
                        failure = svc_failure
                        print(f"  Services Rendered attempt {attempt + 1} detected issue")
                        continue

                    # Step 5: Save & Exit
                    await save_and_exit(page)

                    success = True
                    break

                except Exception as e:
                    failure = str(e)
                    print(f"  Entry {i} attempt {attempt + 1} exception: {e}")

            if success:
                print(f"\n  ✓ Entry {i}: SUCCESS")
                results.append({"entry": i, "data": entry, "status": "success"})
            else:
                print(f"\n  ✗ Entry {i}: FAILED — {failure[:200]}")
                results.append({"entry": i, "data": entry, "status": "failed", "reason": failure})

        # ── Summary ──────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("SUBMISSION SUMMARY")
        print("=" * 70)

        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]

        print(f"Total    : {len(results)}")
        print(f"Success  : {len(successful)}")
        print(f"Failed   : {len(failed)}")

        if failed:
            print(f"\n{'─' * 70}\nFAILED ENTRIES:")
            for r in failed:
                flds = ', '.join(f'{k}="{v}"' for k, v in r["data"].items())
                reason = r["reason"] or "Unknown"
                print(f"\n  Entry {r['entry']}: {flds}")
                print(f"  Reason: {reason[:300]}{'...' if len(reason) > 300 else ''}")

        print("=" * 70 + "\n")
        await page.screenshot(path="final_result.png", full_page=True)
        print("Screenshot saved to final_result.png")

if __name__ == "__main__":
    asyncio.run(run_agent())