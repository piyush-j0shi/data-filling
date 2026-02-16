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

FORM_FIELDS = [
    ("patient_name",       "Patient Name",       "autocomplete"),
    ("service_location",   "Service Location",   "autocomplete"),
    ("primary_biller",     "Primary Biller",     "autocomplete"),
    ("date_of_service",    "Date of Service",    "date"),
    ("primary_provider",   "Primary Provider",   "autocomplete"),
    ("referring_provider", "Referring Provider",  "autocomplete"),
    ("reportable_reason",  "Reportable Reason",  "select"),
    ("diagnoses",          "Diagnoses",          "autocomplete"),
]

_page_holder: dict[str, Page | None] = {"page": None}

def set_page(page: Page):
    _page_holder["page"] = page

def get_page() -> Page:
    if _page_holder["page"] is None:
        raise RuntimeError("Browser session not active. Call set_page first.")
    return _page_holder["page"]

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

_EXTRACT_ELEMENTS_JS = """() => {
    const results = [];
    const seen = new Set();
    const tagSel = 'input, button, a, select, textarea, label, form, option';
    const roleSel = '[role="button"], [role="combobox"], [role="listbox"], [role="option"], [role="textbox"], [role="link"], [role="menuitem"], [role="tab"], [role="checkbox"], [role="radio"], [role="switch"], [role="slider"]';
    const interSel = '[contenteditable="true"], [onclick], [data-testid], [tabindex]:not([tabindex="-1"])';
    const attrs = ['id','name','type','placeholder','value','href','for','role',
                   'aria-label','data-testid','data-field','contenteditable','action','method'];
    for (const el of document.querySelectorAll(`${tagSel}, ${roleSel}, ${interSel}`)) {
        if (el.id && /^focusser-\\d+$/.test(el.id)) continue;
        if (el.classList?.contains('ui-select-search') && el.offsetHeight === 0) continue;
        const st = window.getComputedStyle(el);
        if (st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') continue;
        if (el.offsetWidth === 0 && el.offsetHeight === 0 && el.tagName !== 'INPUT' && el.tagName !== 'OPTION') continue;
        if (seen.has(el)) continue;
        seen.add(el);
        const info = {tag: el.tagName.toLowerCase()};
        for (const a of attrs) { const v = el.getAttribute(a); if (v?.trim()) info[a] = v.trim().substring(0, 80); }
        if (el.disabled) info.disabled = 'true';
        if (el.readOnly) info.readonly = 'true';
        const text = el.textContent?.trim().substring(0, 60);
        if (text) info.text = text;
        results.push(info);
    }
    return results;
}"""

_FIND_INPUT_BY_LABEL_JS = """(labelText) => {
    for (const label of document.querySelectorAll('label')) {
        if (!label.textContent.toLowerCase().includes(labelText.toLowerCase())) continue;
        const forAttr = label.getAttribute('for');
        const parent = label.closest('.form-group') || label.parentElement;
        if (forAttr) {
            const target = document.getElementById(forAttr);
            if (target) {
                const isHidden = target.type === 'hidden' || window.getComputedStyle(target).display === 'none';
                if (!isHidden && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA'))
                    return { type: 'direct_input', selector: '#' + forAttr };
                if (isHidden && parent) {
                    for (const inp of parent.querySelectorAll('input:not([type="hidden"])')) {
                        const s = window.getComputedStyle(inp);
                        if (s.display === 'none' || s.visibility === 'hidden') continue;
                        if (inp.offsetWidth === 0 && inp.offsetHeight === 0) continue;
                        if (inp.id) return { type: 'direct_input', selector: '#' + inp.id };
                        if (inp.placeholder) return { type: 'direct_input', selector: 'input[placeholder="' + inp.placeholder + '"]' };
                    }
                }
                if (!isHidden) return { type: 'ui_select', selector: '#' + forAttr };
            }
        }
        if (parent) {
            for (const inp of parent.querySelectorAll('input:not([type="hidden"])')) {
                const s = window.getComputedStyle(inp);
                if (s.display === 'none' || s.visibility === 'hidden') continue;
                if (inp.offsetWidth === 0 && inp.offsetHeight === 0) continue;
                if (inp.id) return { type: 'direct_input', selector: '#' + inp.id };
                if (inp.placeholder) return { type: 'direct_input', selector: 'input[placeholder="' + inp.placeholder + '"]' };
            }
        }
    }
    return null;
}"""

_FIND_SELECT_BY_LABEL_JS = """(labelText) => {
    for (const label of document.querySelectorAll('label')) {
        if (!label.textContent.toLowerCase().includes(labelText.toLowerCase())) continue;
        const forAttr = label.getAttribute('for');
        if (forAttr) {
            const target = document.getElementById(forAttr);
            if (target?.tagName === 'SELECT') return forAttr;
            if (target) { const s = target.querySelector('select'); if (s?.id) return s.id; }
        }
        const parent = label.closest('.form-group') || label.parentElement;
        if (parent) { const s = parent.querySelector('select'); if (s?.id) return s.id; }
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

async def extract_interactive_elements(page: Page) -> str:
    try:
        elements = await page.evaluate(_EXTRACT_ELEMENTS_JS)
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
        return result[:16000] + '\n... (truncated)' if len(result) > 16000 else result
    except Exception as e:
        return f"Failed to get page HTML: {e}"


async def _fill_autocomplete(page: Page, label_text: str, value: str) -> str:
    field_info = await page.evaluate(_FIND_INPUT_BY_LABEL_JS, label_text)
    if not field_info:
        return f"Failed: Could not find field with label '{label_text}'"

    selector = field_info['selector']

    if field_info['type'] == 'direct_input':
        await page.wait_for_selector(selector, timeout=5000, state="visible")
        await page.click(selector)
        await page.fill(selector, "")
        await page.type(selector, value, delay=50)
        await page.wait_for_timeout(2000)
    else:
        await page.click(selector, timeout=5000)
        await page.wait_for_timeout(500)
        search_input = None
        for sel in [f"{selector} input.ui-select-search",
                    f"{selector} input[type='search']",
                    f"{selector} input:not([type='hidden']):not(.ui-select-focusser)"]:
            try:
                search_input = await page.wait_for_selector(sel, timeout=2000, state="visible")
                if search_input:
                    break
            except Exception:
                continue
        if not search_input:
            return f"Failed: Could not find search input for '{label_text}'"
        await search_input.fill("")
        await search_input.type(value, delay=50)
        await page.wait_for_timeout(2000)

    clicked = await page.evaluate(_CLICK_SUGGESTION_JS)
    if clicked:
        await page.wait_for_timeout(1000)
        return f"Selected: {clicked}"
    return f"No suggestion appeared after typing '{value}' for '{label_text}'"


async def _fill_date(page: Page, label_text: str, value: str) -> str:
    input_selector = await page.evaluate(_FIND_INPUT_BY_LABEL_JS, label_text)
    if not input_selector:
        return f"Failed: Could not find input for label '{label_text}'"
    selector = input_selector['selector']
    await page.wait_for_selector(selector, timeout=5000, state="visible")
    await page.click(selector, click_count=3)
    await page.keyboard.press("Backspace")
    await page.type(selector, value, delay=30)
    await page.wait_for_timeout(200)
    return f"Filled '{label_text}' with '{value}'"


async def _select_dropdown(page: Page, label_text: str, value: str) -> str:
    select_id = await page.evaluate(_FIND_SELECT_BY_LABEL_JS, label_text)
    if not select_id:
        return f"Failed: Could not find select for label '{label_text}'"
    selector = f"#{select_id}"
    await page.wait_for_selector(selector, timeout=5000, state="visible")
    try:
        await page.select_option(selector, value=value, timeout=3000)
    except Exception:
        await page.select_option(selector, label=value, timeout=3000)
    await page.wait_for_timeout(200)
    return f"Selected '{value}' for '{label_text}'"


async def _fill_one_field(page: Page, label: str, value: str, field_type: str) -> str:
    if field_type == "autocomplete":
        return await _fill_autocomplete(page, label, value)
    elif field_type == "date":
        return await _fill_date(page, label, value)
    elif field_type == "select":
        return await _select_dropdown(page, label, value)
    return f"Unknown field type: {field_type}"


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

    print("Logged in")


async def fill_and_create_bill(page: Page, entry: dict) -> tuple[bool, str]:
    """Navigate to Financials, open Create Bill modal, fill all form fields, click Create Bill.

    Returns (success, failure_reason).
    """
    financials_url = APP_URL.rstrip("/").rsplit("/ema", 1)[0] + "/ema/practice/financial/Financials.action#/home/bills"

    await page.goto(financials_url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(2000)
    await page.click("text=Create a Bill", timeout=10000)
    await page.wait_for_timeout(2000)
    print("  Create a Bill modal open")

    await page.click("#patientBillTypeRadio", timeout=10000)
    await page.wait_for_timeout(3000)
    print("  Clicked Patient Bill radio")

    fill_failures = []
    for field_key, label, field_type in FORM_FIELDS:
        value = entry.get(field_key, "")
        if not value:
            continue

        result = ""
        for retry in range(2):
            result = await _fill_one_field(page, label, value, field_type)
            print(f"  {label}: {result}")
            if "Failed" not in str(result):
                break
            if retry == 0:
                print(f"  Retrying {label}...")

        if "Failed" in str(result):
            fill_failures.append(f"{label}: {result}")

    if fill_failures:
        return False, f"Field failures: {'; '.join(fill_failures)}"

    try:
        await page.click(".modal-content button:has-text('Create Bill')", timeout=10000)
    except Exception:
        await page.click("button:has-text('Create Bill'):visible", timeout=10000)
    await page.wait_for_timeout(3000)
    print("  Clicked Create Bill")

    return True, ""


@tool
async def get_page_html() -> str:
    """Returns visible interactive elements on the current page."""
    return await extract_interactive_elements(get_page())

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
    Use this for fields that show a dropdown of suggestions as you type
    (e.g. Service Code).

    Args:
        selector: CSS selector for the input field
        text: Full text to type — should narrow suggestions to one result
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


SYSTEM_PROMPT = """You are a browser automation agent for the Services Rendered section of a medical billing form.

RULES:
1. When form HTML is PROVIDED in the user message, use it to understand the page structure. Do NOT call get_page_html during form filling — the DOM mutates after each interaction.
2. Fill form fields ONE AT A TIME. Wait for each to complete before the next.
3. RETRY ON FAILURE: If a tool call returns "Failed" or an error, IMMEDIATELY retry the same tool call once before moving on.
4. Elements marked disabled="true" or readonly="true" cannot be interacted with — skip them.

FORM FILLING:
- The Code field is an autocomplete — use type_and_select with the CSS selector from the HTML.
- For Units, use fill_field with the CSS selector.
- For DX Ptr buttons (numbered 1, 2, 3, 4), use click_element.
- ONLY use get_page_html after all fields are filled to confirm."""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_automation_agent():
    llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq")
    tools = [get_page_html, fill_field, click_element, press_key, type_and_select]

    workflow = StateGraph(AgentState)

    async def call_model(state):
        all_msgs = list(state["messages"])
        if len(all_msgs) > 10:
            all_msgs = all_msgs[:1] + all_msgs[-8:]
        response = await llm.bind_tools(tools).ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + all_msgs
        )

        if hasattr(response, 'tool_calls') and len(response.tool_calls) > 1:
            print(f"\n  LLM wanted {len(response.tool_calls)} tools, limiting to first:")
            response = AIMessage(content=response.content, tool_calls=[response.tool_calls[0]])

        if hasattr(response, 'tool_calls') and response.tool_calls:
            tc = response.tool_calls[0]
            args_str = ', '.join(
                f"{k}={v[:50] if isinstance(v, str) else v}"
                for k, v in tc.get('args', {}).items()
            )
            print(f"  Tool: {tc.get('name')}({args_str})")

        return {"messages": [response]}

    tool_node = ToolNode(tools)

    async def call_tools(state):
        result = await tool_node.ainvoke(state)
        for msg in result.get("messages", []):
            if getattr(msg, 'name', '') == "get_page_html":
                continue
            content = msg.content if hasattr(msg, 'content') else str(msg)
            print(f"  - {content[:150]}")
        return result

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()


ERROR_KEYWORDS = ["failed", "error", "could not", "unable", "not found", "timeout", "exception"]
MAX_RETRIES = 1

def _detect_failure(message) -> str | None:
    content = message.content if hasattr(message, 'content') else str(message)
    for kw in ERROR_KEYWORDS:
        if kw in content.lower():
            return content
    return None


async def run_phase(graph, messages, phase_name, recursion_limit=30):
    print(f"{phase_name}")
    print(f"{'=' * 70}")
    result = await graph.ainvoke({"messages": messages}, {"recursion_limit": recursion_limit})
    return list(result["messages"])


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
            fields = ', '.join(f'{k}="{v}"' for k, v in entry.items())
            print(f"\n{'=' * 70}")
            print(f"BILL ENTRY {i}/{len(form_data)}")
            print(f"{fields}")
            print(f"{'=' * 70}")

            bill_success = False
            bill_failure = ""

            for attempt in range(1 + MAX_RETRIES):
                try:
                    if attempt > 0:
                        print(f"  Retrying fill & create bill (attempt {attempt + 1})...")
                    success, failure = await fill_and_create_bill(page, entry)
                    if success:
                        bill_success = True
                        break
                    bill_failure = failure
                except Exception as e:
                    bill_failure = str(e)
                    print(f"  Fill & create bill attempt {attempt + 1} exception: {e}")

            if not bill_success:
                print(f"\n  Entry {i}: FAILED (Fill & Create Bill)")
                results.append({"entry": i, "data": entry, "status": "failed", "reason": bill_failure})
                continue

            service_code = entry.get("service_code", "")
            service_units = entry.get("service_units", "1")
            dx_ptrs = entry.get("dx_ptrs", "1")

            svc_success = False
            svc_failure = ""

            for attempt in range(1 + MAX_RETRIES):
                try:
                    if attempt > 0:
                        print(f"  Retrying Services Rendered (attempt {attempt + 1})...")

                    services_html = await extract_interactive_elements(page)

                    svc_msg = f"""Fill the Services Rendered section on the Manage Bill page.

Here is the current page HTML:

{services_html}

Service data:
  - Code: "{service_code}"
  - Units: "{service_units}"
  - DX Ptrs: "{dx_ptrs}"

Instructions:
1. From the HTML above, find the Services Rendered section.
2. The Code field is an autocomplete — use type_and_select with the code value "{service_code}".
3. Fill the Units field with "{service_units}" using fill_field.
4. Click the DX Ptr button(s) for pointer "{dx_ptrs}" (these are numbered buttons like 1, 2, 3, 4).
5. If any tool call fails, retry it once immediately with the same selector.
6. After filling, call get_page_html to confirm the service line is filled."""

                    svc_messages = await run_phase(
                        graph, [("user", svc_msg)],
                        f"  Fill Services Rendered (Entry {i})"
                    )
                    svc_failure = _detect_failure(svc_messages[-1])

                    if svc_failure is None:
                        svc_success = True
                        break
                    else:
                        print(f"  Services Rendered attempt {attempt + 1} detected issue")

                except Exception as e:
                    svc_failure = str(e)
                    print(f"  Services Rendered attempt {attempt + 1} exception: {e}")

            if not svc_success:
                print(f"\n  Entry {i}: FAILED (Services Rendered)")
                results.append({"entry": i, "data": entry, "status": "failed", "reason": svc_failure})
                continue

            try:
                print("  Saving bill")
                await page.click("button:has-text('Save & Exit')", timeout=10000)
                await page.wait_for_timeout(3000)
                print(f"\n  Entry {i}: SUCCESS")
                results.append({"entry": i, "data": entry, "status": "success"})
            except Exception as e:
                print(f"\n  Entry {i}: FAILED (Save & Exit)")
                results.append({"entry": i, "data": entry, "status": "failed", "reason": str(e)})

        print("\n" + "=" * 70)
        print("SUBMISSION SUMMARY")
        print("=" * 70)

        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]

        print(f"Total entries : {len(results)}")
        print(f"Successful    : {len(successful)}")
        print(f"Failed        : {len(failed)}")

        if failed:
            print(f"\n{'─' * 70}")
            print("FAILED ENTRIES:")
            print(f"{'─' * 70}")
            for r in failed:
                fields = ', '.join(f'{k}="{v}"' for k, v in r["data"].items())
                print(f"\n  Entry {r['entry']}: {fields}")
                reason = r["reason"] or "Unknown error"
                print(f"  Reason: {reason[:300]}{'...' if len(reason) > 300 else ''}")

        print("=" * 70 + "\n")

        await page.wait_for_timeout(1000)
        await page.screenshot(path="final_result.png", full_page=True)
        print("Screenshot saved to final_result.png")

if __name__ == "__main__":
    asyncio.run(run_agent())
