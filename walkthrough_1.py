import asyncio
import json
import os

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

_DISCOVER_AND_EXTRACT_FORM_JS = """() => {
    // ── Form discovery ──
    // Priority 1: If a modal is open, ONLY search inside it (ignore background forms)
    let root = null;
    const modal = document.querySelector('.modal-dialog, .modal-content, [uib-modal-window]');
    if (modal) {
        // Look for a named form inside the modal, otherwise use the modal itself
        root = modal.querySelector('form[name], [ng-form]') || modal;
    }
    // Priority 2: Named form or ng-form
    if (!root) {
        const namedForms = [...document.querySelectorAll('form[name], [ng-form]')];
        if (namedForms.length === 1) {
            root = namedForms[0];
        } else if (namedForms.length > 1) {
            let bestCount = 0;
            for (const f of namedForms) {
                const c = f.querySelectorAll('input, select, textarea').length;
                if (c > bestCount) { bestCount = c; root = f; }
            }
        }
    }
    // Priority 3: Largest visible form
    if (!root) {
        let maxInputs = 0;
        for (const f of document.querySelectorAll('form')) {
            const st = window.getComputedStyle(f);
            if (st.display === 'none' || st.visibility === 'hidden') continue;
            const c = f.querySelectorAll('input, select, textarea').length;
            if (c > maxInputs) { maxInputs = c; root = f; }
        }
    }
    // Priority 4: document.body
    if (!root) root = document.body;
    const formName = root.getAttribute
        ? (root.getAttribute('name') || root.getAttribute('ng-form') || root.tagName.toLowerCase())
        : 'body';

    // ── Extract interactive elements ──
    const results = [];
    const seen = new Set();

    for (const el of root.querySelectorAll('input:not([type="hidden"]), select, textarea, button, [role="combobox"], [role="listbox"]')) {
        if (seen.has(el)) continue;
        seen.add(el);
        const st = window.getComputedStyle(el);
        if (st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') continue;
        if (el.offsetWidth === 0 && el.offsetHeight === 0 && el.tagName !== 'INPUT') continue;
        if (el.disabled || el.readOnly) continue;

        // ── Pre-compute CSS selector ──
        const tag = el.tagName.toLowerCase();
        let sel = '';
        const id = el.getAttribute('id');
        const name = el.getAttribute('name');
        const ph = el.getAttribute('placeholder');
        const ngm = el.getAttribute('ng-model');
        if (id) sel = '#' + id;
        else if (name) sel = tag + "[name='" + name + "']";
        else if (ph) sel = tag + "[placeholder='" + ph + "']";
        else if (ngm) sel = tag + "[ng-model='" + ngm + "']";
        else continue; // no usable selector, skip

        // ── Detect interaction type ──
        let interaction = 'input';
        if (el.hasAttribute('uib-typeahead') || el.getAttribute('aria-autocomplete') ||
            el.getAttribute('role') === 'combobox' || el.closest('.ui-select-container')) {
            interaction = 'typeahead';
        } else if (tag === 'select') {
            interaction = 'select';
        } else if (tag === 'button') {
            interaction = 'click';
        }

        // ── Find label ──
        let label = '';
        if (id) {
            const lbl = root.querySelector('label[for="' + id + '"]');
            if (lbl) label = lbl.textContent.trim();
        }
        if (!label) {
            const group = el.closest('.form-group, .form-field, fieldset, .field-container');
            if (group) {
                const lbl = group.querySelector('label');
                if (lbl) label = lbl.textContent.trim();
            }
        }
        if (!label && ph) label = ph;
        if (!label && tag === 'button') label = el.textContent?.trim() || '';
        if (!label) continue; // no way to identify this field, skip

        results.push({ selector: sel, interaction: interaction, label: label.substring(0, 80) });
    }
    return { formName: formName, elementCount: results.length, elements: results };
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


async def extract_form_elements(page: Page) -> str:
    try:
        data = await page.evaluate(_DISCOVER_AND_EXTRACT_FORM_JS)
        if not data or not data.get("elements"):
            return "No form elements discovered on the page."
        form_name = data.get("formName", "unknown")
        element_count = data.get("elementCount", 0)
        lines = [f"[Form: {form_name}] ({element_count} fields)"]
        for el in data["elements"]:
            # Format: [interaction] "label" → selector
            lines.append(f"[{el['interaction']}] \"{el['label']}\" → {el['selector']}")
        result = '\n'.join(lines)
        return result[:16000] + '\n... (truncated)' if len(result) > 16000 else result
    except Exception as e:
        return f"Failed to extract form elements: {e}"


# ── Tools for the LLM agent ──────────────────────────────────────────────────

@tool
async def get_page_html() -> str:
    """Returns interactive elements from the dynamically discovered form on the current page."""
    return await extract_form_elements(get_page())

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
    Use this for fields that show a dropdown of suggestions as you type.

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


# ── LLM Agent ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a browser automation agent. You fill web forms using a pre-built list of fields.

Each field in the list looks like:
  [interaction_type] "Label Text" → css_selector

RULES:
1. When fields are PROVIDED in the user message, use them directly. Do NOT call get_page_html — the DOM mutates after each fill.
2. Fill ONE field at a time. Wait for the result before the next.
3. If a tool call returns "Failed", retry it once.

HOW TO FILL:
1. Match each data key to a field by comparing the key meaning to the Label Text.
2. Use the interaction type to pick the tool:
   - [typeahead] → type_and_select(selector, value)
   - [select] → select_option(selector, value)
   - [input] → fill_field(selector, value)
   - [click] → click_element(selector)
3. Copy the css_selector from the field line EXACTLY as-is. Do NOT modify, shorten, or invent selectors.
4. If no field matches a data key, skip that key.

ONLY use get_page_html after all fields are filled to verify."""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_automation_agent():
    llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq")
    tools = [get_page_html, fill_field, click_element, press_key, type_and_select, select_option]

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


async def run_phase(graph, messages, phase_name, recursion_limit=30):
    print(f"{phase_name}")
    print(f"{'=' * 70}")
    result = await graph.ainvoke({"messages": messages}, {"recursion_limit": recursion_limit})
    return list(result["messages"])


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

    print("Logged in")


async def navigate_to_create_bill(page: Page, bill_type: str):
    financials_url = APP_URL.rstrip("/").rsplit("/ema", 1)[0] + "/ema/practice/financial/Financials.action#/home/bills"
    await page.goto(financials_url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(2000)
    await page.click("text=Create a Bill", timeout=10000)
    await page.wait_for_timeout(2000)
    print("  Create a Bill modal open")

    # Dynamically select the bill type radio by its label text
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
            fields = ', '.join(f'{k}="{v}"' for k, v in entry.items())
            print(f"\n{'=' * 70}")
            print(f"BILL ENTRY {i}/{len(form_data)}")
            print(f"{fields}")
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

                    # Step 2: Extract form HTML and let LLM fill the bill fields
                    form_html = await extract_form_elements(page)
                    print(f"  Extracted form: {form_html[:200]}")

                    bill_fields = {k: v for k, v in entry.items()
                                   if k not in ("bill_type", "service_code", "service_units", "dx_ptrs")}
                    fields_str = '\n'.join(f'  - {k}: "{v}"' for k, v in bill_fields.items())

                    fill_msg = f"""Fill the form fields below. Match each data key to a field by label meaning, use the tool indicated by [type], and copy the selector exactly.

FIELDS:
{form_html}

DATA TO FILL:
{fields_str}"""

                    fill_messages = await run_phase(
                        graph, [("user", fill_msg)],
                        f"  Fill Bill Fields (Entry {i})"
                    )

                    fill_failure = _detect_failure(fill_messages[-1])
                    if fill_failure:
                        failure = fill_failure
                        print(f"  Fill attempt {attempt + 1} detected issue")
                        continue

                    # Verify the LLM actually filled fields (not just called get_page_html)
                    fill_tool_names = {
                        getattr(m, 'name', '') for m in fill_messages
                        if hasattr(m, 'name')
                    }
                    actually_filled = fill_tool_names & {"fill_field", "type_and_select", "select_option", "click_element"}
                    if not actually_filled:
                        failure = "LLM did not fill any form fields"
                        print(f"  Fill attempt {attempt + 1}: no fields were filled, skipping Create Bill")
                        continue

                    # Step 3: Click Create Bill button
                    await click_create_bill(page)

                    # Step 4: Fill Services Rendered via LLM
                    service_fields = {k: entry[k] for k in ("service_code", "service_units", "dx_ptrs") if k in entry}
                    svc_fields_str = '\n'.join(f'  - {k}: "{v}"' for k, v in service_fields.items())

                    services_html = await extract_form_elements(page)
                    print(f"  Extracted services form: {services_html[:200]}")

                    svc_msg = f"""Fill the remaining form fields below. Match each data key to a field by label meaning, use the tool indicated by [type], and copy the selector exactly.

FIELDS:
{services_html}

DATA TO FILL:
{svc_fields_str}"""

                    svc_messages = await run_phase(
                        graph, [("user", svc_msg)],
                        f"  Fill Services Rendered (Entry {i})"
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
                print(f"\n  Entry {i}: SUCCESS")
                results.append({"entry": i, "data": entry, "status": "success"})
            else:
                print(f"\n  Entry {i}: FAILED")
                results.append({"entry": i, "data": entry, "status": "failed", "reason": failure})

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
