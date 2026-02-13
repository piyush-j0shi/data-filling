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

            const tagSelector = 'input, button, a, select, textarea, label, form, option';
            const roleSelector = '[role="button"], [role="combobox"], [role="listbox"], [role="option"], [role="textbox"], [role="link"], [role="menuitem"], [role="tab"], [role="checkbox"], [role="radio"], [role="switch"], [role="slider"]';
            const interactiveSelector = '[contenteditable="true"], [onclick], [data-testid], [tabindex]:not([tabindex="-1"])';

            const combined = `${tagSelector}, ${roleSelector}, ${interactiveSelector}`;
            const attrs = ['id', 'name', 'type', 'placeholder', 'value', 'href', 'for',
                           'role', 'aria-label', 'data-testid', 'data-field',
                           'contenteditable', 'action', 'method'];

            for (const el of document.querySelectorAll(combined)) {
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
                if (el.offsetWidth === 0 && el.offsetHeight === 0 && el.tagName !== 'INPUT' && el.tagName !== 'OPTION') continue;

                if (seen.has(el)) continue;
                seen.add(el);

                const tag = el.tagName.toLowerCase();
                const info = {tag: tag};

                for (const a of attrs) {
                    const v = el.getAttribute(a);
                    if (v && v.trim()) info[a] = v.trim().substring(0, 80);
                }

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
        if len(result) > 16000:
            result = result[:16000] + '\n... (truncated)'
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
    """Fill a form field by CSS selector with a value. Use this for simple text inputs and date fields, NOT for autocomplete/search fields."""
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
    """Select a dropdown option by CSS selector. Use for native <select> elements only."""
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

@tool
async def type_and_select(selector: str, text: str) -> str:
    """Type into an autocomplete/search field and select the first suggestion.
    Use this for fields that show a dropdown of suggestions as you type
    (e.g. Patient Name, Primary Biller, Primary Provider, Referring Provider, Diagnoses, Service Code).

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

        clicked = await page.evaluate("""() => {
            // Strategy 1: Look for common dropdown/suggestion containers
            const dropdownSelectors = [
                '.ui-select-choices-row',
                '.ui-select-choices-row-inner',
                '.dropdown-menu li',
                '.autocomplete-suggestion',
                '.typeahead-result',
                'ul.ui-select-choices li',
                '[role="option"]',
                '.ui-menu-item',
                '.search-result-item',
                '.suggestion-item'
            ];

            for (const sel of dropdownSelectors) {
                const items = document.querySelectorAll(sel);
                for (const el of items) {
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') continue;
                    if (el.offsetWidth === 0 && el.offsetHeight === 0) continue;
                    const text = el.textContent?.trim();
                    if (text && text.length > 2) {
                        el.click();
                        return text;
                    }
                }
            }

            // Strategy 2: Find any visible element with cursor:pointer near an open dropdown
            const all = document.querySelectorAll('li, a, div, span');
            for (const el of all) {
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden') continue;
                if (el.offsetWidth === 0 && el.offsetHeight === 0) continue;

                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute('role') || '';
                const cursor = style.cursor;
                const parent = el.parentElement;
                const parentClass = parent ? parent.className : '';

                const isDropdownItem = (
                    role === 'option' || role === 'menuitem' ||
                    (tag === 'li' && cursor === 'pointer') ||
                    (tag === 'a' && el.closest('ul')) ||
                    (cursor === 'pointer' && (
                        parentClass.includes('select') ||
                        parentClass.includes('dropdown') ||
                        parentClass.includes('autocomplete') ||
                        parentClass.includes('typeahead') ||
                        parentClass.includes('suggestion') ||
                        parentClass.includes('choices')
                    )) ||
                    el.closest('[role="listbox"]') !== null ||
                    el.closest('[role="menu"]') !== null ||
                    el.closest('.ui-select-choices') !== null
                );

                if (!isDropdownItem) continue;

                const text = el.textContent?.trim();
                if (text && text.length > 2 && text.length < 300) {
                    el.click();
                    return text;
                }
            }
            return null;
        }""")

        if clicked:
            await page.wait_for_timeout(1000)
            return f"Selected: {clicked}"
        else:
            return f"No suggestion appeared after typing '{text}'"

    except Exception as e:
        return f"Failed: {e}"

SYSTEM_PROMPT = """You are a browser automation agent for a ModMed EMA medical billing application.

RULES:
1. ALWAYS call get_page_html first to discover the page structure before interacting with any elements.
2. NEVER guess CSS selectors. Only use selectors you found from get_page_html.
3. Build selectors from id, name, data-testid, or role attributes. Prefer #id or [name="x"] over class-based selectors.
4. Map data fields to form fields by meaning, not exact name match.
5. Fill form fields ONE AT A TIME. Wait for each to complete before the next.
6. After clicking submit or navigating, call get_page_html again to see the updated page.
7. Elements marked disabled="true" or readonly="true" cannot be interacted with — skip them.

AUTOCOMPLETE FIELDS:
- For search/autocomplete fields (Patient Name, Primary Biller, Primary Provider, Referring Provider, Diagnoses, Service Code), use type_and_select instead of fill_field.
- type_and_select types the full value character-by-character to trigger suggestions, then clicks the first suggestion.
- Do NOT use fill_field for autocomplete fields — it won't trigger the dropdown.

RADIO BUTTONS:
- To select a radio button, use click_element on the radio input element or its label.

DROPDOWNS:
- For native <select> elements (like Reportable Reason), use select_option.
- For Angular ui-select components, use type_and_select.

MEDICAL DOMAIN:
- Skip the Medical Domain field — it auto-fills when Primary Provider is selected.

MODAL FORMS:
- The Create a Bill form is inside a modal (div.modal-content.printable-container). Selectors still work on the full page DOM."""

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
        type_and_select,
    ]

    workflow = StateGraph(AgentState)

    async def call_model(state):
        all_msgs = list(state["messages"])
        if len(all_msgs) > 10:
            all_msgs = all_msgs[:1] + all_msgs[-8:]
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

    tool_node = ToolNode(tools)

    # Tools whose results are too large to print (page reads)
    _quiet_tools = {"get_page_html", "get_page_text"}

    async def call_tools(state):
        result = await tool_node.ainvoke(state)
        for msg in result.get("messages", []):
            tool_name = getattr(msg, 'name', '')
            if tool_name in _quiet_tools:
                continue
            content = msg.content if hasattr(msg, 'content') else str(msg)
            preview = content[:150] if len(content) > 150 else content
            print(f"    -> {preview}")
        return result

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

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

async def run_phase(graph, messages, phase_name, recursion_limit=30):
    """Run a single phase of the workflow and return updated messages."""
    print(f"  {phase_name}")
    print(f"{'=' * 70}")
    result = await graph.ainvoke(
        {"messages": messages},
        {"recursion_limit": recursion_limit}
    )
    return list(result["messages"])


async def run_agent():
    nest_asyncio.apply()

    try:
        form_data = load_form_data(FORM_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {FORM_DATA_FILE} not found. Create it with a JSON list of entries.")
        return

    print(f"Loaded {len(form_data)} bill entries to submit\n")
    graph = create_automation_agent()

    async with get_bedrock_browser() as page:
        set_page(page)
        print(f"Connected to Bedrock Browser\n")

        username = os.environ.get("LOGIN_USERNAME", "admin")
        password = os.environ.get("LOGIN_PASSWORD", "admin")

        print("  Select Practice Staff")
        
        await page.goto(APP_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)
        await page.click("input[name='redirectToNonPatientLoginPage']", timeout=10000)
        await page.wait_for_timeout(2000)
        print("on login page")
        print("Login")
        await page.fill("#username", username, timeout=5000)
        await page.fill("#password", password, timeout=5000)
        await page.click("button[name='login']", timeout=5000)
        await page.wait_for_timeout(3000)

        try:
            next_btn = await page.wait_for_selector("text=Next", timeout=3000)
            if next_btn:
                await next_btn.click()
                await page.wait_for_timeout(2000)
                print("  Dismissed survey page")
        except Exception:
            pass  

        print("logged in")

        results = []

        for i, entry in enumerate(form_data, 1):
            fields = ', '.join(f'{k}="{v}"' for k, v in entry.items())
            print(f"\n{'=' * 70}")
            print(f"  BILL ENTRY {i}/{len(form_data)}")
            print(f"  {fields}")
            print(f"{'=' * 70}")

            print(f"Navigate to Create Bill (Entry {i})")
            
            financials_url = APP_URL.rstrip("/").rsplit("/ema", 1)[0] + "/ema/practice/financial/Financials.action#/home/bills"
            await page.goto(financials_url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(2000)
            await page.click("text=Create a Bill", timeout=10000)
            await page.wait_for_timeout(2000)
            print("Create a Bill modal open")

            bill_fields = {
                "patient_name": entry.get("patient_name", ""),
                "service_location": entry.get("service_location", ""),
                "primary_biller": entry.get("primary_biller", ""),
                "date_of_service": entry.get("date_of_service", ""),
                "primary_provider": entry.get("primary_provider", ""),
                "referring_provider": entry.get("referring_provider", ""),
                "reportable_reason": entry.get("reportable_reason", ""),
                "diagnoses": entry.get("diagnoses", ""),
            }
            bill_fields_str = '\n'.join(f'  - {k}: "{v}"' for k, v in bill_fields.items() if v)

            phase3_msg = f"""Fill the Create a Bill form and submit it. The modal is already open.

Form data:
{bill_fields_str}

Steps:
1. Call get_page_html to see the modal form
2. FIRST: Click the "Patient Bill" radio button (the form defaults to Claim Bill, we need Patient Bill)
3. For autocomplete fields (patient_name, service_location, primary_biller, primary_provider, referring_provider, diagnoses), use type_and_select with the FULL value provided above
4. For date_of_service, use fill_field on the date input
5. For reportable_reason, use select_option on the dropdown
6. SKIP Medical Domain — it auto-fills when Primary Provider is selected
7. After filling all fields, click the "Create Bill" button
8. Confirm the form was submitted by reading the page — you should be redirected to a Manage Bill page"""

            await run_phase(
                graph,
                [("user", phase3_msg)],
                f"Fill Create Bill Modal (Entry {i})"
            )

            service_code = entry.get("service_code", "")
            service_units = entry.get("service_units", "1")
            dx_ptrs = entry.get("dx_ptrs", "1")

            phase4_msg = f"""Fill the Services Rendered section on the Manage Bill page. You are already on the Manage Bill page.

Service data:
  - Code: "{service_code}"
  - Units: "{service_units}"
  - DX Ptrs: "{dx_ptrs}"

Steps:
1. Call get_page_html to see the Manage Bill page and find the Services Rendered section
2. The Code field is an autocomplete — use type_and_select with the code value "{service_code}"
3. Fill the Units field with "{service_units}"
4. Click the DX Ptr button(s) for pointer "{dx_ptrs}" (these are numbered buttons like 1, 2, 3, 4)
5. Confirm the service line is filled by reading the page"""

            await run_phase(
                graph,
                [("user", phase4_msg)],
                f"Fill Services Rendered (Entry {i})"
            )

            phase5_msg = """Save the bill and exit. You are on the Manage Bill page.

Steps:
1. Call get_page_html to see the current page
2. Click the "Save" button
3. Wait for the save to complete, then call get_page_html to confirm
4. Click the "Save & Exit" button
5. Confirm you are back on the bills list page"""

            success = False
            failure_reason = ""

            for attempt in range(1 + MAX_RETRIES):
                if attempt > 0:
                    print(f"  Retrying entry {i} (attempt {attempt + 1})...")

                try:
                    final_messages = await run_phase(
                        graph,
                        [("user", phase5_msg)],
                        f"Save & Exit (Entry {i})"
                    )
                    final_msg = final_messages[-1]
                    failure_reason = _detect_failure(final_msg)

                    if failure_reason is None:
                        success = True
                        break
                    else:
                        print(f"  Entry {i} attempt {attempt + 1} detected issue")

                except Exception as e:
                    failure_reason = str(e)
                    print(f"  Entry {i} attempt {attempt + 1} exception: {e}")

            if success:
                print(f"\n  Entry {i}: SUCCESS")
                results.append({"entry": i, "data": entry, "status": "success"})
            else:
                print(f"\n  Entry {i}: FAILED")
                results.append({"entry": i, "data": entry, "status": "failed", "reason": failure_reason})

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
                if len(reason) > 300:
                    reason = reason[:300] + "..."
                print(f"  Reason: {reason}")

        print("=" * 70 + "\n")

        await page.wait_for_timeout(1000)
        await page.screenshot(path="final_result.png", full_page=True)
        print("Screenshot saved to final_result.png")

if __name__ == "__main__":
    asyncio.run(run_agent())
