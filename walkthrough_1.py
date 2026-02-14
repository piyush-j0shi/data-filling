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


async def _extract_interactive_elements(page: Page) -> str:
    """Extract visible interactive elements from page as formatted HTML string."""
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
                // Skip Angular ui-select focusser elements (noise that confuses selector matching)
                if (el.id && /^focusser-\\d+$/.test(el.id)) continue;
                // Skip ui-select search inputs that appear dynamically
                if (el.classList && el.classList.contains('ui-select-search') && el.offsetHeight === 0) continue;

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
    return await _extract_interactive_elements(page)

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
            // Only look inside known dropdown/suggestion containers — never scan the whole page
            const dropdownSelectors = [
                '.ui-select-choices-row',
                '.ui-select-choices-row-inner',
                '.dropdown-menu li a',
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
                    // Must be inside a dropdown container, not a random page element
                    const inDropdown = el.closest('.ui-select-choices, .dropdown-menu, [role="listbox"], .typeahead-dropdown, .autocomplete-results');
                    if (!inDropdown) continue;
                    const text = el.textContent?.trim();
                    if (text && text.length > 2 && text.length < 300) {
                        el.click();
                        return text;
                    }
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

_CLICK_SUGGESTION_JS = """() => {
    const dropdownSelectors = [
        '.ui-select-choices-row',
        '.ui-select-choices-row-inner',
        '.dropdown-menu li a',
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
            const inDropdown = el.closest('.ui-select-choices, .dropdown-menu, [role="listbox"], .typeahead-dropdown, .autocomplete-results');
            if (!inDropdown) continue;
            const text = el.textContent?.trim();
            if (text && text.length > 2 && text.length < 300) {
                el.click();
                return text;
            }
        }
    }
    return null;
}"""


@tool
async def fill_autocomplete_by_label(label_text: str, value: str) -> str:
    """Fill an autocomplete/ui-select field by its visible LABEL TEXT, not by CSS selector.
    Use this for Angular ui-select fields: Service Location, Primary Biller,
    Primary Provider, Referring Provider, Diagnoses, and any other autocomplete/search field.

    This tool handles the complex ui-select DOM internally — just provide the label text exactly
    as it appears on the form (e.g. "Service Location", "Primary Provider").

    Args:
        label_text: The visible label text on the form (e.g. "Service Location", "Primary Biller")
        value: The value to type and select from suggestions
    """
    page = get_page()
    try:
        container_info = await page.evaluate("""(labelText) => {
            const labels = document.querySelectorAll('label');
            for (const label of labels) {
                if (!label.textContent.toLowerCase().includes(labelText.toLowerCase())) continue;

                const forAttr = label.getAttribute('for');
                if (forAttr) {
                    const target = document.getElementById(forAttr);
                    if (target) {
                        // Return the id so we can build a selector
                        return { id: forAttr, tagName: target.tagName.toLowerCase() };
                    }
                }

                // Fallback: look for input near the label
                const parent = label.closest('.form-group') || label.parentElement;
                if (parent) {
                    const input = parent.querySelector('input:not([type="hidden"])');
                    if (input && input.id) {
                        return { id: input.id, tagName: 'input' };
                    }
                }
            }
            return null;
        }""", label_text)

        if not container_info:
            return f"Failed: Could not find field with label '{label_text}'"

        target_id = container_info['id']

        await page.click(f"#{target_id}", timeout=5000)
        await page.wait_for_timeout(500)

        search_selectors = [
            f"#{target_id} input.ui-select-search",
            f"#{target_id} input[type='search']",
            f"#{target_id} input:not([type='hidden']):not(.ui-select-focusser)",
        ]

        search_input = None
        for sel in search_selectors:
            try:
                search_input = await page.wait_for_selector(sel, timeout=2000, state="visible")
                if search_input:
                    break
            except Exception:
                continue

        if not search_input:
            if container_info['tagName'] == 'input':
                search_input = await page.wait_for_selector(f"#{target_id}", timeout=2000, state="visible")
            else:
                return f"Failed: Could not find search input for '{label_text}' (container: #{target_id})"

        await search_input.fill("")
        await search_input.type(value, delay=50)
        await page.wait_for_timeout(2000)

        clicked = await page.evaluate(_CLICK_SUGGESTION_JS)

        if clicked:
            await page.wait_for_timeout(1000)
            return f"Selected: {clicked}"
        else:
            return f"No suggestion appeared after typing '{value}' for '{label_text}'"

    except Exception as e:
        return f"Failed: {e}"


@tool
async def fill_field_by_label(label_text: str, value: str) -> str:
    """Fill a simple text/date input field by its visible LABEL TEXT, not by CSS selector.
    Use this for non-autocomplete fields like Date of Service.

    Args:
        label_text: The visible label text on the form (e.g. "Date of Service")
        value: The value to fill in
    """
    page = get_page()
    try:
        input_id = await page.evaluate("""(labelText) => {
            const labels = document.querySelectorAll('label');
            for (const label of labels) {
                if (!label.textContent.toLowerCase().includes(labelText.toLowerCase())) continue;

                const forAttr = label.getAttribute('for');
                if (forAttr) {
                    const target = document.getElementById(forAttr);
                    if (target) {
                        // If target is an input, use it directly
                        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
                            return forAttr;
                        }
                        // Otherwise find input inside
                        const input = target.querySelector('input:not([type="hidden"])');
                        if (input && input.id) return input.id;
                    }
                }

                // Fallback: look for input near the label
                const parent = label.closest('.form-group') || label.parentElement;
                if (parent) {
                    const input = parent.querySelector('input:not([type="hidden"])');
                    if (input && input.id) return input.id;
                }
            }
            return null;
        }""", label_text)

        if not input_id:
            return f"Failed: Could not find input for label '{label_text}'"

        await page.wait_for_selector(f"#{input_id}", timeout=5000, state="visible")
        await page.fill(f"#{input_id}", value, timeout=5000)
        await page.wait_for_timeout(200)
        return f"Filled '{label_text}' with '{value}'"

    except Exception as e:
        return f"Failed: {e}"


@tool
async def select_option_by_label(label_text: str, value: str) -> str:
    """Select a dropdown option by its visible LABEL TEXT, not by CSS selector.
    Use this for native <select> dropdowns like Reportable Reason.

    Args:
        label_text: The visible label text on the form (e.g. "Reportable Reason")
        value: The option value or label to select
    """
    page = get_page()
    try:
        select_id = await page.evaluate("""(labelText) => {
            const labels = document.querySelectorAll('label');
            for (const label of labels) {
                if (!label.textContent.toLowerCase().includes(labelText.toLowerCase())) continue;

                const forAttr = label.getAttribute('for');
                if (forAttr) {
                    const target = document.getElementById(forAttr);
                    if (target && target.tagName === 'SELECT') return forAttr;
                    if (target) {
                        const sel = target.querySelector('select');
                        if (sel && sel.id) return sel.id;
                    }
                }

                const parent = label.closest('.form-group') || label.parentElement;
                if (parent) {
                    const sel = parent.querySelector('select');
                    if (sel && sel.id) return sel.id;
                }
            }
            return null;
        }""", label_text)

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

    except Exception as e:
        return f"Failed: {e}"


SYSTEM_PROMPT = """You are a browser automation agent for a ModMed EMA medical billing application.

RULES:
1. When form HTML is PROVIDED in the user message, use it to understand the page structure. Do NOT call get_page_html during form filling — the DOM mutates after each interaction.
2. Fill form fields ONE AT A TIME. Wait for each to complete before the next.
3. RETRY ON FAILURE: If a tool call returns "Failed" or an error, IMMEDIATELY retry the same tool call once before moving to the next field.
4. After clicking submit or navigating to a NEW page, you MAY call get_page_html to see the new page.
5. Elements marked disabled="true" or readonly="true" cannot be interacted with — skip them.

FORM FILLING — USE LABEL-BASED TOOLS:
- For autocomplete/search fields (Patient Name, Service Location, Primary Biller, Primary Provider, Referring Provider, Diagnoses, Service Code), use fill_autocomplete_by_label with the LABEL TEXT as shown on the form.
  Example: fill_autocomplete_by_label(label_text="Service Location", value="Beavercreek")
  Example: fill_autocomplete_by_label(label_text="Primary Provider", value="Bakos, Matthew, MD")
- For date and text inputs (Date of Service), use fill_field_by_label with the LABEL TEXT.
  Example: fill_field_by_label(label_text="Date of Service", value="02/13/26")
- For native <select> dropdowns (Reportable Reason), use select_option_by_label with the LABEL TEXT.
  Example: select_option_by_label(label_text="Reportable Reason", value="Medical Non-emergency")
- For buttons and radio buttons, use click_element with the CSS selector from the provided HTML.
- ONLY use type_and_select or fill_field with CSS selectors when label-based tools are not applicable (e.g., Services Rendered section where fields may not have labels).

MEDICAL DOMAIN & PROVIDER FEE SCHEDULE:
- Skip Medical Domain and Provider Fee Schedule fields — they auto-fill when Primary Provider is selected.

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
        fill_field_by_label,
        fill_autocomplete_by_label,
        select_option_by_label,
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

            financials_url = APP_URL.rstrip("/").rsplit("/ema", 1)[0] + "/ema/practice/financial/Financials.action#/home/bills"

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

            phase3_success = False
            phase3_failure = ""

            for attempt in range(1 + MAX_RETRIES):
                try:
                    if attempt > 0:
                        print(f"  Retrying Fill Create Bill Modal (attempt {attempt + 1})...")
                    else:
                        print(f"Navigate to Create Bill (Entry {i})")

                    await page.goto(financials_url, wait_until="domcontentloaded", timeout=30000)
                    await page.wait_for_timeout(2000)
                    await page.click("text=Create a Bill", timeout=10000)
                    await page.wait_for_timeout(2000)
                    print("  Create a Bill modal open")

                    await page.click("#patientBillTypeRadio", timeout=10000)
                    await page.wait_for_timeout(1000)
                    print("  Clicked Patient Bill radio")

                    form_html = await _extract_interactive_elements(page)
                    print("  Captured clean form HTML snapshot")

                    phase3_msg = f"""Fill the Create a Bill form and submit it. The modal is already open and Patient Bill is selected.

Here is the current form HTML for reference (do NOT call get_page_html during filling):

{form_html}

Form data to fill:
{bill_fields_str}

Instructions — use LABEL-BASED tools (they find inputs automatically by label text):
1. fill_autocomplete_by_label(label_text="Patient Name", value="<patient_name value>")
2. fill_autocomplete_by_label(label_text="Service Location", value="<service_location value>")
3. fill_autocomplete_by_label(label_text="Primary Biller", value="<primary_biller value>")
4. fill_field_by_label(label_text="Date of Service", value="<date_of_service value>")
5. fill_autocomplete_by_label(label_text="Primary Provider", value="<primary_provider value>")
6. fill_autocomplete_by_label(label_text="Referring Provider", value="<referring_provider value>") — skip if empty
7. select_option_by_label(label_text="Reportable Reason", value="<reportable_reason value>")
8. fill_autocomplete_by_label(label_text="Diagnoses", value="<diagnoses value>")
9. SKIP Medical Domain and Provider Fee Schedule — they auto-fill.
10. Fill fields ONE AT A TIME. If any tool call fails, retry it once immediately.
11. After filling ALL fields, click the "Create Bill" button using click_element.
12. After clicking Create Bill, call get_page_html to confirm you are on the Manage Bill page."""

                    phase3_messages = await run_phase(
                        graph,
                        [("user", phase3_msg)],
                        f"Fill Create Bill Modal (Entry {i})"
                    )
                    phase3_failure = _detect_failure(phase3_messages[-1])

                    if phase3_failure is None:
                        phase3_success = True
                        break
                    else:
                        print(f"  Fill Create Bill attempt {attempt + 1} detected issue")

                except Exception as e:
                    phase3_failure = str(e)
                    print(f"  Fill Create Bill attempt {attempt + 1} exception: {e}")

            if not phase3_success:
                print(f"\n  Entry {i}: FAILED (Fill Create Bill)")
                results.append({"entry": i, "data": entry, "status": "failed", "reason": phase3_failure})
                continue

            service_code = entry.get("service_code", "")
            service_units = entry.get("service_units", "1")
            dx_ptrs = entry.get("dx_ptrs", "1")

            phase4_success = False
            phase4_failure = ""

            for attempt in range(1 + MAX_RETRIES):
                try:
                    if attempt > 0:
                        print(f"  Retrying Fill Services Rendered (attempt {attempt + 1})...")

                    services_html = await _extract_interactive_elements(page)
                    print("  Captured Manage Bill page HTML snapshot")

                    phase4_msg = f"""Fill the Services Rendered section on the Manage Bill page. You are already on the Manage Bill page.

Here is the current page HTML (do NOT call get_page_html during filling):

{services_html}

Service data:
  - Code: "{service_code}"
  - Units: "{service_units}"
  - DX Ptrs: "{dx_ptrs}"

Instructions:
1. From the HTML above, find the Services Rendered section.
2. The Code field is an autocomplete — use type_and_select with the code value "{service_code}".
3. Fill the Units field with "{service_units}".
4. Click the DX Ptr button(s) for pointer "{dx_ptrs}" (these are numbered buttons like 1, 2, 3, 4).
5. If any tool call fails, retry it once immediately with the same selector.
6. After filling, call get_page_html to confirm the service line is filled."""

                    phase4_messages = await run_phase(
                        graph,
                        [("user", phase4_msg)],
                        f"Fill Services Rendered (Entry {i})"
                    )
                    phase4_failure = _detect_failure(phase4_messages[-1])

                    if phase4_failure is None:
                        phase4_success = True
                        break
                    else:
                        print(f"  Fill Services Rendered attempt {attempt + 1} detected issue")

                except Exception as e:
                    phase4_failure = str(e)
                    print(f"  Fill Services Rendered attempt {attempt + 1} exception: {e}")

            if not phase4_success:
                print(f"\n  Entry {i}: FAILED (Fill Services Rendered)")
                results.append({"entry": i, "data": entry, "status": "failed", "reason": phase4_failure})
                continue

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
                    print(f"  Retrying Save & Exit (attempt {attempt + 1})...")

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
                        print(f"  Save & Exit attempt {attempt + 1} detected issue")

                except Exception as e:
                    failure_reason = str(e)
                    print(f"  Save & Exit attempt {attempt + 1} exception: {e}")

            if success:
                print(f"\n  Entry {i}: SUCCESS")
                results.append({"entry": i, "data": entry, "status": "success"})
            else:
                print(f"\n  Entry {i}: FAILED (Save & Exit)")
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
