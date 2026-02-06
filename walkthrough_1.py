import asyncio
import json
import os
import re

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import nest_asyncio
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from playwright.async_api import async_playwright, Page
from bedrock_agentcore.tools.browser_client import browser_session

load_dotenv()

FORM_DATA_FILE = "form_data.json"
APP_URL = os.environ.get("APP_URL", "https://your-public-ngrok-url.ngrok-free.app")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
BROWSER_ID = os.environ.get("BROWSER_ID", "your-bedrock-browser-id")
LOGIN_USERNAME = os.environ.get("LOGIN_USERNAME", "")
LOGIN_PASSWORD = os.environ.get("LOGIN_PASSWORD", "")

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
    with browser_session(AWS_REGION, identifier = BROWSER_ID) as client:
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
async def navigate_to_app() -> str:
    """Navigates to the configured React application URL."""
    page = get_page()
    try:
        await page.goto(APP_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(2000)
        return f"Navigated to {APP_URL}"
    
    except Exception as e:
        return f"Navigation failed: {str(e)}"

@tool
async def login(username: str, password: str) -> str:
    """Logs into the application with the given credentials."""
    page = get_page()
    try:
        await page.wait_for_selector('form', timeout=10000, state="visible")

        # Find username field by type/role — try text/email inputs, skip password fields
        username_field = page.locator(
            'form input[type="text"], form input[type="email"], '
            'form input:not([type])'
        ).first
        await username_field.fill(username, timeout=5000)

        await page.locator('input[type="password"]').fill(password, timeout=5000)

        # Find submit button by role, falling back to any button in the form
        submit_btn = page.get_by_role("button", name=re.compile(r"log\s*in|sign\s*in|submit", re.IGNORECASE))
        if await submit_btn.count() > 0:
            await submit_btn.first.click(timeout=5000)
        else:
            await page.locator('form button, form input[type="submit"]').first.click(timeout=5000)

        await page.wait_for_timeout(3000)
        return f"Successfully logged in as '{username}'"

    except Exception as e:
        return f"Login failed: {str(e)}"

@tool
async def get_form_fields() -> str:
    """Discovers all visible form fields on the current page and returns their details.
    Checks name, id, placeholder, aria-label, aria-labelledby, and both explicit and
    implicit label associations. Call this BEFORE fill_and_submit_form so you know
    what field names the form expects."""
    page = get_page()
    try:
        await page.wait_for_selector(
            'input, textarea, select, [role="textbox"], [role="combobox"], [role="listbox"], [contenteditable="true"]',
            timeout=10000, state="visible"
        )
        await page.wait_for_timeout(500)

        fields = await page.evaluate("""() => {
            const results = [];
            const elements = document.querySelectorAll(
                'input, textarea, select, [role="textbox"], [role="combobox"], [role="listbox"], [contenteditable="true"]'
            );
            for (const el of elements) {
                if (el.type === 'hidden' || el.type === 'submit' || el.type === 'button') continue;
                if (el.offsetParent === null) continue;

                // Explicit label: <label for="id">
                let labelText = '';
                if (el.id) {
                    const explicitLabel = document.querySelector('label[for="' + el.id + '"]');
                    if (explicitLabel) labelText = explicitLabel.textContent.trim();
                }

                // Implicit label: <label><input/></label>
                if (!labelText) {
                    const parentLabel = el.closest('label');
                    if (parentLabel) {
                        const clone = parentLabel.cloneNode(true);
                        clone.querySelectorAll('input, textarea, select').forEach(c => c.remove());
                        labelText = clone.textContent.trim();
                    }
                }

                // aria-labelledby
                let ariaLabelledBy = '';
                const labelledById = el.getAttribute('aria-labelledby');
                if (labelledById) {
                    const refEl = document.getElementById(labelledById);
                    if (refEl) ariaLabelledBy = refEl.textContent.trim();
                }

                results.push({
                    tag: el.tagName.toLowerCase(),
                    type: el.type || el.getAttribute('role') || '',
                    name: el.name || '',
                    id: el.id || '',
                    placeholder: el.placeholder || '',
                    label: labelText,
                    ariaLabel: el.getAttribute('aria-label') || '',
                    ariaLabelledBy: ariaLabelledBy
                });
            }
            return results;
        }""")

        if not fields:
            return "No visible form fields found on the page."
        return json.dumps(fields, indent=2)

    except Exception as e:
        return f"Error discovering form fields: {str(e)}"

def _scan_fields_js():
    """JS snippet to discover visible form fields with full label detection."""
    return """() => {
        const results = [];
        const elements = document.querySelectorAll(
            'input, textarea, select, [role="textbox"], [role="combobox"], [contenteditable="true"]'
        );
        for (const el of elements) {
            if (el.type === 'hidden' || el.type === 'submit' || el.type === 'button') continue;
            if (el.offsetParent === null) continue;

            let labelText = '';
            if (el.id) {
                const explicitLabel = document.querySelector('label[for="' + el.id + '"]');
                if (explicitLabel) labelText = explicitLabel.textContent.trim();
            }
            if (!labelText) {
                const parentLabel = el.closest('label');
                if (parentLabel) {
                    const clone = parentLabel.cloneNode(true);
                    clone.querySelectorAll('input, textarea, select').forEach(c => c.remove());
                    labelText = clone.textContent.trim();
                }
            }

            let ariaLabelledBy = '';
            const labelledById = el.getAttribute('aria-labelledby');
            if (labelledById) {
                const refEl = document.getElementById(labelledById);
                if (refEl) ariaLabelledBy = refEl.textContent.trim();
            }

            results.push({
                tag: el.tagName.toLowerCase(),
                type: el.type || el.getAttribute('role') || '',
                name: el.name || '',
                id: el.id || '',
                placeholder: el.placeholder || '',
                label: labelText,
                ariaLabel: el.getAttribute('aria-label') || '',
                ariaLabelledBy: ariaLabelledBy
            });
        }
        return results;
    }"""

@tool
async def fill_and_submit_form(form_data: str) -> str:
    """Fills out and submits ONE form entry. form_data is a JSON string of key-value pairs
    (e.g. '{"name": "Alice", "email": "alice@example.com"}'). Keys must match the actual
    form field names you discovered with get_form_fields. Wait for this to complete
    before calling again."""
    page = get_page()
    try:
        data = json.loads(form_data)
    except json.JSONDecodeError:
        return f"Invalid JSON in form_data: {form_data}"
    try:
        await page.wait_for_selector(
            'input, textarea, select, [role="textbox"], [role="combobox"], [contenteditable="true"]',
            timeout=10000, state="visible"
        )
        await page.wait_for_timeout(500)

        filled = []
        for key, value in data.items():
            # Re-scan fields before each fill to catch dynamically rendered fields
            fields = await page.evaluate(_scan_fields_js())
            key_lower = key.lower()
            matched = False

            for field in fields:
                field_identifiers = [
                    field['name'].lower(),
                    field['id'].lower(),
                    field['placeholder'].lower(),
                    field['label'].lower(),
                    field.get('ariaLabel', '').lower(),
                    field.get('ariaLabelledBy', '').lower()
                ]
                if key_lower in field_identifiers or any(key_lower in ident for ident in field_identifiers if ident):
                    tag = field['tag']
                    selector = None
                    if field['name']:
                        selector = f"{tag}[name=\"{field['name']}\"]"
                    elif field['id']:
                        selector = f"#{field['id']}"
                    elif field.get('ariaLabel'):
                        selector = f"[aria-label=\"{field['ariaLabel']}\"]"

                    if selector:
                        await page.fill(selector, str(value), timeout=5000)
                        await page.wait_for_timeout(300)
                        filled.append(key)
                        matched = True
                        break
            if not matched:
                filled.append(f"{key} (no match)")

        # Smart submit: try role-based, then type="submit", then any button with submit-like text
        submit_btn = page.get_by_role("button", name=re.compile(r"submit|save|send|create|add", re.IGNORECASE))
        if await submit_btn.count() > 0:
            await submit_btn.first.click(timeout=5000)
        elif await page.locator('button[type="submit"], input[type="submit"]').count() > 0:
            await page.locator('button[type="submit"], input[type="submit"]').first.click(timeout=5000)
        else:
            # Last resort: click the last button in the form (often the submit)
            form_buttons = page.locator('form button, form [role="button"]')
            if await form_buttons.count() > 0:
                await form_buttons.last.click(timeout=5000)
            else:
                return f"Filled fields {filled} but could not find a submit button"

        await page.wait_for_timeout(1000)
        return f"Submitted form with fields: {filled}"

    except Exception as e:
        return f"Form submission failed: {str(e)}"

@tool
async def get_page_content() -> str:
    """Returns the current page text to verify state."""
    page = get_page()
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=5000)
        body_text = await page.inner_text("body", timeout=5000)
        
        if len(body_text) > 500:
            return body_text[:500] + "... (truncated)"
        return body_text
    
    except Exception as e:
        return f"Error retrieving page content: {str(e)}"

@tool
async def click_submit_form() -> str:
    """Clicks the 'Submit Form' button to go back to the form page."""
    page = get_page()
    try:
        await page.click('button:has-text("Submit Form")', timeout=5000)
        await page.wait_for_timeout(1000)
        return "Navigated back to Submit Form page"
    
    except Exception as e:
        return f"Failed to click Submit Form: {str(e)}"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_automation_agent():
    llm = init_chat_model(
        model="llama-3.3-70b-versatile",
        model_provider="groq"
    )
    
    tools = [
        navigate_to_app,
        login,
        get_form_fields,
        fill_and_submit_form,
        get_page_content,
        click_submit_form
    ]
    
    workflow = StateGraph(AgentState)
    
    async def call_model(state):
        sys_msg = SystemMessage(content="""You are a browser automation agent that can fill and submit any web form.

CRITICAL RULES FOR FORM SUBMISSION:
1. You MUST call get_form_fields() FIRST to discover what fields the form has
2. You MUST then map the user's data keys to the actual field names on the page
3. You MUST call fill_and_submit_form() ONE AT A TIME for each entry
4. You MUST wait for each submission to complete before starting the next one
5. NEVER call fill_and_submit_form() multiple times in parallel - this causes DOM detachment errors

WORKFLOW:
Step 1: navigate_to_app()
Step 2: If login credentials are provided, call login(). Otherwise skip.
Step 3: Call get_form_fields() to discover the actual field names/ids on the form.
Step 4: For EACH entry, map the user's data keys to the actual form field names you discovered. Build a JSON object using the ACTUAL field names as keys, then call fill_and_submit_form(form_data). WAIT for success before the next entry.
Step 5: get_page_content() to verify

IMPORTANT - FIELD MAPPING:
The user's data keys may NOT match the form's field names exactly.
For example, user data might say "name" but the form field might be called "full_name".
Or user data says "phone" but the form has "phone_number".
Use get_form_fields() to see the real field names (name, id, placeholder, label, ariaLabel, ariaLabelledBy), then remap the user's data accordingly.
The form_data JSON you pass to fill_and_submit_form MUST use the actual field names (the "name" or "id" attribute) from the form.

Your tools:
- navigate_to_app() - Goes to the app URL
- login(username, password) - Logs in (only use if credentials are provided)
- get_form_fields() - Returns all visible form fields with name, id, placeholder, label, ariaLabel, ariaLabelledBy. Also detects ARIA roles (textbox, combobox) and contenteditable elements. CALL THIS FIRST before filling any form.
- fill_and_submit_form(form_data) - Submits ONE entry. form_data is a JSON string where keys must match the actual field names from get_form_fields. The tool re-scans fields before each fill to handle dynamically rendered fields. It also auto-detects the submit button.
- get_page_content() - Gets current page text
- click_submit_form() - Clicks a "Submit Form" navigation button if available

Remember: DISCOVER fields first, MAP data to real field names, then fill SEQUENTIALLY!""")
        
        messages = [sys_msg] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"\ntool calling {len(response.tool_calls)} tool(s):")
            for tc in response.tool_calls:
                args_str = ', '.join(f"{k}={v[:30] if isinstance(v, str) else v}" 
                                   for k, v in tc.get('args', {}).items())
                print(f"   - {tc.get('name')}({args_str})")
        
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
        print("="*70)

        entries_list = []
        for i, entry in enumerate(form_data, 1):
            entries_list.append(f"Entry {i}: {json.dumps(entry)}")

        login_step = ""
        if LOGIN_USERNAME and LOGIN_PASSWORD:
            login_step = f'2. Log in with username="{LOGIN_USERNAME}" and password="{LOGIN_PASSWORD}"\n'
            step_offset = 3
        else:
            step_offset = 2

        user_message = f"""Please complete this workflow step-by-step:

1. Navigate to the application
{login_step}{step_offset}. Call get_form_fields() to discover what fields the form has
{step_offset + 1}. Submit these {len(form_data)} form entries ONE AT A TIME (wait for each to complete).
   For each entry, MAP the data keys below to the actual form field names you discovered, then call fill_and_submit_form with the correctly mapped JSON:

{chr(10).join(entries_list)}

{step_offset + 2}. After ALL entries are submitted, use get_page_content() to verify
{step_offset + 3}. Report the results

IMPORTANT: Discover fields FIRST, then map data keys to real field names, then submit SEQUENTIALLY!"""

        result = await graph.ainvoke({"messages": [("user", user_message)]})

        print("="*70)
        final_message = result["messages"][-1]
        print(final_message.content if hasattr(final_message, 'content') else str(final_message))
        print("="*70 + "\n")
        
        await page.wait_for_timeout(1000)
        await page.screenshot(path="final_result.png", full_page=True)
        print("Screenshot saved to final_result.png")

if __name__ == "__main__":
    asyncio.run(run_agent())