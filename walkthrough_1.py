import asyncio
import json
import os

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
MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

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
    with browser_session(AWS_REGION) as client:
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
        await page.locator('form input').first.fill(username, timeout=5000)
        await page.locator('input[type="password"]').fill(password, timeout=5000)
        await page.click('button[type="submit"]', timeout=5000)
        await page.wait_for_selector('input[name="name"]', timeout=10000, state="visible")
        await page.wait_for_timeout(1500)
        return f"Successfully logged in as '{username}'"
    
    except Exception as e:
        return f"✗ Login failed: {str(e)}"

@tool
async def fill_and_submit_form(name: str, email: str, phone: str, address: str, message: str) -> str:
    """Fills out and submits ONE form entry. Wait for this to complete before calling again."""
    page = get_page()
    try:
        await page.wait_for_selector('input[name="name"]', timeout=10000, state="visible")
        await page.wait_for_timeout(500)
        
        await page.fill('input[name="name"]', name, timeout=5000)
        await page.wait_for_timeout(200)
        
        await page.fill('input[name="email"]', email, timeout=5000)
        await page.wait_for_timeout(200)
        
        await page.fill('input[name="phone"]', phone, timeout=5000)
        await page.wait_for_timeout(200)
        
        await page.fill('input[name="address"]', address, timeout=5000)
        await page.wait_for_timeout(200)
        
        await page.fill('textarea[name="message"]', message, timeout=5000)
        await page.wait_for_timeout(200)
        
        await page.click('button[type="submit"]', timeout=5000)
        await page.wait_for_timeout(1000)
        
        name_value = await page.input_value('input[name="name"]')
        
        if name_value == "":
            return f"Successfully submitted form for '{name}' (form cleared)"
        else:
            return f"Submitted form for '{name}' but form may not have cleared"
            
    except Exception as e:
        return f"Form submission failed for '{name}': {str(e)}"

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
        return f"✗ Error retrieving page content: {str(e)}"

@tool
async def get_submission_count() -> str:
    """Returns the number of submissions shown in the table."""
    page = get_page()
    try:
        await page.wait_for_timeout(1000)
        rows = await page.locator('table tbody tr').count()
        return f"Found {rows} submission(s) in the table"
    
    except Exception as e:
        return f"Could not count submissions: {str(e)}"

@tool
async def click_view_submissions() -> str:
    """Clicks the 'View Submissions' button to see all submitted data."""
    page = get_page()
    try:
        await page.click('button:has-text("View Submissions")', timeout=5000)
        await page.wait_for_timeout(1000)
        return "✓ Navigated to View Submissions page"
    
    except Exception as e:
        return f"✗ Failed to click View Submissions: {str(e)}"

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
        fill_and_submit_form,
        get_page_content,
        get_submission_count,
        click_view_submissions,
        click_submit_form
    ]
    
    workflow = StateGraph(AgentState)
    
    async def call_model(state):
        sys_msg = SystemMessage(content="""You are a browser automation agent for a React form application.

CRITICAL RULES FOR FORM SUBMISSION:
1. You MUST call fill_and_submit_form() ONE AT A TIME for each entry
2. You MUST wait for each submission to complete before starting the next one
3. NEVER call fill_and_submit_form() multiple times in parallel - this causes DOM detachment errors
4. After each fill_and_submit_form() call, the tool will tell you if it succeeded

CORRECT WORKFLOW:
Step 1: navigate_to_app()
Step 2: login("admin", "admin")  
Step 3: fill_and_submit_form() for Entry 1 → WAIT for success
Step 4: fill_and_submit_form() for Entry 2 → WAIT for success
Step 5: fill_and_submit_form() for Entry 3 → WAIT for success
... (repeat for all entries, ONE AT A TIME)
Step N: click_view_submissions()
Step N+1: get_submission_count()

WRONG APPROACH (DO NOT DO THIS):
Calling fill_and_submit_form() multiple times in one response
Example: [fill_and_submit_form(entry1), fill_and_submit_form(entry2)]
This causes elements to detach from the DOM

Your tools:
- navigate_to_app() - Goes to the app URL
- login(username, password) - Logs in (use "admin", "admin")
- fill_and_submit_form(name, email, phone, address, message) - Submits ONE entry at a time
- get_page_content() - Gets current page text
- get_submission_count() - Counts rows in table
- click_view_submissions() - Switches to view page
- click_submit_form() - Switches back to form page

Remember: Process entries SEQUENTIALLY, not in parallel!""")
        
        messages = [sys_msg] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"\nAgent calling {len(response.tool_calls)} tool(s):")
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
            entries_list.append(
                f"Entry {i}: name='{entry['name']}', email='{entry['email']}', "
                f"phone='{entry['phone']}', address='{entry['address']}', "
                f"message='{entry['message'][:50]}...'"
            )

        user_message = f"""Please complete this workflow step-by-step:

1. Navigate to the application
2. Log in with username="admin" and password="admin"
3. Submit these {len(form_data)} form entries ONE AT A TIME (wait for each to complete):

{chr(10).join(entries_list)}

4. After ALL entries are submitted, click "View Submissions"
5. Get the submission count and verify all {len(form_data)} entries are there
6. Report the final count

IMPORTANT: Submit forms SEQUENTIALLY - wait for each submission to complete before starting the next one!"""

        print("Starting automation...\n")
        result = await graph.ainvoke({"messages": [("user", user_message)]})

        print("\n" + "="*70)
        print("AUTOMATION COMPLETE")
        print("="*70)
        final_message = result["messages"][-1]
        print(final_message.content if hasattr(final_message, 'content') else str(final_message))
        print("="*70 + "\n")
        
        await page.wait_for_timeout(1000)
        await page.screenshot(path="final_result.png", full_page=True)
        print("📸 Screenshot saved to final_result.png")

if __name__ == "__main__":
    asyncio.run(run_agent())