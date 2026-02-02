import asyncio
import json
import os
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import boto3
import nest_asyncio
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from playwright.async_api import async_playwright, Page

from bedrock_agentcore.tools.browser_client import browser_session

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Layer 1: AgentCore Browser Connection with Custom Headers
# ---------------------------------------------------------------------------
@asynccontextmanager
async def get_bedrock_browser():
    """Starts an AWS Bedrock browser session and connects via CDP with custom headers."""
    with browser_session(AWS_REGION) as client:
        ws_url, headers = client.generate_ws_headers()
        
        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(ws_url, headers=headers)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            
            # Set custom headers to bypass ngrok warning
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

# ---------------------------------------------------------------------------
# Layer 2: Tools Specifically for Your React App
# ---------------------------------------------------------------------------
@tool
async def navigate_to_app() -> str:
    """Navigates to the configured React application URL."""
    page = get_page()
    await page.goto(APP_URL, wait_until="load", timeout=30000)
    await page.wait_for_timeout(1000)  # Give React time to render
    return f"Navigated to {APP_URL}. Title: {await page.title()}"

@tool
async def login(username: str, password: str) -> str:
    """Logs into the application with the given credentials.
    
    The login page has:
    - First input (no name attribute) for username
    - Second input with type='password' for password
    - Submit button with type='submit'
    """
    page = get_page()
    try:
        # Wait for the login form to be visible
        await page.wait_for_selector('form', timeout=5000)
        
        # Fill username - it's the first input in the form (no name attribute)
        username_input = page.locator('form input').first
        await username_input.fill(username)
        
        # Fill password - has type='password'
        password_input = page.locator('input[type="password"]')
        await password_input.fill(password)
        
        # Click the login button
        await page.click('button[type="submit"]')
        
        # Wait for navigation/form submission
        await page.wait_for_timeout(1500)
        
        return f"Successfully logged in as {username}"
    except Exception as e:
        return f"Login failed: {str(e)}"

@tool
async def fill_and_submit_form(name: str, email: str, phone: str, address: str, message: str) -> str:
    """Fills out and submits the form with all 5 fields.
    
    The form has these fields with name attributes:
    - input[name="name"]
    - input[name="email"]
    - input[name="phone"]
    - input[name="address"]
    - textarea[name="message"]
    """
    page = get_page()
    try:
        # Wait for the form page to load
        await page.wait_for_selector('input[name="name"]', timeout=5000)
        
        # Fill all form fields using their name attributes
        await page.fill('input[name="name"]', name)
        await page.fill('input[name="email"]', email)
        await page.fill('input[name="phone"]', phone)
        await page.fill('input[name="address"]', address)
        await page.fill('textarea[name="message"]', message)
        
        # Click submit button
        await page.click('button[type="submit"]')
        
        # Wait for the submission to process
        await page.wait_for_timeout(1000)
        
        return f"Successfully submitted form for {name}"
    except Exception as e:
        return f"Form submission failed: {str(e)}"

@tool
async def get_page_content() -> str:
    """Returns the current page text to verify state."""
    page = get_page()
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=5000)
        body_text = await page.inner_text("body", timeout=5000)
        return body_text
    except Exception as e:
        return f"Error retrieving page content: {str(e)}"

@tool
async def get_submission_count() -> str:
    """Returns the number of submissions shown in the table."""
    page = get_page()
    try:
        # Look for the submissions table
        rows = await page.locator('table tbody tr').count()
        return f"Found {rows} submissions in the table"
    except Exception as e:
        return f"Could not count submissions: {str(e)}"

@tool
async def click_view_submissions() -> str:
    """Clicks the 'View Submissions' button to see all submitted data."""
    page = get_page()
    try:
        # The button text contains "View Submissions"
        await page.click('button:has-text("View Submissions")')
        await page.wait_for_timeout(500)
        return "Navigated to View Submissions page"
    except Exception as e:
        return f"Failed to click View Submissions: {str(e)}"

@tool
async def click_submit_form() -> str:
    """Clicks the 'Submit Form' button to go back to the form page."""
    page = get_page()
    try:
        await page.click('button:has-text("Submit Form")')
        await page.wait_for_timeout(500)
        return "Navigated to Submit Form page"
    except Exception as e:
        return f"Failed to click Submit Form: {str(e)}"

# ---------------------------------------------------------------------------
# Layer 3: LangGraph Implementation
# ---------------------------------------------------------------------------
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

Your tools are specifically designed for this app:
1. navigate_to_app() - Goes to the app URL
2. login(username, password) - Logs in (use "admin", "admin")
3. fill_and_submit_form(name, email, phone, address, message) - Fills and submits one complete form entry
4. get_page_content() - Gets current page text
5. get_submission_count() - Counts rows in the submissions table
6. click_view_submissions() - Switches to the view submissions page
7. click_submit_form() - Switches back to the form page

WORKFLOW:
1. Navigate to the app
2. Log in with provided credentials
3. After login, you'll be on the "Submit Form" page
4. For each form entry in the data:
   - Call fill_and_submit_form() with all 5 fields at once
   - After each submission, the form clears automatically
5. After all submissions, click_view_submissions() to see the full table
6. Use get_submission_count() to verify all entries were saved
7. Report the final count

IMPORTANT:
- Use fill_and_submit_form() to submit each entry - it handles all 5 fields at once
- Don't try to fill individual fields - the tool does it all in one call
- The app stores data in localStorage, so all submissions persist""")
        
        messages = [sys_msg] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        print(f"\n=== Agent Response ===")
        print(f"Content: {response.content if hasattr(response, 'content') else response}")
        if hasattr(response, 'tool_calls'):
            print(f"Tool calls: {response.tool_calls}")
        return {"messages": [response]}
        
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
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

    print(f"Loaded {len(form_data)} form entries to submit")
    graph = create_automation_agent()

    async with get_bedrock_browser() as page:
        set_page(page)
        print(f"Connected to Bedrock Browser (AgentCore)\n")

        # Create a detailed task description
        entries_text = "\n".join([
            f"Entry {i+1}: name={e.get('name')}, email={e.get('email')}, phone={e.get('phone')}, address={e.get('address')}, message={e.get('message')}"
            for i, e in enumerate(form_data)
        ])

        user_message = f"""Please automate this workflow:

1. Navigate to the application
2. Log in with username="admin" and password="admin"
3. Submit these {len(form_data)} form entries (one at a time):

{entries_text}

4. After submitting all entries, view the submissions table
5. Count and report how many submissions are in the table
6. Confirm that all {len(form_data)} entries were successfully saved"""

        print("Starting agent execution...\n")
        result = await graph.ainvoke({"messages": [("user", user_message)]})

        print("\n" + "="*60)
        print("FINAL RESPONSE")
        print("="*60)
        final_message = result["messages"][-1]
        print(final_message.content if hasattr(final_message, 'content') else str(final_message))
        print("="*60)
        
        # Take a final screenshot
        await page.wait_for_timeout(1000)
        await page.screenshot(path="final_result.png", full_page=True)
        print("\nFinal screenshot saved to final_result.png")

if __name__ == "__main__":
    asyncio.run(run_agent())