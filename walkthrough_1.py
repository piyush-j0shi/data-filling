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
# Ensure your ngrok tunnel is running: ngrok http 5173 --host-header=rewrite
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
# Layer 1: AgentCore Browser Connection
# ---------------------------------------------------------------------------
@asynccontextmanager
async def get_bedrock_browser():
    """Starts an AWS Bedrock browser session and connects via CDP."""
    # This creates the AWS side of the session
    with browser_session(AWS_REGION) as client:
        ws_url, headers = client.generate_ws_headers()
        
        async with async_playwright() as pw:
            # Connect Playwright to the remote AWS browser
            browser = await pw.chromium.connect_over_cdp(ws_url, headers=headers)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = context.pages[0] if context.pages else await context.new_page()
            try:
                yield page
            finally:
                await page.close()
                await browser.close()

# ---------------------------------------------------------------------------
# Layer 2: Smart Tools for React Forms
# ---------------------------------------------------------------------------
@tool
async def navigate_to_app() -> str:
    """Navigates to the configured React application URL."""
    page = get_page()
    # Using 'load' is often safer than 'networkidle' for apps with background traffic
    await page.goto(APP_URL, wait_until="load", timeout=30000)
    return f"Navigated to {APP_URL}. Title: {await page.title()}"

@tool
async def fill_form_field(field_identifier: str, value: str) -> str:
    """Finds and fills a field using ID, Name, or Placeholder."""
    page = get_page()
    selector = f"#{field_identifier}, [name='{field_identifier}'], [placeholder*='{field_identifier}' i]"
    try:
        await page.wait_for_selector(selector, timeout=8000)
        await page.fill(selector, value)
        return f"Successfully filled '{field_identifier}' with '{value}'"
    except Exception:
        return f"Error: Could not find field matching '{field_identifier}'"

@tool
async def click_action(button_text_or_type: str) -> str:
    """Clicks a button by its type (e.g. 'submit') or its visible text."""
    page = get_page()
    selector = f"button[type='{button_text_or_type}'], button:has-text('{button_text_or_type}')"
    try:
        await page.click(selector)
        return f"Clicked '{button_text_or_type}'"
    except Exception as e:
        return f"Failed to click '{button_text_or_type}': {str(e)}"

@tool
async def get_page_text() -> str:
    """Returns a text summary of the current page to verify the form state."""
    page = get_page()
    try:
        # Ensure the page has actually reached a stable state
        await page.wait_for_load_state("domcontentloaded", timeout=10000)
        # Check if we are on a blank page
        if page.url == "about:blank":
            return "The browser is currently on a blank page. You need to navigate to the app first."
        
        return await page.inner_text("body", timeout=5000)
    except Exception as e:
        return f"Error retrieving page text: {str(e)}. The page might still be loading."

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
    
    tools = [navigate_to_app, fill_form_field, click_action, get_page_text]
    
    workflow = StateGraph(AgentState)
    
    async def call_model(state):
        sys_msg = SystemMessage(content="""You are a browser automation agent. You interact with a web application
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
        messages = [sys_msg] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        print(f"response is : {response}")
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
    # Apply nest_asyncio to prevent loop conflicts
    nest_asyncio.apply()
    
    try:
        form_data = load_form_data(FORM_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {FORM_DATA_FILE} not found. Create it with a JSON list of entries.")
        return

    graph = create_automation_agent()

    # FIX: Use get_bedrock_browser context, NOT create_automation_agent
    async with get_bedrock_browser() as page:
        set_page(page)
        print(f"Connected to Bedrock Browser (AgentCore)")

        user_message = f"""Please perform the following tasks on the web application at {APP_URL}:

1. Navigate to {APP_URL}
2. Log in with username "admin" and password "admin"
3. After logging in, you should see a form page titled "Submit Form"
4. For each of the following form entries, fill in all 5 fields and click Submit:

{json.dumps(form_data, indent=2)}

5. After submitting ALL entries, use get_page_text() to verify that all {len(form_data)} entries appear in the submissions table.
6. Report what you see in the table to confirm success."""

        print("Starting agent execution...")
        result = await graph.ainvoke({"messages": [("user", user_message)]})

        print("\n--- FINAL RESPONSE ---")
        print(result["messages"][-1].content)
        
        await page.screenshot(path="final_result.png")
        print("\nFinal result saved to final_result.png")

if __name__ == "__main__":
    asyncio.run(run_agent())