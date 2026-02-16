import asyncio
import json
import os
import warnings

from contextlib import suppress
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from browser_use import Agent
from browser_use.browser.session import BrowserSession
from browser_use.browser import BrowserProfile
from browser_use.llm.openai.chat import ChatOpenAI as BrowserUseChatOpenAI
from bedrock_agentcore.tools.browser_client import BrowserClient

load_dotenv()

FORM_DATA_FILE = "form_data.json"
APP_URL = os.environ.get("APP_URL", "https://your-app-url.com")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
BROWSER_ID = os.environ.get("BROWSER_ID", "your-bedrock-browser-id")
LOGIN_USERNAME = os.environ.get("LOGIN_USERNAME", "admin")
LOGIN_PASSWORD = os.environ.get("LOGIN_PASSWORD", "admin")


def get_llm():
    # --- Option 1: Groq (active) ---
    # Groq exposes an OpenAI-compatible API, so browser-use's built-in ChatOpenAI works directly.
    return BrowserUseChatOpenAI(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    # --- Option 2: OpenRouter ---
    # return BrowserUseChatOpenAI(
    #     model="openai/gpt-4o-mini",
    #     api_key=os.environ["OPENROUTER_API_KEY"],
    #     base_url="https://openrouter.ai/api/v1",
    # )

    # --- Option 3: OpenAI direct ---
    # return BrowserUseChatOpenAI(
    #     model="gpt-4o-mini",
    #     api_key=os.environ["OPENAI_API_KEY"],
    # )


def build_login_task() -> str:
    return (
        f"Go to {APP_URL}. "
        f"On the login page, click on 'Continue as Practice Staff'. "
        f"Wait for the login form. Fill username '{LOGIN_USERNAME}' and password '{LOGIN_PASSWORD}', "
        f"then click the Login button. "
        f"If a survey or 'Next' button appears, click it to dismiss. "
        f"IMPORTANT: After clicking Login, the page takes 10-15 seconds to load. "
        f"Wait for it using the wait action (wait 10 seconds). "
        f"If the page looks empty or has no interactive elements, that means it is still loading — "
        f"DO NOT navigate away or refresh. Just wait and mark the task as done. "
        f"Once you see any dashboard content OR after waiting, mark the task as complete immediately. "
        f"Do NOT explore the dashboard."
    )


def build_bill_task(entry: dict, entry_num: int, total: int) -> str:
    financials_path = APP_URL.rstrip("/").rsplit("/ema", 1)[0] + "/ema/practice/financial/Financials.action#/home/bills"
    fields_desc = "\n".join(f"  - {key}: {value}" for key, value in entry.items())

    return (
        f"BILL ENTRY {entry_num}/{total}\n\n"
        f"1. Navigate to: {financials_path}\n"
        f"   IMPORTANT: After navigating, wait 10 seconds for the page to fully load before doing anything.\n"
        f"2. Look for a button or link with text 'Create a Bill' — it is NOT the 'Patient Search' box. "
        f"Click 'Create a Bill' and wait 3 seconds for the modal to open.\n"
        f"3. In the modal, click the 'Patient Bill' radio button (id='patientBillTypeRadio') and wait 3 seconds.\n"
        f"4. Fill in the form fields below ONE AT A TIME. For each field:\n"
        f"   - Find the field by its label text\n"
        f"   - For autocomplete fields: type the value, wait 2 seconds for suggestions, click the first suggestion\n"
        f"   - For date fields: clear the field and type the date value\n"
        f"   - For dropdowns/selects: click to open and select the matching option\n\n"
        f"Form fields to fill:\n{fields_desc}\n\n"
        f"5. After all fields are filled, click the 'Create Bill' button inside the modal.\n"
        f"6. Wait 5 seconds for the bill page to load, then fill the Services Rendered section if "
        f"service_code, service_units, or dx_ptrs fields were provided above.\n"
        f"7. Finally, click 'Save & Exit' and wait for confirmation.\n"
        f"8. Mark the task as done.\n"
    )


async def run_browser_task(browser_session: BrowserSession, llm, task: str, task_name: str, max_steps: int = 25):
    try:
        print(f"\n{'=' * 70}")
        print(f"Task: {task_name}")
        print(f"{'=' * 70}")

        agent = Agent(
            task=task,
            llm=llm,
            browser_session=browser_session,
            max_steps=max_steps,
        )

        result = await agent.run()

        print(f"Task completed: {task_name}")
        return result

    except Exception as e:
        print(f"Error during '{task_name}': {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    try:
        with open(FORM_DATA_FILE, "r") as f:
            form_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {FORM_DATA_FILE} not found.")
        return

    print(f"Loaded {len(form_data)} bill entries to submit\n")

    client = BrowserClient(AWS_REGION)
    client.start()
    ws_url, headers = client.generate_ws_headers()
    browser_session = None

    try:
        browser_profile = BrowserProfile(
            headers=headers,
            timeout=300000,
        )

        browser_session = BrowserSession(
            cdp_url=ws_url,
            browser_profile=browser_profile,
            keep_alive=True,
        )

        print("Initializing browser session...")
        await browser_session.start()
        print("Browser session ready.\n")

        llm = get_llm()

        await run_browser_task(browser_session, llm, build_login_task(), "Login", max_steps=8)
        results = []

        for i, entry in enumerate(form_data, 1):
            task = build_bill_task(entry, i, len(form_data))
            result = await run_browser_task(browser_session, llm, task, f"Bill Entry {i}/{len(form_data)}")
            status = "success" if result else "failed"
            results.append({"entry": i, "data": entry, "status": status})
            print(f"  Entry {i}: {status.upper()}")

        print(f"\n{'=' * 70}")
        print("SUBMISSION SUMMARY")
        print(f"{'=' * 70}")

        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]

        print(f"Total entries : {len(results)}")
        print(f"Successful    : {len(successful)}")
        print(f"Failed        : {len(failed)}")

        if failed:
            print(f"\n{'─' * 70}")
            print("FAILED ENTRIES:")
            for r in failed:
                fields = ", ".join(f'{k}="{v}"' for k, v in r["data"].items())
                print(f"  Entry {r['entry']}: {fields}")

        print(f"{'=' * 70}\n")

    finally:
        if browser_session:
            print("Closing browser session...")
            with suppress(Exception):
                await browser_session.close()
            print("Browser session closed.")
        client.stop()
        print("Browser client stopped.")


if __name__ == "__main__":
    asyncio.run(main())
