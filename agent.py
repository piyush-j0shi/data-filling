import asyncio
import json
import logging
import os

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from playwright.async_api import Page

from client import get_bedrock_browser, set_page
from tools import fill_field, type_and_select, select_option, fill_ui_select, add_diagnosis, fill_dx_pointers, done_filling

load_dotenv()

logger = logging.getLogger(__name__)

FORM_DATA_FILE = "form_data.json"
APP_URL = os.environ.get("APP_URL", "https://your-public-ngrok-url.ngrok-free.app")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
BROWSER_ID = os.environ.get("BROWSER_ID", "your-bedrock-browser-id")
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "groq")

if "your-public-ngrok-url" in APP_URL:
    raise ValueError("APP_URL is not configured. Set the APP_URL environment variable.")
if "your-bedrock-browser-id" in BROWSER_ID:
    raise ValueError("BROWSER_ID is not configured. Set the BROWSER_ID environment variable.")

if MODEL_PROVIDER == "openai" and not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not configured. Set the OPENAI_API_KEY environment variable.")


def initialize_model(model_provider: str = MODEL_PROVIDER):
    provider = model_provider.lower()

    if provider == "groq":
        return init_chat_model(
            model="llama-3.3-70b-versatile",
            model_provider="groq"
        )

    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set")

        model_name = os.getenv("MODEL_NAME", "openai:gpt-4o")
        return init_chat_model(model_name)

    else:
        raise ValueError(f"Unsupported model provider: {provider}")


SERVICES_FIELDS = """[Form: servicesRendered] — fill in this exact numbered order, ONE tool call per turn:
1.  [fill_ui_select]   "Service Code (CPT)"  → #cptQuickKeySelect
2.  [fill_field]       "Units"               → input[name="serviceLineUnits1"]
3.  [fill_field]       "Unit Charge"         → input[name="serviceLineUnitCharge1"]
4.  [fill_dx_pointers] "DX Pointers"         → dx_ptrs=<value>
5.  [done_filling]     — call after ALL present fields are filled"""


CREATE_BILL_FIELDS = """[Form: createBillForm] — fill in this exact numbered order, ONE tool call per turn:
1.  [type_and_select] "Patient Name"       → #emaPatientQuickSearch
2.  [fill_ui_select]  "Service Location"   → #placeOfServiceSelect
3.  [fill_ui_select]  "Primary Biller"     → #renderingProviderSelect
4.  [fill_field]      "Date of Service"    → input#dateOfServiceInput
5.  [fill_ui_select]  "Primary Provider"   → #primaryProviderSelect
6.  [fill_ui_select]  "Referring Provider" → #referringProviderSelect
7.  [select_option]   "Reportable Reason"  → #reportableReasonSelect
8.  [add_diagnosis]   "Diagnoses"          → call add_diagnosis(text=<value>) for EACH diagnosis entry
                                             If diagnoses is a comma-separated list, split it and call
                                             add_diagnosis once per item.
9.  [done_filling]    — call with NO arguments after ALL diagnoses have been added"""


SYSTEM_PROMPT = """You are a browser automation agent that fills web forms.

STRICT RULES — follow exactly:
1. Call ONLY ONE tool per turn. Never batch or combine tool calls.
2. Fill ONLY the fields present in DATA TO FILL — skip any field not listed there.
   Fill them in the order they appear in DATA TO FILL.
3. Use the tool shown in [brackets] next to each field in the FIELDS reference:
     [type_and_select]   → type_and_select(selector=..., text=<value>)
     [fill_ui_select]    → fill_ui_select(container_selector=..., text=<value>)
     [fill_field]        → fill_field(selector=..., value=<value>)
     [select_option]     → select_option(selector=..., value=<value>)
     [add_diagnosis]     → add_diagnosis(text=<single_diagnosis>)
     [fill_dx_pointers]  → fill_dx_pointers(dx_ptrs=<value>)
     [done_filling]      → done_filling()
4. Copy selectors EXACTLY as listed — never modify them.
5. Pass values EXACTLY as given in DATA TO FILL — do not alter them.
6. For the Diagnoses field specifically:
     - If the value is a single entry, call add_diagnosis once.
     - If the value is a list (e.g. ["J06.9", "Z87.39"]) or comma-separated string,
       call add_diagnosis ONCE PER ITEM, waiting for the result each time.
     - Each add_diagnosis call adds to the multi-select without clearing previous ones.
7. If a tool returns "Failed", retry ONCE. If it still fails, move to the next field.
8. After the LAST present field is filled, call done_filling() immediately.
9. NEVER restart from field 1 after reaching done_filling. It is the final call."""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_automation_agent():
    llm = initialize_model()
    tools = [fill_field, type_and_select, select_option, fill_ui_select, add_diagnosis, fill_dx_pointers, done_filling]

    workflow = StateGraph(AgentState)

    async def call_model(state):
        all_msgs = list(state["messages"])
        if len(all_msgs) > 60:
            all_msgs = all_msgs[:1] + all_msgs[-58:]

        response = await llm.bind_tools(tools, parallel_tool_calls=False).ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + all_msgs
        )

        if hasattr(response, 'tool_calls') and response.tool_calls:
            tc = response.tool_calls[0]
            args_str = ', '.join(
                f"{k}={repr(v[:60]) if isinstance(v, str) else v}"
                for k, v in tc.get('args', {}).items()
            )
            logger.debug("Tool: %s(%s)", tc.get('name'), args_str)

        return {"messages": [response]}

    tool_node = ToolNode(tools)

    async def call_tools(state):
        result = await tool_node.ainvoke(state)
        messages_to_add = list(result.get("messages", []))

        for msg in messages_to_add:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            logger.debug("  → %s", content[:150])

        if messages_to_add:
            last_content = getattr(messages_to_add[-1], 'content', '') or ''
            is_failure = (
                last_content.startswith("Failed") or
                last_content.startswith("No ")
            )
            if is_failure:
                failed_tool = ""
                for m in reversed(state["messages"]):
                    if hasattr(m, 'tool_calls') and m.tool_calls:
                        failed_tool = m.tool_calls[-1].get('name', '')
                        break
                messages_to_add.append(HumanMessage(content=(
                    f"'{failed_tool}' just failed: \"{last_content[:200]}\". "
                    "Do NOT call done_filling yet. Retry that same tool once with "
                    "corrected arguments (e.g. a different selector, shorter search "
                    "text, or adjusted value) to fix the issue. "
                    "Only move to the next field if it fails a second time."
                )))

        return {"messages": messages_to_add}

    def route_after_agent(state):
        """If the agent called done_filling, still execute it (so the tool runs),
        then route_after_tools will stop the graph."""
        msgs = state["messages"]
        for msg in reversed(msgs):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                return "tools"
            if hasattr(msg, 'content'):
                return END
        return END

    def route_after_tools(state):
        """Stop the graph as soon as a tool returned 'DONE' (done_filling result)."""
        msgs = state["messages"]
        for msg in reversed(msgs):
            if hasattr(msg, 'content') and msg.content == "DONE":
                return END
            if hasattr(msg, 'tool_calls'):
                break
        return "agent"

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_after_agent)
    workflow.add_conditional_edges("tools", route_after_tools)

    return workflow.compile()


async def run_phase(graph, messages, phase_name, recursion_limit=50):
    logger.info("%s", phase_name)
    logger.info("%s", "─" * 60)
    collected = []
    try:
        async for chunk in graph.astream(
            {"messages": messages},
            {"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            if "messages" in chunk:
                collected = chunk["messages"]
        return collected
    except Exception:
        if collected:
            return collected
        raise


async def login(page: Page):
    username = os.environ.get("LOGIN_USERNAME", "admin")
    password = os.environ.get("LOGIN_PASSWORD", "admin")

    logger.info("Navigating to app & selecting Practice Staff")
    await page.goto(APP_URL, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(2000)
    await page.click("input[name='redirectToNonPatientLoginPage']", timeout=10000)
    await page.wait_for_timeout(2000)

    logger.info("Logging in")
    await page.fill("#username", username, timeout=5000)
    await page.fill("#password", password, timeout=5000)
    await page.click("button[name='login']", timeout=5000)
    await page.wait_for_timeout(3000)

    try:
        next_btn = await page.wait_for_selector("text=Next", timeout=3000)
        if next_btn:
            await next_btn.click()
            await page.wait_for_timeout(2000)
            logger.info("Dismissed survey page")
    except Exception:
        pass

    logger.info("Logged in successfully")


async def navigate_to_create_bill(page: Page, bill_type: str):
    financials_url = (APP_URL.rstrip("/").rsplit("/ema", 1)[0]
                      + "/ema/practice/financial/Financials.action#/home/bills")
    logger.info("  Navigating to: %s", financials_url)
    await page.goto(financials_url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(4000)

    current_url = page.url
    logger.info("  Current URL after nav: %s", current_url)

    visible_buttons = await page.evaluate("""() => {
        const els = [...document.querySelectorAll('button, a')];
        return els
            .filter(e => {
                const s = window.getComputedStyle(e);
                return s.display !== 'none' && s.visibility !== 'hidden' && e.offsetWidth > 0;
            })
            .map(e => e.textContent.trim())
            .filter(t => t.length > 0 && t.length < 60);
    }""")
    logger.info("  Visible buttons/links: %s", visible_buttons[:20])

    create_bill_selectors = [
        "button:has-text('Create a Bill')",
        "a:has-text('Create a Bill')",
        "text=Create a Bill",
        "[ng-click*='createBill']",
        "[ng-click*='create']",
        "button:has-text('Create')",
    ]
    clicked = False
    for sel in create_bill_selectors:
        try:
            await page.wait_for_selector(sel, timeout=4000, state="visible")
            await page.click(sel, timeout=4000)
            clicked = True
            logger.info("  Clicked Create a Bill via: %s", sel)
            break
        except Exception:
            continue

    if not clicked:
        body_text = await page.evaluate("() => document.body.innerText.substring(0, 500)")
        logger.error("  Could not find Create a Bill button. Page text preview:\n  %s", body_text)
        raise RuntimeError("Could not find 'Create a Bill' button — see logs above for page state")

    await page.wait_for_timeout(2000)
    logger.info("  Create a Bill modal open")
    await page.click(f".modal-content label:has-text('{bill_type}')", timeout=10000)
    await page.wait_for_timeout(3000)
    logger.info("  Selected bill type: %s", bill_type)


async def _get_services_dump(page: Page) -> str:
    """Capture only service-line-relevant elements from the Services Rendered page."""
    try:
        result = await page.evaluate("""() => {
            const lines = [];

            const svcPatterns = [
                /serviceLine/i, /cpt/i, /dxPointer/i, /diagnosisPointer/i,
                /selectedCode/i, /item\\.units/i, /item\\.unitCharge/i, /modifier/i,
            ];
            const relevant = (str) => svcPatterns.some(p => p.test(str));

            const excludedIds = new Set(['cptCodeSelect']);

            function findLabel(el) {
                if (el.id) {
                    const lbl = document.querySelector('label[for="' + el.id + '"]');
                    if (lbl) return lbl.textContent.trim();
                }
                let node = el.parentElement;
                for (let i = 0; i < 5 && node; i++) {
                    const lbl = node.querySelector('label, .control-label');
                    if (lbl && lbl.textContent.trim()) return lbl.textContent.trim();
                    node = node.parentElement;
                }
                return '';
            }

            document.querySelectorAll('.ui-select-container').forEach(el => {
                const ngModel = el.getAttribute('ng-model') || '';
                const id = el.id || '';
                if (excludedIds.has(id)) return;
                if (!relevant(id) && !relevant(ngModel)) return;
                const placeholder = el.querySelector('.ui-select-placeholder');
                lines.push('UI-SELECT: ' + JSON.stringify({
                    id: id,
                    'ng-model': ngModel,
                    label: findLabel(el),
                    placeholder: placeholder ? placeholder.textContent.trim() : ''
                }));
            });

            document.querySelectorAll('input:not([type=hidden]), select').forEach(el => {
                const ngModel = el.getAttribute('ng-model') || '';
                const name = el.name || '';
                const id = el.id || '';
                if (!relevant(id) && !relevant(name) && !relevant(ngModel)) return;
                lines.push('INPUT: ' + JSON.stringify({
                    tag: el.tagName,
                    id: id,
                    name: name,
                    'ng-model': ngModel,
                    label: findLabel(el),
                    placeholder: el.placeholder || ''
                }));
            });

            return lines.join('\\n');
        }""")
        logger.info("\n  [SERVICES DUMP]\n%s\n  [END SERVICES DUMP]", result)
        return result
    except Exception as e:
        msg = f"[dump failed: {e}]"
        logger.warning("  %s", msg)
        return msg


async def click_create_bill(page: Page):
    """Always done by Python after LLM signals done — never delegated to the LLM."""
    try:
        await page.click(".modal-content button:has-text('Create Bill')", timeout=10000)
    except Exception:
        await page.click("button:has-text('Create Bill'):visible", timeout=10000)
    await page.wait_for_timeout(3000)
    logger.info("  Clicked Create Bill")


async def save_and_exit(page: Page):
    for selector in [
        "button:has-text('Save & Exit')",
        "button:has-text('Post Charges & Close')",
        "button:has-text('Post Charges')",
    ]:
        try:
            await page.click(selector, timeout=5000)
            await page.wait_for_timeout(3000)
            label = selector.split("'")[1]
            logger.info("  Clicked '%s'", label)
            return
        except Exception:
            continue
    raise RuntimeError("Could not find Save & Exit / Post Charges & Close button")


MAX_RETRIES = 1


def _any_field_filled(messages) -> bool:
    """Return True only if at least one fill tool call returned a non-failure result."""
    fill_tools = {"fill_field", "type_and_select", "select_option",
                  "fill_ui_select", "add_diagnosis"}
    for m in messages:
        if getattr(m, 'name', '') in fill_tools:
            content = getattr(m, 'content', '') or ''
            if not content.startswith("Failed") and not content.startswith("No "):
                return True
    return False


def _normalize_diagnoses(raw) -> list[str]:
    """Ensure diagnoses is always a list of strings, regardless of input format."""
    if isinstance(raw, list):
        return [str(d).strip() for d in raw if str(d).strip()]
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(d).strip() for d in parsed if str(d).strip()]
            except json.JSONDecodeError:
                pass
        return [d.strip() for d in raw.split(",") if d.strip()]
    return []


async def run_agent():
    try:
        with open(FORM_DATA_FILE, "r") as f:
            form_data = json.load(f)
    except FileNotFoundError:
        logger.error("Error: %s not found.", FORM_DATA_FILE)
        return

    logger.info("Loaded %d bill entries to submit", len(form_data))
    graph = create_automation_agent()

    async with get_bedrock_browser(AWS_REGION, BROWSER_ID) as page:
        set_page(page)
        logger.info("Connected to Bedrock Browser")

        await login(page)
        results = []

        for i, entry in enumerate(form_data, 1):
            fields_display = ', '.join(f'{k}="{v}"' for k, v in entry.items())
            logger.info("%s", "─" * 80)
            logger.info("BILL ENTRY %d/%d", i, len(form_data))
            logger.info("%s", fields_display)
            logger.info("%s", "─" * 80)

            success = False
            failure = ""

            keys = list(entry.keys())
            split_idx = keys.index("diagnoses") + 1 if "diagnoses" in keys else len(keys)
            bill_fields = {k: entry[k] for k in keys[:split_idx] if k != "bill_type"}

            if "diagnoses" in bill_fields:
                diag_list = _normalize_diagnoses(bill_fields["diagnoses"])
                bill_fields["diagnoses"] = json.dumps(diag_list)

            bill_type = entry.get("bill_type", "Patient")

            navigated = False
            for attempt in range(1 + MAX_RETRIES):
                try:
                    if attempt > 0:
                        logger.warning("  [Phase 1] Retrying navigation (attempt %d)...", attempt + 1)
                    await navigate_to_create_bill(page, bill_type)
                    navigated = True
                    break
                except Exception as e:
                    failure = str(e)
                    logger.warning("  [Phase 1] Navigation attempt %d failed: %s", attempt + 1, e)

            if not navigated:
                results.append({"entry": i, "data": entry, "status": "failed", "reason": failure})
                logger.error("  Entry %d: FAILED (navigation) — %s", i, failure[:200])
                continue

            bill_created = False
            bill_fields_str = '\n'.join(f'  - {k}: {v}' for k, v in bill_fields.items())
            fill_msg = f"""Fill ONLY the fields listed in DATA TO FILL, in the order shown.
For the Diagnoses field, call add_diagnosis() once for EACH item in the list.
Call done_filling() after ALL present fields (including all diagnoses) are complete.

FIELDS REFERENCE:
{CREATE_BILL_FIELDS}

DATA TO FILL:
{bill_fields_str}"""

            fill_messages = []
            for attempt in range(1 + MAX_RETRIES):
                try:
                    if attempt > 0:
                        logger.warning("  [Phase 2] Retrying bill fields (attempt %d)...", attempt + 1)

                    input_msgs = (
                        [("user", fill_msg)] if attempt == 0 or not fill_messages
                        else fill_messages + [("user",
                            "An error interrupted the previous attempt. "
                            "Review the tool calls above to see what was already filled. "
                            "Retry only the failed tool and complete any remaining fields.")]
                    )

                    fill_messages = await run_phase(
                        graph, input_msgs,
                        f"Fill Bill Fields (Entry {i})"
                    )

                    if not _any_field_filled(fill_messages):
                        failure = "LLM did not fill any form fields"
                        logger.warning("  [Phase 2] Attempt %d: no fields filled, retrying...", attempt + 1)
                        continue

                    await click_create_bill(page)
                    page_dump = await _get_services_dump(page)
                    bill_created = True
                    break

                except Exception as e:
                    failure = str(e)
                    logger.warning("  [Phase 2] Bill fields attempt %d failed: %s", attempt + 1, e)

            if not bill_created:
                results.append({"entry": i, "data": entry, "status": "failed", "reason": failure})
                logger.error("  Entry %d: FAILED (bill fields) — %s", i, failure[:200])
                continue

            phase3_fields = {k: entry[k] for k in keys[split_idx:]}
            svc_fields_str = '\n'.join(f'  - {k}: "{v}"' for k, v in phase3_fields.items())
            svc_msg = f"""You are on the Services Rendered form.

FIELDS REFERENCE (primary — use these exact tools and selectors):
{SERVICES_FIELDS}

PAGE ELEMENTS (fallback — use only for fields not listed in FIELDS REFERENCE):
{page_dump}

DATA TO FILL:
{svc_fields_str}

Fill EVERY field present in DATA TO FILL using the tool and selector from FIELDS REFERENCE.
Match each DATA TO FILL key to its closest entry in FIELDS REFERENCE by name.
Call done_filling() only after ALL fields in DATA TO FILL have been filled."""

            svc_messages = []
            for attempt in range(1 + MAX_RETRIES):
                try:
                    if attempt > 0:
                        logger.warning("  [Phase 3] Retrying services (attempt %d)...", attempt + 1)

                    input_msgs = (
                        [("user", svc_msg)] if attempt == 0 or not svc_messages
                        else svc_messages + [("user",
                            "An error interrupted the previous attempt. "
                            "Review the tool calls above to see what was already filled. "
                            "Retry only the failed tool and complete any remaining fields.")]
                    )

                    svc_messages = await run_phase(
                        graph, input_msgs,
                        f"Fill Services Rendered (Entry {i})"
                    )

                    if phase3_fields and not _any_field_filled(svc_messages):
                        failure = "LLM did not fill any service fields"
                        logger.warning("  [Phase 3] Attempt %d: no service fields filled, retrying...", attempt + 1)
                        continue

                    await save_and_exit(page)
                    success = True
                    break

                except Exception as e:
                    failure = str(e)
                    logger.warning("  [Phase 3] Services attempt %d failed: %s", attempt + 1, e)

            if success:
                logger.info("  Entry %d: SUCCESS", i)
                results.append({"entry": i, "data": entry, "status": "success"})
            else:
                logger.error("  Entry %d: FAILED — %s", i, failure[:200])
                results.append({"entry": i, "data": entry, "status": "failed", "reason": failure})

        logger.info("%s", "=" * 70)
        logger.info("SUBMISSION SUMMARY")
        logger.info("%s", "=" * 70)
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]
        logger.info("Total    : %d", len(results))
        logger.info("Success  : %d", len(successful))
        logger.info("Failed   : %d", len(failed))

        if failed:
            logger.info("%s", "─" * 70)
            logger.info("FAILED ENTRIES:")
            for r in failed:
                flds = ', '.join(f'{k}="{v}"' for k, v in r["data"].items())
                reason = r["reason"] or "Unknown"
                logger.error("  Entry %d: %s", r['entry'], flds)
                logger.error("  Reason: %s%s", reason[:300], '...' if len(reason) > 300 else '')

        logger.info("%s", "=" * 70)
        await page.screenshot(path="final_result.png", full_page=True)
        logger.info("Screenshot saved to final_result.png")
