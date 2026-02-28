import json
import logging
from collections.abc import Callable

from client import get_bedrock_browser, set_page
from config import FORM_DATA_FILE, AWS_REGION, BROWSER_ID, SCREENSHOT_PATH, validate_config
from prompts import CREATE_BILL_FIELDS, SERVICES_FIELDS
from workflow import create_automation_agent, run_phase
from navigation import login, navigate_to_create_bill, get_services_dump, click_create_bill, save_and_exit

logger = logging.getLogger(__name__)

MAX_RETRIES = 1

_RETRY_MSG = (
    "An error interrupted the previous attempt. "
    "Review the tool calls above to see what was already filled. "
    "Retry only the failed tool and complete any remaining fields."
)


def _any_field_filled(messages: list) -> bool:
    fill_tools = {"fill_field", "type_and_select", "select_option", "fill_ui_select", "add_diagnosis"}
    return any(
        getattr(m, 'name', '') in fill_tools
        and not (getattr(m, 'content', '') or '').startswith(("Failed", "No "))
        for m in messages
    )


def _normalize_diagnoses(raw) -> list[str]:
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


def _build_bill_msg(bill_fields: dict) -> str:
    fields_str = '\n'.join(f'  - {k}: {v}' for k, v in bill_fields.items())
    return (
        "Fill ONLY the fields listed in DATA TO FILL, in the order shown.\n"
        "For the Diagnoses field, call add_diagnosis() once for EACH item in the list.\n"
        "Call done_filling() after ALL present fields (including all diagnoses) are complete.\n\n"
        f"FIELDS REFERENCE:\n{CREATE_BILL_FIELDS}\n\n"
        f"DATA TO FILL:\n{fields_str}"
    )


def _build_services_msg(phase3_fields: dict, page_dump: str) -> str:
    fields_str = '\n'.join(f'  - {k}: "{v}"' for k, v in phase3_fields.items())
    return (
        "You are on the Services Rendered form.\n\n"
        f"FIELDS REFERENCE (primary — use these exact tools and selectors):\n{SERVICES_FIELDS}\n\n"
        f"PAGE ELEMENTS (fallback — use only for fields not listed in FIELDS REFERENCE):\n{page_dump}\n\n"
        f"DATA TO FILL:\n{fields_str}\n\n"
        "Fill EVERY field present in DATA TO FILL using the tool and selector from FIELDS REFERENCE.\n"
        "Match each DATA TO FILL key to its closest entry in FIELDS REFERENCE by name.\n"
        "Call done_filling() only after ALL fields in DATA TO FILL have been filled."
    )


async def _run_with_retry(
    graph,
    build_msg_fn: Callable[[], str],
    phase_label: str,
    check_filled: Callable[[list], bool],
) -> tuple[list, bool, str]:
    """Run a graph phase with one retry on failure. Returns (messages, succeeded, failure_reason)."""
    messages: list = []
    failure = ""
    for attempt in range(1 + MAX_RETRIES):
        try:
            if attempt > 0:
                logger.warning("  [%s] Retrying (attempt %d)...", phase_label, attempt + 1)
            input_msgs = (
                [("user", build_msg_fn())] if attempt == 0 or not messages
                else messages + [("user", _RETRY_MSG)]
            )
            messages = await run_phase(graph, input_msgs, phase_label)
            if not check_filled(messages):
                failure = f"LLM did not fill any fields in {phase_label}"
                logger.warning("  [%s] Attempt %d: no fields filled, retrying...", phase_label, attempt + 1)
                continue
            return messages, True, failure
        except Exception as e:
            failure = str(e)
            logger.warning("  [%s] Attempt %d failed: %s", phase_label, attempt + 1, e)
    return messages, False, failure


def _log_summary(results: list[dict]) -> None:
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    logger.info("%s", "=" * 70)
    logger.info("SUBMISSION SUMMARY")
    logger.info("Total: %d  |  Success: %d  |  Failed: %d", len(results), len(successful), len(failed))
    if failed:
        logger.info("FAILED ENTRIES:")
        for r in failed:
            reason = r.get("reason") or "Unknown"
            logger.error("  Entry %d: %s", r['entry'], ', '.join(f'{k}="{v}"' for k, v in r["data"].items()))
            logger.error("  Reason: %s%s", reason[:300], "..." if len(reason) > 300 else "")
    logger.info("%s", "=" * 70)


async def run_agent() -> list[dict]:
    validate_config()

    try:
        with open(FORM_DATA_FILE) as f:
            form_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Form data file not found: {FORM_DATA_FILE}")

    logger.info("Loaded %d bill entries to submit", len(form_data))
    graph = create_automation_agent()

    async with get_bedrock_browser(AWS_REGION, BROWSER_ID) as page:
        set_page(page)
        await login(page)
        results: list[dict] = []

        for i, entry in enumerate(form_data, 1):
            logger.info("%s", "─" * 80)
            logger.info("BILL ENTRY %d/%d — %s", i, len(form_data),
                        ', '.join(f'{k}="{v}"' for k, v in entry.items()))
            logger.info("%s", "─" * 80)

            keys = list(entry.keys())
            split_idx = keys.index("diagnoses") + 1 if "diagnoses" in keys else len(keys)
            bill_fields = {k: entry[k] for k in keys[:split_idx] if k != "bill_type"}
            if "diagnoses" in bill_fields:
                bill_fields["diagnoses"] = json.dumps(_normalize_diagnoses(bill_fields["diagnoses"]))
            bill_type = entry.get("bill_type", "Patient")
            phase3_fields = {k: entry[k] for k in keys[split_idx:]}

            navigated, failure = False, ""
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

            _, bill_created, failure = await _run_with_retry(
                graph,
                lambda: _build_bill_msg(bill_fields),
                f"Fill Bill Fields (Entry {i})",
                _any_field_filled,
            )

            if not bill_created:
                results.append({"entry": i, "data": entry, "status": "failed", "reason": failure})
                logger.error("  Entry %d: FAILED (bill fields) — %s", i, failure[:200])
                continue

            await click_create_bill(page)
            page_dump = await get_services_dump(page)

            _, success, failure = await _run_with_retry(
                graph,
                lambda: _build_services_msg(phase3_fields, page_dump),
                f"Fill Services Rendered (Entry {i})",
                lambda msgs: not phase3_fields or _any_field_filled(msgs),
            )

            if success:
                await save_and_exit(page)
                logger.info("  Entry %d: SUCCESS", i)
                results.append({"entry": i, "data": entry, "status": "success"})
            else:
                logger.error("  Entry %d: FAILED — %s", i, failure[:200])
                results.append({"entry": i, "data": entry, "status": "failed", "reason": failure})

        _log_summary(results)
        await page.screenshot(path=SCREENSHOT_PATH, full_page=True)
        logger.info("Screenshot saved to %s", SCREENSHOT_PATH)
        return results
