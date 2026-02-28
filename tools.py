import logging

from langchain_core.tools import tool
from client import get_page

logger = logging.getLogger(__name__)

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


@tool
async def fill_field(selector: str, value: str) -> str:
    """Fill a plain input/textarea field by CSS selector."""
    page = get_page()
    logger.debug("fill_field: selector=%s value=%s", selector, value)
    try:
        await page.wait_for_selector(selector, timeout=10000, state="visible")
        await page.locator(selector).scroll_into_view_if_needed(timeout=3000)
        try:
            await page.fill(selector, value, timeout=10000)
        except Exception:
            actual = await page.locator(selector).input_value()
            if actual != value:
                raise
        await page.wait_for_timeout(200)
        return f"Filled {selector} with '{value}'"
    except Exception as e:
        return f"Failed {selector}: {e}"


@tool
async def type_and_select(selector: str, text: str) -> str:
    """Type into an autocomplete/typeahead field and select the first suggestion.
    Pass the value EXACTLY as given — do not modify or shorten it.

    Args:
        selector: CSS selector for the input field
        text: Exact text to type
    """
    page = get_page()
    logger.debug("type_and_select: selector=%s text=%s", selector, text)
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
    logger.debug("select_option: selector=%s value=%s", selector, value)
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


@tool
async def fill_ui_select(container_selector: str, text: str) -> str:
    """Fill a ui-select dropdown widget. Clicks the toggle to open it, types to search,
    and selects the first match.

    Args:
        container_selector: CSS selector for the ui-select container (e.g. #placeOfServiceSelect)
        text: Text to type to search and filter options
    """
    page = get_page()
    logger.debug("fill_ui_select: container_selector=%s text=%s", container_selector, text)
    try:
        await page.wait_for_selector(container_selector, timeout=5000, state="visible")

        toggle = f"{container_selector} .ui-select-toggle"
        try:
            await page.wait_for_selector(toggle, timeout=3000, state="visible")
            await page.click(toggle)
        except Exception:
            await page.click(container_selector)
        await page.wait_for_timeout(700)

        search_selectors = [
            f"{container_selector} input.ui-select-search",
            f"{container_selector} input[type='search']",
            f"{container_selector} input",
            "div.ui-select-dropdown input.ui-select-search",
        ]
        search_el = None
        for sel in search_selectors:
            try:
                await page.wait_for_selector(sel, timeout=2000, state="visible")
                search_el = sel
                break
            except Exception:
                continue

        if not search_el:
            return f"Failed {container_selector}: could not find search input after clicking"

        await page.fill(search_el, "")
        await page.type(search_el, text, delay=50)
        await page.wait_for_timeout(2000)

        clicked = await page.evaluate(_CLICK_SUGGESTION_JS)
        if clicked:
            await page.wait_for_timeout(1000)
            return f"Selected: {clicked}"
        return f"No suggestion appeared after typing '{text}' in {container_selector}"
    except Exception as e:
        return f"Failed {container_selector}: {e}"


@tool
async def add_diagnosis(text: str) -> str:
    """Add a single diagnosis to the Diagnoses multi-select field.
    Call this once per diagnosis. For multiple diagnoses, call this tool
    once per entry — each call types into the search input and selects
    the first suggestion from the dropdown.

    The diagnoses field keeps previously selected items, so calling this
    multiple times builds up the list without clearing earlier entries.

    Args:
        text: ICD code or diagnosis name to search and select (e.g. 'J06.9' or 'Anxiety')
    """
    page = get_page()
    logger.debug("add_diagnosis: text=%s", text)
    selector = "#diagnosesSelect input.ui-select-search"
    try:
        await page.wait_for_selector(selector, timeout=5000, state="visible")
        await page.click(selector)
        await page.fill(selector, "")
        await page.type(selector, text, delay=50)
        await page.wait_for_timeout(2500)

        clicked = await page.evaluate(_CLICK_SUGGESTION_JS)
        if clicked:
            await page.wait_for_timeout(800)
            return f"Added diagnosis: {clicked}"
        return f"No diagnosis suggestion appeared after typing '{text}'"
    except Exception as e:
        return f"Failed to add diagnosis '{text}': {e}"


@tool
async def fill_dx_pointers(dx_ptrs: str) -> str:
    """Fill the DX Pointer slots in the Services Rendered table.
    Each slot is a separate ui-select dropdown. Pass a comma-separated string
    of pointer numbers — one per slot (e.g. '1' or '1,2').

    Args:
        dx_ptrs: Comma-separated pointer values, e.g. '1' or '1,2'
    """
    page = get_page()
    logger.debug("fill_dx_pointers: dx_ptrs=%s", dx_ptrs)
    try:
        values = [v.strip() for v in dx_ptrs.split(",") if v.strip()]
        results = []

        for idx, val in enumerate(values):
            status = await page.evaluate("""([idx]) => {
                const containers = document.querySelectorAll(
                    '.ui-select-container[ng-model="diagnosisPointerSelectorCtrl.dxPointer"]'
                );
                if (idx >= containers.length) return 'out_of_range';
                const toggle = containers[idx].querySelector('.ui-select-toggle');
                if (!toggle) return 'no_toggle';
                toggle.click();
                return 'clicked';
            }""", [idx])

            if status == "out_of_range":
                break
            if status != "clicked":
                results.append(f"slot {idx + 1}: {status}")
                continue

            await page.wait_for_timeout(700)

            search_sel = "div.ui-select-dropdown.open input.ui-select-search"
            try:
                await page.wait_for_selector(search_sel, timeout=2000, state="visible")
                await page.fill(search_sel, val)
                await page.wait_for_timeout(1000)
                clicked = await page.evaluate(_CLICK_SUGGESTION_JS)
                results.append(f"slot {idx + 1}: {'selected ' + clicked if clicked else 'no suggestion for ' + val}")
            except Exception as e:
                results.append(f"slot {idx + 1}: search input error - {e}")

            await page.wait_for_timeout(400)

        return "DX Pointers: " + "; ".join(results) if results else "No DX pointer values provided"
    except Exception as e:
        return f"Failed to fill DX pointers '{dx_ptrs}': {e}"


@tool
def done_filling() -> str:
    """Call this tool ONCE when every field has been filled, including all diagnoses.
    This signals the agent to stop — do NOT call any other tool after this."""
    return "DONE"
