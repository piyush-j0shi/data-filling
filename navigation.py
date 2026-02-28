import logging

from playwright.async_api import Page

from config import APP_URL

logger = logging.getLogger(__name__)

_SERVICES_DUMP_JS = """() => {
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
            id, 'ng-model': ngModel, label: findLabel(el),
            placeholder: placeholder ? placeholder.textContent.trim() : ''
        }));
    });

    document.querySelectorAll('input:not([type=hidden]), select').forEach(el => {
        const ngModel = el.getAttribute('ng-model') || '';
        const name = el.name || '';
        const id = el.id || '';
        if (!relevant(id) && !relevant(name) && !relevant(ngModel)) return;
        lines.push('INPUT: ' + JSON.stringify({
            tag: el.tagName, id, name, 'ng-model': ngModel,
            label: findLabel(el), placeholder: el.placeholder || ''
        }));
    });

    return lines.join('\\n');
}"""


async def login(page: Page):
    import os
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
    logger.info("  Current URL after nav: %s", page.url)

    visible_buttons = await page.evaluate("""() => {
        return [...document.querySelectorAll('button, a')]
            .filter(e => {
                const s = window.getComputedStyle(e);
                return s.display !== 'none' && s.visibility !== 'hidden' && e.offsetWidth > 0;
            })
            .map(e => e.textContent.trim())
            .filter(t => t.length > 0 && t.length < 60);
    }""")
    logger.info("  Visible buttons/links: %s", visible_buttons[:20])

    for sel in [
        "button:has-text('Create a Bill')", "a:has-text('Create a Bill')",
        "text=Create a Bill", "[ng-click*='createBill']",
        "[ng-click*='create']", "button:has-text('Create')",
    ]:
        try:
            await page.wait_for_selector(sel, timeout=4000, state="visible")
            await page.click(sel, timeout=4000)
            logger.info("  Clicked Create a Bill via: %s", sel)
            break
        except Exception:
            continue
    else:
        body_text = await page.evaluate("() => document.body.innerText.substring(0, 500)")
        logger.error("  Could not find Create a Bill button. Page text:\n  %s", body_text)
        raise RuntimeError("Could not find 'Create a Bill' button — see logs above for page state")

    await page.wait_for_timeout(2000)
    logger.info("  Create a Bill modal open")
    await page.click(f".modal-content label:has-text('{bill_type}')", timeout=10000)
    await page.wait_for_timeout(3000)
    logger.info("  Selected bill type: %s", bill_type)


async def get_services_dump(page: Page) -> str:
    try:
        result = await page.evaluate(_SERVICES_DUMP_JS)
        logger.info("\n  [SERVICES DUMP]\n%s\n  [END SERVICES DUMP]", result)
        return result
    except Exception as e:
        msg = f"[dump failed: {e}]"
        logger.warning("  %s", msg)
        return msg


async def click_create_bill(page: Page):
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
            logger.info("  Clicked '%s'", selector.split("'")[1])
            return
        except Exception:
            continue
    raise RuntimeError("Could not find Save & Exit / Post Charges & Close button")
