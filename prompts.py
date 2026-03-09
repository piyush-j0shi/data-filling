# FIELD_REGISTRY — single source of truth for all form fields.
#
# To add a new field:
#   1. Add one entry here
#   2. Add the key under "bill" or "services" in form_data.json
#   That's it — nothing else needs to change.
#
# Keys:
#   key      — matches the key in form_data.json
#   label    — human-readable name (for reference only)
#   selector — CSS selector (None for tools that handle their own targeting)
#   tool     — which tool to call
#   form     — "bill" or "services"
#   optional — if True, skip when the value is empty or absent

FIELD_REGISTRY = [
    {"key": "patient_name",          "label": "Patient Name",          "selector": "#emaPatientQuickSearch",               "tool": "type_and_select",  "form": "bill",     "optional": False},
    {"key": "service_location",      "label": "Service Location",      "selector": "#placeOfServiceSelect",                "tool": "fill_ui_select",   "form": "bill",     "optional": False},
    {"key": "primary_biller",        "label": "Primary Biller",        "selector": "#renderingProviderSelect",             "tool": "fill_ui_select",   "form": "bill",     "optional": False},
    {"key": "date_of_service",       "label": "Date of Service",       "selector": "input#dateOfServiceInput",             "tool": "fill_field",       "form": "bill",     "optional": False},
    {"key": "primary_provider",      "label": "Primary Provider",      "selector": "#primaryProviderSelect",               "tool": "fill_ui_select",   "form": "bill",     "optional": False},
    {"key": "referring_provider",    "label": "Referring Provider",    "selector": "#referringProviderSelect",             "tool": "fill_ui_select",   "form": "bill",     "optional": True},
    {"key": "reportable_reason",     "label": "Reportable Reason",     "selector": "#reportableReasonSelect",              "tool": "select_option",    "form": "bill",     "optional": True},
    {"key": "provider_fee_schedule", "label": "Provider Fee Schedule", "selector": "#providerFeeScheduleSelect",           "tool": "select_option",    "form": "bill",     "optional": True},
    {"key": "medical_domain",        "label": "Medical Domain",        "selector": "#medicalSubdomain",                    "tool": "select_option",    "form": "bill",     "optional": True},
    {"key": "diagnoses",             "label": "Diagnoses",             "selector": None,                                   "tool": "add_diagnosis",    "form": "bill",     "optional": False},

    {"key": "service_code",          "label": "Service Code (CPT)",    "selector": "#cptQuickKeySelect",                   "tool": "fill_ui_select",   "form": "services", "optional": False},
    {"key": "service_units",         "label": "Units",                 "selector": 'input[name="serviceLineUnits1"]',       "tool": "fill_field",       "form": "services", "optional": True},
    {"key": "unit_charge",           "label": "Unit Charge",           "selector": 'input[name="serviceLineUnitCharge1"]',  "tool": "fill_field",       "form": "services", "optional": True},
    {"key": "dx_pointers",           "label": "DX Pointers",           "selector": None,                                   "tool": "fill_dx_pointers", "form": "services", "optional": True},
]


def _build_field_map(form: str) -> str:
    """Build a key→tool+selector lookup table from FIELD_REGISTRY for the given form."""
    lines = []
    for f in FIELD_REGISTRY:
        if f["form"] != form:
            continue
        tool, sel = f["tool"], f["selector"]
        note = "  [optional]" if f["optional"] else ""
        if tool == "add_diagnosis":
            lines.append(f'  {f["key"]:<26}→ add_diagnosis(text=VALUE)  — split comma lists, call once per item')
        elif tool == "fill_dx_pointers":
            lines.append(f'  {f["key"]:<26}→ fill_dx_pointers(dx_ptrs=VALUE){note}')
        else:
            lines.append(f'  {f["key"]:<26}→ {tool}  ·  selector: {sel}{note}')
    return "\n".join(lines)


def build_bill_msg(bill_data: dict) -> str:
    data_str = "\n".join(f"  {k}: {v}" for k, v in bill_data.items())
    return (
        "Fill the bill creation form.\n\n"
        "FIELD MAP — look up each data key here to find the right tool and selector:\n"
        f"{_build_field_map('bill')}\n\n"
        f"YOUR DATA:\n{data_str}\n\n"
        "Fill every key in YOUR DATA using the FIELD MAP. Call one tool per turn.\n"
        "For 'diagnoses': if the value contains commas, call add_diagnosis() once per item.\n"
        "When all keys are filled, call done_filling()."
    )


def build_services_msg(services_data: dict, page_dump: str) -> str:
    data_str = "\n".join(f"  {k}: {v}" for k, v in services_data.items())
    return (
        "Fill the Services Rendered form.\n\n"
        "FIELD MAP — look up each data key here to find the right tool and selector:\n"
        f"{_build_field_map('services')}\n\n"
        f"YOUR DATA:\n{data_str}\n\n"
        f"PAGE ELEMENTS (fallback — use only if a key is not found in the FIELD MAP):\n{page_dump}\n\n"
        "Fill every key in YOUR DATA using the FIELD MAP. Call one tool per turn.\n"
        "When all keys are filled, call done_filling()."
    )


SYSTEM_PROMPT = """You are a browser automation agent that fills web forms.

RULES:
1. Call ONLY ONE tool per turn. Never batch tool calls.
2. Fill ONLY the keys present in YOUR DATA — skip anything not listed.
3. For each key, look it up in the FIELD MAP to find the correct tool and selector.
4. Copy selectors EXACTLY as shown — never modify them.
5. Pass values EXACTLY as given — do not alter them.
6. For diagnoses: call add_diagnosis() once per item (split comma-separated lists).
7. If a tool returns "Failed", retry ONCE with corrected arguments, then move on.
8. When ALL keys in YOUR DATA are filled, call done_filling() immediately.
9. NEVER restart from the beginning after calling done_filling."""
