import asyncio
import json
import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from mangum import Mangum

from agent import run_agent
from config import FORM_DATA_FILE, initialize_model, validate_config
from extractor import OpenAIMedicalDocumentExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
EXECUTION_TIMEOUT = 900

_MAPPING_PROMPT_BASE = """You are given raw extracted data from a medical fee ticket PDF.
Your job is to map this data to the billing form fields listed below.

FORM FIELDS
-----------
Required fields (always include, use empty string if not found):
  - patient_name     : "Last, First" format only — no DOB, no MRN suffix
  - service_location : location name or code from the document
  - primary_biller   : billing provider name (empty string if not found)
  - date_of_service  : MM/DD/YYYY format
  - primary_provider : provider name or initials
  - diagnoses        : comma-separated ICD codes ONLY (e.g. "C44.49, L72.0")
                       Use ONLY codes from ACTIVE arrow connections (not crossed out).
                       ICD codes have a letter prefix (C44.49, L72.0). NOT CPT codes (17311, 17315).

Optional fields (omit the key entirely if not found in the data):
  - referring_provider
  - reportable_reason
  - medical_domain
  - provider_fee_schedule

Services (omit the key entirely if not found):
  - service_code : single primary CPT procedure code as a string

RULES
-----
1. Return a JSON object with only "bill" and "services" keys — nothing else.
2. "services" must be a flat object e.g. {"service_code": "17311"} — NOT a list.
3. For diagnoses — if multiple active ICD codes exist, include all as comma-separated string.
4. For patient_name — format must be "Last, First" only. No DOB. No extra suffixes.
5. For date_of_service — convert "7/30/24" or any format to MM/DD/YYYY.
6. Return ONLY the JSON object. No explanation, no markdown.

EXTRACTED DATA:
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()
    logger.info("App startup — config validated, writable dir: %s", FORM_DATA_FILE.parent)
    yield
    logger.info("App shutdown")


app = FastAPI(lifespan=lifespan)
handler = Mangum(app, lifespan="auto")


@app.post("/run")
async def run(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()

    if filename.endswith(".pdf"):
        return await _handle_pdf(file)
    elif filename.endswith(".json"):
        return await _handle_json(file)
    else:
        raise HTTPException(status_code=400, detail="Only .pdf or .json files are accepted")


async def _handle_pdf(file: UploadFile) -> dict:
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit")

    logger.info("Received PDF: %s (%d bytes)", file.filename, len(contents))

    # Step 1 — Extract
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        extractor = OpenAIMedicalDocumentExtractor()
        loop = asyncio.get_event_loop()
        extraction = await loop.run_in_executor(None, extractor.extract_from_pdf, tmp_path)
    except Exception as e:
        logger.error("Extraction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    logger.info("Extraction complete. Mapping to form fields via LLM...")

    # Step 2 — LLM maps extracted JSON → form_data structure
    try:
        form_data = await _map_to_form_data(extraction)
    except Exception as e:
        logger.error("LLM mapping failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM mapping failed: {e}")

    logger.info("Mapped form data: %s", json.dumps(form_data, indent=2))

    # Step 3 — Save and run agent
    return await _save_and_run(form_data)


async def _map_to_form_data(extraction: dict) -> list:
    prompt = _MAPPING_PROMPT_BASE + json.dumps(extraction, indent=2)

    model = initialize_model()
    loop = asyncio.get_event_loop()

    def _call():
        return model.invoke(prompt).content

    raw = await loop.run_in_executor(None, _call)

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw response:\n{raw}")

    entries = result if isinstance(result, list) else [result]
    for entry in entries:
        entry["bill_type"] = "Patient"
    return entries


async def _handle_json(file: UploadFile) -> dict:
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit")

    logger.info("Received JSON: %s (%d bytes)", file.filename, len(contents))

    try:
        data = json.loads(contents)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in uploaded file")

    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="JSON must be a list of bill entry objects")

    for entry in data:
        diagnoses = entry.get("bill", {}).get("diagnoses")
        if isinstance(diagnoses, list):
            for dx in diagnoses:
                dx_str = str(dx).strip()
                if not (1 < len(dx_str) <= 12):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Diagnosis '{dx_str}' must be between 2 and 12 characters",
                    )

    return await _save_and_run(data)


async def _save_and_run(form_data: list) -> dict:
    FORM_DATA_FILE.write_text(json.dumps(form_data, indent=2))
    logger.info("Saved form data (%d entries), starting agent", len(form_data))

    try:
        results = await asyncio.wait_for(run_agent(), timeout=EXECUTION_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error("Execution timed out after %ds", EXECUTION_TIMEOUT)
        raise HTTPException(status_code=504, detail="Execution timed out")
    except Exception as e:
        logger.error("Execution failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("Execution completed — %d entries processed", len(results or []))
    return {"status": "success", "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
