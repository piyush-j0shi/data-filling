import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from contextlib import asynccontextmanager
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, HTTPException, UploadFile
from mangum import Mangum

from agent import run_agent
from config import (
    FORM_DATA_FILE, initialize_model, validate_config,
    S3_INPUT_BUCKET, S3_RESULTS_BUCKET,
)
from extractor import OpenAIMedicalDocumentExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
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
_mangum = Mangum(app, lifespan="auto")

s3_client = boto3.client("s3")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit")

    job_id = uuid.uuid4().hex
    s3_key = f"{job_id}.pdf"

    s3_client.put_object(Bucket=S3_INPUT_BUCKET, Key=s3_key, Body=contents)
    logger.info("Job %s queued: s3://%s/%s", job_id, S3_INPUT_BUCKET, s3_key)

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
async def status(job_id: str):
    result_key = f"{job_id}.json"
    try:
        obj = s3_client.get_object(Bucket=S3_RESULTS_BUCKET, Key=result_key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {"job_id": job_id, "status": "pending"}
        logger.error("S3 error checking status for job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=str(e))


async def _process_s3_event(bucket: str, key: str, job_id: str) -> None:
    validate_config()

    obj = s3_client.get_object(Bucket=bucket, Key=key)
    contents = obj["Body"].read()

    try:
        form_data = await _process_pdf_bytes(contents)

        FORM_DATA_FILE.write_text(json.dumps(form_data, indent=2))
        logger.info("Job %s: saved %d entries, starting agent", job_id, len(form_data))
        logger.info("Job %s: extracted form_data = %s", job_id, json.dumps(form_data, indent=2))

        results = await asyncio.wait_for(run_agent(), timeout=EXECUTION_TIMEOUT)
        payload = {"job_id": job_id, "status": "success", "results": results}

    except asyncio.TimeoutError:
        logger.error("Job %s timed out after %ds", job_id, EXECUTION_TIMEOUT)
        payload = {"job_id": job_id, "status": "failed", "error": f"Timed out after {EXECUTION_TIMEOUT}s"}
    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e, exc_info=True)
        payload = {"job_id": job_id, "status": "failed", "error": str(e)}

    result_key = f"{job_id}.json"
    s3_client.put_object(
        Bucket=S3_RESULTS_BUCKET,
        Key=result_key,
        Body=json.dumps(payload, indent=2),
        ContentType="application/json",
    )
    logger.info("Job %s: result written to s3://%s/%s", job_id, S3_RESULTS_BUCKET, result_key)


async def _process_pdf_bytes(contents: bytes) -> list:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        extractor = OpenAIMedicalDocumentExtractor()
        loop = asyncio.get_event_loop()
        extraction = await loop.run_in_executor(None, extractor.extract_from_pdf, tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return await _map_to_form_data(extraction)



async def _map_to_form_data(extraction: dict) -> list:
    prompt = _MAPPING_PROMPT_BASE + json.dumps(extraction, indent=2)

    model = initialize_model()
    loop = asyncio.get_event_loop()

    def _call():
        return model.invoke(prompt).content

    raw = await loop.run_in_executor(None, _call)

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


def handler(event, context):
    records = event.get("Records", [])

    if not records or records[0].get("eventSource") != "aws:s3":
        logger.info("No S3 records in event — routing to FastAPI via Mangum")
        asyncio.set_event_loop(asyncio.new_event_loop())
        return _mangum(event, context)

    record = records[0]["s3"]
    bucket = record["bucket"]["name"]
    key = unquote_plus(record["object"]["key"])

    logger.info("S3 trigger: bucket=%s key=%s", bucket, key)

    job_id = key.rsplit(".", 1)[0]
    logger.info("S3 event accepted: job_id=%s", job_id)

    asyncio.run(_process_s3_event(bucket, key, job_id))
    return {"statusCode": 200, "body": json.dumps({"job_id": job_id})}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
