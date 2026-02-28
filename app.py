import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from mangum import Mangum

from agent import run_agent
from config import FORM_DATA_FILE, validate_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 1 * 1024 * 1024  
EXECUTION_TIMEOUT = 900             

@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()
    logger.info("App startup — config validated, writable dir: %s", FORM_DATA_FILE.parent)
    yield
    logger.info("App shutdown")


app = FastAPI(lifespan=lifespan)
handler = Mangum(app, lifespan="auto")


@app.post("/run")
async def run_walkthrough(file: UploadFile = File(...)):
    if not (file.filename or "").endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are accepted")

    contents = await file.read()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 1 MB limit")

    logger.info("Received upload: %s (%d bytes)", file.filename, len(contents))

    try:
        data = json.loads(contents)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in uploaded file")

    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="JSON must be a list of bill entry objects")

    FORM_DATA_FILE.write_text(json.dumps(data, indent=2))
    logger.info("Saved form data (%d entries), starting execution",
                len(data) if isinstance(data, list) else 1)

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
