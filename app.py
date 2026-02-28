import json
import logging
import subprocess
import sys
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI()

FORM_DATA_FILE = Path(__file__).parent / "form_data.json"
WALKTHROUGH_SCRIPT = Path(__file__).parent / "main.py"


@app.post("/run")
async def run_walkthrough(file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are accepted")

    contents = await file.read()
    logger.info("Received file upload: %s (%d bytes)", file.filename, len(contents))

    try:
        data = json.loads(contents)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in uploaded file")

    FORM_DATA_FILE.write_text(json.dumps(data, indent=2))
    logger.info("Saved form data to %s, starting execution", FORM_DATA_FILE)

    result = subprocess.run(
        [sys.executable, str(WALKTHROUGH_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(WALKTHROUGH_SCRIPT.parent),
    )

    if result.returncode != 0:
        logger.error("Execution failed (returncode=%d): %s", result.returncode, result.stderr[:500])
        raise HTTPException(
            status_code=500,
            detail={"error": "Script failed", "stderr": result.stderr},
        )

    logger.info("Execution completed successfully")
    return {"status": "success", "output": result.stdout}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
