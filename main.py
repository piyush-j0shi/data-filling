import json
import subprocess
import sys
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI()

FORM_DATA_FILE = Path(__file__).parent / "form_data.json"
WALKTHROUGH_SCRIPT = Path(__file__).parent / "walkthrough_1.py"


@app.post("/run")
async def run_walkthrough(file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are accepted")

    contents = await file.read()
    try:
        data = json.loads(contents)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in uploaded file")

    # Write the uploaded JSON to form_data.json
    FORM_DATA_FILE.write_text(json.dumps(data, indent=2))

    # Run walkthrough_1.py as a subprocess
    result = subprocess.run(
        [sys.executable, str(WALKTHROUGH_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(WALKTHROUGH_SCRIPT.parent),
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={"error": "Walkthrough script failed", "stderr": result.stderr},
        )

    return {"status": "success", "output": result.stdout}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
