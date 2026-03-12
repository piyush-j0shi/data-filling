import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

logger = logging.getLogger(__name__)

_ON_LAMBDA = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
_WRITABLE_DIR = Path("/tmp") if _ON_LAMBDA else Path(__file__).parent

FORM_DATA_FILE = _WRITABLE_DIR / "form_data.json"
SCREENSHOT_PATH = str(_WRITABLE_DIR / "final_result.png")

APP_URL = os.environ.get("APP_URL", "https://your-public-ngrok-url.ngrok-free.app")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
BROWSER_ID = os.environ.get("BROWSER_ID", "your-bedrock-browser-id")
BROWSER_PROFILE_ID = os.environ.get("BROWSER_PROFILE_ID", "")
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "groq")

S3_INPUT_BUCKET = os.environ.get("S3_INPUT_BUCKET", "")
S3_RESULTS_BUCKET = os.environ.get("S3_RESULTS_BUCKET", "")


def validate_config() -> None:
    """Raise ValueError if any required environment variable is missing or unset."""
    if "your-public-ngrok-url" in APP_URL:
        raise ValueError("APP_URL is not configured. Set the APP_URL environment variable.")
    if "your-bedrock-browser-id" in BROWSER_ID:
        raise ValueError("BROWSER_ID is not configured. Set the BROWSER_ID environment variable.")
    if not S3_INPUT_BUCKET:
        raise ValueError("S3_INPUT_BUCKET is not configured. Set the S3_INPUT_BUCKET environment variable.")
    if not S3_RESULTS_BUCKET:
        raise ValueError("S3_RESULTS_BUCKET is not configured. Set the S3_RESULTS_BUCKET environment variable.")
    if MODEL_PROVIDER == "groq" and not os.environ.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not configured. Set the GROQ_API_KEY environment variable.")
    if MODEL_PROVIDER == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not configured. Set the OPENAI_API_KEY environment variable.")


def initialize_model(model_provider: str = MODEL_PROVIDER):
    provider = model_provider.lower()

    if provider == "groq":
        return init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq")

    if provider == "openai":
        return init_chat_model(model=(os.getenv("MODEL_NAME") or "").strip() or "gpt-4o", model_provider="openai")

    raise ValueError(f"Unsupported model provider: {provider}")
