import os
import logging

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

logger = logging.getLogger(__name__)

FORM_DATA_FILE = "form_data.json"
APP_URL = os.environ.get("APP_URL", "https://your-public-ngrok-url.ngrok-free.app")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
BROWSER_ID = os.environ.get("BROWSER_ID", "your-bedrock-browser-id")
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "groq")

if "your-public-ngrok-url" in APP_URL:
    raise ValueError("APP_URL is not configured. Set the APP_URL environment variable.")
if "your-bedrock-browser-id" in BROWSER_ID:
    raise ValueError("BROWSER_ID is not configured. Set the BROWSER_ID environment variable.")
if MODEL_PROVIDER == "openai" and not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not configured. Set the OPENAI_API_KEY environment variable.")


def initialize_model(model_provider: str = MODEL_PROVIDER):
    provider = model_provider.lower()

    if provider == "groq":
        return init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq")

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set")
        return init_chat_model(os.getenv("MODEL_NAME", "openai:gpt-4o"))

    raise ValueError(f"Unsupported model provider: {provider}")
