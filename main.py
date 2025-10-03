import os
import uuid
import logging
from typing import Optional


# Third-party libraries
import fal_client
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Configuration ---
# Fal AI uses FAL_KEY for authentication. Ensure this is set.
# You can get your key from https://fal.ai/dashboard/keys
FAL_KEY = os.environ.get("FAL_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Continuous Image Generation API (Fal AI)",
    description="An API that generates a single image using Fal AI and returns its URL.",
    version="1.3.0"
)

# --- CORS Configuration ---
# Allows the frontend to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
# These define the structure of the API requests and responses.
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt to generate an image from.")

class GenerateResponse(BaseModel):
    id: str
    status: str
    url: Optional[str] = None
    error_message: Optional[str] = None

# --- API Endpoint ---
@app.post("/api/generate-image", response_model=GenerateResponse)
async def generate_single_image(request: GenerateRequest):
    """
    Generates a single image using the fal-ai/flux-1-schnell model and returns its URL.
    """
    if not FAL_KEY:
        logger.critical("FAL_KEY environment variable is not set. Service is unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Image generation service is not configured."
        )

    attempt_id = str(uuid.uuid4())
    logger.info(f"Received request for prompt: '{request.prompt}' (ID: {attempt_id})")
    
    try:
        # --- Call the Fal AI API using the synchronous run function ---
        result = fal_client.run(
            "fal-ai/flux-1/schnell",
            arguments={
                "prompt": request.prompt,
                # FLUX models typically use 1024x1024
            }
        )

        # --- Validate and Extract URL from the Fal AI Response ---
        if not isinstance(result, dict) or "images" not in result:
             raise ValueError(f"Unexpected response format from Fal AI API: {result}")

        images = result["images"]
        if not isinstance(images, list) or len(images) == 0 or "url" not in images[0]:
            raise ValueError(f"No image URL found in Fal AI response: {result}")

        image_url = images[0]["url"]
        logger.info(f"Successfully generated image for ID {attempt_id}. URL: {image_url}")
        
        return GenerateResponse(id=attempt_id, status="success", url=image_url)

    except Exception as e:
        # Catching a generic exception as fal_client might raise various errors
        error_message = f"An error occurred with Fal AI: {e}"
        logger.critical(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail=error_message)