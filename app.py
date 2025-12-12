from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import io
import base64
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model once at startup (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

try:
    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    ).to(device)
    feature_extractor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

class ImageData(BaseModel):
    image_base64: str

    @validator('image_base64')
    def validate_base64(cls, v):
        if not v:
            raise ValueError("image_base64 cannot be empty")
        
        # Check if it has data URL prefix
        if ',' not in v:
            raise ValueError("Invalid base64 format - missing data URL prefix")
        
        # Basic length check (prevent huge uploads)
        if len(v) > 10_000_000:  # ~10MB limit
            raise ValueError("Image too large (max 10MB)")
        
        return v

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None
    }

@app.post("/caption")
async def generate_caption(data: ImageData):
    try:
        # Decode base64 image
        try:
            # Split on first comma to handle data:image/jpeg;base64,... format
            base64_data = data.image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(base64_data)
        except (IndexError, base64.binascii.Error) as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid base64 image data: {str(e)}"
            )

        # Open and validate image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {str(e)}"
            )

        # Generate caption
        try:
            pixel_values = feature_extractor(
                images=image, 
                return_tensors="pt"
            ).pixel_values.to(device)
            
            output_ids = model.generate(
                pixel_values, 
                max_length=16, 
                num_beams=4,
                early_stopping=True
            )
            
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            logger.info(f"Generated caption: {caption}")
            return {"caption": caption}
            
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory")
            raise HTTPException(
                status_code=503,
                detail="Server overloaded - try again in a moment"
            )
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate caption"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)