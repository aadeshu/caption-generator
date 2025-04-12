from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import uvicorn
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # You can change to DEBUG for more verbosity
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS setup (modify as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific frontend domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model & processor
model = None
processor = None

# Load model and processor only once at startup
@app.on_event("startup")
def load_model():
    global model, processor
    logger.info("Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    logger.info("Model loaded successfully!")

# Updated system prompt template without extra text
SYSTEM_PROMPT_TEMPLATE = """You are an intelligent visual assistant trained to analyze images and extract detailed information about clothing items.

Your goal is to analyze the image for the clothing item belonging to the following category: {category}.

For the clothing item, output a JSON array that contains exactly one JSON object with the following keys:
- "category": the clothing category (as provided).
- "caption": a highly detailed, image-friendly description of the item that includes details such as color, material, pattern, style, design features, shape, and any visible accessories.

IMPORTANT:
- Your entire response MUST be exactly one JSON array containing one JSON object.
- DO NOT include any additional text, explanation, markdown formatting, triple backticks, or any other characters before or after the JSON.
- Output only raw, valid JSON with no extra spaces, newlines, or formatting characters.
"""

# Post-processing function to clean output if needed
def clean_output(raw_text: str) -> str:
    # Remove markdown code block markers and language identifiers
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()

# Inference logic
def caption_from_url(image_url: str, category: str, user_prompt="Describe the image.", max_tokens=2048*8) -> str:
    global model, processor

    # Format the system prompt with the provided category
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(category=category)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

# Allowed categories set
ALLOWED_CATEGORIES = {"Topwear", "Bottomwear", "Full Body Dress", "Caps and Hats", "Belts", "Footwear", "Handbags"}

# API endpoint: GET request with image_url and category query params
@app.get("/caption/")
def caption_endpoint(
    image_url: str = Query(..., description="URL of the image to caption"),
    category: str = Query(..., description="Clothing category to generate caption for. Allowed values: Topwear, Bottomwear, Full Body Dress, Caps and Hats, Belts, Footwear, Handbags")
):
    if category not in ALLOWED_CATEGORIES:
        logger.warning(f"Invalid category received: {category}")
        raise HTTPException(status_code=400, detail=f"Invalid category. Allowed values are: {', '.join(ALLOWED_CATEGORIES)}")
    
    try:
        raw_output = caption_from_url(image_url, category)
        logger.debug(f"Raw output from model: {raw_output}")
        try:
            captions = json.loads(raw_output)
        except json.JSONDecodeError:
            cleaned_output = clean_output(raw_output)
            logger.warning("Initial JSON parse failed. Trying cleaned output.")
            captions = json.loads(cleaned_output)

        if not isinstance(captions, list) or len(captions) != 1:
            raise ValueError("Output does not contain exactly one item as expected.")

        return JSONResponse(content={"captions": captions})

    except json.JSONDecodeError as je:
        logger.error(f"JSON parsing error: {je}")
        return JSONResponse(content={"error": "Failed to parse model output as JSON", "raw_output": raw_output}, status_code=500)
    except Exception as e:
        logger.exception("Unhandled exception occurred during captioning.")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Run server with: uvicorn filename:app --reload
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
