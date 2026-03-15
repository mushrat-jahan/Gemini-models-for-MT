import os
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
# import google.generativeai as genai 
from google import genai
from google.genai import types
from dotenv import load_dotenv
from loguru import logger
  
logger.add("logs/file.log")

logger.info("Starting the application")


load_dotenv()

api_key = os.getenv("Gemini_API_KEY")
if not api_key:
    raise ValueError("Gemini_API_KEY is not found in environment variables. Please check your .env file.")

#client = 
client = genai.Client(api_key = api_key)
logger.info("Gemini API Key configured")

# Define Pydantic models for structured translation outputs
class DefaultTranslation(BaseModel):
    original_text: str = Field(..., description="Original Bangla text")
    translated_text: str = Field(..., description="Translated text in target language")
    formal_alternative: Optional[str] = Field(None, description="More formal translation if applicable")
    # domains: List[str] = Field(..., description="List of domains")
    notes: Optional[str] = Field(None, description="Translation notes or cultural context")

class WordPair(BaseModel):
    bangla: str = Field(..., description="Bangla word")
    translated: str = Field(..., description="Translated word")
    part_of_speech: Optional[str] = Field(None, description="Part of speech")

class TranslationError(BaseModel):
    error: str = Field(..., description="Error message")
    original_text: Optional[str] = Field(None, description="Original text that caused the error")

class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_language: str = Field(..., description="Source language for translation")
    target_language: str = Field(..., description="Target language for translation")

# Factory function to create the appropriate model based on structure type
def create_translation_model(structure_type: str):
    models = {
        "default": DefaultTranslation,
    }
    return models.get(structure_type, DefaultTranslation)

def translate_text_structured(text: str, source_language: str, target_language: str):
    """
    Translate text from Bangla to the target language using Gemini with structured output
    validated by Pydantic models.
    """
    try:
        # System instructions 
        sys_instr = (
            f"""You are a professional translator and a domain classification expert. Translate the {source_language} text to {target_language}. Also Identify the domain(s) of the following paragraph (language: {source_language}).
            Return a JSON object with the following structure:
            {{
                              "original_text": "The original {source_language} text",
                              "translated_text": "The translated text in {target_language}",
                              "formal_alternative": "A more formal translation if applicable",
                              "notes": "Any translation notes or cultural context"
            }}
            """
        )

        # Model config
        response = client.models.generate_content(
            model="gemini-3.1-pro-preview",  
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=sys_instr,
                temperature=0.3,
                response_mime_type="application/json",
                response_schema=DefaultTranslation,  
            )
        )

        # Use .model_dump() to return a dictionary to the FastAPI endpoint
        if response.parsed:
            return response.parsed.model_dump()
        else:
            return json.loads(response.text)

    except Exception as e:
        logger.error(f"Gemini API Error: {str(e)}")
        return {"error": f"Model Processing Error: {str(e)}"}


# Create FastAPI appS
app = FastAPI(
    title="Structured Translator API",
    description="API for translating text to other languages with structured output",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/translate", response_model=Dict[str, Any])
async def translate(request: TranslationRequest):
    """
    Translate source text to the target language with structured output.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text to translate is required")
    if not request.target_language:
        raise HTTPException(status_code=400, detail="Target language is required")
    if not request.source_language:
        raise HTTPException(status_code=400, detail="Source language is required")
   
    logger.info("Translating text")
    result = translate_text_structured(
        text=request.text,
        source_language=request.source_language,
        target_language=request.target_language
    )

    logger.info(
        f"""
        Translation result:
        {result}
        """
    )

    logger.info("Translation complete")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result

@app.get("/")
async def root():
    """
    Root endpoint returning API information
    """
    return {
        "name": "Structured Bangla Translator API",
        "version": "1.0.0",
        "description": "API for translating Bangla text to other languages with structured output",
        "endpoints": {
            "/translate": "POST - Translate text with structured output",
            "/docs": "GET - API documentation (Swagger UI)",
            "/redoc": "GET - Alternative API documentation (ReDoc)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



#     sample_text = "আপনার দিনটি কেমন কাটছে?" 
