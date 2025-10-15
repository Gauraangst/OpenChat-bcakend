from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
import httpx
from io import BytesIO
import uuid # Added for generating dummy conversation IDs

# PDF / Image processing
import PyPDF2
from PIL import Image
import pytesseract

# NOTE: Supabase client and related imports/variables have been removed.

# Load environment variables first
load_dotenv()

# -------------------------------
# ENV variables & Initialization
# -------------------------------

# 1. Check for required environment variables (Only OpenRouter is strictly needed now)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise EnvironmentError("Missing required environment variable: OPENROUTER_API_KEY. Check your .env file.")

# NOTE: Supabase client initialization has been removed.

# -------------------------------
# FastAPI App Setup
# -------------------------------
app = FastAPI()

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------------------
# Models
# -------------------------------
class ChatRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: Optional[str] = None
    persona_id: Optional[str] = None
    persona_prompt: Optional[str] = None
    history: List[Dict] = []

# -------------------------------
# Helper: Call OpenRouter LLM
# -------------------------------
async def get_llm_reply(messages: List[Dict]):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": "gpt-4o-mini", "messages": messages},
            timeout=60.0
        )
    response.raise_for_status()
    
    data = response.json()
    if "choices" not in data:
        error_detail = data.get("error", "Unknown API error")
        raise HTTPException(status_code=500, detail=f"LLM API Error: {error_detail}")
        
    return data["choices"][0]["message"]["content"]

# -------------------------------
# /chat endpoint
# -------------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Create a DUMMY conversation ID since we removed Supabase
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Build LLM messages
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if request.persona_prompt:
            messages[0]["content"] += f" Persona prompt: {request.persona_prompt}"
        messages += request.history
        messages.append({"role": "user", "content": request.message})

        # Get LLM reply
        reply = await get_llm_reply(messages)

        # NOTE: Supabase storage logic removed.

        return {"reply": reply, "conversation_id": conversation_id}

    except httpx.HTTPStatusError as e:
        # Handling for HTTP errors from OpenRouter
        raise HTTPException(status_code=e.response.status_code, detail=f"OpenRouter API HTTP Error: {e}")
    except Exception as e:
        # General catch-all for other errors
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# -------------------------------
# /chat/upload endpoint
# -------------------------------
@app.post("/chat/upload")
async def upload_chat_file(
    file: UploadFile = File(...),
    user_id: str = File(...),
    conversation_id: Optional[str] = File(None),
    persona_id: Optional[str] = File(None),
    persona_prompt: Optional[str] = File(None),
    title: Optional[str] = File(None)
):
    try:
        content_bytes = await file.read()
        file_summary = ""

        # PDF
        if file.content_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(content_bytes))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                file_summary = f"Text of PDF '{file.filename}':\n{text[:4000]}..."
            else:
                file_summary = f"PDF '{file.filename}' is empty or unreadable."

        # Image
        elif file.content_type.startswith("image/"):
            image = Image.open(BytesIO(content_bytes))
            image.load() 
            text = pytesseract.image_to_string(image)
            if text.strip():
                file_summary = f"Text from image '{file.filename}':\n{text[:4000]}..."
            else:
                file_summary = f"Image '{file.filename}' has no readable text."

        else:
            file_summary = f"File '{file.filename}' of type '{file.content_type}' uploaded (not PDF or image)."

        # Create a DUMMY conversation ID since we removed Supabase
        conversation_id = conversation_id or str(uuid.uuid4())

        # LLM messages
        messages = [{"role": "system", "content": "You are a helpful assistant analyzing uploaded files."}]
        if persona_prompt:
            messages[0]["content"] += f" Persona prompt: {persona_prompt}"
        messages.append({"role": "user", "content": file_summary})

        # Get LLM reply
        reply = await get_llm_reply(messages)

        # NOTE: Supabase storage logic removed.

        return {
            "reply": reply,
            "conversation_id": conversation_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content_bytes)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")