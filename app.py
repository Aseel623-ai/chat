from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pytesseract
import google.generativeai as genai
import os
import io
from dotenv import load_dotenv
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env variables - important for local development
load_dotenv()


# Configuration class
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TESSERACT_PATH = os.getenv("TESSERACT_PATH",
                               "/usr/bin/tesseract" if os.name != 'nt' else
                               r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    PORT = int(os.getenv("PORT", 8000))


# Validate configuration
if not Config.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Configure Gemini
try:
    genai.configure(api_key=Config.GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel(model_name="models/gemini-2.0-flash")  # Updated to newer model
    logger.info("Gemini AI configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini: {str(e)}")
    raise

# Configure Tesseract
try:
    pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH
    logger.info(f"Tesseract configured at: {Config.TESSERACT_PATH}")
except Exception as e:
    logger.error(f"Tesseract configuration failed: {str(e)}")
    raise

# Initialize FastAPI
app = FastAPI(
    title="Medical Chatbot API",
    description="API for medical analysis and consultation",
    version="1.0.0"
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class SymptomRequest(BaseModel):
    question: str
    language: Optional[str] = "ar"  # Default to Arabic


class AnalysisResponse(BaseModel):
    analysis_text: str
    ai_response: str
    success: bool = True


class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    success: bool = False


# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "Medical Chatbot API"}


@app.post("/chat", response_model=AnalysisResponse, responses={400: {"model": ErrorResponse}})
async def chat_with_bot(request: SymptomRequest):
    """Endpoint for medical consultation"""
    try:
        prompt = f"""أنت مساعد طبي ذكي تابع لإدارة طبية.

مهمتك هي:
1. فهم الأعراض أو السؤال المقدم من المستخدم
2. تقديم تشخيص مبدأي إن أمكن
3. إعطاء نصيحة طبية مناسبة
4. تحديد التخصص الطبي الذي يجب على المستخدم زيارته

السؤال أو الأعراض: {request.question}""" if request.language == "ar" else f"""You are an intelligent medical assistant.

Your tasks are:
1. Understand the user's symptoms or question
2. Provide preliminary diagnosis if possible
3. Give appropriate medical advice
4. Recommend the medical specialty to consult

Question/Symptoms: {request.question}"""

        response = model_gemini.generate_content(prompt)
        return {
            "analysis_text": request.question,
            "ai_response": response.text
        }
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={"error": "Processing failed", "details": str(e)}
        )


@app.post("/upload-analysis", response_model=AnalysisResponse, responses={400: {"model": ErrorResponse}})
async def upload_analysis(file: UploadFile = File(...)):
    """Endpoint for medical analysis upload"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Only image files are allowed"
            )

        # Process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Extract text (Arabic + English)
        extracted_text = pytesseract.image_to_string(
            image,
            lang='ara+eng',
            config='--psm 6'  # Assume a single uniform block of text
        )

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the image"
            )

        # Generate medical analysis
        prompt = f"""أنت مساعد طبي ذكي تابع للإدارة الطبية.

مهمتك هي:
1. قراءة نتائج التحاليل المرفقة
2. تلخيص النتائج بشكل تقرير طبي واضح
3. تقديم نصيحة طبية عامة حسب النتائج
4. اقتراح التخصص الطبي المناسب إن لزم

نتائج التحاليل المستخرجة: {extracted_text}"""

        response = model_gemini.generate_content(prompt)

        return {
            "analysis_text": extracted_text,
            "ai_response": response.text
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={"error": "Analysis failed", "details": str(e)}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.PORT,
        timeout_keep_alive=60  # Important for Render's free tier
    )
