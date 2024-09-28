import os
import logging
import json
import tempfile
import time
from pathlib import Path
import shutil
import nest_asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
import uuid
from typing import List, Dict
from openai import OpenAI
from llama_index.llms.anthropic import Anthropic

# LlamaIndex imports
from llama_index.core import  VectorStoreIndex,SimpleDirectoryReader,StorageContext,load_index_from_storage,Document
from llama_index.core.llms import ChatMessage, MessageRole

# Import API clients and configuration
from api_clients import get_api_client
from config import Config

from llama_index.core import SimpleDirectoryReader, Settings

from llama_index.core.llms import ChatMessage, MessageRole
import uuid
from config import Config
from typing import List, Dict
# Apply nest_asyncio to allow nested event loops (useful in some async environments)
nest_asyncio.apply()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up loggin
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI clients for RAG, TTS, STT
import openai
#import anthropic

openai.api_key = Config.OPENAI_API_KEY_RAG
client = OpenAI(api_key = Config.OPENAI_API_KEY_TTS)

# FastAPI app setup
app = FastAPI(title="Simple Vector Store API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize or load index
INDEX_STORAGE_PATH = "./storage"
index = None

if os.path.exists(INDEX_STORAGE_PATH):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_PATH)
        index = load_index_from_storage(storage_context, index_id="vector_index")
        logger.info("Loaded existing index from storage.")
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
else:
    logger.info("No existing index found. A new index will be created when needed.")

# Define the base directory and uploads directory
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / 'pdfs' / 'uploads'
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Uploads directory set to: {UPLOADS_DIR}")

# Rate limiter class
class RateLimiter:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill_time = time.time()

    def refill_tokens(self):
        now = time.time()
        time_passed = now - self.last_refill_time
        tokens_to_add = time_passed * self.rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill_time = now

    def consume_token(self):
        self.refill_tokens()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

# Initialize rate limiter
rate_limiter = RateLimiter(rate=1, capacity=5)

# Request counter
request_counter = 0

# Pydantic models
class BookSelection(BaseModel):
    bookId: str

class ChatRequest(BaseModel):
    message: str
    bookId: str
    systemPromptType: str = 'default'

class Flashcard(BaseModel):
    question: str
    answer: str

class FlashcardRequest(BaseModel):
    bookId: str

class FlashcardResponse(BaseModel):
    flashcards: List[Flashcard]

class InitChatRequest(BaseModel):
    bookId: str
    systemPromptType: str = 'default'

class ChatMessageContent(BaseModel):
    text: str

class ChatMessageRequest(BaseModel):
    sessionId: str
    message: str

## Helper functions
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def create_speech_with_retry(text, model, voice):
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    return response

def get_embedding_model():

    if Config.EMBEDDING_TYPE == 'openai':
        logger.info("Using OpenAI embeddings")
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(api_key=Config.OPENAI_API_KEY_RAG)
    else:
        raise ValueError(f"Invalid EMBEDDING_TYPE: {Config.EMBEDDING_TYPE}")

# Set up LlamaIndex settings
from llama_index.core import Settings

if Config.API_PROVIDER == 'anthropic':
    llm = Anthropic(api_key=Config.ANTHROPIC_API_KEY, model=Config.MODEL_NAME)
elif Config.API_PROVIDER == 'openai':
    llm = openai.ChatCompletion(api_key=Config.OPENAI_API_KEY_RAG, model=Config.MODEL_NAME)
else:
    raise ValueError(f"Unsupported API provider: {Config.API_PROVIDER}")

Settings.llm = llm
Settings.embed_model = get_embedding_model()

# In-memory storage for chat sessions
chat_sessions: Dict[str, VectorStoreIndex] = {}

# Endpoint to transcribe audio
@app.post("/voice/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await audio.read())
            temp_audio_path = temp_audio.name
        
        if Config.USE_WHISPER_API:
            with open(temp_audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            text = transcript
        else:
            # Placeholder for local Whisper implementation if needed
            text = "Transcription functionality is not enabled."
        
        os.remove(temp_audio_path)
        return {"text": text}
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for Text-to-Speech
@app.post("/voice/tts")
async def text_to_speech(request: Request):
    global request_counter
    request_counter += 1
    
    logger.debug(f"Received request #{request_counter}")
    
    try:
        request_data = await request.json()
        text = request_data.get("text")
        tts_engine = request_data.get("tts_engine", "openai")
        
        if not isinstance(text, str):
            logger.error(f"Request #{request_counter} - Invalid input: text is not a string")
            raise HTTPException(status_code=400, detail="Text input must be a string")
        
        if tts_engine == "openai":
            logger.debug(f"Request #{request_counter} - Using OpenAI TTS")
            try:
                response = await create_speech_with_retry(text, "tts-1", "alloy")
                audio_bytes = response['audio']
            except Exception as e:
                logger.error(f"Request #{request_counter} - OpenAI API error: {str(e)}")
                raise HTTPException(status_code=500, detail="Error generating audio from OpenAI API")
        else:
            logger.debug(f"Request #{request_counter} - Using open-source TTS")
            # Placeholder for open-source TTS implementation
            audio_bytes = b''  # Replace with actual audio bytes from TTS model
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        logger.info(f"Request #{request_counter} - Successfully generated audio")
        return FileResponse(temp_audio_path, media_type="audio/wav")
    except Exception as e:
        logger.error(f"Request #{request_counter} - Error generating TTS audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to upload and index PDF
@app.post("/pdf/upload")
async def upload_pdf(pdf: UploadFile = File(...)):
    try:
        filename = pdf.filename
        file_path = UPLOADS_DIR / filename
        
        if file_path.exists():
            raise HTTPException(status_code=400, detail="A file with this name already exists")
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(pdf.file, buffer)
        
        logger.info(f"PDF uploaded: {filename}")
        
        documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
        if not documents:
            raise HTTPException(status_code=500, detail="Failed to load document content")
        
        global index
        if index is None:
            index = VectorStoreIndex.from_documents(documents)
        else:
            index.insert(documents)
        
        index.storage_context.persist(persist_dir=INDEX_STORAGE_PATH)
        
        logger.info(f"Book indexed with ID: {filename}")
        
        return {"message": "File uploaded and indexed successfully", "book_id": filename}
    except Exception as e:
        logger.error(f"Error during PDF upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to list uploaded PDF
@app.get("/pdfs")
async def list_pdfs():
    try:
        pdfs = [f for f in os.listdir(UPLOADS_DIR) if f.endswith('.pdf')]
        logger.info(f"Retrieved PDF list: {pdfs}")
        return pdfs
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve PDFs: {str(e)}")

# Endpoint to select a book for chat
@app.post("/book/select")
async def select_book(selection: BookSelection):
    try:
        logger.info(f"Attempting to select book: {selection.bookId}")
        file_path = UPLOADS_DIR / selection.bookId
        if not file_path.exists():
            logger.warning(f"Book not found: {selection.bookId}")
            raise HTTPException(status_code=404, detail="Book not found")

        documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
        if not documents:
            raise HTTPException(status_code=500, detail="Failed to load document content")

        # Convert the loaded data into Document objects if necessary
        if isinstance(documents, list) and not isinstance(documents[0], Document):
            documents = [Document(text=doc.text, metadata=doc.metadata) for doc in documents]

        global index
        if index is None:
            index = VectorStoreIndex.from_documents(documents)
        else:
            index.insert_nodes(documents)
        
        index.storage_context.persist(persist_dir=INDEX_STORAGE_PATH)

        logger.info(f"Book selected and indexed successfully: {selection.bookId}")
        return {"message": "Book selected and indexed successfully", "book_id": selection.bookId}
    except Exception as e:
        logger.error(f"Error selecting book: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to serve PDF files
@app.get("/pdf/file/{filename}")
async def get_pdf(filename: str):
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        logger.warning(f"PDF not found: {filename}")
        raise HTTPException(status_code=404, detail="PDF not found")
    logger.info(f"Serving PDF: {filename}")
    return FileResponse(str(file_path), media_type="application/pdf", filename=filename)

# In-memory storage for chat sessions
chat_sessions: Dict[str, VectorStoreIndex] = {}

# Endpoint to initialize chat session
@app.post("/chat/init")
async def init_chat(request: InitChatRequest):
    try:
        file_path = UPLOADS_DIR / request.bookId
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")

        documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
        if not documents:
            raise HTTPException(status_code=500, detail="Failed to load document content")

        # Convert the loaded data into Document objects if necessary
        if isinstance(documents, list) and not isinstance(documents[0], Document):
            documents = [Document(text=doc.text, metadata=doc.metadata) for doc in documents]

        # Insert documents into the index
        index.insert_nodes(documents)
        index.storage_context.persist(persist_dir=INDEX_STORAGE_PATH)

        # Initialize chat engine
        chat_engine = index.as_chat_engine(verbose=True)

        system_prompt = Config.get_system_prompt(request.systemPromptType)
        chat_engine.chat_history.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))

        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = chat_engine

        logger.info(f"Chat session initialized with ID: {session_id}")
        return {"sessionId": session_id, "message": "Chat session initialized"}
    except Exception as e:
        logger.error(f"Error initializing chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to handle chat messages
@app.post("/chat/message")
async def chat_message(request: ChatMessageRequest):
    try:
        chat_engine = chat_sessions.get(request.sessionId)
        if not chat_engine:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        response = chat_engine.chat(request.message)
        result = response.response
        
        return {"reply": result}
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to get total request count
@app.get("/debug/request_count")
async def get_request_count():
    return {"total_requests": request_counter}

# Endpoint to get rate limiter status
@app.get("/debug/rate_limit_status")
async def get_rate_limit_status():
    return {
        "available_tokens": rate_limiter.tokens,
        "last_refill_time": rate_limiter.last_refill_time
    }

# Function to generate flashcards
def generate_flashcards_for_book(book_id: str) -> List[Flashcard]:
    try:
        # Retrieve the index
        chat_engine = chat_sessions.get(book_id)
        if not chat_engine:
            raise ValueError(f"No chat session found for book: {book_id}")

        # Use the index to generate flashcards
        response = chat_engine.chat(
            "Generate 10 flashcards based on the key concepts in this document. "
            "Each flashcard should have a question and an answer. "
            "Format the output as a list of JSON objects, each with 'question' and 'answer' fields."
        )
        
        logger.debug(f"Raw response from chat engine: {response.response}")

        # Parse the response and create Flashcard objects
        flashcards = []
        try:
            # Attempt to parse JSON array from response
            json_start = response.response.find('[')
            json_end = response.response.rfind(']') + 1
            if json_start != -1 and json_end != -1:
                json_str = response.response[json_start:json_end]
                parsed_response = json.loads(json_str)
                for item in parsed_response:
                    if isinstance(item, dict) and 'question' in item and 'answer' in item:
                        flashcards.append(Flashcard(question=item['question'], answer=item['answer']))
            else:
                raise ValueError("Could not find JSON array in response")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            # Fallback: Extract flashcards using regex
            import re
            matches = re.findall(r'Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)', response.response, re.DOTALL)
            flashcards = [Flashcard(question=q.strip(), answer=a.strip()) for q, a in matches]

        logger.debug(f"Parsed flashcards: {flashcards}")
        return flashcards
    except Exception as e:
        logger.error(f"Error generating flashcards for book {book_id}: {str(e)}")
        raise

# Endpoint to generate flashcards
@app.post("/generate-flashcards", response_model=FlashcardResponse)
async def generate_flashcards(request: FlashcardRequest):
    try:
        flashcards = generate_flashcards_for_book(request.bookId)
        if not flashcards:
            logger.warning(f"No flashcards generated for book {request.bookId}")
        return FlashcardResponse(flashcards=flashcards)
    except Exception as e:
        logger.error(f"Failed to generate flashcards: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
#
# Middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Received request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Returning response: Status {response.status_code}")
    return response

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting FastAPI server on port {Config.FASTAPI_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=Config.FASTAPI_PORT)
