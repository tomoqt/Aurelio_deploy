import os
import logging
import json
import tempfile
import time
from pathlib import Path
import shutil
import nest_asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse  # Add StreamingResponse here
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
import uuid
from typing import List, Dict, Optional
from openai import OpenAI
from llama_index.llms.anthropic import Anthropic
import pyodbc
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import uvicorn
from config import Config, User, UserRole, Material, AssignMaterialRequest
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

### Rate limiter class
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
class User(BaseModel):
    id: int  # Add this line
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    hashed_password: str  # Keep existing fields
    role: UserRole

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[UserRole] = None

class Token(BaseModel):
    access_token: str
    token_type: str

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

class UserRegistration(BaseModel):
    username: str
    password: str
    email: str
    full_name: str
    role: UserRole = UserRole.student  # Default to student role

# New imports
from fastapi import Body
from pydantic import BaseModel, EmailStr

# New Pydantic model for profile update
class ProfileUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None

# Add these imports if not already present
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer


# Add this function to get the current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# Add this function to get the current active user
async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.get("/profile", response_model=User)
async def get_profile(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.put("/profile", response_model=User)
async def update_profile(
    profile_update: ProfileUpdate,
    current_user: User = Depends(get_current_active_user)
):
    try:
        conn = pyodbc.connect(Config.AZURE_SQL_CONNECTION_STRING)
        cursor = conn.cursor()

        update_fields = []
        update_values = []

        if profile_update.email is not None:
            update_fields.append("email = ?")
            update_values.append(profile_update.email)
            current_user.email = profile_update.email

        if profile_update.full_name is not None:
            update_fields.append("full_name = ?")
            update_values.append(profile_update.full_name)
            current_user.full_name = profile_update.full_name

        if update_fields:
            update_query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
            update_values.append(current_user.id)
            cursor.execute(update_query, update_values)
            conn.commit()

        conn.close()
        return current_user
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")

### Helper functions
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def create_speech_with_retry(text, model, voice):
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    return response.content

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
            text = transcript.text  # Extract the transcribed text
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

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(Config.AZURE_STORAGE_CONNECTION_STRING)

# Function to get or create user-specific container
def get_user_container_client(username: str):
    container_name = Config.get_user_container_name(username)
    container_client = blob_service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()
    return container_client

# Add these functions after the User model definition
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/pdf/upload")
async def upload_pdf(pdf: UploadFile = File(...), current_user: User = Depends(get_current_active_user)):
    try:
        container_client = get_user_container_client(current_user.username)
        filename = pdf.filename
        blob_client = container_client.get_blob_client(filename)
        
        content = await pdf.read()
        blob_client.upload_blob(content, overwrite=True)
        
        logger.info(f"PDF uploaded to Azure Blob Storage: {filename} for user {current_user.username}")
        
        # Always save the uploaded PDF to UPLOADS_DIR
        upload_path = UPLOADS_DIR / filename
        with open(upload_path, "wb") as f:
            f.write(content)
        logger.info(f"PDF saved locally at: {upload_path}")
        
        # Load documents from the saved PDF file
        documents = SimpleDirectoryReader(input_files=[str(upload_path)]).load_data()
        
        global index
        if index is None:
            index = VectorStoreIndex.from_documents(documents)
        else:
            index.insert(documents)
        
        index.storage_context.persist(persist_dir=INDEX_STORAGE_PATH)
        
        logger.info(f"Book indexed with ID: {filename} for user {current_user.username}")
        
        return {"message": "File uploaded and indexed successfully", "book_id": filename}
    except Exception as e:
        logger.error(f"Error during PDF upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdfs")
async def list_pdfs(current_user: User = Depends(get_current_active_user)):
    try:
        container_client = get_user_container_client(current_user.username)
        pdfs = [blob.name for blob in container_client.list_blobs()]
        logger.info(f"Retrieved PDF list for user {current_user.username}: {pdfs}")
        return pdfs
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve PDFs: {str(e)}")

@app.get("/pdf/file/{filename}")
async def get_pdf(filename: str, current_user: User = Depends(get_current_active_user)):
    try:
        container_client = get_user_container_client(current_user.username)
        blob_client = container_client.get_blob_client(filename)
        
        if not blob_client.exists():
            logger.warning(f"PDF not found: {filename} for user {current_user.username}")
            raise HTTPException(status_code=404, detail="PDF not found")
        
        stream = blob_client.download_blob()
        return StreamingResponse(stream.chunks(), media_type="application/pdf")
    except Exception as e:
        logger.error(f"Error retrieving PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to select a book for chat
@app.post("/book/select")
async def select_book(selection: BookSelection, current_user: User = Depends(get_current_active_user)):
    try:
        logger.info(f"Attempting to select book: {selection.bookId}")
        file_path = UPLOADS_DIR / selection.bookId

        if not file_path.exists():
            logger.info(f"File {selection.bookId} not found in UPLOADS_DIR. Attempting to download from Blob Storage.")
            # Attempt to download from Blob Storage
            container_client = get_user_container_client(current_user.username)
            blob_client = container_client.get_blob_client(selection.bookId)
            if blob_client.exists():
                try:
                    with open(file_path, "wb") as f:
                        download_stream = blob_client.download_blob()
                        f.write(download_stream.readall())
                    logger.info(f"Downloaded {selection.bookId} from Blob Storage to UPLOADS_DIR.")
                except Exception as download_error:
                    logger.error(f"Error downloading file from Blob Storage: {str(download_error)}")
                    raise HTTPException(status_code=500, detail="Failed to download file from storage")
            else:
                logger.warning(f"Book not found in Blob Storage: {selection.bookId}")
                raise HTTPException(status_code=404, detail="Book not found in storage")

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting book: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Endpoint to initialize chat session
@app.post("/chat/init")
async def init_chat(request: InitChatRequest, current_user: User = Depends(get_current_active_user)):
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

## Endpoint to handle chat messages
@app.post("/chat/message")
async def chat_message(request: ChatMessageRequest, current_user: User = Depends(get_current_active_user)):
    try:
        chat_engine = chat_sessions.get(request.sessionId)
        if not chat_engine:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        response = chat_engine.chat(request.message)
        result = str(response)
        
        return {"reply": result}
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

## #Endpoint to get total request count
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
async def generate_flashcards(request: FlashcardRequest, current_user: User = Depends(get_current_active_user)):
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

# Authentication utilities
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, Config.SECRET_KEY, algorithm=Config.ALGORITHM)
    return encoded_jwt

# Function to get user from database
def get_user(username: str):
    conn = pyodbc.connect(Config.AZURE_SQL_CONNECTION_STRING)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", username)
    user_data = cursor.fetchone()
    conn.close()
    if user_data:
        return User(
            id=user_data.id,  # Map the 'id' from the database to the User model
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            disabled=user_data.disabled,
            hashed_password=user_data.hashed_password,
            role=UserRole(user_data.role)
        )

# Function to authenticate user
def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token_expires = timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during login: {str(e)}")

@app.post("/register")
async def register_user(user: UserRegistration):
    try:
        conn = pyodbc.connect(Config.AZURE_SQL_CONNECTION_STRING)
        cursor = conn.cursor()

        # Check if the users table exists, if not, create it
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='users' AND xtype='U')
        CREATE TABLE users (
            id INT IDENTITY(1,1) PRIMARY KEY,
            username NVARCHAR(50) UNIQUE NOT NULL,
            hashed_password NVARCHAR(100) NOT NULL,
            email NVARCHAR(100) UNIQUE NOT NULL,
            full_name NVARCHAR(100) NOT NULL,
            disabled BIT NOT NULL DEFAULT 0,
            role NVARCHAR(10) NOT NULL DEFAULT 'student'
        )
        """)
        conn.commit()

        # Check if the role column exists, if not, add it
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID('users') AND name = 'role')
        BEGIN
            ALTER TABLE users ADD role NVARCHAR(10) NOT NULL DEFAULT 'student'
        END
        """)
        conn.commit()

        hashed_password = get_password_hash(user.password)
        cursor.execute("INSERT INTO users (username, hashed_password, email, full_name, disabled, role) VALUES (?, ?, ?, ?, ?, ?)",
                       user.username, hashed_password, user.email, user.full_name, False, user.role.value)
        conn.commit()

        cursor.execute("SELECT id FROM users WHERE username = ?", user.username)
        user_id = cursor.fetchone().id

        conn.close()
        return {"message": "User registered successfully", "user_id": user_id}
    except Exception as e:
        logger.error(f"Error during user registration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to register user: {str(e)}")

def get_current_user_with_role(required_role: UserRole):
    async def current_user_with_role(token: str = Depends(oauth2_scheme)):
        try:
            payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            token_data = TokenData(username=username, role=payload.get("role"))
        except JWTError:
            raise credentials_exception
        user = Config.get_user(username=token_data.username)
        if user is None:
            raise credentials_exception
        if user.role != required_role:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return current_user_with_role

@app.get("/control-panel/data", response_model=Dict[str, List[User] or List[Material]])
async def get_control_panel_data(current_user: User = Security(get_current_user_with_role(UserRole.teacher))):
    try:
        conn = pyodbc.connect(Config.AZURE_SQL_CONNECTION_STRING)
        cursor = conn.cursor()

        # Fetch students assigned by this teacher
        cursor.execute("""
            SELECT DISTINCT u.id, u.username, u.email, u.full_name, u.disabled, u.role
            FROM users u
            JOIN assignments a ON u.id = a.student_id
            WHERE a.teacher_id = ?
        """, current_user.id)
        students = cursor.fetchall()
        student_list = [User(id=stu.id, username=stu.username, email=stu.email, full_name=stu.full_name, disabled=stu.disabled, role=UserRole(stu.role)) for stu in students]

        # Fetch all available materials
        cursor.execute("SELECT id, title, description FROM materials")
        materials = cursor.fetchall()
        material_list = [Material(id=mat.id, title=mat.title, description=mat.description) for mat in materials]

        conn.close()
        return {"students": student_list, "materials": material_list}
    except Exception as e:
        logger.error(f"Error fetching control panel data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch control panel data")

@app.post("/assign-material")
async def assign_material(request: AssignMaterialRequest, current_user: User = Security(get_current_user_with_role(UserRole.teacher))):
    try:
        conn = pyodbc.connect(Config.AZURE_SQL_CONNECTION_STRING)
        cursor = conn.cursor()
        
        # Check if the student exists and is assigned to this teacher
        cursor.execute("SELECT 1 FROM assignments WHERE student_id = ? AND teacher_id = ?", request.studentId, current_user.id)
        if not cursor.fetchone():
            raise HTTPException(status_code=400, detail="Student not found or not assigned to this teacher")
        
        # Insert the new assignment
        cursor.execute("INSERT INTO assignments (student_id, material_id, teacher_id, assigned_at) VALUES (?, ?, ?, GETDATE())",
                       request.studentId, request.materialId, current_user.id)
        conn.commit()
        conn.close()
        
        return {"message": "Material assigned successfully"}
    except Exception as e:
        logger.error(f"Error assigning material: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to assign material")

@app.get("/materials", response_model=List[Material])
async def get_all_materials(current_user: User = Security(get_current_user_with_role(UserRole.teacher))):
    try:
        conn = pyodbc.connect(Config.AZURE_SQL_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, description FROM materials")
        materials = cursor.fetchall()
        conn.close()
        return [Material(id=mat.id, title=mat.title, description=mat.description) for mat in materials]
    except Exception as e:
        logger.error(f"Error fetching materials: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch materials")

@app.get("/students/{student_id}/assigned-materials", response_model=List[Material])
async def get_assigned_materials(student_id: int, current_user: User = Depends(get_current_active_user)):
    try:
        if current_user.role == UserRole.student and current_user.id != student_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        conn = pyodbc.connect(Config.AZURE_SQL_CONNECTION_STRING)
        cursor = conn.cursor()
        
        if current_user.role == UserRole.teacher:
            # Check if the student is assigned to this teacher
            cursor.execute("SELECT 1 FROM assignments WHERE student_id = ? AND teacher_id = ?", student_id, current_user.id)
            if not cursor.fetchone():
                raise HTTPException(status_code=403, detail="Access denied")
        
        cursor.execute("""
            SELECT m.id, m.title, m.description
            FROM materials m
            JOIN assignments a ON m.id = a.material_id
            WHERE a.student_id = ?
        """, student_id)
        materials = cursor.fetchall()
        conn.close()
        
        return [Material(id=mat.id, title=mat.title, description=mat.description) for mat in materials]
    except Exception as e:
        logger.error(f"Error fetching assigned materials: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch assigned materials")

###
if __name__ == "__main__":
    logger.info(f"Starting FastAPI server on port {Config.FASTAPI_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=Config.FASTAPI_PORT)