# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    LLAMAPARSE_API_KEY = os.getenv('LLAMAPARSE_API_KEY')
    OPENAI_API_KEY_TTS = os.getenv('OPENAI_API_KEY_TTS')
    OPENAI_API_KEY_RAG = os.getenv('OPENAI_API_KEY_RAG')
    OPENAI_API_KEY_STT = os.getenv('OPENAI_API_KEY_STT')
    
    # Model Configurations
    MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4')
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', 150))
    TOP_K = int(os.getenv('TOP_K', 3))
    SIMILARITY_CUTOFF = float(os.getenv('SIMILARITY_CUTOFF', 0.7))
    FASTAPI_PORT = int(os.getenv('FASTAPI_PORT', 3002))
    EMBEDDING_TYPE = os.getenv('EMBEDDING_TYPE', 'api')  # Set to 'api' for API-based embeddings
    USE_SIMPLE_API = os.getenv('USE_SIMPLE_API', 'false').lower() == 'true'
    API_PROVIDER = os.getenv('API_PROVIDER', 'openai').lower()
    USE_LLAMAPARSE = os.getenv('USE_LLAMAPARSE', 'true').lower() == 'true'
    USE_RETRIES = os.getenv('USE_RETRIES', 'false').lower() == 'true'
    
    # Add the USE_WHISPER_API attribute with a default value of True
    USE_WHISPER_API = os.getenv('USE_WHISPER_API', 'true').lower() == 'true'
    
    # LlamaIndex configurations
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', 1536))  # Dimension for ada-002
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
    
    # System prompts and query templates
    SYSTEM_PROMPTS = {
        'default': "You are a helpful AI assistant. Provide concise and accurate responses based on the given context.",
        'teacher': "You are an experienced teacher. Explain concepts clearly and provide examples when necessary.",
        'researcher': "You are a research assistant. Provide detailed and factual information, citing sources when possible.",
    }

    QUERY_TEMPLATES = {
        'default': "Provide a concise and helpful response based on the given context: {context_str} User: {query_str}",
        'elaborate': "Based on the context: {context_str}, provide a detailed explanation for the following question: {query_str}",
        'summarize': "Summarize the key points from the context: {context_str} that are relevant to answering: {query_str}",
    }

    @classmethod
    def get_system_prompt(cls, prompt_type='default'):
        return cls.SYSTEM_PROMPTS.get(prompt_type, cls.SYSTEM_PROMPTS['default'])

    @classmethod
    def get_query_template(cls, template_type='default'):
        return cls.QUERY_TEMPLATES.get(template_type, cls.QUERY_TEMPLATES['default'])
    
    @classmethod
    def get_embedding_model(cls):
        return cls.EMBEDDING_MODEL
