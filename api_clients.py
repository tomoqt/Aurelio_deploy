# api_clients.py
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
import anthropic
import os

# Read the USE_RETRIES environment variable
USE_RETRIES = os.getenv('USE_RETRIES', 'true').lower() == 'true'

def optional_retry(func):
    """Decorator to optionally apply retry logic"""
    if USE_RETRIES:
        return retry(
            wait=wait_exponential(multiplier=1, min=4, max=10),
            stop=stop_after_attempt(3)
        )(func)
    return func

class APIClient(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = self.create_client()

    @abstractmethod
    def create_client(self):
        pass

    @abstractmethod
    def generate_response(self, messages, temperature, max_tokens):
        pass

class OpenAIClient(APIClient):
    @optional_retry
    def create_client(self):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        return openai

    @optional_retry
    def generate_response(self, messages, temperature, max_tokens):
        response = self.client.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False  # Set to False for simplicity
        )
        return response.choices[0].message['content']

class AnthropicClient(APIClient):
    @optional_retry
    def create_client(self):
        return anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    @optional_retry
    def generate_response(self, messages, temperature, max_tokens):
        system_message = next((m['content'] for m in messages if m['role'] == 'system'), '')
        user_messages = [m for m in messages if m['role'] != 'system']
        
        anthropic_messages = []
        for msg in user_messages:
            anthropic_messages.append({
                "role": msg['role'],
                "content": msg['content']
            })

        response = self.client.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt=system_message + "\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in anthropic_messages]),
        )
        return response.completion.strip()

def get_api_client(api_provider, model_name):
    if api_provider == 'openai':
        return OpenAIClient(model_name)
    elif api_provider == 'anthropic':
        return AnthropicClient(model_name)
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")
