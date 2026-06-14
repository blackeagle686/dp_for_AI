from base import BaseLLM
from enum import Enum 
from openai import OpenAI
from google import genai 
from anthropic import Anthropic
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()

class OpenAIClient(BaseLLM): 
    class OpenaAIModles(Enum):
        GPT_4_O = "gpt-4o"
        GPT_3_5_TURBO = "gpt-3.5-turbo"
        GPT_4_O_MINI = "gpt-4o-mini"
        GPT_5 = "gpt-5"
        GPT_5_MINI = "gpt-5-mini"    
        GPT_5_PRO = "gpt-5-pro"
    
    def _check_model(self, name:str) -> bool:
        return name in [m.value for m in self.OpenaAIModles]

    def __init__(self, api_key:str, model: str): 
        if not api_key: 
            raise ValueError("api key is required")

        if api_key[:2] != "sk-": 
            raise ValueError("invalid api key")

        if not model: 
            model = "gpt-4o"    

        if not self._check_model(model): 
            raise ValueError("model not found")

        self.api_key = api_key
        self.model = model 
        self.client = OpenAI(api_key=self.api_key)
        
    def generate(self, prompt: str): 
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response.choices[0].message.content
        except Exception as e: 
            print(f"Error: {str(e)}")
            return None    
    
class GeminiClient(BaseLLM): 
    
    class GeminiModles(Enum):
        GEMINI_2_5_FLASH = "gemini-2.5-flash"
        GEMINI_2_5_PRO = "gemini-2.5-pro"
        GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
        GEMINI_2_5_FLASH_TEXT = "gemini-2.5-flash-text"
        GEMINI_2_5_FLASH_IMAGE = "gemini-2.5-flash-image"
        GEMINI_2_5_FLASH_VIDEO = "gemini-2.5-flash-video"
        GEMINI_2_5_FLASH_AUDIO = "gemini-2.5-flash-audio"
        GEMINI_2_5_FLASH_PDF = "gemini-2.5-flash-pdf"  

    def _check_model(self, name:str) -> bool:
        return name in [m.value for m in self.GeminiModles]    

    def __init__(self, api_key:str, model: str): 
        if not api_key: 
            raise ValueError("api key is required")

        if not model: 
            model = "gemini-2.5-flash"    

        if not self._check_model(model): 
            raise ValueError("model not found")


        self.api_key = api_key
        self.model = model 
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str): 
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )

            return response.text
        except Exception as e: 
            print(f"Error: {str(e)}")
            return None    

class AnthropicClient(BaseLLM): 
    class AnthropicModles(Enum):
        CLAUDE_SONNET_4_5 = "claude-sonnet-4.5"    
        CLAUDE_OPUS_4_5 = "claude-opus-4.5"    
        CLAUDE_HAIKU_4_5 = "claude-haiku-4.5"    
        CLAUDE_SONNET_4_5_1 = "claude-sonnet-4.5.1"    
        CLAUDE_OPUS_4_5_1 = "claude-opus-4.5.1"    
        CLAUDE_HAIKU_4_5_1 = "claude-haiku-4.5.1"    

    def _check_model(self, name:str) -> bool:
        return name in [m.value for m in self.AnthropicModles]    

    def __init__(self, api_key:str, model: str): 
        if not api_key: 
            raise ValueError("api key is required")

        if not model: 
            model = "claude-sonnet-4.5"    

        if not self._check_model(model): 
            raise ValueError("model not found")

        self.api_key = api_key
        self.model = model 
        self.client = Anthropic(api_key=self.api_key)

    def generate(self, prompt: str): 
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response.content[0].text
        except Exception as e: 
            print(f"Error: {str(e)}")
            return None    

class LLMFactory:
    """
        This class is a factory for creating LLM clients
        It uses the register_client method to register new clients
        It uses the get_llm method to get clients    
    """
    __registry = {}

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls.__registry.keys()
        
    @classmethod
    def register_client(cls, name:str, client_class:BaseLLM):
        if not issubclass(client_class, BaseLLM):
            raise TypeError("client_class must be a subclass of BaseLLM")

        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        
        if cls.is_registered(name):
            raise ValueError(f"Client provider {name} already registered")
            
        cls.__registry[name] = client_class

    @classmethod 
    def get_llm(
            provider: str, 
            cls, 
            model: str = None,
            api_key: str = None,)-> BaseLLM:
        try:
            if not provider or not isinstance(provider, str):
                raise ValueError("provider must be a non-empty string")

            if not cls.is_registered(provider):
                raise ValueError(f"Client provider {provider} not registered")

            api_key = os.getenv(provider.upper() + "_API_KEY")
            model = os.getenv(provider.upper() + "_MODEL")

            if not api_key:
                raise ValueError("api key is required")

            client = cls.__registry.get(provider)
            if not client: 
                raise ValueError(f"Client provider {provider} not found")
            return client(api_key=api_key, model=model)

        except Exception as e: 
            raise ValueError(f"Error: {str(e)}")    
