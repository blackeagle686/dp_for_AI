from abc import ABC, abstractmethod 
from typing import List, Any, Dict 

class BaseLLM(ABC): 
    @abstractmethod
    def generate(self, prompt: str): 
        pass 

