from abc import ABC, abstractmethod
from typing import List    

class Observer(ABC):
    @abstractmethod
    def update(self, subject): 
        pass

class Subject(ABC): 
    @abstractmethod
    def attach(self, observer: Observer): 
        pass 

    @abstractmethod
    def detach(self, observer: Observer): 
        pass 

    @abstractmethod
    def notify(self, context: str): 
        pass 
