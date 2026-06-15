from base import Subject, Observer    
from typing import List

class AiAgent(Subject):

    def __init__(self, state: str): 
        self._state = state
        self._observers: List[Observer] = []

    def set_state(self, state:str):
        try:
            if not state or not isinstance(state, str):
                raise ValueError("state must be a non-empty string")
            self._state = state
        except Exception as e: 
            raise ValueError(f"Filed to set state: {state}: error: {e}")    

    def attach(self, observer:Observer): 
        try:
            if not observer or not isinstance(observer, Observer):
                raise TypeError("Observer must be an instance of Observer")
            
            if observer in self._observers:
                return
            
            self._observers.append(observer)
        except Exception as e: 
            raise ValueError(f"Filed to attach observer: {observer}: error: {e}")
        
    def detach(self, observer:Observer): 
        try:
            if not observer or not isinstance(observer, Observer):
                raise TypeError("Observer must be an instance of Observer")
            
            if observer in self._observers:
                self._observers.remove(observer)
        except Exception as e: 
            raise ValueError(f"Filed to detach observer: {observer}: error: {e}")
    
    def notify(self, context): 
        try: 
            if not context: 
                raise ValueError("context is required")

            if self._observers is None:
                raise ValueError("No observers to notify")
            
            for obs in self._observers: 
                obs.update(self)

        except Exception as e: 
            raise ValueError(f"Filed to notify observer: {context}: error: {e}")
            

class ConsoleLogger(Observer):
    def __init__(self): 
        pass 

    def update(self, subject:Subject):
        try: 
            if not subject or not isinstance(subject, Subject):
                raise TypeError("Subject must be an instance of Subject")

            print(f"[Console]-> Agent state: {subject._state}")
        except Exception as e: 
            raise ValueError(f"Filed to update observer: {subject}: error: {e}")    

class FileLogger(Observer):
    def __init__(self):
        pass 

    def update(self, subject: Subject): 
        try:
            if not subject or not isinstance(subject, Subject):
                raise TypeError("Subject must be an instance of Subject")
            print(f"[FileLogger]-> Agent state: {subject._state}")
        except Exception as e: 
            raise ValueError(f"Filed to update observer: {subject}: error: {e}")    
        

if __name__ == "__main__":
    agent = AiAgent(state="initialized")
    agent.attach(ConsoleLogger())
    agent.attach(FileLogger())

    states = ["start", "ready", "processing", "thinking"]
    for state in states: 
        agent.set_state(state)
        agent.notify(state)

    
    


            


    


    