from base import BaseRAGChunking
from typing import List

class FixedSizeChunking(BaseRAGChunking):
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        if not max_tokens: 
            self.max_tokens = 500 

    def chunk(self, text: str) -> List[str]: 
        chunk_list = []
        if not text:
            raise ValueError("text cannot be empty")

        words = text.split(" ")
        for i in range(0, len(words), self.max_tokens): 
            chunk_list.append(" ".join(words[i: i + self.max_tokens]))

        return chunk_list

class SentenceChunking(BaseRAGChunking):
    def __init__(self):
        pass 

    def chunk(self, text: str) -> List[str]:
        chunk_list = []
        if not text: 
            raise ValueError("text cannot be empty")

        docs = text.split(".")
        for doc in docs: 
            if doc.strip():
                chunk_list.append(doc.strip())

        return chunk_list

class MarkDownChunking(BaseRAGChunking):
    def __init__(self):
        pass 

    def chunk(self, text: str) -> List[str]:
        chunk_list = []
        if not text: 
            raise ValueError("text cannot be empty")

        docs = text.split("#")
        for i, chunk in enumerate(docs):
            if not chunk.strip():
                continue
            if i == 0: 
                chunk_list.append(chunk.strip())
            else: 
                chunk_list.append("# " + chunk.strip())

        return chunk_list


class Chunking:
    def __init__(self, strategy: BaseRAGChunking):
        self.strategy = strategy

    def set_strategy(self, strategy: BaseRAGChunking):
        self.strategy = strategy
    
    def chunk(self, text: str) -> List[str]:
        return self.strategy.chunk(text)

            


        
