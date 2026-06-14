# Strategy Pattern in AI

## Scenario: RAG Text Chunking Pipeline

In a Retrieval-Augmented Generation (RAG) system, large documents must be split into smaller "chunks" before being embedded and stored in a Vector Database. Different document types require different chunking strategies to get the best search results.

1. Fixed-Size Chunking: Splits text into chunks of exactly N words.
2. Sentence-Based Chunking: Splits text at periods to preserve grammatical context.
3. Markdown Chunking: Splits text based on Markdown headers to preserve logical sections.

Using the Strategy Pattern, we can define a common interface `BaseRAGChunking` and implement these three strategies as separate classes. A context class `Chunking` can then switch between these strategies dynamically at runtime without changing the main application logic.

### Example Usage

```python
from rag_scenario import Chunking, FixedSizeChunking, MarkDownChunking

document = "This is a very long text. # Header 1. It has content. # Header 2. More content."

# 1. Initialize Context with a Strategy
pipeline = Chunking(FixedSizeChunking(max_tokens=10))

# 2. Execute
basic_chunks = pipeline.chunk(document)
print(basic_chunks)

# 3. Swap strategy at runtime without changing the pipeline
pipeline.set_strategy(MarkDownChunking())
markdown_chunks = pipeline.chunk(document)
print(markdown_chunks)
```
