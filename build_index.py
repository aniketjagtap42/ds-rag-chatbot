from ingestion.loader import load_document
from ingestion.chunker import chunk_documents
from retrieval.vectorstore import build_vectorstore

print('=== Building DS Notes Vector Index ===')
docs = load_document()
chunks = chunk_documents(docs)
build_vectorstore(chunks)
print('=== Done! Your vector database is ready ===')