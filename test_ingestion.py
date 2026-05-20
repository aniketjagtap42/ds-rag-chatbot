from ingestion.loader import load_document
from ingestion.chunker import chunk_documents

# Step 1: Load the document
docs = load_document()

# Step 2: Chunk the document
chunks = chunk_documents(docs)

# Step 3: Print some chunks to verify
print("\n--- CHUNK 5 ---")
print(chunks[5].page_content)

print("\n--- CHUNK 10 ---")
print(chunks[10].page_content)

print("\n--- CHUNK 20 ---")
print(chunks[20].page_content)