import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

print("Starting ingestion process...")

# 1. Load configuration from .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. Load the source documents
print("Loading documents from clean_docs.json...")
with open("clean_docs.json") as f:
    raw_docs = json.load(f)
docs = [Document(**doc) for doc in raw_docs]
print(f"Loaded {len(docs)} documents.")

# 3. Split documents into chunks
print("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks.")

# 4. Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# 5. Create the index if it doesn't exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new one...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Index created successfully.")
else:
    # If the index already exists, it's a good practice to clear it
    # before re-ingesting to avoid duplicates and outdated data.
    print(f"Index '{PINECONE_INDEX_NAME}' already exists. Clearing old data...")
    index_to_clear = pc.Index(PINECONE_INDEX_NAME)
    index_to_clear.delete(delete_all=True)
    print("Index cleared.")


# 6. Initialize embeddings and vector store
print("Initializing OpenAI embeddings model...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print(f"Connecting to index: {PINECONE_INDEX_NAME}")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

# ==================== MANUAL BATCHING SOLUTION ====================

# Set a batch size
batch_size = 100
total_chunks = len(chunks)

print(f"Embedding and upserting {total_chunks} chunks to Pinecone in batches of {batch_size}...")

# Loop through the chunks in batches
for i in range(0, total_chunks, batch_size):
    # Get the current batch of chunks
    batch = chunks[i : i + batch_size]
    
    # Process and upsert the current batch
    vectorstore.add_documents(batch)
    
    # Print progress
    print(f"  Processed batch {i // batch_size + 1} / {(total_chunks + batch_size - 1) // batch_size}")

# =================================================================

print("\n Ingestion complete!")