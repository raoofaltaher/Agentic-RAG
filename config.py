# config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# --- Environment Variables ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Needed for Embeddings and LLMs

# --- Behavior Control ---
# Set to True to allow the agent to search the web if retrieved context is not relevant.
# Set to False to prevent web search and only answer from ingested documents.
ALLOW_WEB_SEARCH_FALLBACK = True
# <<< ------------------ >>>


# --- Model Selection ---
# LLM Models (using Google Gemini)
LLM_DECISION_MODEL = "gemini/gemini-1.5-flash-latest" # Using litellm prefix
LLM_ANSWER_MODEL = "gemini/gemini-1.5-flash-latest"   # Using litellm prefix

# Embedding Model (using Google's model)
EMBEDDING_PROVIDER = "google"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
VECTOR_SIZE = 768 # Standard dimension for text-embedding-004

# --- API Key Checks ---
if not GOOGLE_API_KEY:
     print("CRITICAL WARNING: GOOGLE_API_KEY not found in environment variables.")
     print("                 This is required for BOTH embedding generation and LLM calls in this configuration.")
     # raise ValueError("Missing required GOOGLE_API_KEY for embeddings and LLM calls.")
else:
    print("Google API Key found (used for embeddings and LLM tasks).")

# --- LLM Configuration ---
LLM_MAX_TOKENS = 800

# --- Qdrant Configuration ---
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "agent_rag_index_py_google_emb" # Reflects Google embedding model

# --- Text Processing Configuration ---
CHUNK_SIZE = 150
CHUNK_OVERLAP = 0
TEXT_SPLITTER_MODEL = "gpt-4" # For tiktoken length estimate

# --- Data Paths ---
PDF_FOLDER_PATH = "./rag_data"

# --- Search Configuration ---
WEB_SEARCH_MAX_RESULTS = 5

# --- Agent Configuration ---
RETRIEVAL_TOP_K = 3

# --- Prompts ---
DECISION_SYSTEM_PROMPT = """Your job is decide if a given question can be answered with a given context.
If the context contains information that can directly answer the question, return 1.
If the context does not contain information to answer the question, return 0.

Respond ONLY with 0 or 1. Do not provide any explanation, preamble, or justification. Just the single digit.

Context:
{context}
"""

DECISION_USER_PROMPT = """
Question: {question}

Answer:"""


ANSWER_SYSTEM_PROMPT = """You are an expert Q&A system. Your task is to answer the question based *only* on the provided context below.
Do not use any external knowledge or information you might have. Focus solely on the text provided in the 'Context'.
If the question cannot be answered using the provided context, respond exactly with: "Based on the provided context, I cannot answer this question."
Do not try to infer or make up information not present in the context.
Your answer should be informative and concise, directly addressing the question using only the context information. Format your response in Markdown.

Context:
{context}
"""

ANSWER_USER_PROMPT = """
Question: {question}

Answer:"""

# --- URLs for Ingestion (Optional) ---
# --- URLs for Ingestion (Example - Add more if needed) ---
# --- Add URLs to be ingested into the vector store for RAG ---
# These URLs should point to relevant content that can be used for answering questions.
INGEST_URLS = [
    # Add URLs here if you want to ingest them initially
    # e.g., "https://example-site.com/about-ai"
]

# --- Final Config Printout ---
print(f"--- Configuration Loaded ---")
print(f"  LLM Decision Model: {LLM_DECISION_MODEL} (Requires GOOGLE_API_KEY)")
print(f"  LLM Answer Model:   {LLM_ANSWER_MODEL} (Requires GOOGLE_API_KEY)")
print(f"  Embedding Provider: {EMBEDDING_PROVIDER}")
print(f"  Embedding Model:    {EMBEDDING_MODEL_NAME} (Requires GOOGLE_API_KEY)")
print(f"  Embedding Dim:      {VECTOR_SIZE}")
print(f"  Qdrant Collection:  {COLLECTION_NAME} at {QDRANT_URL}")
print(f"  PDF Folder:         {PDF_FOLDER_PATH}")
print(f"  Web Search Fallback: {'ENABLED' if ALLOW_WEB_SEARCH_FALLBACK else 'DISABLED'}")
print(f"--------------------------")