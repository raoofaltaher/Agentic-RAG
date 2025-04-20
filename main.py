# main.py
import argparse
import time
import config
import data_loader
import text_processing
import embedding_utils
import vector_store_interface
from agent import AgenticRAG
import os

def ingest_data():
    """Loads data, processes it, generates embeddings (using Google), and uploads to Qdrant."""
    print("--- Starting Data Ingestion Process ---")

    # Check for essential Google API Key early
    if not config.GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY is missing. Cannot proceed with ingestion (needed for embeddings).")
        return

    # 1. Initialize Vector Store and check/create collection
    print("\n--- Initializing Vector Store ---")
    try:
        qdrant_store = vector_store_interface.QdrantVectorStore()
        # Creation now uses vector size from config (e.g., 768)
        if not qdrant_store.create_collection():
            print("ERROR: Failed to create or ensure collection exists. Aborting ingestion.")
            return
    except ConnectionError as e:
         print(f"ERROR: Could not connect to Qdrant at {config.QDRANT_URL}. Aborting ingestion.")
         return

    # 2. Load data from sources
    print("\n--- Loading Data Sources ---")
    documents = data_loader.load_data_sources(
        urls=config.INGEST_URLS,
        pdf_folder=config.PDF_FOLDER_PATH
    )
    if not documents:
        print("No documents loaded. Ingestion finished.")
        return

    # 3. Split documents
    print("\n--- Splitting Documents ---")
    text_splitter = text_processing.get_text_splitter()
    all_chunks = []
    payloads = []
    for doc in documents:
        content = doc.get('content')
        source = doc.get('source', 'unknown')
        if content and isinstance(content, str):
            cleaned_content = text_processing.clean_text(content)
            if cleaned_content:
                chunks = text_splitter.split_text(cleaned_content)
                for chunk_content in chunks:
                    all_chunks.append(chunk_content) # Store only the text for embedding
                    payloads.append({"source": source, "content": chunk_content}) # Payload retains original chunk
        else:
            print(f"Warning: Skipping document from source '{source}' due to missing or invalid content.")

    if not all_chunks:
        print("No text chunks generated after splitting. Ingestion finished."); return
    print(f"Generated {len(all_chunks)} chunks.")

    # 4. Generate Embeddings using Google API
    print("\n--- Generating Embeddings (using Google API) ---")
    # <<< Calls the updated embedding_utils.get_embeddings >>>
    all_embeddings = embedding_utils.get_embeddings(
        all_chunks,
        task_type="RETRIEVAL_DOCUMENT" # Specify task type for storage
    )

    # Check if embedding generation was successful
    if all_embeddings is None:
        print(f"ERROR: Failed to generate embeddings using Google API. Aborting ingestion.")
        # embedding_utils function already printed detailed errors
        return

    if len(all_embeddings) != len(all_chunks):
         print(f"ERROR: Mismatch between number of chunks ({len(all_chunks)}) and generated embeddings ({len(all_embeddings)}). Aborting.")
         return

    print(f"Successfully generated {len(all_embeddings)} embeddings.")

    # 5. Upload to Vector Store
    print("\n--- Uploading to Vector Store ---")
    ids = list(range(len(all_chunks)))
    if len(payloads) != len(ids): # Sanity check
         print(f"ERROR: Mismatch between number of IDs ({len(ids)}) and payloads ({len(payloads)}). Aborting upload.")
         return

    uploaded = qdrant_store.upload_data(ids=ids, vectors=all_embeddings, payloads=payloads)
    if uploaded:
        print("\n--- Data Ingestion Process Completed Successfully ---")
        final_count = qdrant_store.count(); print(f"Collection '{config.COLLECTION_NAME}' now contains {final_count} points.")
    else:
        print("\n--- Data Ingestion Process Failed During Upload ---")


def run_agent(question: str):
    """Initializes and runs the agent for a given question."""
    print("--- Initializing and Running Agent ---")
    # Check for Google Key, needed for BOTH embeddings search AND LLM calls now
    if not config.GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY is missing. Agent requires it for embeddings and LLM calls. Cannot function.")
        return

    try:
        agent = AgenticRAG()
        start_time = time.time()
        answer = agent.process_query(question)
        end_time = time.time()

        print("\n--- Final Answer ---")
        print(answer)
        print(f"\nProcessed in {end_time - start_time:.2f} seconds.")
    except ConnectionError as e:
        print(f"ERROR: Could not connect to Qdrant during agent run: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during agent run:")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG CLI Tool (Google Embedding + Gemini LLM)")
    parser.add_argument("--ingest", action="store_true", help="Run the data ingestion pipeline.")
    parser.add_argument("--query", type=str, help="Ask a question to the Agentic RAG system.")
    parser.add_argument("--clear", action="store_true", help="Delete the existing Qdrant collection before any other action.")
    args = parser.parse_args()

    # Ensure necessary directories/config exist
    if not os.path.exists(config.PDF_FOLDER_PATH):
        try: os.makedirs(config.PDF_FOLDER_PATH); print(f"Created data directory: {config.PDF_FOLDER_PATH}")
        except OSError as e: print(f"ERROR creating directory {config.PDF_FOLDER_PATH}: {e}"); return

    qdrant_store = None # Define store initially

    if args.clear:
         print(f"--- Clearing Collection '{config.COLLECTION_NAME}' ---")
         try:
             # Need to initialize client even for delete
             qdrant_store = vector_store_interface.QdrantVectorStore()
             qdrant_store.delete_collection()
             print(f"Collection '{config.COLLECTION_NAME}' cleared.")
         except ConnectionError as e:
              print(f"ERROR: Could not connect to Qdrant to clear collection: {e}")
         except Exception as e:
              print(f"ERROR during collection clearing: {e}")
              traceback.print_exc()
         # Exit after clear if no other args provided
         if not args.ingest and not args.query: return

    if args.ingest:
        ingest_data()
    elif args.query:
        run_agent(args.query)
    else:
        if not args.clear: # Only show help if no other action specified
             parser.print_help()
             print("\nExample Usage:")
             print(f"  python main.py --ingest           # Ingest data (uses Google Embeddings)")
             print(f"  python main.py --query \"Your Question?\" # Ask a question (uses Google Embed+LLM)")
             print(f"  python main.py --clear            # Delete the '{config.COLLECTION_NAME}' collection")
             print(f"  python main.py --clear --ingest   # Clear collection then ingest data")

if __name__ == "__main__":
    main()