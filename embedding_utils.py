# embedding_utils.py
import config
from typing import List, Optional
import google.generativeai as genai
import google.api_core.exceptions
import time
import traceback

# Configure the Google Generative AI client
try:
    if config.GOOGLE_API_KEY:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        print("Google Generative AI SDK configured successfully.")
    else:
        print("Warning: GOOGLE_API_KEY not found during initial configuration. Embedding calls will fail.")
except Exception as e:
    print(f"Error configuring Google Generative AI SDK: {e}")
    traceback.print_exc()

# Rate Limiting (Free tier for text-embedding-004 is often generous, e.g., 1500 QPM)
# Adjust these if you hit rate limits
GOOGLE_EMBEDDING_BATCH_SIZE = 100 # Google's batch embed API can handle up to 100 texts per call
REQUESTS_PER_MINUTE_LIMIT = 1400 # Slightly below the typical 1500 QPM limit
SECONDS_PER_MINUTE = 60
DELAY_BETWEEN_BATCHES = SECONDS_PER_MINUTE / REQUESTS_PER_MINUTE_LIMIT * GOOGLE_EMBEDDING_BATCH_SIZE


def get_embeddings(texts: List[str], model: str = config.EMBEDDING_MODEL_NAME, task_type: str = "RETRIEVAL_DOCUMENT") -> Optional[List[List[float]]]:
    """
    Generates embeddings for a list of texts using the Google Generative AI SDK.

    Args:
        texts (List[str]): A list of text strings to embed.
        model (str): The embedding model name (e.g., "models/text-embedding-004").
        task_type (str): The type of task for the embedding ("RETRIEVAL_DOCUMENT" for storage,
                         "RETRIEVAL_QUERY" for search queries, "SEMANTIC_SIMILARITY", etc.).

    Returns:
        Optional[List[List[float]]]: A list where each element is an embedding vector (list of floats),
                                     or None if a fatal error occurs. Returns empty list for empty input.
    """
    if not texts:
        print("Warning: No texts provided for embedding.")
        return []

    if not config.GOOGLE_API_KEY:
        print(f"ERROR: Cannot generate embeddings. GOOGLE_API_KEY is not configured.")
        return None

    all_embeddings = []
    num_texts = len(texts)
    print(f"Requesting Google embeddings for {num_texts} texts using model '{model}' (Task: {task_type}, Batch size: {GOOGLE_EMBEDDING_BATCH_SIZE})...")

    for i in range(0, num_texts, GOOGLE_EMBEDDING_BATCH_SIZE):
        batch_texts = texts[i:i + GOOGLE_EMBEDDING_BATCH_SIZE]
        batch_num = (i // GOOGLE_EMBEDDING_BATCH_SIZE) + 1
        print(f"  Processing batch {batch_num} ({len(batch_texts)} texts)...")

        try:
            # Use embed_content for batch processing
            result = genai.embed_content(
                model=model,
                content=batch_texts,
                task_type=task_type
            )

            # Check if the expected 'embedding' key exists and is a list
            if 'embedding' in result and isinstance(result['embedding'], list):
                 batch_embeddings = result['embedding']
                 # Verify that we received the correct number of embeddings
                 if len(batch_embeddings) == len(batch_texts):
                      all_embeddings.extend(batch_embeddings)
                      print(f"  Successfully received embeddings for batch {batch_num}.")
                 else:
                     print(f"ERROR: Embedding count mismatch for batch {batch_num}.")
                     print(f"       Expected {len(batch_texts)}, got {len(batch_embeddings)}.")
                     # Decide whether to abort or continue. Let's abort for now.
                     return None
            else:
                 print(f"ERROR: Unexpected response structure from Google embedding API for batch {batch_num}.")
                 print(f"       Result: {result}")
                 # Abort on unexpected structure
                 return None

        except google.api_core.exceptions.ResourceExhausted as e:
            print(f"ERROR: Rate limit likely exceeded (ResourceExhausted) during Google embedding request for batch {batch_num}: {e}")
            print(f"       Consider adjusting rate limit settings in config or check your Google Cloud quotas.")
            # Potentially add a retry mechanism here with backoff
            return None # Abort on rate limit error for now
        except google.api_core.exceptions.InvalidArgument as e:
            print(f"ERROR: Invalid argument during Google embedding request for batch {batch_num}: {e}")
            print(f"       Check model name ('{model}'), task type ('{task_type}'), and input text content.")
            traceback.print_exc()
            return None # Abort on invalid argument
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during Google embedding generation for batch {batch_num}:")
            traceback.print_exc()
            return None # Abort on other errors

        # Optional delay to respect rate limits if processing many batches quickly
        if num_texts > GOOGLE_EMBEDDING_BATCH_SIZE and i + GOOGLE_EMBEDDING_BATCH_SIZE < num_texts:
             print(f"  Delaying for ~{DELAY_BETWEEN_BATCHES:.2f} seconds before next batch...")
             time.sleep(DELAY_BETWEEN_BATCHES)


    print(f"Successfully generated {len(all_embeddings)} embeddings in total.")
    return all_embeddings


if __name__ == '__main__':
    # Example Usage - requires GOOGLE_API_KEY set in .env
    print("\n--- Testing Embedding Utils (Google) ---")
    sample_texts_doc = [
        "This is the first document chunk.",
        "Here is another piece of text for storage.",
        "Embeddings represent text numerically."
    ]
    sample_text_query = "How is text represented?"


    if not config.GOOGLE_API_KEY:
         print("\nSkipping embedding generation test: GOOGLE_API_KEY not set in config or .env file.")
    else:
        print(f"\nRequesting DOCUMENT embeddings for {len(sample_texts_doc)} sample texts...")
        doc_embeddings = get_embeddings(sample_texts_doc, task_type="RETRIEVAL_DOCUMENT")

        if doc_embeddings:
            print(f"\nSuccessfully obtained {len(doc_embeddings)} document embeddings.")
            print(f"Dimension of the first embedding: {len(doc_embeddings[0]) if doc_embeddings[0] else 'N/A'}")
            # print("First embedding vector:", doc_embeddings[0][:10], "...") # Print first few dims
        else:
            print("\nFailed to obtain document embeddings.")

        print(f"\nRequesting QUERY embedding for: '{sample_text_query}'")
        query_embedding_list = get_embeddings([sample_text_query], task_type="RETRIEVAL_QUERY")

        if query_embedding_list and query_embedding_list[0]:
            print(f"\nSuccessfully obtained query embedding.")
            print(f"Dimension of the query embedding: {len(query_embedding_list[0])}")
        else:
            print("\nFailed to obtain query embedding.")


    print("\n--- Testing with empty input ---")
    empty_result = get_embeddings([])
    print(f"Result for empty input: {empty_result}")

    print("\n--- Testing without API key (should print error and return None) ---")
    # Temporarily unset the configured key for testing the check
    original_key = config.GOOGLE_API_KEY
    config.GOOGLE_API_KEY = None
    no_key_result = get_embeddings(["test text"])
    print(f"Result without API key: {no_key_result}")
    config.GOOGLE_API_KEY = original_key # Restore the key