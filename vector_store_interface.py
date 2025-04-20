# vector_store_interface.py

# ...(imports and class definition as before)...
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, UpdateStatus
from qdrant_client.http import models as rest
from typing import List, Dict, Optional, Any
import config
import time
import embedding_utils # Import for embedding generation during search
import traceback

class QdrantVectorStore:
    """
    A wrapper class for interacting with a Qdrant vector store collection.
    Uses VECTOR_SIZE from config.py.
    """
    def __init__(self, url: str = config.QDRANT_URL, collection_name: str = config.COLLECTION_NAME):
        """
        Initializes the Qdrant client and sets the collection name.
        Args:
            url (str): The URL of the Qdrant instance.
            collection_name (str): The name of the collection to interact with.
        """
        print(f"Initializing Qdrant client for URL: {url}")
        try:
            self.client = QdrantClient(url=url, timeout=60)
            print("Qdrant client initialized. Connection will be verified by subsequent operations.")
        except Exception as e:
            print(f"ERROR during Qdrant client initialization for {url}:")
            traceback.print_exc()
            raise ConnectionError(f"Could not initialize or connect to Qdrant client at {url}") from e

        self.collection_name = collection_name
        print(f"Target collection: {self.collection_name}")
        # Crucially uses config.VECTOR_SIZE when creating collection below

    def collection_exists(self) -> bool:
        # ... (Keep this method as previously updated - without health_check) ...
        """Checks if the configured collection exists by trying to list collections."""
        # print(f"Checking if collection '{self.collection_name}' exists...") # reduce verbosity
        try:
            collections_response = self.client.get_collections()
            exists = any(col.name == self.collection_name for col in collections_response.collections)
            # print(f"Collection '{self.collection_name}' exists: {exists}") # reduce verbosity
            return exists
        except Exception as e:
            print(f"Error checking for collection '{self.collection_name}' (potential connection issue): {e}")
            # traceback.print_exc() # Only show traceback if needed
            return False

    def create_collection(self, vector_size: int = config.VECTOR_SIZE, distance: Distance = Distance.COSINE) -> bool:
        # ... (Keep this method as previously updated - without health_check) ...
        """
        Creates the collection in Qdrant if it doesn't already exist. Uses VECTOR_SIZE from config.
        Args:
            vector_size (int): The dimensionality of the vectors (defaults to config.VECTOR_SIZE).
            distance (Distance): The distance metric to use (e.g., COSINE, EUCLID).
        Returns:
            bool: True if the collection was created or already exists, False otherwise.
        """
        try:
            if self.collection_exists():
                return True
        except Exception as e:
             print(f"Failed to check collection existence before creating '{self.collection_name}'. Error: {e}")
             return False

        print(f"Attempting to create collection '{self.collection_name}' with vector size {vector_size} and distance {distance}...")
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
                timeout=30
            )
            time.sleep(1)
            if self.collection_exists():
                print(f"Collection '{self.collection_name}' created successfully.")
                return True
            else:
                print(f"ERROR: Collection '{self.collection_name}' not found immediately after creation attempt.")
                return False
        except Exception as e:
            print(f"ERROR creating collection '{self.collection_name}': {e}")
            traceback.print_exc()
            try: # Fallback check
                if self.collection_exists():
                     print(f"Warning: Creation API call failed, but collection '{self.collection_name}' seems to exist now.")
                     return True
            except Exception: pass
            return False

    def delete_collection(self) -> bool:
        # ... (Keep this method as previously updated) ...
        """Deletes the configured collection."""
        print(f"Attempting to delete collection '{self.collection_name}'...")
        try:
             if not self.collection_exists():
                 print(f"Collection '{self.collection_name}' does not exist. Skipping deletion.")
                 return True
             print(f"Deleting existing collection '{self.collection_name}'...")
             result = self.client.delete_collection(collection_name=self.collection_name, timeout=60)
             time.sleep(1)
             if result:
                 print(f"Collection '{self.collection_name}' delete operation returned success.")
                 if not self.collection_exists(): return True
                 else: print(f"ERROR: Collection '{self.collection_name}' still exists after delete success."); return False
             else:
                 print(f"Warning: Delete operation for '{self.collection_name}' returned False. Checking status...")
                 if not self.collection_exists(): print(f"Confirmed deleted despite False return."); return True
                 else: print(f"ERROR: Collection '{self.collection_name}' still exists after delete returned False."); return False
        except Exception as e:
            print(f"ERROR during deletion of collection '{self.collection_name}': {e}"); traceback.print_exc(); return False


    def upload_data(self, ids: List[int], vectors: List[List[float]], payloads: List[Dict[str, Any]], batch_size: int = 128) -> bool:
        # ... (Keep this method as previously updated) ...
        """
        Uploads vectors and payloads to the Qdrant collection.
        Args:
            ids (List[int]): A list of unique integer IDs for the points.
            vectors (List[List[float]]): A list of embedding vectors.
            payloads (List[Dict[str, Any]]): A list of corresponding metadata payloads.
            batch_size (int): Max points per GRPC message (Qdrant default is often good).
        Returns:
            bool: True if the upload was successful (or partially successful), False if a major error occurred.
        """
        print(f"Preparing to upload data to '{self.collection_name}'...")
        try:
            if not self.collection_exists():
                print(f"Error: Collection '{self.collection_name}' does not exist. Attempting to create...")
                if not self.create_collection(): print("ERROR: Failed to create collection. Aborting upload."); return False
                else: print(f"Collection '{self.collection_name}' created successfully.")

            if not (len(ids) == len(vectors) == len(payloads)):
                print("ERROR: Mismatch in lengths of ids, vectors, and payloads."); return False
            if not ids:
                print("No data provided for upload."); return True

            print(f"Uploading {len(ids)} points to collection '{self.collection_name}'...") # Batch size info removed, client handles it
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(ids=ids, vectors=vectors, payloads=payloads),
                wait=True
            )
            if operation_info.status == UpdateStatus.COMPLETED:
                print(f"Successfully uploaded/upserted {len(ids)} points."); return True
            else:
                print(f"Warning: Upload/upsert finished with status: {operation_info.status}. Checking count...")
                count_after = self.count(); print(f"Current collection count: {count_after}"); return True
        except Exception as e:
            print(f"ERROR during data upload to Qdrant collection '{self.collection_name}': {e}"); traceback.print_exc(); return False


    def search(self, query_text: str, top_k: int = config.RETRIEVAL_TOP_K) -> List[models.ScoredPoint]:
        """
        Performs a semantic search in the collection using the configured embedding model.
        Args:
            query_text (str): The text to search for.
            top_k (int): The maximum number of results to return.
        Returns:
            List[models.ScoredPoint]: A list of search results, or an empty list on error.
        """
        print(f"\nPerforming search in '{self.collection_name}'...")
        try:
            if not self.collection_exists():
                 print(f"Warning: Cannot search. Collection '{self.collection_name}' not found."); return []

            print(f"  Generating QUERY embedding for: '{query_text[:50]}...'")
            # <<< Uses the new embedding_utils.get_embeddings with RETRIEVAL_QUERY >>>
            query_embedding_list = embedding_utils.get_embeddings(
                [query_text],
                task_type="RETRIEVAL_QUERY" # Specify task type for query
            )

            if not query_embedding_list or not query_embedding_list[0]:
                 print("ERROR: Could not generate a valid embedding for the query text.")
                 return []
            query_embedding = query_embedding_list[0] # Get the single embedding vector

            print(f"  Searching collection for top {top_k} results...")
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=None,
                limit=top_k,
                with_payload=True
            )
            print(f"  Search returned {len(search_result)} results.")
            return search_result
        except Exception as e:
            print(f"ERROR during search in Qdrant collection '{self.collection_name}': {e}"); traceback.print_exc(); return []

    def count(self) -> int:
        # ... (Keep this method as previously updated) ...
        """Returns the number of points in the collection."""
        try:
            if not self.collection_exists(): return 0
            count_result = self.client.count(collection_name=self.collection_name, exact=True); return count_result.count
        except Exception as e: print(f"Error counting points in collection '{self.collection_name}': {e}"); return -1


    @staticmethod
    def format_search_results(docs: List[models.ScoredPoint]) -> str:
        # ... (Keep this method as previously updated) ...
        """Formats Qdrant search results into a single string context."""
        if not docs: return "No relevant context found in the vector store."
        formatted = []
        for i, doc in enumerate(docs):
            content, source, score = "Payload missing or invalid", "Unknown Source", 0.0
            if hasattr(doc, 'score'): score = doc.score
            if doc.payload is not None:
                 content = doc.payload.get("content", "Content not available in payload")
                 source = doc.payload.get("source", "Source not available in payload")
            formatted.append(f"Retrieved Document {i+1} (Source: {source}, Score: {score:.4f}):\n{content}")
        return "\n\n---\n\n".join(formatted)

# <<< Updated `if __name__ == '__main__'` block >>>
if __name__ == '__main__':
    # Example Usage (requires Qdrant running and GOOGLE_API_KEY)
    print("--- Testing Vector Store Interface (with Google Embeddings) ---")
    try:
        # Ensure necessary key is set for testing embedding calls
        if not config.GOOGLE_API_KEY:
            print("CRITICAL: GOOGLE_API_KEY not found. Cannot run embedding-dependent tests.")
            exit()

        qdrant_store = QdrantVectorStore() # Uses COLLECTION_NAME and VECTOR_SIZE from config

        # 1. Delete collection if exists
        print("\nStep 1: Ensuring clean state")
        qdrant_store.delete_collection()
        time.sleep(1)

        # 2. Create collection
        print("\nStep 2: Creating collection")
        # Uses VECTOR_SIZE (e.g., 768) from config now
        created = qdrant_store.create_collection()
        if not created:
            print("Failed to create collection. Aborting test.")
            exit()
        print(f"Collection count after creation: {qdrant_store.count()}")

        # 3. Prepare data
        print("\nStep 3: Preparing dummy data")
        dummy_texts = ["alpha document content", "beta document info", "gamma text test"]
        print("Getting embeddings for dummy data using Google API (RETRIEVAL_DOCUMENT task)...")

        # Uses the updated embedding_utils function with correct task type
        dummy_embeddings = embedding_utils.get_embeddings(dummy_texts, task_type="RETRIEVAL_DOCUMENT")

        if not dummy_embeddings or len(dummy_embeddings) != len(dummy_texts):
             print("Failed to get embeddings for dummy data. Aborting upload/search test.")
        else:
            dummy_ids = list(range(len(dummy_texts))) # Simple 0, 1, 2 IDs
            dummy_payloads = [{"content": text, "source": f"dummy_{i}.txt"} for i, text in enumerate(dummy_texts)]
            print(f"Dummy IDs: {dummy_ids}")
            print(f"Dummy Payloads: {dummy_payloads}")

            # 4. Upload data
            print("\nStep 4: Uploading dummy data")
            uploaded = qdrant_store.upload_data(ids=dummy_ids, vectors=dummy_embeddings, payloads=dummy_payloads)
            if uploaded:
                print(f"Data upload finished. Collection count: {qdrant_store.count()}")
            else:
                print("Data upload failed.")

            # 5. Search
            print("\nStep 5: Searching for 'beta information'")
            # Search now uses Google Embeddings via embedding_utils (RETRIEVAL_QUERY task)
            search_results = qdrant_store.search("beta information", top_k=2)
            if search_results:
                print(f"Found {len(search_results)} results:")
                for result in search_results:
                    print(f" - ID: {result.id}, Score: {result.score:.4f}, Payload: {result.payload}")
                formatted_context = QdrantVectorStore.format_search_results(search_results)
                print("\nFormatted Context:\n", formatted_context)
            else:
                print("Search failed or returned no results.")

        # 6. Count
        print("\nStep 6: Final count check")
        final_count = qdrant_store.count()
        print(f"Final collection count: {final_count}")

    except ConnectionError as e:
        print(f"\nTest failed: Could not connect to or initialize Qdrant client. Ensure it's running at {config.QDRANT_URL}")
        print(f"   Error Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing:")
        traceback.print_exc()