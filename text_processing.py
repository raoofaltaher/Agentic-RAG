import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import config # Import config for parameters

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Initializes and returns a RecursiveCharacterTextSplitter based on config.

    Returns:
        RecursiveCharacterTextSplitter: An instance of the text splitter.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=config.TEXT_SPLITTER_MODEL,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

def clean_text(text: str) -> str:
    """
    Cleans text by removing excessive newlines, whitespace, and stripping ends.

    Args:
        text (str): The input text string.

    Returns:
        str: The cleaned text string.
    """
    if not isinstance(text, str):
        return "" # Return empty string if input is not a string

    # Remove all newline characters
    text = text.replace('\n', ' ').replace('\r', ' ')

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading and trailing spaces
    text = text.strip()

    return text

def split_documents(documents: List[dict], text_splitter: RecursiveCharacterTextSplitter) -> List[str]:
    """
    Splits the content of multiple documents into smaller text chunks.

    Args:
        documents (List[dict]): A list of documents, where each dict has a 'content' key.
        text_splitter (RecursiveCharacterTextSplitter): The splitter instance to use.

    Returns:
        List[str]: A list of all text chunks from all documents.
    """
    all_chunks = []
    print(f"Splitting {len(documents)} documents...")
    for doc in documents:
        if 'content' in doc and isinstance(doc['content'], str):
            cleaned_content = clean_text(doc['content'])
            if cleaned_content: # Proceed only if content is not empty after cleaning
                chunks = text_splitter.split_text(cleaned_content)
                all_chunks.extend(chunks)
        else:
            source = doc.get('source', 'Unknown source')
            print(f"Warning: Document from '{source}' has missing or invalid 'content'. Skipping.")

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

def split_text_direct(text: str, text_splitter: RecursiveCharacterTextSplitter) -> List[str]:
    """
    Splits a single string of text into chunks.

    Args:
        text (str): The input text.
        text_splitter (RecursiveCharacterTextSplitter): The splitter instance to use.

    Returns:
        List[str]: A list of text chunks.
    """
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return []
    chunks = text_splitter.split_text(cleaned_text)
    print(f"Split text into {len(chunks)} chunks.")
    return chunks


if __name__ == '__main__':
    # Example Usage
    print("--- Testing Text Processing ---")
    splitter = get_text_splitter()
    print(f"Text splitter configured: chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}")

    sample_text = """
    This is the first sentence. It has some length.

    This is the second sentence. It is shorter.


    This is a third sentence, which might be split depending on the chunk size. It contains multiple words and punctuation.
    Another line.
    """

    print(f"\nOriginal Text:\n{sample_text}")
    cleaned = clean_text(sample_text)
    print(f"\nCleaned Text:\n{cleaned}")

    chunks = split_text_direct(sample_text, splitter)
    print(f"\nSplit into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")

    # Test splitting documents (using dummy data)
    dummy_docs = [
        {"content": "Document one content. Quite short.", "source": "doc1.txt"},
        {"content": sample_text, "source": "doc2.txt"},
        {"content": "", "source": "empty_doc.txt"}, # Empty content
        {"content": None, "source": "none_doc.txt"}, # None content
        {"source": "missing_content.txt"} # Missing content key
    ]
    print("\n--- Testing Document Splitting ---")
    doc_chunks = split_documents(dummy_docs, splitter)
    print(f"\nTotal chunks from documents: {len(doc_chunks)}")
    # print("First 5 chunks:", doc_chunks[:5]) # Uncomment to see chunks