import os
import PyPDF2
import requests
from typing import List, Dict, Optional, Union
from tqdm import tqdm
import config

def read_pdfs_from_folder(folder_path: str) -> List[Dict[str, str]]:
    """
    Reads all PDF files from a specified folder and extracts their text content.

    Args:
        folder_path (str): The path to the folder containing PDF files.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each dictionary
                               contains the 'content' and 'filename' of a PDF.
                               Returns an empty list if the folder doesn't exist.
    """
    pdf_list = []
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return pdf_list

    print(f"Reading PDFs from folder: {folder_path}")
    try:
        file_list = os.listdir(folder_path)
        pdf_files = [f for f in file_list if f.lower().endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return pdf_list

        for filename in tqdm(pdf_files, desc="Reading PDFs"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    content = ""
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        page_content = page.extract_text()
                        if page_content: # Add content only if text extraction was successful
                            content += page_content + "\n" # Add newline between pages

                    if content: # Only add if content was extracted
                         pdf_list.append({
                            "content": content.strip(),
                            "source": filename # Use 'source' for consistency
                         })
                    else:
                         print(f"Warning: No text extracted from {filename}")

            except Exception as e:
                print(f"Error reading PDF file {filename}: {e}")
    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")

    print(f"Successfully read {len(pdf_list)} PDF documents.")
    return pdf_list

def fetch_url_content(url: str) -> Optional[str]:
    """
    Fetches content from a URL using Jina AI's reader service.

    Args:
        url (str): The endpoint or URL to fetch content from.

    Returns:
        Optional[str]: The content retrieved from the URL as a string,
                       or None if the request fails.
    """
    prefix_url: str = "https://r.jina.ai/"
    full_url: str = prefix_url + url  # Concatenate the prefix URL

    print(f"Fetching content from URL: {url} (via Jina AI)")
    try:
        response = requests.get(full_url, timeout=30) # Add a timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        print(f"Successfully fetched content from {url}")
        return response.content.decode('utf-8', errors='ignore') # Ignore decoding errors

    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch URL {full_url}. Exception: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching {full_url}: {e}")
        return None

def load_data_sources(urls: List[str] = None, pdf_folder: str = None) -> List[Dict[str, str]]:
    """
    Loads data from specified URLs and PDF folder.

    Args:
        urls (List[str], optional): A list of URLs to fetch content from. Defaults to None.
        pdf_folder (str, optional): Path to the folder containing PDFs. Defaults to None.

    Returns:
        List[Dict[str, str]]: A list of documents, each with 'content' and 'source'.
    """
    all_documents = []

    # Load from URLs
    if urls:
        for url in urls:
            content = fetch_url_content(url)
            if content:
                all_documents.append({"content": content, "source": url})

    # Load from PDFs
    if pdf_folder:
        pdf_documents = read_pdfs_from_folder(pdf_folder)
        all_documents.extend(pdf_documents) # Extend the list with PDF results

    return all_documents

if __name__ == '__main__':
    # Example usage (optional: for testing the module directly)
    print("--- Testing Data Loader ---")

    # Test URL fetching
    test_url = config.INGEST_URLS[0] if config.INGEST_URLS else "https://example.com"
    url_content = fetch_url_content(test_url)
    if url_content:
        print(f"\nSuccessfully fetched content from {test_url} (first 200 chars):")
        print(url_content[:200] + "...")
    else:
        print(f"\nFailed to fetch content from {test_url}")

    # Test PDF reading (requires a PDF in ./rag_data)
    # Create a dummy PDF for testing if needed, or place real PDFs there
    if not os.path.exists(config.PDF_FOLDER_PATH):
        os.makedirs(config.PDF_FOLDER_PATH)
        print(f"\nCreated directory: {config.PDF_FOLDER_PATH}")
        print("Place PDF files in this directory to test PDF loading.")

    pdf_docs = read_pdfs_from_folder(config.PDF_FOLDER_PATH)
    if pdf_docs:
        print(f"\nRead {len(pdf_docs)} PDF documents.")
        print(f"Content from first page of '{pdf_docs[0]['source']}':")
        print(pdf_docs[0]['content'][:500] + "...") # Print first 500 chars
    else:
        print(f"\nNo PDFs found or read from {config.PDF_FOLDER_PATH}")

    # Test combined loading
    print("\n--- Testing Combined Loading ---")
    combined_docs = load_data_sources(urls=config.INGEST_URLS, pdf_folder=config.PDF_FOLDER_PATH)
    print(f"Total documents loaded: {len(combined_docs)}")
    if combined_docs:
        print(f"Sources: {[doc['source'] for doc in combined_docs]}")