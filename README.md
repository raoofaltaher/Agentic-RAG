# Agentic RAG Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Example Badge -->

## Overview

This project implements an Agentic Retrieval-Augmented Generation (RAG) system in Python. It allows users to ask questions against a collection of ingested documents (PDFs and web URLs). The agent intelligently decides whether the retrieved local context is sufficient to answer the question or if it needs to perform a web search (this fallback can be toggled on/off).

The system leverages vector embeddings (currently using Google's `text-embedding-004` model) stored in a Qdrant vector database and uses a Large Language Model (LLM) (currently Google Gemini via LiteLLM) for decision-making and final answer generation.

## Features

*   **Retrieval-Augmented Generation (RAG):** Answers questions based on information retrieved from ingested documents.
*   **Data Ingestion:** Supports loading and processing text from PDF files and web URLs.
*   **Vector Store Integration:** Uses Qdrant for efficient similarity search of document chunks.
*   **Agentic Decision Making:** An LLM determines if the retrieved context is relevant enough to answer the query.
*   **Configurable Web Search Fallback:** Option to enable or disable fallback to web search (using DuckDuckGo) if local context is insufficient.
*   **Modular Design:** Code is separated into logical components (data loading, embedding, vector store, LLM interface, agent logic).
*   **Free Tier Focused:** Configured to use Google's free tier embedding model and LLM (Gemini Flash).

## Architecture

**Core Agentic RAG Flow:**
![Core Flow Diagram](images\agentic_rag_0.png) 

## Prerequisites

*   **Python:** Version 3.8 or higher recommended.
*   **Pip:** Python package installer.
*   **Docker:** Required for running the Qdrant vector database locally. Install Docker Desktop (Windows/Mac) or Docker Engine (Linux). Ensure it's running and properly configured (including WSL2/Virtualization on Windows).
*   **Git:** (Optional) For cloning the repository.
*   **Google API Key:** An active API key from Google AI Studio or Google Cloud Console with the "Generative Language API" enabled. This is needed for *both* embeddings and LLM calls.

## Setup

1.  **Clone the Repository (or download files):**
    ```bash
    git clone <your-repo-url>
    cd agentic-rag-project
    ```

2.  **Create Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # or
    .\venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Start Qdrant Database:**
    Open a **separate** terminal window and run:
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```
    Keep this terminal running. You can access the Qdrant dashboard at `http://localhost:6333/dashboard`.

5.  **Configure API Keys:**
    *   Create a file named `.env` in the project root.
    *   Add your Google API Key:
        ```dotenv
        # .env
        GOOGLE_API_KEY="..." # Replace with your actual key
        ```
    *   **IMPORTANT:** Ensure the `.env` file is listed in your `.gitignore` to prevent accidentally committing keys.

6.  **Prepare Data:**
    *   Create a `rag_data` folder in the project root if it doesn't exist: `mkdir rag_data`
    *   Place any PDF files you want to ingest into the `./rag_data/` directory.
    *   (Optional) Edit the `INGEST_URLS` list in `config.py` to add web pages for ingestion.

## Configuration

Key settings can be adjusted in `config.py`:

*   `GOOGLE_API_KEY`: Loaded from the `.env` file (essential).
*   `ALLOW_WEB_SEARCH_FALLBACK` (boolean): Set to `True` to allow web search if local documents aren't relevant, `False` to disable web search and only use ingested data.
*   `LLM_DECISION_MODEL`, `LLM_ANSWER_MODEL`: Specifies the language models used (via LiteLLM prefix).
*   `EMBEDDING_MODEL_NAME`, `VECTOR_SIZE`: Configures the embedding model and its dimensions. Must match the Qdrant collection setup.
*   `COLLECTION_NAME`: The name of the Qdrant collection used. Changing the embedding model *requires* changing this or clearing the old collection.
*   `PDF_FOLDER_PATH`, `INGEST_URLS`: Define data sources for ingestion.
*   `RETRIEVAL_TOP_K`: Number of document chunks retrieved from Qdrant.
*   `WEB_SEARCH_MAX_RESULTS`: Number of results fetched from DuckDuckGo.
*   Prompts (`DECISION_SYSTEM_PROMPT`, `ANSWER_SYSTEM_PROMPT`, etc.): Can be customized to guide the LLM behavior.

## Usage

Run commands from the project root directory in the terminal where your virtual environment is activated.

1.  **Clear Existing Data (Recommended before first ingest or after config changes):**
    *   This deletes the *currently configured* Qdrant collection (`COLLECTION_NAME` in `config.py`).
    ```bash
    python main.py --clear
    ```

2.  **Ingest Data:**
    *   Loads PDFs from `./rag_data` and URLs from `config.py`, processes them, generates embeddings using the configured Google model, and uploads to Qdrant.
    ```bash
    python main.py --ingest
    ```
    *   You can combine clear and ingest: `python main.py --clear --ingest`

3.  **Ask Questions:**
    *   Processes your query through the RAG pipeline.
    ```bash
    python main.py --query "Your question about the ingested documents?"
    ```
    *   Example: `python main.py --query "What is the main idea behind the STORM paper?"`

    *   If `ALLOW_WEB_SEARCH_FALLBACK` is `True` and the LLM decides local context is insufficient (returns '0'), it will search the web:
        ```bash
        python main.py --query "What is the latest news about autonomous vehicles?"
        ```

## How it Works

1.  **Query Embedding:** The user's query is converted into a vector embedding using the configured Google model (`RETRIEVAL_QUERY` task).
2.  **Retrieval:** The query vector is used to search the Qdrant vector database for the most semantically similar document chunks (based on cosine distance) previously ingested (`RETRIEVAL_DOCUMENT` task).
3.  **Relevance Check (Agentic Step):** The retrieved text chunks are presented to a Decision LLM (Gemini). The LLM is prompted to decide if this context is sufficient to answer the original query ('1' for yes, '0' for no).
4.  **Context Selection & Fallback:**
    *   If Decision == '1', the retrieved chunks are used as the context for the final answer.
    *   If Decision == '0' **and** `ALLOW_WEB_SEARCH_FALLBACK` is `True`, a DuckDuckGo web search is performed using the original query. The snippets from the web results become the context.
    *   If Decision == '0' **and** `ALLOW_WEB_SEARCH_FALLBACK` is `False`, the process stops and informs the user that the answer isn't in the local documents and web search is off.
5.  **Answer Generation:** The selected context (either from Qdrant or the web) is passed to an Answer LLM (Gemini) along with the original query. The LLM is prompted to answer *solely* based on the provided context.
6.  **Response:** The generated answer is returned to the user.

## Troubleshooting

*   **Connection Errors (Qdrant):** Ensure the Qdrant Docker container is running (`docker ps`). Check `config.QDRANT_URL`.
*   **Authentication Errors (Google API):** Verify the `GOOGLE_API_KEY` in `.env` is correct and active in your Google AI Studio / Cloud Console for the Generative Language API. Check for typos or extra spaces.
*   **Embedding Errors:** Usually related to the API key or Google API quotas/limits. Check error messages in the console.
*   **LLM Errors (LiteLLM/Gemini):** Could be API key issues, rate limits, or problems with the prompt/response structure. Check console logs and `litellm` documentation if needed. Ensure the explicit `api_key` pass in `llm_interface.py` is working.
*   **`main.py` Command Not Found:** Ensure your virtual environment is activated and you are in the correct project directory.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements. (Provide more details if needed).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.