from duckduckgo_search import DDGS
from typing import List, Dict, Optional
import config

def web_search(query: str, max_results: int = config.WEB_SEARCH_MAX_RESULTS) -> Optional[List[Dict]]:
    """
    Performs a web search using DuckDuckGo.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to retrieve.

    Returns:
        Optional[List[Dict]]: A list of search result dictionaries (typically
                              containing 'title', 'href', 'body'), or None on error.
    """
    print(f"Performing web search for: '{query}' (max {max_results} results)")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            # Sometimes ddgs.text might yield fewer results than requested or none
            if results:
                 print(f"Found {len(results)} results from web search.")
                 return results
            else:
                 print("Web search returned no results.")
                 return []
    except Exception as e:
        print(f"Error during web search: {e}")
        return None

def format_search_results(results: Optional[List[Dict]]) -> str:
    """
    Formats DuckDuckGo search results into a single string context.

    Args:
        results (Optional[List[Dict]]): The list of result dictionaries from DDGS().text.

    Returns:
        str: A formatted string containing the body of each search result,
             or a message indicating no results were found or an error occurred.
    """
    if results is None:
        return "An error occurred during the web search."
    if not results:
        return "No relevant information found from web search."

    # Extract the 'body' (snippet) from each result
    snippets = [doc.get("body", "") for doc in results if doc.get("body")]
    if not snippets:
         return "Web search results did not contain usable content snippets."

    return "\n\n".join(snippets)


if __name__ == '__main__':
    # Example Usage
    print("--- Testing Search Tools ---")
    test_query = "What is OpenAI o1 model?"

    results = web_search(test_query)

    if results is not None:
        print(f"\nRaw results for '{test_query}':")
        for i, res in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Title: {res.get('title')}")
            print(f"  Href: {res.get('href')}")
            print(f"  Body: {res.get('body', '')[:100]}...") # Show first 100 chars of body

        formatted_context = format_search_results(results)
        print(f"\nFormatted Context for LLM:\n{formatted_context}")

    else:
        print("\nWeb search failed.")

    # Test formatting with empty/None results
    print("\n--- Testing Formatting Edge Cases ---")
    print("Formatting None:", format_search_results(None))
    print("Formatting empty list:", format_search_results([]))
    print("Formatting list with no body:", format_search_results([{'title': 't', 'href': 'h'}]))