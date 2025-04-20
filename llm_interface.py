
# llm_interface.py
from litellm import completion
from typing import Optional, Dict, List, Tuple
import config
import re # Import regex for parsing decision
import traceback # For detailed error logging

def call_llm(model: str, system_prompt: str, user_prompt: str, max_tokens: int = config.LLM_MAX_TOKENS) -> Optional[str]:
    """
    Makes a call to the specified LLM using litellm.
    Checks for necessary API keys based on model prefix before calling.
    Explicitly passes the API key for Gemini calls.

    Args:
        model (str): The name of the LLM model to use (e.g., 'gemini/gemini-1.5-flash-latest').
        system_prompt (str): The system message/instruction for the LLM.
        user_prompt (str): The user message/query for the LLM.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        Optional[str]: The content of the LLM's response, or None on error.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Determine required API key and check if it's available
    api_key_to_use = None
    model_provider = "Unknown"

    if model.startswith("gemini/"):
        api_key_to_use = config.GOOGLE_API_KEY
        model_provider = "Google Gemini"
        if not api_key_to_use:
             print(f"ERROR: LLM Call failed for model '{model}'. GOOGLE_API_KEY is missing.")
             return None
    elif model.startswith("gpt-"):
        api_key_to_use = config.OPENAI_API_KEY # Example if you switch back
        model_provider = "OpenAI GPT"
        if not api_key_to_use:
            print(f"ERROR: LLM Call failed for model '{model}'. OPENAI_API_KEY is missing.")
            return None
    # Add elif blocks here for other providers

    print(f"\nAttempting LLM call to '{model}' ({model_provider})...")

    try:
        # <<< MODIFICATION: Explicitly pass api_key >>>
        response = completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            api_key=api_key_to_use # Pass the key directly
        )

        # Validate response structure before accessing content
        if response and response.choices and response.choices[0].message and response.choices[0].message.content is not None:
             content = response.choices[0].message.content.strip()
             print(f"LLM '{model}' responded successfully.")
             return content
        else:
             print(f"ERROR: Unexpected LLM response structure from model '{model}'.")
             print(f"       Raw response object: {response}")
             return None

    except Exception as e:
        print(f"ERROR: Exception during LLM call to '{model}':")
        traceback.print_exc()
        return None


# --- Functions get_llm_decision and get_llm_answer remain unchanged ---
# --- They use call_llm which now includes the explicit api_key ---

def get_llm_decision(context: str, question: str) -> Optional[str]:
    """
    Asks the LLM to decide if the context can answer the question.
    Returns '0' (cannot answer) or '1' (can answer), robustly parsing the LLM output.
    Returns None on failure to get a valid decision.
    """
    system_prompt = config.DECISION_SYSTEM_PROMPT.format(context=context)
    user_prompt = config.DECISION_USER_PROMPT.format(question=question)

    raw_decision = call_llm(
        model=config.LLM_DECISION_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=20 # Allow a bit more buffer
    )

    if raw_decision is None:
        print("Error: Failed to get decision response from LLM.")
        return None # Propagate the error/failure state

    match = re.search(r'\b([01])\b', raw_decision) # Search for whole digit
    if not match: match = re.search(r'([01])', raw_decision) # Fallback to any digit

    if match:
        decision = match.group(1) # Get the matched digit ('0' or '1')
        print(f"  LLM Decision Raw Output: '{raw_decision[:50]}...' -> Parsed Decision: '{decision}'")
        return decision
    else:
        print(f"WARNING: LLM decision response did not contain a clear '0' or '1'. Received: '{raw_decision[:100]}...'")
        print(f"         Defaulting decision to '0' (search web) for safety.")
        return '0'

def get_llm_answer(context: str, question: str) -> Optional[str]:
    """
    Asks the LLM to generate an answer based *only* on the provided context.
    """
    system_prompt = config.ANSWER_SYSTEM_PROMPT.format(context=context)
    user_prompt = config.ANSWER_USER_PROMPT.format(question=question)

    answer = call_llm(
        model=config.LLM_ANSWER_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=config.LLM_MAX_TOKENS
    )

    if answer is None:
        print("Error: Failed to get answer response from LLM.")
        return "Sorry, I encountered an error while generating the answer." # Return error message
    else:
        return answer


if __name__ == '__main__':
    # Example Usage - requires GOOGLE_API_KEY set in .env
    print("--- Testing LLM Interface (Google LLM) ---")
    sample_context = "Llama 3 is a Large Language Model from Meta AI."
    sample_question_answered = "Who made Llama 3?"
    sample_question_not_answered = "What's the capital of Canada?"

    if not config.GOOGLE_API_KEY:
         print(f"\nSkipping LLM call tests: GOOGLE_API_KEY not set in config or .env file.")
    else:
        print("Using Google API Key for testing...")
        # Test Decision LLM
        print(f"\n--- Testing Decision LLM (Q: '{sample_question_answered}') ---")
        decision1 = get_llm_decision(sample_context, sample_question_answered)
        print(f"Final Parsed Decision: {decision1}")

        print(f"\n--- Testing Decision LLM (Q: '{sample_question_not_answered}') ---")
        decision2 = get_llm_decision(sample_context, sample_question_not_answered)
        print(f"Final Parsed Decision: {decision2}")

        # Test Answer LLM
        print(f"\n--- Testing Answer LLM (Answerable Question) ---")
        answer1 = get_llm_answer(sample_context, sample_question_answered)
        print(f"Answer:\n{answer1}")

        print(f"\n--- Testing Answer LLM (Unanswerable Question in Context) ---")
        answer2 = get_llm_answer(sample_context, sample_question_not_answered)
        print(f"Answer:\n{answer2}")