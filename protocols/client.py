"""OpenAI client wrapper for TAMU API"""

import openai
import time
import json
from config import (
    TAMU_BASE_URL,
    TAMU_API_KEY,
    CF_AUTHORIZATION_COOKIE,
    MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    MAX_RETRIES,
    RETRY_DELAY
)


def get_client():
    """Initialize and return OpenAI client configured for TAMU API"""
    return openai.OpenAI(
        base_url=TAMU_BASE_URL,
        api_key=TAMU_API_KEY,
        default_headers={"Cookie": CF_AUTHORIZATION_COOKIE}
    )


def call_llm(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, model=MODEL):
    """
    Call LLM with retry logic.

    Args:
        messages: list of message dicts with 'role' and 'content'
        temperature: sampling temperature
        max_tokens: max output tokens
        model: model identifier

    Returns:
        (response_text, usage_dict) on success
        (None, None) on failure after retries
    """
    client = get_client()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )

            # Handle case where response is a string (error response)
            if isinstance(response, str):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    raise Exception(f"API returned string instead of response object: {response[:200]}")

            # Handle case where response is None
            if response is None:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    raise Exception("API returned None")

            # Handle API error responses (e.g., budget exceeded)
            if hasattr(response, 'error') and response.error:
                error_msg = response.error.get('message', 'Unknown error') if isinstance(response.error, dict) else str(response.error)
                raise Exception(f"API Error: {error_msg}")

            # Extract usage information
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }

            # Extract response content
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                elif hasattr(choice, 'text'):
                    content = choice.text
                else:
                    content = str(choice)
            else:
                raise Exception(f"Unexpected response structure: {response}")

            return content, usage

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed to call LLM after {MAX_RETRIES} attempts")
                print(f"Final error: {e}")
                return None, None
