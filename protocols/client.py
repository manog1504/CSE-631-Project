"""Anthropic API client wrapper"""

import anthropic
import time
from config import ANTHROPIC_API_KEY, MODEL, TEMPERATURE, MAX_TOKENS, MAX_RETRIES, RETRY_DELAY


def call_llm(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, model=MODEL):
    """
    Call Claude via Anthropic SDK with retry logic.

    Args:
        messages: list of dicts with 'role' and 'content' (system role supported)
        temperature: sampling temperature
        max_tokens: max output tokens
        model: model identifier

    Returns:
        (response_text, usage_dict) on success
        (None, None) on failure after retries
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Anthropic's API takes the system prompt as a top-level `system` kwarg,
    # not as a message with role="system". Split it out here so callers can
    # use the familiar OpenAI-style messages list format.
    system = None
    conv_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            conv_messages.append(msg)

    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": conv_messages,
            }
            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)

            text = response.content[0].text if response.content else None
            if text is None:
                raise Exception("Empty response from API")

            usage = {
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }
            return text, usage

        except Exception as e:
            print(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None, None
