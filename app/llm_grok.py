# app/llm_grok.py
from __future__ import annotations

import logging
import random
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from llm_base import LLMClient
from settings import GROK_MAX_RETRIES, GROK_MAX_TOKENS, GROK_MODEL, api_key

logger = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1"

SYSTEM_PROMPT = (
    "Answer using the provided sources. Cite pages like (PDF p. 3)."
)


class GrokLLM(LLMClient):
    """
    xAI Grok through the OpenAI-compatible endpoint.

    Requires XAI_API_KEY (or GROK_API_KEY). A full handbook run issues thirty or
    more calls, so transient rate limits and connection resets are retried with
    exponential backoff rather than aborting the whole document.
    """

    def __init__(self) -> None:
        key = api_key()
        if not key:
            raise RuntimeError("Missing XAI_API_KEY (or GROK_API_KEY) in environment")

        self.client = OpenAI(api_key=key, base_url=XAI_BASE_URL)
        self.model = GROK_MODEL
        self.max_tokens = GROK_MAX_TOKENS
        self.max_retries = max(1, GROK_MAX_RETRIES)

    def _call(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def generate(self, prompt: str) -> str:
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return self._call(prompt)

            except PermissionDeniedError as exc:
                # Not retryable: the credential itself is the problem.
                raise RuntimeError(
                    "Grok API call failed (403 PermissionDenied). Check that your "
                    "xAI account has API access and billing enabled."
                ) from exc

            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                last_error = exc

            except APIStatusError as exc:
                # Server-side faults are worth retrying; 4xx client errors are not.
                if exc.status_code < 500:
                    raise RuntimeError(
                        f"Grok API rejected the request ({exc.status_code}): {exc}"
                    ) from exc
                last_error = exc

            if attempt < self.max_retries - 1:
                # Exponential backoff with jitter, so parallel sections do not
                # retry in lockstep and re-trigger the same rate limit.
                delay = (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "Grok call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    last_error,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"Grok API unavailable after {self.max_retries} attempts: {last_error}"
        ) from last_error
