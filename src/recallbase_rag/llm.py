import logging
import time
from typing import Protocol

import openai
from openai import OpenAI

from recallbase_rag.errors import LLMError
from recallbase_rag.prompt_builder import Prompt

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60.0


class LLM(Protocol):
    def generate(self, prompt: Prompt) -> str: ...


class OpenAILLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not api_key.strip():
            raise LLMError("OPENAI_API_KEY is not set. Add it to your .env file.")

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=2,
        )

    def generate(self, prompt: Prompt) -> str:
        started = time.perf_counter()

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=prompt.system,
                input=prompt.user,
            )
        except openai.AuthenticationError as error:
            raise LLMError(
                "OpenAI rejected the API key. Check OPENAI_API_KEY in .env."
            ) from error
        except openai.RateLimitError as error:
            raise LLMError(
                "OpenAI rate limit or quota reached. Try again later."
            ) from error
        except openai.APIConnectionError as error:
            raise LLMError(
                "Could not reach OpenAI (network problem or timeout)."
            ) from error
        except openai.APIError as error:
            raise LLMError(f"OpenAI request failed: {error}") from error

        answer = response.output_text.strip()
        if not answer:
            raise LLMError("OpenAI returned an empty answer.")

        elapsed = time.perf_counter() - started
        logger.info("Model %s answered in %.1f seconds", self.model, elapsed)
        return answer