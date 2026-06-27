"""PromptGate LangChain callback handler.

Screens LLM inputs and outputs during chain execution using LangChain's
callback system. Drop-in integration — attach to any LLM, chain, or agent
without modifying existing code.

Usage::

    from promptgate import PromptGate
    from promptgate.integrations.langchain import PromptGateCallbackHandler

    handler = PromptGateCallbackHandler(
        gate=PromptGate(),
        on_block="raise",      # "raise" | "warn" | "skip"
        screen_outputs=True,   # also run check_output() on LLM responses
    )

    # Attach to any LangChain LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(callbacks=[handler])

    # Or attach to a chain
    chain = prompt | llm | output_parser
    result = chain.invoke({"input": user_message}, config={"callbacks": [handler]})

Exceptions::

    from promptgate.integrations.langchain import (
        PromptInjectionError,
        OutputScreeningError,
    )

on_block behaviour:
    "raise" (default) — raises PromptInjectionError / OutputScreeningError
    "warn"            — logs a warning, execution continues
    "skip"            — silently allows, useful for monitoring-only mode
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class PromptInjectionError(Exception):
    """Raised when PromptGate blocks an LLM input due to injection risk.

    Attributes:
        result: The full 7-key PromptGate response dict.
    """

    def __init__(self, result: dict) -> None:
        self.result = result
        decision = result.get("decision", "BLOCK")
        confidence = result.get("confidence", 0.0)
        message = result.get("message", "Prompt injection detected.")
        super().__init__(
            f"PromptGate blocked input [{decision}, confidence={confidence:.2f}]: {message}"
        )


class OutputScreeningError(Exception):
    """Raised when PromptGate blocks an LLM output due to sensitive content.

    Attributes:
        result: The full 7-key PromptGate check_output() response dict.
    """

    def __init__(self, result: dict) -> None:
        self.result = result
        decision = result.get("decision", "BLOCK")
        confidence = result.get("confidence", 0.0)
        super().__init__(
            f"PromptGate blocked output [{decision}, confidence={confidence:.2f}]: "
            "sensitive content detected in LLM response."
        )


class PromptGateCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that screens inputs and outputs.

    Hooks into ``on_llm_start`` to screen prompts before they are sent
    to the LLM, and into ``on_llm_end`` to screen responses before they
    are returned to the chain.

    Args:
        gate: A configured ``PromptGate`` instance. Required.
        on_block: Action when a decision is BLOCK.
            ``"raise"`` (default) — raises ``PromptInjectionError`` or
            ``OutputScreeningError``.
            ``"warn"`` — logs a warning at WARNING level and continues.
            ``"skip"`` — silently continues (monitoring-only mode).
        screen_outputs: If ``True``, run ``check_output()`` on LLM
            responses in ``on_llm_end``. Defaults to ``True``.
    """

    raise_error = True  # BaseCallbackHandler: propagate exceptions from callbacks

    def __init__(
        self,
        gate,
        on_block: str = "raise",
        screen_outputs: bool = True,
    ) -> None:
        super().__init__()
        if on_block not in ("raise", "warn", "skip"):
            raise ValueError(
                f"on_block must be 'raise', 'warn', or 'skip', got {on_block!r}"
            )
        self.gate = gate
        self.on_block = on_block
        self.screen_outputs = screen_outputs

    def _handle_block(
        self,
        result: dict,
        error_class: type,
        context: str,
    ) -> None:
        """Apply the configured on_block action."""
        if self.on_block == "raise":
            raise error_class(result)
        elif self.on_block == "warn":
            logger.warning(
                "PromptGate %s: decision=%s confidence=%.2f categories=%s",
                context,
                result.get("decision"),
                result.get("confidence", 0.0),
                result.get("threat_categories", []),
            )
        # "skip" — do nothing

    def _extract_text_from_messages(
        self, messages: List[Union[BaseMessage, List[BaseMessage]]]
    ) -> Optional[str]:
        """Extract the last human/user message text from a messages list.

        LangChain passes prompts to on_llm_start as either:
        - List[str] (legacy string prompts)
        - List[List[BaseMessage]] (chat message lists)

        We screen the last user-role message, which carries the actual
        user input. System messages and prior assistant turns are skipped.
        """
        if not messages:
            return None

        # Flatten if nested list
        flat: list = []
        for item in messages:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)

        # Find last human/user message
        for msg in reversed(flat):
            if hasattr(msg, "type") and msg.type in ("human", "user"):
                content = msg.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    # Multi-modal content — extract text parts
                    parts = [
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    ]
                    return " ".join(parts).strip() or None

        # Fall back to last message of any role
        last = flat[-1]
        if hasattr(last, "content"):
            content = last.content
            return content if isinstance(content, str) else None

        return None

    # ── LangChain callback hooks ──────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Screen string prompts before they are sent to the LLM."""
        for prompt in prompts:
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            result = self.gate.check(prompt)
            if result["decision"] == "BLOCK":
                self._handle_block(result, PromptInjectionError, "input blocked")

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Screen chat messages before they are sent to the LLM."""
        text = self._extract_text_from_messages(messages)
        if not text:
            return
        result = self.gate.check(text)
        if result["decision"] == "BLOCK":
            self._handle_block(result, PromptInjectionError, "input blocked")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Screen LLM response text before it is returned to the chain."""
        if not self.screen_outputs:
            return

        for generations in response.generations:
            for generation in generations:
                text = getattr(generation, "text", None)
                if not text or not isinstance(text, str):
                    # Try message content for ChatGenerations
                    msg = getattr(generation, "message", None)
                    if msg is not None:
                        text = getattr(msg, "content", None)

                if not text or not isinstance(text, str) or not text.strip():
                    continue

                result = self.gate.check_output(text)
                if result["decision"] == "BLOCK":
                    self._handle_block(
                        result, OutputScreeningError, "output blocked"
                    )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Pass through LLM errors without interference."""
        pass