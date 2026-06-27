"""
promptgate/integrations/openai.py
-----------------------------------
Drop-in wrapper around the OpenAI Python client that transparently screens
inputs and outputs via PromptGate before and after every API call.

Usage::

    from promptgate.integrations.openai import PromptGateOpenAI
    from promptgate import PromptGate
    import openai

    client = PromptGateOpenAI(
        openai_client=openai.OpenAI(api_key="..."),
        gate=PromptGate(),
        screen_outputs=True,
    )

    # Drop-in replacement — same API as openai.OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_message}]
    )

The wrapper screens the last user message in the ``messages`` list before
calling the OpenAI API. On BLOCK, ``PromptInjectionError`` is raised and
the API is NOT called — saving cost and latency. If ``screen_outputs=True``,
the response content is screened via ``check_output()`` after the API call.

The original ``openai.ChatCompletion`` response object is returned unchanged
on ALLOW — the wrapper is transparent to the caller.
"""

from __future__ import annotations

from typing import Any


class PromptGateOpenAI:
    """Wrapper around ``openai.OpenAI`` that screens inputs and outputs.

    Parameters
    ----------
    openai_client:
        A configured ``openai.OpenAI`` instance. The wrapper delegates all
        API calls to this client unchanged.
    gate:
        A configured ``PromptGate`` instance used for screening.
    screen_outputs:
        If ``True``, run ``gate.check_output()`` on the first response
        choice's message content after each API call. Default ``False``.
    """

    def __init__(
        self,
        openai_client: Any,
        gate: Any,
        screen_outputs: bool = False,
    ) -> None:
        self._client = openai_client
        self._gate = gate
        self._screen_outputs = screen_outputs
        # Expose chat.completions.create as the primary wrapped surface
        self.chat = _ChatProxy(self)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _screen_input(self, messages: list[dict]) -> None:
        """Extract and screen the last user message from ``messages``.

        Only the last message with ``role == "user"`` is screened.  System
        messages and assistant turns are trusted content controlled by the
        developer.  If no user message is present, the call passes through
        without screening.

        Raises
        ------
        PromptInjectionError
            If the last user message is classified as BLOCK by ``gate.check()``.
        """
        from promptgate.integrations.exceptions import PromptInjectionError

        last_user = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user = msg.get("content", "")
                break

        if last_user is None:
            return  # No user message — pass through

        result = self._gate.check(last_user)
        if result["decision"] == "BLOCK":
            raise PromptInjectionError(
                f"PromptGate blocked the input before calling the OpenAI API. "
                f"Decision: {result['decision']}, "
                f"confidence: {result['confidence']:.2f}, "
                f"categories: {result['threat_categories']}",
                gate_result=result,
            )

    def _screen_output(self, response: Any) -> None:
        """Screen the first choice's message content via ``check_output()``.

        Extracts ``response.choices[0].message.content``.  If the content is
        ``None`` or not a string (e.g., a tool call response), screening is
        skipped silently.

        Raises
        ------
        OutputScreeningError
            If the response content is classified as not ALLOW by
            ``gate.check_output()``.
        """
        from promptgate.integrations.exceptions import OutputScreeningError

        if not self._screen_outputs:
            return

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError):
            return  # Unexpected response shape — skip screening

        if not isinstance(content, str) or not content:
            return

        result = self._gate.check_output(content)
        if result["decision"] != "ALLOW":
            raise OutputScreeningError(
                f"PromptGate withheld the OpenAI response — sensitive content "
                f"detected. Decision: {result['decision']}, "
                f"confidence: {result['confidence']:.2f}, "
                f"categories: {result['threat_categories']}",
                gate_result=result,
            )


class _ChatProxy:
    """Proxy for ``client.chat`` that exposes a screened ``completions``."""

    def __init__(self, wrapper: PromptGateOpenAI) -> None:
        self._wrapper = wrapper
        self.completions = _CompletionsProxy(wrapper)


class _CompletionsProxy:
    """Proxy for ``client.chat.completions`` with screened ``create()``."""

    def __init__(self, wrapper: PromptGateOpenAI) -> None:
        self._wrapper = wrapper

    def create(self, *, messages: list[dict], **kwargs: Any) -> Any:
        """Screen input, call OpenAI, screen output, return response.

        Parameters
        ----------
        messages:
            The messages list passed to the OpenAI chat completions API.
            The last message with ``role == "user"`` is screened before
            the API call.
        **kwargs:
            All remaining keyword arguments (``model``, ``temperature``,
            ``max_tokens``, etc.) are forwarded unchanged to the underlying
            OpenAI client.

        Returns
        -------
        openai.types.chat.ChatCompletion
            The original OpenAI response object, unchanged.

        Raises
        ------
        PromptInjectionError
            If the last user message is BLOCK.  The API is not called.
        OutputScreeningError
            If ``screen_outputs=True`` and the response content is not ALLOW.
        """
        # Screen input — raises before API call on BLOCK (saves cost)
        self._wrapper._screen_input(messages)

        # Call the real OpenAI API
        response = self._wrapper._client.chat.completions.create(
            messages=messages, **kwargs
        )

        # Screen output if configured
        self._wrapper._screen_output(response)

        return response