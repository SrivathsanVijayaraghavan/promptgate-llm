"""
promptgate/integrations/exceptions.py
---------------------------------------
Custom exceptions raised by PromptGate integrations.

All exceptions carry the full ``gate_result`` dict so callers can
inspect which signals fired, the confidence score, and the decision
without having to re-run the check.

Importable directly from ``promptgate``::

    from promptgate import PromptInjectionError, OutputScreeningError
"""


class PromptGateError(Exception):
    """Base class for all PromptGate integration exceptions."""

    def __init__(self, message: str, gate_result: dict | None = None) -> None:
        super().__init__(message)
        self.gate_result: dict | None = gate_result


class PromptInjectionError(PromptGateError):
    """Raised when an input is classified as BLOCK by ``gate.check()``.

    Attributes
    ----------
    gate_result:
        The full 7-key response dict from ``gate.check()``.  Provides
        decision, confidence, signals, threat_categories, etc.

    Example
    -------
    ::

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": user_message}]
            )
        except PromptInjectionError as exc:
            print(exc.gate_result["threat_categories"])
    """


class OutputScreeningError(PromptGateError):
    """Raised when an LLM response fails ``gate.check_output()`` screening.

    Attributes
    ----------
    gate_result:
        The full 7-key response dict from ``gate.check_output()``.

    Example
    -------
    ::

        try:
            response = client.chat.completions.create(...)
        except OutputScreeningError as exc:
            log_security_event(exc.gate_result)
            return "Response withheld — sensitive content detected."
    """