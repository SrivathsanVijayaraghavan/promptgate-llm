"""PromptGate FastAPI middleware.

Drop-in middleware that wraps any existing FastAPI application with
PromptGate input (and optionally output) screening. Zero changes to
existing route handlers required.

Usage::

    from fastapi import FastAPI
    from promptgate import PromptGate
    from promptgate.integrations.fastapi import PromptGateMiddleware

    app = FastAPI()
    app.add_middleware(
        PromptGateMiddleware,
        gate=PromptGate(),
        input_fields=["message", "prompt", "query"],
        block_status_code=403,
        review_action="allow",
        flag_action="allow",
        screen_output=False,
    )

All existing routes work unchanged. PromptGate intercepts JSON request
bodies, screens the configured fields, and either blocks the request
before it reaches the route handler or passes it through unmodified.

Design notes:
- Non-JSON requests pass through without screening (no crash).
- Requests with missing or non-string fields pass through without
  screening (graceful degradation — missing field = no signal).
- Body is read once, buffered, and re-injected so the route handler
  receives the original request unchanged.
- screen_output=True reads the response body after the route handler
  returns, runs check_output(), and replaces the body with a JSON
  error if the output is not ALLOW.
"""

from __future__ import annotations

import json
from typing import Callable, Sequence

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


class PromptGateMiddleware(BaseHTTPMiddleware):
    """Starlette/FastAPI middleware that screens requests and responses.

    Args:
        app: The ASGI application (supplied automatically by FastAPI).
        gate: A configured ``PromptGate`` instance. Required.
        input_fields: JSON body fields to screen on every request.
            Defaults to ``["message", "prompt", "query"]``.
            Fields not present in the body are silently skipped.
        block_status_code: HTTP status code returned when a request is
            blocked. Defaults to 403.
        review_action: What to do when ``check()`` returns REVIEW.
            ``"allow"`` (default) — passes the request through.
            ``"block"`` — blocks with ``block_status_code``.
        flag_action: What to do when ``check()`` returns FLAG.
            ``"allow"`` (default) — passes the request through.
            ``"block"`` — blocks with ``block_status_code``.
        screen_output: If ``True``, run ``check_output()`` on the
            response body after the route handler returns. Responses
            that are not ALLOW are replaced with a JSON error.
            Defaults to ``False``.
    """

    def __init__(
        self,
        app: ASGIApp,
        gate,
        input_fields: Sequence[str] = ("message", "prompt", "query"),
        block_status_code: int = 403,
        review_action: str = "allow",
        flag_action: str = "allow",
        screen_output: bool = False,
    ) -> None:
        super().__init__(app)
        self.gate = gate
        self.input_fields = list(input_fields)
        self.block_status_code = block_status_code
        self.review_action = review_action
        self.flag_action = flag_action
        self.screen_output = screen_output

    def _should_block(self, decision: str) -> bool:
        """Return True if this decision should result in a blocked response."""
        if decision == "BLOCK":
            return True
        if decision == "REVIEW" and self.review_action == "block":
            return True
        if decision == "FLAG" and self.flag_action == "block":
            return True
        return False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # ── Input screening ───────────────────────────────────────────────
        body_bytes = await request.body()

        if body_bytes:
            try:
                body_json = json.loads(body_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                body_json = None

            if isinstance(body_json, dict):
                for field in self.input_fields:
                    value = body_json.get(field)
                    if not isinstance(value, str) or not value.strip():
                        continue

                    result = self.gate.check(value)

                    if self._should_block(result["decision"]):
                        return JSONResponse(
                            status_code=self.block_status_code,
                            content={
                                "blocked": True,
                                "field": field,
                                "decision": result["decision"],
                                "confidence": result["confidence"],
                                "risk_level": result["risk_level"],
                                "threat_categories": result["threat_categories"],
                                "message": result["message"],
                            },
                        )

        # Re-inject body so the route handler can read it normally.
        # Starlette's BaseHTTPMiddleware buffers the body automatically,
        # but we set _body explicitly to be safe.
        async def receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}

        request._receive = receive  # type: ignore[attr-defined]

        # ── Route handler ─────────────────────────────────────────────────
        response = await call_next(request)

        # ── Output screening ──────────────────────────────────────────────
        if not self.screen_output:
            return response

        # Read response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk if isinstance(chunk, bytes) else chunk.encode()

        # Try to extract text from JSON response
        output_text: str | None = None
        try:
            resp_json = json.loads(response_body)
            if isinstance(resp_json, dict):
                # Look for common response text fields
                for key in ("response", "message", "content", "text", "output"):
                    val = resp_json.get(key)
                    if isinstance(val, str) and val.strip():
                        output_text = val
                        break
            elif isinstance(resp_json, str):
                output_text = resp_json
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Plain text response
            try:
                output_text = response_body.decode("utf-8")
            except UnicodeDecodeError:
                output_text = None

        if output_text:
            output_result = self.gate.check_output(output_text)
            if self._should_block(output_result["decision"]):
                return JSONResponse(
                    status_code=self.block_status_code,
                    content={
                        "blocked": True,
                        "source": "output_screening",
                        "decision": output_result["decision"],
                        "confidence": output_result["confidence"],
                        "risk_level": output_result["risk_level"],
                        "message": "Response withheld — sensitive content detected.",
                    },
                )

        # Return original response with body re-streamed
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )