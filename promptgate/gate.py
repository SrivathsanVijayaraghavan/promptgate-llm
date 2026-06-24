"""Main PromptGate middleware entry point."""

import asyncio
import hashlib
import json
from datetime import datetime, timezone

from promptgate.aggregator import aggregate
from promptgate.detector.intent import IntentClassifier
from promptgate.detector.output_filter import OutputFilter
from promptgate.detector.rule_based import RuleBasedDetector
from promptgate.detector.semantic import SemanticDetector
from promptgate.parser.input_parser import parse_input
from promptgate.policy import evaluate
from promptgate.response import build_response
from promptgate.scorer import score


class PromptGate:
    """AI security middleware that classifies prompt injection risk before LLM access.

    Runs a three-layer detection pipeline:
      1. Rule-based — fast keyword/phrase matching against pattern files.
      2. Semantic   — sentence-embedding similarity against known attack library.
      3. Intent     — fine-tuned DistilBERT classifier for implicit/conversational
                      injections that bypass vocabulary-based detection entirely.

    All layers feed signals into the same accumulation model. Each layer degrades
    gracefully when its optional dependencies are not installed.

    Phase 6 additions:
      - check_batch()  — efficient multi-input processing
      - acheck()       — async wrapper for FastAPI / LangChain integration
      - acheck_batch() — async batch wrapper
      - history        — conversation context for multi-turn attack detection
      - log_mode       — privacy-safe JSONL audit logging
      - on_block / on_flag / on_review / on_allow / on_error — callback hooks

    Phase 8 additions:
      - check_output() — screen LLM-generated responses before returning to user
      - sanitize()     — strip character-level attack primitives from input
    """

    def __init__(
        self,
        thresholds: dict | None = None,
        skip_semantic: bool = False,
        skip_intent: bool = False,
        semantic_threshold: float = 0.65,
        intent_threshold: float = 0.70,
        log_mode: bool = False,
        log_path: str = "./promptgate_audit.jsonl",
        on_block=None,
        on_flag=None,
        on_review=None,
        on_allow=None,
        on_error=None,
    ) -> None:
        """Initialise PromptGate with optional configuration.

        Args:
            thresholds: Optional dict to override DEFAULT_THRESHOLDS.
                        Accepted keys: block, review, flag.
                        Unspecified keys fall back to DEFAULT_THRESHOLDS.
            skip_semantic: If True, the semantic detector is never called
                           regardless of whether it is installed.
            skip_intent: If True, the intent classifier is never called
                         regardless of whether it is installed.
            semantic_threshold: Cosine similarity cutoff passed to
                                SemanticDetector. Default 0.65.
            intent_threshold: INJECTION probability cutoff passed to
                              IntentClassifier. Default 0.70.
            log_mode: If True, appends a privacy-safe audit record to
                      log_path after every check() call. Raw input is
                      never logged — only its sha256 hash and metadata.
                      Default False.
            log_path: Path to the JSONL audit log file. Only used when
                      log_mode=True. Default './promptgate_audit.jsonl'.
            on_block: Optional callable(result: dict) called when the
                      decision is BLOCK.
            on_flag: Optional callable(result: dict) called when the
                     decision is FLAG.
            on_review: Optional callable(result: dict) called when the
                       decision is REVIEW.
            on_allow: Optional callable(result: dict) called when the
                      decision is ALLOW.
            on_error: Optional callable(exc: Exception) called when any
                      hook raises an exception. If on_error itself raises,
                      the exception is silently swallowed. Hook failures
                      never affect the detection result.
        """
        self.thresholds = thresholds
        self.skip_semantic = skip_semantic
        self.skip_intent = skip_intent
        self.log_mode = log_mode
        self.log_path = log_path
        self.on_block = on_block
        self.on_flag = on_flag
        self.on_review = on_review
        self.on_allow = on_allow
        self.on_error = on_error
        self.rule_detector = RuleBasedDetector()
        self.semantic_detector = SemanticDetector(threshold=semantic_threshold)
        self.intent_detector = IntentClassifier(threshold=intent_threshold)
        self.output_filter = OutputFilter(semantic_threshold=semantic_threshold)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        cleaned: str,
        intent_input: str,
    ) -> dict:
        """Run detection layers and build response for one pre-parsed input.

        Shared by check(), check_batch(), and the async wrappers. Accepts
        the already-cleaned text and a (possibly history-enriched) intent
        input string so callers control what each layer sees.

        Args:
            cleaned: Normalised lowercase text from the parser. Used by
                     rule-based and semantic layers.
            intent_input: Text passed to the intent classifier. May include
                          prepended conversation history.

        Returns:
            Structured response dict with exactly 7 keys.
        """
        # Layer 1 — Rule-based
        rule_signals = self.rule_detector.detect(cleaned)
        rule_checked = (
            f"rule_based: {len(rule_signals)} pattern{'s' if len(rule_signals) != 1 else ''} matched"
            if rule_signals
            else "rule_based: no injection patterns found"
        )

        # Layer 2 — Semantic
        semantic_signals = []
        if self.skip_semantic:
            semantic_checked = "semantic: skipped by configuration"
        elif not self.semantic_detector.is_available():
            semantic_checked = "semantic: skipped (not installed)"
        else:
            semantic_signals = self.semantic_detector.detect(cleaned)
            semantic_checked = (
                "semantic: similar attack found above threshold"
                if semantic_signals
                else "semantic: no similar attacks found"
            )

        # Layer 3 — Intent
        intent_signals = []
        if self.skip_intent:
            intent_checked = "intent: skipped by configuration"
        elif not self.intent_detector.is_available():
            intent_checked = "intent: skipped (model not trained or not installed)"
        else:
            intent_signals = self.intent_detector.detect(intent_input)
            intent_checked = (
                "intent: injection intent detected above threshold"
                if intent_signals
                else "intent: no injection intent detected"
            )

        all_signals = rule_signals + semantic_signals + intent_signals
        signals_checked = [rule_checked, semantic_checked, intent_checked]

        aggregated = aggregate(all_signals)
        signals = aggregated["signals"]
        threat_categories = aggregated["threat_categories"]

        risk_score = score(signals)
        decision = evaluate(risk_score, self.thresholds)

        return build_response(
            decision=decision,
            risk_score=risk_score,
            threat_categories=threat_categories,
            signals=signals,
            signals_checked=signals_checked,
        )

    def _build_intent_input(self, cleaned: str, history: list[dict] | None) -> str:
        """Prepend the last 3 conversation turns to the cleaned input.

        Passes enriched context to the intent classifier only. Rule-based
        and semantic layers always receive the raw cleaned input — history
        increases noise for pattern matching and embedding comparison.

        DistilBERT truncates at 512 tokens. By prepending history and
        appending the current message last, truncation (when it occurs)
        removes older context rather than the current input. This ensures
        the current message is always represented in the classification.

        Invalid turn dicts (missing 'role' or 'content') are silently
        skipped — malformed history never crashes detection.

        Args:
            cleaned: Normalised current user input.
            history: Optional list of prior conversation turns.
                     Each turn should have 'role' and 'content' keys.

        Returns:
            String to pass to the intent classifier. Equals cleaned when
            history is None or empty.
        """
        if not history:
            return cleaned

        recent = history[-3:]
        context_parts = [
            f"{turn['role']}: {turn['content']}"
            for turn in recent
            if isinstance(turn, dict) and "role" in turn and "content" in turn
        ]
        if not context_parts:
            return cleaned

        context_parts.append(f"user: {cleaned}")
        return " | ".join(context_parts)

    def _log_decision(self, raw_input: str, result: dict) -> None:
        """Append one privacy-safe audit record to the JSONL log file.

        Never logs raw input text. Logs the sha256 hash of raw input so
        identical inputs can be correlated across requests without exposing
        prompt content. All other fields are metadata only.

        Logging failures (disk full, permission error, etc.) are silently
        swallowed — they must never crash or delay the detection pipeline.

        Args:
            raw_input: Original unmodified user input. Hashed, never stored.
            result: The result dict from the detection pipeline.
        """
        if not self.log_mode:
            return

        input_hash = "sha256:" + hashlib.sha256(
            raw_input.encode("utf-8")
        ).hexdigest()

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_hash": input_hash,
            "decision": result["decision"],
            "confidence": result["confidence"],
            "risk_level": result["risk_level"],
            "threat_categories": result["threat_categories"],
            "signal_count": len(result["signals"]),
            "signals_checked": result["signals_checked"],
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    def _call_hook(self, hook, payload) -> None:
        """Call a callback hook safely, routing exceptions to on_error.

        Hook exceptions are passed to on_error if set; otherwise silently
        swallowed. If on_error itself raises, that exception is also
        swallowed. Detection results are never affected by hook behavior.

        Args:
            hook: Callable to invoke, or None.
            payload: Argument to pass to the hook.
        """
        if hook is None:
            return
        try:
            hook(payload)
        except Exception as exc:
            if self.on_error is not None:
                try:
                    self.on_error(exc)
                except Exception:
                    pass

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        user_input: str,
        history: list[dict] | None = None,
    ) -> dict:
        """Run the full three-layer risk classification pipeline.

        Args:
            user_input: Raw user prompt text.
            history: Optional list of previous conversation turns for
                     multi-turn attack detection. Each turn is a dict
                     with keys 'role' (str) and 'content' (str).

        Returns:
            Structured response dict with exactly 7 keys:
            decision, confidence, risk_level, threat_categories,
            signals, signals_checked, message.
        """
        parsed = parse_input(user_input)
        cleaned = parsed["cleaned_text"]
        intent_input = self._build_intent_input(cleaned, history)

        result = self._run_pipeline(cleaned, intent_input)

        self._log_decision(user_input, result)

        hook_map = {
            "BLOCK":  self.on_block,
            "FLAG":   self.on_flag,
            "REVIEW": self.on_review,
            "ALLOW":  self.on_allow,
        }
        self._call_hook(hook_map.get(result["decision"]), result)

        return result

    def check_batch(self, inputs: list[str]) -> list[dict]:
        """Check multiple prompts efficiently in a single call.

        Runs the full three-layer pipeline across all inputs. The semantic
        detector encodes all inputs in one batch, making this significantly
        faster than calling check() in a loop when the semantic layer is
        active.

        Args:
            inputs: List of raw user prompt strings.

        Returns:
            List of result dicts in the same order as inputs.
            Returns [] if inputs is empty.
        """
        if not inputs:
            return []

        parsed_list = [parse_input(inp) for inp in inputs]
        cleaned_list = [p["cleaned_text"] for p in parsed_list]

        if not self.skip_semantic and self.semantic_detector.is_available():
            semantic_signals_list = self.semantic_detector.detect_batch(cleaned_list)
            if not semantic_signals_list:
                semantic_signals_list = [[] for _ in inputs]
        else:
            semantic_signals_list = [[] for _ in inputs]

        results = []
        for i, (raw_input, cleaned) in enumerate(zip(inputs, cleaned_list)):
            rule_signals = self.rule_detector.detect(cleaned)
            rule_checked = (
                f"rule_based: {len(rule_signals)} pattern{'s' if len(rule_signals) != 1 else ''} matched"
                if rule_signals
                else "rule_based: no injection patterns found"
            )

            semantic_signals = semantic_signals_list[i] if i < len(semantic_signals_list) else []
            if self.skip_semantic:
                semantic_checked = "semantic: skipped by configuration"
            elif not self.semantic_detector.is_available():
                semantic_checked = "semantic: skipped (not installed)"
            else:
                semantic_checked = (
                    "semantic: similar attack found above threshold"
                    if semantic_signals
                    else "semantic: no similar attacks found"
                )

            intent_signals = []
            if self.skip_intent:
                intent_checked = "intent: skipped by configuration"
            elif not self.intent_detector.is_available():
                intent_checked = "intent: skipped (model not trained or not installed)"
            else:
                intent_signals = self.intent_detector.detect(cleaned)
                intent_checked = (
                    "intent: injection intent detected above threshold"
                    if intent_signals
                    else "intent: no injection intent detected"
                )

            all_signals = rule_signals + semantic_signals + intent_signals
            signals_checked = [rule_checked, semantic_checked, intent_checked]

            aggregated = aggregate(all_signals)
            risk_score = score(aggregated["signals"])
            decision = evaluate(risk_score, self.thresholds)

            result = build_response(
                decision=decision,
                risk_score=risk_score,
                threat_categories=aggregated["threat_categories"],
                signals=aggregated["signals"],
                signals_checked=signals_checked,
            )

            self._log_decision(raw_input, result)

            hook_map = {
                "BLOCK":  self.on_block,
                "FLAG":   self.on_flag,
                "REVIEW": self.on_review,
                "ALLOW":  self.on_allow,
            }
            self._call_hook(hook_map.get(decision), result)
            results.append(result)

        return results

    def check_output(self, llm_output: str) -> dict:
        """Screen an LLM-generated response for leaked secrets, system
        prompt echoes, or harmful content before returning it to the user.

        Mirrors check() in pipeline shape and response format: same 7-key
        structured response, same 0.0-1.0 signal accumulation, same
        ALLOW/FLAG/REVIEW/BLOCK decision bands. Uses the output_filter
        detector instead of the three input-side detectors.

        Intended usage pattern::

            gate = PromptGate()

            # Input side — screen before sending to LLM
            input_result = gate.check(user_message)
            if input_result["decision"] != "ALLOW":
                return input_result["message"]

            llm_response = call_your_llm(user_message)

            # Output side — screen before returning to user
            output_result = gate.check_output(llm_response)
            if output_result["decision"] != "ALLOW":
                return "Response withheld — sensitive content detected."

            return llm_response

        Args:
            llm_output: Raw text produced by the protected LLM.

        Returns:
            Same 7-key structured response dict as check():
            decision, confidence, risk_level, threat_categories,
            signals, signals_checked, message.
        """
        if not isinstance(llm_output, str):
            llm_output = str(llm_output)

        signals = self.output_filter.detect(llm_output)
        aggregated = aggregate(signals)
        risk_score = score(aggregated["signals"])
        decision = evaluate(risk_score, self.thresholds)

        output_checked = (
            "output_filter: "
            + (f"{len(signals)} signal(s) detected" if signals else "no leaks detected")
        )

        result = build_response(
            decision=decision,
            risk_score=risk_score,
            threat_categories=aggregated["threat_categories"],
            signals=aggregated["signals"],
            signals_checked=[output_checked],
        )

        self._log_decision(llm_output, result)

        hook_map = {
            "BLOCK":  self.on_block,
            "FLAG":   self.on_flag,
            "REVIEW": self.on_review,
            "ALLOW":  self.on_allow,
        }
        self._call_hook(hook_map.get(decision), result)

        return result

    def sanitize(self, user_input: str) -> dict:
        """Return a sanitized version of the input with known dangerous
        character-level primitives neutralized, alongside the standard
        risk assessment.

        Sanitization scope is STRICTLY LIMITED to character-level attack
        primitives that input_parser already detects:
          - Zero-width / invisible unicode characters: stripped entirely
          - Homoglyph characters (Cyrillic and other lookalike scripts):
            normalized to closest ASCII equivalent via unicodedata

        This does NOT rewrite, paraphrase, or remove injection PHRASES.
        Removing "ignore previous instructions" as a phrase is a detection
        problem handled by check() — not a sanitization problem. Phrase-
        level rewriting risks silently altering legitimate inputs.

        Base64-looking payloads are deliberately NOT decoded or removed —
        the intent is ambiguous (legitimate base64 exists), and decoding
        attacker-controlled data introduces its own risks. They are
        flagged by check() via encoding_obfuscation signals instead.

        Args:
            user_input: Raw user prompt text.

        Returns:
            Dict with exactly three keys:
                sanitized_text  (str)  — input with character primitives
                                         removed/normalized. Equals
                                         user_input if nothing modified.
                modifications   (list) — human-readable list of changes,
                                         e.g. ["stripped 2 zero-width char(s)",
                                               "normalized 3 homoglyph char(s)"].
                                         Empty list if no changes were made.
                original_check  (dict) — same as calling check(user_input)
                                         on the ORIGINAL unsanitized input.
        """
        import re
        import unicodedata

        modifications: list[str] = []
        text = user_input

        # Step 1 — Strip zero-width and invisible unicode characters.
        # These split keywords across invisible boundaries:
        # "ignore\u200bprevious" looks identical to "ignoreprevious" visually
        # but may fool substring matchers that operate on raw codepoints.
        _ZW_PATTERN = re.compile(
            "[\u200b\u200c\u200d\u2060\ufeff\u00ad]"
        )
        zw_matches = _ZW_PATTERN.findall(text)
        if zw_matches:
            text = _ZW_PATTERN.sub("", text)
            modifications.append(f"stripped {len(zw_matches)} zero-width char(s)")

        # Step 2 — Normalize homoglyph characters to ASCII equivalents.
        # Attackers replace Latin chars with visually identical Cyrillic or
        # other script lookalikes: і (Cyrillic, U+0456) instead of i (Latin).
        #
        # NFKD normalization alone is insufficient — Cyrillic homoglyphs are
        # standalone Unicode characters with no ASCII decomposition, so NFKD
        # returns them unchanged. We use an explicit lookup table for the most
        # common visual lookalikes used in injection attacks, then fall back
        # to NFKD for composed characters (e.g., accented Latin letters).
        _HOMOGLYPH_MAP: dict[str, str] = {
            # Lowercase Cyrillic lookalikes
            "\u0430": "a",   # Cyrillic а
            "\u0435": "e",   # Cyrillic е
            "\u0456": "i",   # Cyrillic і (byelorussian-ukrainian i)
            "\u0457": "i",   # Cyrillic ї
            "\u043e": "o",   # Cyrillic о
            "\u0440": "r",   # Cyrillic р
            "\u0441": "c",   # Cyrillic с
            "\u0445": "x",   # Cyrillic х
            "\u0443": "y",   # Cyrillic у
            "\u0455": "s",   # Cyrillic dze
            # Uppercase Cyrillic lookalikes
            "\u0410": "A",   # Cyrillic А
            "\u0412": "B",   # Cyrillic В
            "\u0415": "E",   # Cyrillic Е
            "\u0406": "I",   # Cyrillic І
            "\u041c": "M",   # Cyrillic М
            "\u041d": "H",   # Cyrillic Н
            "\u041e": "O",   # Cyrillic О
            "\u0420": "R",   # Cyrillic Р
            "\u0421": "C",   # Cyrillic С
            "\u0422": "T",   # Cyrillic Т
            "\u0425": "X",   # Cyrillic Х
        }

        normalized_chars: list[str] = []
        homoglyph_count = 0
        for char in text:
            if char in _HOMOGLYPH_MAP:
                # Explicit homoglyph hit
                normalized_chars.append(_HOMOGLYPH_MAP[char])
                homoglyph_count += 1
            elif ord(char) > 127:
                # Try NFKD for composed characters (e.g., accented Latin)
                nfkd = unicodedata.normalize("NFKD", char)
                ascii_equiv = nfkd.encode("ascii", errors="ignore").decode("ascii")
                if ascii_equiv:
                    normalized_chars.append(ascii_equiv)
                    homoglyph_count += 1
                else:
                    # Legitimate non-ASCII (e.g., Chinese, Arabic) — keep
                    normalized_chars.append(char)
            else:
                normalized_chars.append(char)

        if homoglyph_count > 0:
            text = "".join(normalized_chars)
            modifications.append(f"normalized {homoglyph_count} homoglyph char(s)")

        return {
            "sanitized_text": text,
            "modifications": modifications,
            "original_check": self.check(user_input),
        }

    async def acheck(
        self,
        user_input: str,
        history: list[dict] | None = None,
    ) -> dict:
        """Async version of check().

        Runs the synchronous detection pipeline in a thread pool executor
        to avoid blocking the event loop during model inference.

        NOTE: PyTorch inference holds the GIL. Concurrent acheck() calls
        on the same instance will serialize during intent classification.
        For high-throughput concurrent workloads, use multiple instances.

        Args:
            user_input: Raw user prompt text.
            history: Optional conversation history (see check() for details).

        Returns:
            Same 7-key result dict as check().
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.check(user_input, history)
        )

    async def acheck_batch(self, inputs: list[str]) -> list[dict]:
        """Async version of check_batch().

        Runs check_batch() in a thread pool executor. Inherits the same
        semantic batching optimisation as check_batch().

        Args:
            inputs: List of raw user prompt strings.

        Returns:
            List of result dicts in the same order as inputs.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.check_batch, inputs)