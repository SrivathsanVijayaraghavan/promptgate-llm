"""
promptgate/detector/output_filter.py
--------------------------------------
Output-side detection layer — screens LLM-generated responses before
they reach the end user.

Architecture decision: separate regex-only detector for this layer
rather than extending RuleBasedDetector. Reasons:
  1. Secret-leak patterns (API keys, tokens) are inherently regex-shaped
     — there is no substring use case to preserve.
  2. Extending RuleBasedDetector to support mixed substring/regex patterns
     would touch code on which 186 existing tests depend, for a feature
     only the new output layer needs.
  3. A separate, smaller detector is easier to reason about and test
     independently.

Mirrors the input detection pattern: rule-based + optional semantic,
producing the same {"signal", "severity", "matched", "category"} signal
structure as input detectors. Aggregator, scorer, and policy are SHARED
with input-side — output risk uses the identical 0.0-1.0 accumulation
model and ALLOW/FLAG/REVIEW/BLOCK decision bands as input risk.

Usage (via gate.py):
    result = gate.check_output(llm_response)
"""

import re
import warnings
from pathlib import Path
from typing import Any

_IMPORT_ERROR: Exception | None = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]
    import numpy as np  # type: ignore[import]
except ImportError as exc:
    _IMPORT_ERROR = exc
    SentenceTransformer = None  # type: ignore[assignment, misc]
    cosine_similarity = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

import json

from promptgate.config import SIGNAL_SEVERITIES, SIGNAL_TO_CATEGORY

# Resolve paths relative to the package — works after pip install
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PATTERNS_FILE = _PACKAGE_ROOT / "data" / "patterns" / "output_leaks.txt"
_LEAKS_PATH = _PACKAGE_ROOT / "data" / "embeddings" / "known_leaks.json"

_MODEL_NAME = "all-MiniLM-L6-v2"
_SEMANTIC_THRESHOLD = 0.70  # Slightly higher than input semantic (0.65)
                             # — output scanning has higher FP risk on
                             # legitimate verbose responses


def _load_output_patterns(patterns_file: Path) -> dict[str, list[re.Pattern]]:
    """Parse output_leaks.txt into {signal_name: [compiled_regex, ...]}."""
    if not patterns_file.is_file():
        warnings.warn(
            f"OutputFilter: patterns file not found at {patterns_file}. "
            "Output rule detection disabled.",
            RuntimeWarning,
            stacklevel=3,
        )
        return {}

    patterns: dict[str, list[re.Pattern]] = {}
    current_signal: str | None = None

    for line in patterns_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            if line.startswith("# signal:"):
                current_signal = line.split("# signal:", 1)[1].strip()
                patterns.setdefault(current_signal, [])
            continue
        if current_signal is None:
            continue
        try:
            compiled = re.compile(line, re.IGNORECASE | re.MULTILINE)
            patterns[current_signal].append(compiled)
        except re.error as exc:
            warnings.warn(
                f"OutputFilter: invalid regex in {patterns_file.name}: "
                f"{line!r} — {exc}. Skipping.",
                RuntimeWarning,
                stacklevel=3,
            )

    return patterns


class OutputFilter:
    """Detect sensitive or policy-violating content in LLM-generated output.

    Two detection sub-layers:
      1. Regex rule-based — high-precision patterns for secrets (API keys,
         tokens, credentials) and structural markers (system prompt echoes,
         harmful instruction templates). Runs on raw output text.
      2. Semantic similarity — optional sentence-transformer layer comparing
         output against known_leaks.json examples. Catches paraphrased or
         partial leaks that don't match regex patterns exactly.

    Both sub-layers produce the same signal dict structure as input
    detectors, enabling the shared aggregator/scorer/policy to treat input
    and output risks identically.
    """

    def __init__(
        self,
        semantic_threshold: float = _SEMANTIC_THRESHOLD,
    ) -> None:
        self._threshold = semantic_threshold
        self._patterns = _load_output_patterns(_PATTERNS_FILE)
        self._semantic_available = False
        self._model: Any = None
        self._leak_embeddings: Any = None
        self._leak_metadata: list[dict] = []

        if SentenceTransformer is None:
            return  # Degrade gracefully

        try:
            self._model = SentenceTransformer(_MODEL_NAME)
            self._load_known_leaks()
            self._semantic_available = True
        except Exception as exc:
            warnings.warn(
                f"OutputFilter: semantic layer unavailable: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _load_known_leaks(self) -> None:
        """Pre-compute embeddings for known_leaks.json examples."""
        if not _LEAKS_PATH.is_file():
            warnings.warn(
                f"OutputFilter: known_leaks.json not found at {_LEAKS_PATH}. "
                "Semantic output detection disabled.",
                RuntimeWarning,
                stacklevel=3,
            )
            return

        leaks = json.loads(_LEAKS_PATH.read_text(encoding="utf-8"))
        texts = [entry["text"] for entry in leaks]
        self._leak_metadata = leaks
        self._leak_embeddings = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )

    def detect(self, output_text: str) -> list[dict[str, Any]]:
        """Run regex + semantic detection on LLM-generated text.

        Args:
            output_text: Raw text produced by the protected LLM.

        Returns:
            List of signal dicts: [{"signal", "severity", "matched", "category"}]
            Empty list if no signals fired or all layers unavailable.
        """
        if not output_text or not output_text.strip():
            return []

        signals: list[dict[str, Any]] = []
        fired_types: set[str] = set()

        # Layer 1 — Regex rule-based
        for signal_name, compiled_patterns in self._patterns.items():
            if signal_name in fired_types:
                continue
            for pattern in compiled_patterns:
                match = pattern.search(output_text)
                if match:
                    matched_text = match.group(0)
                    # Truncate matched credential values for safety —
                    # don't echo the full secret in the signal's matched field
                    if signal_name == "secret_leak" and len(matched_text) > 40:
                        matched_text = matched_text[:20] + "...[truncated]"
                    signals.append({
                        "signal": signal_name,
                        "severity": SIGNAL_SEVERITIES.get(signal_name, 0.80),
                        "matched": matched_text,
                        "category": SIGNAL_TO_CATEGORY.get(
                            signal_name, "output_" + signal_name
                        ),
                    })
                    fired_types.add(signal_name)
                    break  # One signal per type, deduplication handled here

        # Layer 2 — Semantic similarity (optional)
        if (
            self._semantic_available
            and self._leak_embeddings is not None
            and len(self._leak_metadata) > 0
        ):
            try:
                output_emb = self._model.encode(
                    [output_text], convert_to_numpy=True, show_progress_bar=False
                )
                sims = cosine_similarity(output_emb, self._leak_embeddings)[0]

                # Group by category, find best per category
                best_per_cat: dict[str, tuple[float, str]] = {}
                for idx, sim in enumerate(sims):
                    cat = self._leak_metadata[idx].get("category", "unknown")
                    if sim > best_per_cat.get(cat, (0.0, ""))[0]:
                        best_per_cat[cat] = (sim, self._leak_metadata[idx]["text"])

                for cat, (best_sim, best_text) in best_per_cat.items():
                    if best_sim >= self._threshold:
                        signal_name = "semantic_output_similarity"
                        if signal_name not in fired_types:
                            signals.append({
                                "signal": signal_name,
                                "severity": SIGNAL_SEVERITIES.get(
                                    signal_name, 0.60
                                ),
                                "matched": (
                                    f"similar to known leak: "
                                    f"{best_text[:60]!r} ({best_sim:.2f})"
                                ),
                                "category": cat,
                            })
                            fired_types.add(signal_name)
            except Exception:
                pass  # Semantic layer failure never crashes output check

        return signals

    def is_available(self) -> bool:
        """True if at least the regex layer has loaded patterns."""
        return bool(self._patterns)