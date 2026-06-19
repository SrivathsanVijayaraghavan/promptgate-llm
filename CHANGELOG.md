# Changelog

## [0.4.2] - 2025-06-19

### Fixed
- Intent classifier falsely blocking benign coding requests at 0.93-0.99 confidence ("Write a function to reverse a string in Python" -> BLOCK)
- Intent classifier falsely allowing malicious coding requests after one-sided counter-example fix ("Write code that exfiltrates config files" -> ALLOW)
- Retrained on balanced 3-source dataset: deepset (529) + benign coding (300) + malicious coding (150) = 979 examples
- Promoted test_regression.py and test_malicious_coding.py into tests/ suite (186 tests total)
- Fixed placeholder Homepage URL in pyproject.toml

### Added
- scripts/generate_benign_coding_dataset.py
- scripts/generate_malicious_coding_dataset.py
- scripts/train_intent_classifier_v3.py
- injectionbench/datasets/benign/coding_requests.json (300 examples)
- injectionbench/datasets/attacks/malicious_coding.json (150 examples)
- tests/test_regression.py (10 tests)
- tests/test_malicious_coding.py (12 tests)

## [0.4.0] - 2025-06-17

### Added
- check_batch() -- semantic layer batches all inputs in one model.encode() call
- acheck() and acheck_batch() -- async support via run_in_executor
- history parameter on check()/acheck() -- last 3 turns prepended to intent classifier
- log_mode -- privacy-safe JSONL audit logging (sha256 hash only, raw text never logged)
- Callback hooks: on_block, on_flag, on_review, on_allow, on_error

## [0.3.0] - 2025-06-15

### Fixed
- Data files moved inside package (promptgate/data/, injectionbench/datasets/) -- pip install now ships all patterns and embeddings
- Path resolution fixed: parents[2] -> parents[1] in all detectors
- intent.py model resolution rewritten: 3-tier fallback (local -> cache -> HF Hub auto-download)
- HF Hub repo casing fixed: SrivathsanVijayaraghavan -> srivathsan-vijayaraghavan
- Stale global pip registration removed (old 0.1.0 from Desktop path)

### Added
- MIT LICENSE
- .gitignore (models/, __pycache__/, dist/, results/*.json)
- HuggingFace Hub model hosting (auto-download on first use, ~267MB)
- Package renamed to promptgate-llm (original name taken on PyPI)

## [0.2.0] - 2025-06-12

### Added
- SemanticDetector: sentence-transformers/all-MiniLM-L6-v2, 77 known attack embeddings
- 12-word sliding window with 4-word overlap for long input handling
- InjectionBench benchmarking framework (dataset loader, mutator, runner, scorer, reporter)
- CLI: python -m injectionbench run --source huggingface|manual|combined
- Fine-tuned DistilBERT intent classifier (F1 INJECTION 0.97, accuracy 0.98)
- Detection rate: 15.2% (rule+semantic) -> 98.3% (full pipeline)
- signals_checked expanded to 3 entries (rule_based, semantic, intent)

## [0.1.0] - 2025-06-10

### Added
- Three-layer detection pipeline (InputParser, RuleBasedDetector, Aggregator, Scorer, Policy, ResponseBuilder)
- 191 patterns across 5 files (direct_injection, jailbreaks, system_override, social_engineering, encoding_tricks)
- Pattern format: # signal: headers map patterns to signal types
- Response always exactly 7 keys: decision, confidence, risk_level, threat_categories, signals, signals_checked, message
- Signal accumulation scoring: score = min(1.0, sum of severities)
- Policy thresholds: 0.00-0.30 ALLOW, 0.30-0.55 FLAG, 0.55-0.75 REVIEW, 0.75-1.00 BLOCK
- Configurable thresholds per deployment
- Graceful degradation when optional dependencies absent