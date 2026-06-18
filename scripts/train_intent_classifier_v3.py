"""
scripts/train_intent_classifier_v3.py
--------------------------------------
Fine-tune DistilBERT on three merged datasets:

  1. deepset/prompt-injections          529 examples  (real-world injections)
  2. benign coding requests             300 examples  (imperative code, safe tasks)
  3. malicious coding requests          ~150 examples (imperative code, malicious tasks)

WHY THREE DATASETS
------------------
v1 (Phase 4): deepset only
  Problem: model learned "write a function" → INJECTION (spurious correlation)
  Result: "Write a function to sort a list" blocked at 0.99

v2 (Phase 6.5): deepset + benign coding
  Problem: model learned "write a function" → BENIGN (opposite spurious correlation)
  Result: "Write code to exfiltrate secrets" allowed at 0.00

v3 (this script): deepset + benign coding + malicious coding
  Model sees BOTH sides of "write a function that X":
    benign version (sort, reverse, validate) → BENIGN
    malicious version (exfiltrate, bypass, leak) → INJECTION
  Model must learn to distinguish WHAT the code does, not just HOW it's phrased.

EVAL STRATEGY
-------------
Three sources split independently (80/20 each), evaluated separately:
  eval_deepset   → regression check (should match Phase 4: F1 ~0.97+)
  eval_benign    → benign coding still passes (should be ~100% BENIGN)
  eval_malicious → malicious coding now caught (should be ~90%+ INJECTION)

Run:
    python scripts/generate_benign_coding_dataset.py    # if not already done
    python scripts/generate_malicious_coding_dataset.py
    python scripts/train_intent_classifier_v3.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR             = PROJECT_ROOT / "models" / "intent_classifier"
BENIGN_CODING_PATH    = PROJECT_ROOT / "injectionbench" / "datasets" / "benign"  / "coding_requests.json"
MALICIOUS_CODING_PATH = PROJECT_ROOT / "injectionbench" / "datasets" / "attacks" / "malicious_coding.json"


def load_json_dataset(path: Path) -> tuple[list[str], list[int]]:
    data   = json.loads(path.read_text(encoding="utf-8"))
    texts  = [d["text"]  for d in data]
    labels = [d["label"] for d in data]
    return texts, labels


def main() -> None:
    try:
        import torch
        import numpy as np
        from datasets import load_dataset
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        sys.exit(1)

    print("=" * 60)
    print("PromptGate — Intent Classifier Training v3")
    print("deepset + benign coding + malicious coding")
    print("=" * 60)

    # ── 1. deepset/prompt-injections ─────────────────────────────────────────
    print("\n[1/7] Loading deepset/prompt-injections...")
    ds = load_dataset("deepset/prompt-injections")
    ds_texts  = list(ds["train"]["text"])  + list(ds["test"]["text"])
    ds_labels = list(ds["train"]["label"]) + list(ds["test"]["label"])
    print(f"      {len(ds_texts)} samples  |  injections: {sum(ds_labels)}  |  benign: {len(ds_labels)-sum(ds_labels)}")

    tr_ds_t, ev_ds_t, tr_ds_l, ev_ds_l = train_test_split(
        ds_texts, ds_labels, test_size=0.20, random_state=42, stratify=ds_labels
    )

    # ── 2. Benign coding requests ─────────────────────────────────────────────
    print("\n[2/7] Loading benign coding-request dataset...")
    if not BENIGN_CODING_PATH.is_file():
        print(f"      NOT FOUND: {BENIGN_CODING_PATH}")
        print("      Run scripts/generate_benign_coding_dataset.py first.")
        sys.exit(1)
    bc_texts, bc_labels = load_json_dataset(BENIGN_CODING_PATH)
    print(f"      {len(bc_texts)} samples  |  all label=0 (benign)")
    tr_bc_t, ev_bc_t, tr_bc_l, ev_bc_l = train_test_split(
        bc_texts, bc_labels, test_size=0.20, random_state=42
    )

    # ── 3. Malicious coding requests ──────────────────────────────────────────
    print("\n[3/7] Loading malicious coding-request dataset...")
    if not MALICIOUS_CODING_PATH.is_file():
        print(f"      NOT FOUND: {MALICIOUS_CODING_PATH}")
        print("      Run scripts/generate_malicious_coding_dataset.py first.")
        sys.exit(1)
    mc_texts, mc_labels = load_json_dataset(MALICIOUS_CODING_PATH)
    print(f"      {len(mc_texts)} samples  |  all label=1 (injection)")
    tr_mc_t, ev_mc_t, tr_mc_l, ev_mc_l = train_test_split(
        mc_texts, mc_labels, test_size=0.20, random_state=42
    )

    # ── Combine ───────────────────────────────────────────────────────────────
    train_texts  = tr_ds_t + tr_bc_t + tr_mc_t
    train_labels = tr_ds_l + tr_bc_l + tr_mc_l
    eval_texts   = ev_ds_t + ev_bc_t + ev_mc_t
    eval_labels  = ev_ds_l + ev_bc_l + ev_mc_l

    print(f"\n      Combined train: {len(train_texts)}")
    print(f"      Combined eval:  {len(eval_texts)}")
    print(f"      Train injections: {sum(train_labels)} | benign: {len(train_labels)-sum(train_labels)}")

    # ── 4. Tokenise ───────────────────────────────────────────────────────────
    print("\n[4/7] Tokenising...")
    BASE_MODEL = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenise(texts):
        return tokenizer(texts, truncation=True, padding=True,
                         max_length=128, return_tensors="pt")

    class InjectionDataset(torch.utils.data.Dataset):
        def __init__(self, enc, labels):
            self.enc    = enc
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_ds = InjectionDataset(tokenise(train_texts), train_labels)
    eval_ds  = InjectionDataset(tokenise(eval_texts),  eval_labels)

    # ── 5. Model ──────────────────────────────────────────────────────────────
    print(f"\n[5/7] Loading base model: {BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2,
        id2label={0: "BENIGN", 1: "INJECTION"},
        label2id={"BENIGN": 0, "INJECTION": 1},
    )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("\n[6/7] Fine-tuning...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        r = classification_report(labels, preds, labels=[0,1],
                                   target_names=["BENIGN","INJECTION"],
                                   output_dict=True, zero_division=0)
        return {"f1_injection": r["INJECTION"]["f1-score"],
                "f1_macro":     r["macro avg"]["f1-score"],
                "accuracy":     r["accuracy"]}

    import inspect
    _params = inspect.signature(TrainingArguments.__init__).parameters
    _eval_key = "eval_strategy" if "eval_strategy" in _params else "evaluation_strategy"

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=50,
        weight_decay=0.01,
        **{_eval_key: "epoch"},
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_injection",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    # ── 7. Evaluation — three separate reports ────────────────────────────────
    print("\n[7/7] Evaluation on held-out splits:")

    def eval_subset(texts, labels, title):
        if not texts:
            return
        ds = InjectionDataset(tokenise(texts), labels)
        out = trainer.predict(ds)
        preds = np.argmax(out.predictions, axis=-1)
        print(f"\n--- {title} ---")
        print(classification_report(labels, preds, labels=[0,1],
                                     target_names=["BENIGN","INJECTION"],
                                     zero_division=0))
        if all(l == 0 for l in labels):
            n = sum(1 for p in preds if p == 0)
            print(f"  {n}/{len(preds)} correctly BENIGN ({100*n/len(preds):.1f}%)")
            if n < len(preds):
                for t, p in zip(texts, preds):
                    if p != 0:
                        print(f"    STILL BLOCKED: {t!r}")
        elif all(l == 1 for l in labels):
            n = sum(1 for p in preds if p == 1)
            print(f"  {n}/{len(preds)} correctly INJECTION ({100*n/len(preds):.1f}%)")
            if n < len(preds):
                for t, p in zip(texts, preds):
                    if p != 1:
                        print(f"    STILL ALLOWED: {t!r}")

    eval_subset(ev_ds_t, ev_ds_l, "deepset eval (regression check)")
    eval_subset(ev_bc_t, ev_bc_l, "benign coding eval (FP check)")
    eval_subset(ev_mc_t, ev_mc_l, "malicious coding eval (fix verification)")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving model to: {MODEL_DIR}")
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print("\nDone.")
    print("Next:")
    print("  python test_regression.py")
    print("  python test_malicious_coding.py")
    print("  python -m pytest tests/ -q")


if __name__ == "__main__":
    main()