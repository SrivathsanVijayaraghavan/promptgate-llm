"""
Phase 8, Part 1 — Diff which deepset/prompt-injections eval examples
regressed between the v2 and v3 intent classifier checkpoints.

v2: trained on deepset (529) + benign coding (300)
v3: trained on deepset (529) + benign coding (300) + malicious coding (150)

Answers: which SPECIFIC examples did v2 get right that v3 gets wrong
(and vice versa)? Aggregate F1 (0.99 -> 0.95) cannot answer this alone.

REQUIRES the exact same train/eval split that train_intent_classifier_v2.py
and v3.py used (seed=42, stratified, test_size=0.20) — otherwise this diff
compares different eval sets and is meaningless.

Usage:
    python scripts/diff_model_regressions.py

Before running, you MUST have BOTH checkpoints available locally:
    models/intent_classifier_v2/   <- see recovery note below
    models/intent_classifier/      <- current v3, already present

RECOVERING v2 WEIGHTS
----------------------
v2 was overwritten in-place by v3 (Mistake 5). It may only be recoverable
from the HF Hub model repo's git history, IF a commit/tag exists between
the v2 push and the v3 push. Check this FIRST:

    from huggingface_hub import HfApi
    api = HfApi()
    commits = api.list_repo_commits("srivathsan-vijayaraghavan/promptgate-intent-classifier")
    for c in commits:
        print(c.commit_id, c.created_at, c.title)

Find the commit whose message matches the v2 push (search for the
exact commit message used when v2 was uploaded — check your shell
history / earlier conversation logs for the upload_folder() call's
commit_message argument). Do NOT assume the most recent non-v3 commit
is v2 without checking the title/timestamp — verify it explicitly.

Then:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="srivathsan-vijayaraghavan/promptgate-intent-classifier",
        revision="<verified v2 commit SHA>",
        local_dir="models/intent_classifier_v2",
    )

If no v2 commit exists in the Hub history (e.g. the v2 push and v3 push
were somehow combined, or history was force-pushed), v2 is NOT
recoverable. In that case this diff cannot be run retroactively — report
that back rather than approximating with a retrain, since a fresh
"v2-like" retrain would not be bit-for-bit identical to the original v2
checkpoint (different random initialization unless seeds are controlled
end-to-end, including dataloader shuffling).
"""

import json
from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

V2_PATH = Path("models/intent_classifier_v2")
V3_PATH = Path("models/intent_classifier")
OUT_PATH = Path("results/model_regression_diff.json")


def load_model(path: Path):
    if not path.is_dir():
        raise FileNotFoundError(
            f"Model not found at {path}. See recovery instructions in this "
            f"script's module docstring before running."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model) -> tuple[int, float]:
    """Returns (predicted_label, injection_probability)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    injection_prob = probs[1].item()  # label 1 = INJECTION, per id2label in config.json
    pred = 1 if injection_prob >= 0.5 else 0
    return pred, injection_prob


def load_deepset_eval_split() -> list[dict]:
    """Reproduce the EXACT eval split used by train_intent_classifier_v2.py
    and v3.py: load deepset/prompt-injections, concatenate train+test,
    stratified 80/20 split with random_state=42. Returns only the eval
    (20%) half — this must match what both v2 and v3 were evaluated on
    during their original training runs.
    """
    ds = load_dataset("deepset/prompt-injections")
    texts  = list(ds["train"]["text"])  + list(ds["test"]["text"])
    labels = list(ds["train"]["label"]) + list(ds["test"]["label"])

    _, eval_texts, _, eval_labels = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    return [{"text": t, "true_label": l} for t, l in zip(eval_texts, eval_labels)]


def main() -> None:
    print("Loading v2 checkpoint...")
    tok_v2, model_v2 = load_model(V2_PATH)
    print("Loading v3 checkpoint...")
    tok_v3, model_v3 = load_model(V3_PATH)

    print("Reproducing deepset eval split (seed=42, same as training)...")
    eval_set = load_deepset_eval_split()
    print(f"Eval set size: {len(eval_set)}")

    flipped_wrong:  list[dict] = []   # v2 correct -> v3 incorrect (regression)
    flipped_right:  list[dict] = []   # v2 incorrect -> v3 correct (improvement)
    both_wrong:     list[dict] = []
    both_correct_count = 0

    print("Running both models on every eval example...")
    for ex in eval_set:
        true_label = ex["true_label"]
        pred_v2, prob_v2 = predict(ex["text"], tok_v2, model_v2)
        pred_v3, prob_v3 = predict(ex["text"], tok_v3, model_v3)

        v2_correct = pred_v2 == true_label
        v3_correct = pred_v3 == true_label

        record = {
            "text": ex["text"],
            "true_label": true_label,
            "v2_pred": pred_v2, "v2_prob": round(prob_v2, 4),
            "v3_pred": pred_v3, "v3_prob": round(prob_v3, 4),
        }

        if v2_correct and not v3_correct:
            flipped_wrong.append(record)
        elif not v2_correct and v3_correct:
            flipped_right.append(record)
        elif not v2_correct and not v3_correct:
            both_wrong.append(record)
        else:
            both_correct_count += 1

    print()
    print("=" * 60)
    print(f"Both correct:                       {both_correct_count}")
    print(f"v2 correct, v3 WRONG (regressions):  {len(flipped_wrong)}")
    print(f"v2 wrong, v3 correct (improvements): {len(flipped_right)}")
    print(f"Both wrong:                          {len(both_wrong)}")
    print("=" * 60)

    print("\n=== REGRESSIONS (v2 right, v3 wrong) — REVIEW EACH ===")
    if not flipped_wrong:
        print("  (none)")
    for r in flipped_wrong:
        kind = "ATTACK" if r["true_label"] == 1 else "benign"
        print(f"  [{kind}] {r['text'][:80]!r}")
        print(f"    v2: pred={r['v2_pred']} prob={r['v2_prob']:.3f} (correct)")
        print(f"    v3: pred={r['v3_pred']} prob={r['v3_prob']:.3f} (WRONG)")

    print("\n=== IMPROVEMENTS (v2 wrong, v3 right) ===")
    if not flipped_right:
        print("  (none)")
    for r in flipped_right:
        kind = "ATTACK" if r["true_label"] == 1 else "benign"
        print(f"  [{kind}] {r['text'][:80]!r}  v2_prob={r['v2_prob']:.3f} -> v3_prob={r['v3_prob']:.3f}")

    # Flag the highest-priority items explicitly: missed real attacks
    missed_attacks = [r for r in flipped_wrong if r["true_label"] == 1]
    if missed_attacks:
        print(f"\n*** {len(missed_attacks)} REAL ATTACKS regressed from caught to missed. ***")
        print("*** Review each one below before deciding whether this matters. ***")
        for r in missed_attacks:
            distance_from_threshold = abs(r["v3_prob"] - 0.70)
            calibration_note = (
                "CLOSE to threshold (likely calibration issue)"
                if distance_from_threshold < 0.10
                else "FAR from threshold (signal likely genuinely lost)"
            )
            print(f"\n  TEXT: {r['text']!r}")
            print(f"  v3 injection probability: {r['v3_prob']:.3f}  (threshold=0.70)")
            print(f"  {calibration_note}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "regressions": flipped_wrong,
        "improvements": flipped_right,
        "both_wrong": both_wrong,
        "summary": {
            "eval_set_size": len(eval_set),
            "both_correct": both_correct_count,
            "regressions": len(flipped_wrong),
            "improvements": len(flipped_right),
            "both_wrong": len(both_wrong),
            "missed_real_attacks": len(missed_attacks),
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull diff saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()