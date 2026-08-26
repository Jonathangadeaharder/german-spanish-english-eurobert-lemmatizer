"""Error analysis for zh UPOS tagger — builds confusion matrix.

Identifies which UPOS tags are most confused and prints example errors.
Used to guide targeted data augmentation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from lemmatizer.data.conllu import read_conllu
from lemmatizer.eval.zh_char_utils import (
    MAX_LENGTH,
    _predict_sentence_chars,
    label_to_upos,
)

GOLD_TEST = "data/gold/zh/test.conllu"
GOLD_TRAIN = "data/gold/zh/train.conllu"
BERT_PATH = "models/bert-base-chinese-mlx"
ZH_BIO_CHECKPOINT = "runs/mlx-zh-bio-pos/best.safetensors"
BIO_LABELS_PATH = "data/processed/zh_bio/labels.json"


def load_label_maps():
    meta = json.loads(Path(BIO_LABELS_PATH).read_text(encoding="utf-8"))
    label2id = meta["label2id"]
    id2label = {int(v): k for k, v in label2id.items()}
    return label2id, id2label


def _score_words(
    gold_words: list[str],
    gold_upos: list[str],
    word_start_offsets: list[int],
    n_chars: int,
    char_label: list[int | None],
    id2label: dict[int, str],
    sent_idx: int,
    confusion: dict[str, Counter],
    errors_by_tag: dict[str, list[dict]],
    upos_total: Counter,
    upos_correct: Counter,
) -> None:
    """Score predicted UPOS for each gold word against the gold tag."""
    for i, (word, gold_pos) in enumerate(zip(gold_words, gold_upos, strict=True)):
        offset = word_start_offsets[i] if i < len(word_start_offsets) else 0
        # Skip words beyond truncation boundary.
        if offset >= n_chars:
            continue
        if char_label[offset] is not None:
            raw_label = id2label.get(char_label[offset], "O")
            pred_pos = label_to_upos(raw_label)
        else:
            pred_pos = "X"

        upos_total[gold_pos] += 1
        confusion[gold_pos][pred_pos] += 1
        if pred_pos == gold_pos:
            upos_correct[gold_pos] += 1
        elif len(errors_by_tag[gold_pos]) < 10:
            errors_by_tag[gold_pos].append(
                {
                    "word": word,
                    "gold": gold_pos,
                    "pred": pred_pos,
                    "sent_idx": sent_idx,
                }
            )


def _process_sentence(
    sent_idx: int,
    sent: dict,
    tokenizer,
    model,
    id2label: dict[int, str],
    confusion: dict[str, Counter],
    errors_by_tag: dict[str, list[dict]],
    upos_total: Counter,
    upos_correct: Counter,
) -> None:
    """Tokenize, predict, and score one sentence."""
    gold_words = sent["words"]
    gold_upos = sent["upos"]

    char_label, word_start_offsets = _predict_sentence_chars(
        tokenizer, model, gold_words, sent_idx
    )
    n_chars = min(sum(len(w) for w in gold_words), MAX_LENGTH - 2)

    _score_words(
        gold_words, gold_upos, word_start_offsets, n_chars,
        char_label, id2label, sent_idx,
        confusion, errors_by_tag, upos_total, upos_correct,
    )


def _print_confusion_report(
    confusion: dict[str, Counter],
    errors_by_tag: dict[str, list[dict]],
    upos_total: Counter,
    upos_correct: Counter,
) -> None:
    """Print per-tag accuracy, top confusions, and example errors."""
    print(f"\n{'Tag':8s} {'Total':>6s} {'Correct':>7s} {'Acc':>6s} {'Top confusion':>30s}")
    print("-" * 70)
    for tag in sorted(upos_total.keys(), key=lambda t: -upos_total[t]):
        total = upos_total[tag]
        correct = upos_correct[tag]
        acc = correct / max(total, 1)
        # Top confusion (excluding self)
        top_conf = confusion[tag].most_common(3)
        conf_str = ", ".join(f"{p}({c})" for p, c in top_conf if p != tag)
        print(f"{tag:8s} {total:6d} {correct:7d} {acc:6.2f} {conf_str:>30s}")

    # Print example errors for worst tags
    print(f"\n{'=' * 60}")
    print("Example errors for worst-performing tags:")
    print(f"{'=' * 60}")
    worst = sorted(
        upos_total.keys(),
        key=lambda t: upos_correct[t] / max(upos_total[t], 1),
    )
    for tag in worst[:5]:
        acc = upos_correct[tag] / max(upos_total[tag], 1)
        if acc >= 0.9:
            break
        print(f"\n--- {tag} (acc={acc:.2f}, {upos_correct[tag]}/{upos_total[tag]}) ---")
        for err in errors_by_tag[tag][:5]:
            print(
                f"  {err['word']:8s} gold={err['gold']:6s} "
                f"pred={err['pred']:6s} (sent {err['sent_idx']})"
            )


def _print_training_distribution() -> None:
    """Print UPOS distribution in the training data."""
    train_sentences = read_conllu(GOLD_TRAIN, lang="zh")
    train_upos = Counter()
    for sent in train_sentences:
        for pos in sent["upos"]:
            train_upos[pos] += 1
    print(f"\n{'=' * 60}")
    print("Training data UPOS distribution:")
    print(f"{'=' * 60}")
    for tag in sorted(train_upos.keys(), key=lambda t: -train_upos[t]):
        print(f"  {tag:8s} {train_upos[tag]:6d}")


def run() -> None:
    _, id2label = load_label_maps()
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)

    from lemmatizer.train.zh_bio import _load_bert_model

    model = _load_bert_model(BERT_PATH)
    model.load_weights(ZH_BIO_CHECKPOINT, strict=False)
    model.eval()

    sentences = read_conllu(GOLD_TEST, lang="zh")
    print(f"Gold test: {len(sentences)} sentences")

    # Confusion matrix: confusion[gold][pred] = count
    confusion: dict[str, Counter] = defaultdict(Counter)
    errors_by_tag: dict[str, list[dict]] = defaultdict(list)
    upos_total = Counter()
    upos_correct = Counter()

    for sent_idx, sent in enumerate(sentences):
        _process_sentence(
            sent_idx, sent, tokenizer, model, id2label,
            confusion, errors_by_tag, upos_total, upos_correct,
        )

    _print_confusion_report(confusion, errors_by_tag, upos_total, upos_correct)
    _print_training_distribution()


if __name__ == "__main__":
    run()
