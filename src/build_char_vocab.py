from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from conllu_reader import read_conllu

LANGS = ["de", "es", "en"]

ALL_FILES = {
    "de": [
        "data/gold/de/train.conllu",
        "data/gold/de/dev.conllu",
        "data/gold/de/test.conllu",
    ],
    "es": [
        "data/gold/es/train.conllu",
        "data/gold/es/dev.conllu",
        "data/gold/es/test.conllu",
    ],
    "en": [
        "data/gold/en/train.conllu",
        "data/gold/en/dev.conllu",
        "data/gold/en/test.conllu",
    ],
}

OUT_DIR = Path("artifacts")
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
MAX_LEMMA_LENGTH = 32


def main():
    char_counter: Counter[str] = Counter()

    for lang in LANGS:
        for path in ALL_FILES[lang]:
            sentences = read_conllu(path, lang=lang)
            for sent in sentences:
                for word, lemma in zip(sent["words"], sent["lemmas"], strict=True):
                    for ch in word:
                        char_counter[ch] += 1
                    for ch in lemma:
                        char_counter[ch] += 1

    vocab = list(SPECIAL_TOKENS)
    for ch, _count in char_counter.most_common():
        if ch not in vocab:
            vocab.append(ch)

    char2id = {ch: i for i, ch in enumerate(vocab)}
    id2char = {str(i): ch for ch, i in char2id.items()}

    out = {
        "char2id": char2id,
        "id2char": id2char,
        "special_tokens": SPECIAL_TOKENS,
        "vocab_size": len(vocab),
        "max_lemma_length": MAX_LEMMA_LENGTH,
    }

    (OUT_DIR / "char_vocab.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Character vocabulary: {len(vocab)} tokens")
    print(f"  Special: {SPECIAL_TOKENS}")
    print(f"  Data chars: {len(vocab) - len(SPECIAL_TOKENS)}")
    print(f"  Max lemma length: {MAX_LEMMA_LENGTH}")


if __name__ == "__main__":
    main()
