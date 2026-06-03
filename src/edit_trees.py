def longest_common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def longest_common_suffix(a: str, b: str, prefix_len: int) -> int:
    max_len = min(len(a), len(b)) - prefix_len
    i = 0
    while i < max_len and a[len(a) - 1 - i] == b[len(b) - 1 - i]:
        i += 1
    return i


def make_edit_label(word: str, lemma: str) -> str:
    if word == lemma:
        return "IDENTITY"

    lower = word.lower()
    if lower == lemma:
        return "LOWERCASE"

    prefix_len = longest_common_prefix(word, lemma)
    suffix_len = longest_common_suffix(word, lemma, prefix_len)

    word_mid_start = prefix_len
    word_mid_end = len(word) - suffix_len if suffix_len > 0 else len(word)

    lemma_mid_start = prefix_len
    lemma_mid_end = len(lemma) - suffix_len if suffix_len > 0 else len(lemma)

    delete_mid = word[word_mid_start:word_mid_end]
    insert_mid = lemma[lemma_mid_start:lemma_mid_end]

    return f"P{prefix_len}|S{suffix_len}|D{delete_mid}|I{insert_mid}"


def apply_edit_label(word: str, label: str):
    if label == "IDENTITY":
        return word

    if label == "LOWERCASE":
        return word.lower()

    if not label.startswith("P"):
        return None

    try:
        parts = label.split("|")
        prefix_len = int(parts[0][1:])
        suffix_len = int(parts[1][1:])
        delete_mid = parts[2][1:]
        insert_mid = parts[3][1:]
    except Exception:
        return None

    if prefix_len + suffix_len > len(word):
        return None

    mid_start = prefix_len
    mid_end = len(word) - suffix_len if suffix_len > 0 else len(word)

    actual_delete = word[mid_start:mid_end]

    if actual_delete != delete_mid:
        return None

    prefix = word[:prefix_len]
    suffix = word[len(word) - suffix_len :] if suffix_len > 0 else ""

    return prefix + insert_mid + suffix
