import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoTokenizer

from edit_trees import apply_edit_label
from multitask_model import EuroBertForUposLemma, EuroBertUposLemmaConfig

MODEL_DIR = "runs/eurobert-multilingual-lemma-210m-lora"
CHAR_MODEL_DIR = "runs/eurobert-multilingual-lemma-210m-char-gen"
DATASET_PATH = "data/processed/eurobert_multilingual_lemma_dataset"
CHAR_DATASET_PATH = "data/processed/eurobert_char_lemma_dataset"
LABEL2ID_PATH = "artifacts/label2id.json"
LABEL2ID_TOP300_PATH = "artifacts/label2id_top300.json"
ID2LABEL_PATH = "artifacts/id2label.json"
ID2LABEL_TOP300_PATH = "artifacts/id2label_top300.json"
UPOS_LABEL2ID_PATH = "artifacts/upos_label2id.json"
UPOS_ID2LABEL_PATH = "artifacts/upos_id2label.json"
LEXICON_PATH = "artifacts/lexicon.json"
CHAR_VOCAB_PATH = "artifacts/char_vocab.json"
REPORT_PATH = Path("artifacts/eval_report.json")
MAX_LENGTH = 256

LANGS = ["de", "es", "en"]
LANG_TOKENS = {
    "de": "[LANG_DE]",
    "es": "[LANG_ES]",
    "en": "[LANG_EN]",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def env_bool(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_candidate_label_ids(id2label):
    candidate_ids = {}

    for lang in LANGS:
        ids = [
            int(label_id)
            for label_id, label in id2label.items()
            if label != "UNKNOWN" and label.startswith(f"{lang}::")
        ]
        candidate_ids[lang] = np.array(sorted(ids), dtype=np.int64)

    return candidate_ids


def load_lexicon(path):
    if not Path(path).exists():
        return {lang: {} for lang in LANGS}

    lexicon = load_json(path)
    return {lang: lexicon.get(lang, {}) for lang in LANGS}


def select_best_label_id(logits_row, candidate_ids):
    candidate_logits = logits_row[candidate_ids]
    best_offset = int(np.argmax(candidate_logits))
    return int(candidate_ids[best_offset])


def collect_word_predictions(
    upos_logits_row,
    lemma_logits_row,
    word_ids,
    candidate_ids_by_lang,
    lang,
    id2label,
    upos_id2label,
):
    raw_labels_by_word = {}
    constrained_ids_by_word = {}
    upos_by_word = {}
    seen_word_ids = set()
    candidate_ids = candidate_ids_by_lang[lang]

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == 0 or word_id in seen_word_ids:
            continue

        seen_word_ids.add(word_id)

        raw_label_id = int(np.argmax(lemma_logits_row[token_idx]))
        raw_labels_by_word[word_id] = id2label.get(str(raw_label_id), "UNKNOWN")

        constrained_ids_by_word[word_id] = select_best_label_id(
            lemma_logits_row[token_idx], candidate_ids
        )
        upos_label_id = int(np.argmax(upos_logits_row[token_idx]))
        upos_by_word[word_id] = upos_id2label.get(str(upos_label_id), "X")

    return raw_labels_by_word, constrained_ids_by_word, upos_by_word


def resolve_prediction(word, upos, base_label, lexicon):
    if upos == "PROPN":
        return None, "propn", False

    if base_label is not None:
        applied = apply_edit_label(word, base_label)

        if applied is not None:
            return applied, "edit", False

        edit_failed = True
    else:
        edit_failed = False

    lexicon_lemma = lexicon.get(word)

    if lexicon_lemma is not None:
        return lexicon_lemma, "lexicon", edit_failed

    return word, "identity", edit_failed


def ensure_upos(words, upos):
    if not upos:
        return ["_"] * len(words)

    if len(upos) < len(words):
        upos = list(upos) + ["_"] * (len(words) - len(upos))

    return upos[: len(words)]


def decode_char_lemma(
    char_generator,
    encoder_hidden,
    encoder_mask,
    word_chars_tensor,
    word_char_mask_tensor,
    char_vocab,
    device,
):
    id2char = {v: k for k, v in char_vocab["char2id"].items()}
    bos_id = char_vocab["char2id"]["<BOS>"]
    eos_id = char_vocab["char2id"]["<EOS>"]
    unk_id = char_vocab["char2id"]["<UNK>"]

    with torch.inference_mode():
        encoder_hidden_single = encoder_hidden.unsqueeze(0)
        encoder_mask_single = encoder_mask.unsqueeze(0) if encoder_mask is not None else None
        word_chars_single = (
            word_chars_tensor.unsqueeze(0) if word_chars_tensor is not None else None
        )
        word_char_mask_single = (
            word_char_mask_tensor.unsqueeze(0) if word_char_mask_tensor is not None else None
        )

        outputs = char_generator(
            encoder_outputs=encoder_hidden_single,
            encoder_mask=encoder_mask_single,
            word_chars=word_chars_single,
            word_char_mask=word_char_mask_single,
        )

    char_logits = outputs["char_logits"][0]
    copy_logits = outputs.get("copy_logits")
    p_gen = outputs.get("p_gen")

    if copy_logits is not None and p_gen is not None and word_chars_tensor is not None:
        vocab_logits = char_logits
        p_gen_val = p_gen[0, :, 0]
        copy_weights = torch.softmax(copy_logits[0], dim=-1)

        for t in range(vocab_logits.shape[0]):
            gen_dist = torch.softmax(vocab_logits[t], dim=0) * p_gen_val[t]
            copy_dist = copy_weights[t] * (1 - p_gen_val[t])

            word_char_ids = word_chars_tensor[t]
            word_char_mask_t = word_char_mask_tensor[t]

            for ci, char_id in enumerate(word_char_ids):
                if word_char_mask_t[ci] == 0 or char_id.item() < 4:
                    continue
                if char_id.item() < vocab_logits.shape[0]:
                    gen_dist[char_id.item()] += copy_dist[ci].item()

            char_logits[t] = gen_dist

    predicted_ids = char_logits.argmax(dim=-1)
    chars = []
    for cid in predicted_ids:
        cid = cid.item()
        if cid == eos_id:
            break
        if cid == bos_id or cid == 0:
            continue
        chars.append(id2char.get(cid, id2char.get(unk_id, "?")))

    return "".join(chars)


def main():
    use_char_gen = env_bool("EVAL_USE_CHAR_GENERATOR", False)

    label2id_path = LABEL2ID_TOP300_PATH if use_char_gen else LABEL2ID_PATH
    id2label_path = ID2LABEL_TOP300_PATH if use_char_gen else ID2LABEL_PATH
    dataset_path = CHAR_DATASET_PATH if use_char_gen else DATASET_PATH

    label2id = load_json(label2id_path)
    id2label = load_json(id2label_path)
    upos_label2id = load_json(UPOS_LABEL2ID_PATH)
    upos_id2label = load_json(UPOS_ID2LABEL_PATH)
    lexicon = load_lexicon(LEXICON_PATH)
    candidate_ids_by_lang = build_candidate_label_ids(id2label)
    model_dir = os.getenv("MODEL_DIR", CHAR_MODEL_DIR if use_char_gen else MODEL_DIR)

    char_vocab = None
    if use_char_gen:
        char_vocab = load_json(CHAR_VOCAB_PATH)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    device = get_device()

    config_kwargs = {
        "base_model_name_or_path": "EuroBERT/EuroBERT-210m",
        "upos_label2id": upos_label2id,
        "lemma_label2id": label2id,
    }

    if use_char_gen:
        config_kwargs["use_char_generator"] = True
        config_kwargs["char_vocab_size"] = char_vocab["vocab_size"]
        config_kwargs["max_lemma_length"] = char_vocab["max_lemma_length"]

    config = EuroBertUposLemmaConfig(**config_kwargs)

    base_model = EuroBertForUposLemma.from_pretrained(
        "EuroBERT/EuroBERT-210m",
        config=config,
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, model_dir)
    model.to(device)
    model.eval()

    dataset = load_from_disk(dataset_path)

    eval_limit = int(os.getenv("EVAL_LIMIT", "0"))
    per_lang_limit = int(os.getenv("EVAL_PER_LANG_LIMIT", "0"))
    batch_size = max(1, int(os.getenv("EVAL_BATCH_SIZE", "8")))

    rows = list(dataset["test"])

    if eval_limit > 0:
        rows = rows[: min(eval_limit, len(rows))]

    if per_lang_limit > 0:
        grouped = {lang: [] for lang in LANGS}

        for row in rows:
            lang = row["lang"]

            if len(grouped[lang]) < per_lang_limit:
                grouped[lang].append(row)

        rows = [row for lang in LANGS for row in grouped[lang]]

    stats = {
        lang: {
            "total": 0,
            "upos_correct": 0,
            "lemma_total": 0,
            "lemma_correct": 0,
            "unknown": 0,
            "raw_wrong_language": 0,
            "constrained_wrong_language": 0,
            "failed_apply": 0,
            "missing_prediction": 0,
            "oov_total": 0,
            "oov_correct": 0,
            "in_vocab_total": 0,
            "in_vocab_correct": 0,
            "char_gen_total": 0,
            "char_gen_correct": 0,
            "edit_tree_total": 0,
            "edit_tree_correct": 0,
            "source_counts": Counter(),
            "pred_upos": Counter(),
            "upos": defaultdict(lambda: {"total": 0, "correct": 0}),
        }
        for lang in LANGS
    }

    unwrapped = model
    if hasattr(model, "base_model"):
        unwrapped = model.base_model.model
    char_gen_module = getattr(unwrapped, "char_generator", None) if use_char_gen else None

    with torch.inference_mode():
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            batch_words = [[LANG_TOKENS[row["lang"]], *row["words"]] for row in batch_rows]

            encoded = tokenizer(
                batch_words,
                is_split_into_words=True,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**model_inputs)
            upos_logits = outputs.logits[0].detach().cpu().numpy()
            lemma_logits = outputs.logits[1].detach().cpu().numpy()
            route_logits = outputs.logits[2].detach().cpu() if len(outputs.logits) > 2 else None

            for batch_index, row in enumerate(batch_rows):
                lang = row["lang"]
                words = row["words"]
                lemmas = row["lemmas"]
                gold_upos = ensure_upos(words, row.get("upos"))
                word_ids = encoded.word_ids(batch_index=batch_index)

                (
                    raw_labels_by_word,
                    constrained_ids_by_word,
                    predicted_upos_by_word,
                ) = collect_word_predictions(
                    upos_logits[batch_index],
                    lemma_logits[batch_index],
                    word_ids,
                    candidate_ids_by_lang,
                    lang,
                    id2label,
                    upos_id2label,
                )

                route_by_word = {}
                if route_logits is not None:
                    seen = set()
                    for token_idx, wid in enumerate(word_ids):
                        if wid is None or wid == 0 or wid in seen:
                            continue
                        seen.add(wid)
                        route_by_word[wid] = route_logits[batch_index, token_idx].item() > 0

                for word_index, (word, gold_lemma) in enumerate(
                    zip(words, lemmas, strict=True), start=1
                ):
                    stats[lang]["total"] += 1

                    upos_tag = gold_upos[word_index - 1] if word_index - 1 < len(gold_upos) else "_"
                    upos_bucket = stats[lang]["upos"][upos_tag]
                    upos_bucket["total"] += 1

                    predicted_upos = predicted_upos_by_word.get(word_index, "X")
                    stats[lang]["pred_upos"][predicted_upos] += 1

                    if predicted_upos == upos_tag:
                        stats[lang]["upos_correct"] += 1
                        upos_bucket["correct"] += 1

                    if upos_tag != "PROPN":
                        stats[lang]["lemma_total"] += 1
                        if word not in lexicon[lang]:
                            stats[lang]["oov_total"] += 1
                        else:
                            stats[lang]["in_vocab_total"] += 1

                        use_char_route = route_by_word.get(word_index, False)

                        if (
                            use_char_route
                            and char_gen_module is not None
                            and char_vocab is not None
                        ):
                            stats[lang]["char_gen_total"] += 1

                            unk_id = char_vocab["char2id"]["<UNK>"]
                            word_char_ids = [char_vocab["char2id"].get(c, unk_id) for c in word]
                            word_char_ids = word_char_ids[: char_vocab.get("max_word_length", 64)]
                            max_wl = char_vocab.get("max_word_length", 64)
                            pad_len = max_wl - len(word_char_ids)
                            wc_tensor = torch.tensor(
                                [word_char_ids + [0] * pad_len],
                                dtype=torch.long,
                                device=device,
                            )
                            wcm_tensor = torch.tensor(
                                [[1] * len(word_char_ids) + [0] * pad_len],
                                dtype=torch.long,
                                device=device,
                            )

                            token_positions = [
                                i for i, wid in enumerate(word_ids) if wid == word_index
                            ]
                            if token_positions:
                                token_idx = token_positions[0]
                                hidden = (
                                    outputs.hidden_states[-1][batch_index, token_idx]
                                    if outputs.hidden_states
                                    else None
                                )
                                if hidden is None:
                                    slice_inputs = {
                                        k: v[batch_index : batch_index + 1]
                                        for k, v in model_inputs.items()
                                    }
                                    backbone_out = model.model.model(
                                        **slice_inputs,
                                        output_hidden_states=True,
                                    )
                                    hidden = backbone_out.hidden_states[-1][0, token_idx]
                            else:
                                hidden = None

                            if hidden is not None:
                                predicted_lemma = decode_char_lemma(
                                    char_gen_module,
                                    hidden,
                                    model_inputs["attention_mask"][batch_index],
                                    wc_tensor[0],
                                    wcm_tensor[0],
                                    char_vocab,
                                    device,
                                )
                                source = "char_gen"
                                failed_apply = False
                            else:
                                predicted_lemma = word
                                source = "char_gen_fail"
                                failed_apply = False
                        elif predicted_upos == "PROPN":
                            predicted_lemma, source, failed_apply = resolve_prediction(
                                word,
                                predicted_upos,
                                None,
                                lexicon[lang],
                            )
                        else:
                            stats[lang]["edit_tree_total"] += 1

                            raw_label = raw_labels_by_word.get(word_index)

                            if raw_label == "UNKNOWN":
                                stats[lang]["unknown"] += 1
                            elif raw_label is not None and not raw_label.startswith(f"{lang}::"):
                                stats[lang]["raw_wrong_language"] += 1

                            constrained_id = constrained_ids_by_word.get(word_index)

                            if constrained_id is None:
                                stats[lang]["missing_prediction"] += 1
                                base_label = None
                            else:
                                constrained_label = id2label.get(str(constrained_id), "UNKNOWN")

                                if constrained_label == "UNKNOWN":
                                    stats[lang]["unknown"] += 1

                                expected_prefix = f"{lang}::"
                                if not constrained_label.startswith(expected_prefix):
                                    stats[lang]["constrained_wrong_language"] += 1
                                    base_label = None
                                else:
                                    base_label = constrained_label[len(expected_prefix) :]

                            predicted_lemma, source, failed_apply = resolve_prediction(
                                word,
                                predicted_upos,
                                base_label,
                                lexicon[lang],
                            )

                        stats[lang]["source_counts"][source] += 1

                        if failed_apply:
                            stats[lang]["failed_apply"] += 1

                        if predicted_lemma == gold_lemma:
                            stats[lang]["lemma_correct"] += 1

                            if source == "char_gen":
                                stats[lang]["char_gen_correct"] += 1
                            else:
                                stats[lang]["edit_tree_correct"] += 1

                            if word not in lexicon[lang]:
                                stats[lang]["oov_correct"] += 1
                            else:
                                stats[lang]["in_vocab_correct"] += 1

    report = {
        "config": {
            "eval_limit": eval_limit,
            "per_lang_limit": per_lang_limit,
            "batch_size": batch_size,
            "device": str(device),
        },
        "languages": {},
    }

    for lang in LANGS:
        total = stats[lang]["total"] or 1
        lemma_total = stats[lang]["lemma_total"] or 1
        oov_total = stats[lang]["oov_total"] or 1
        in_vocab_total = stats[lang]["in_vocab_total"] or 1

        upos_report = {}

        for upos_tag, bucket in sorted(
            stats[lang]["upos"].items(),
            key=lambda item: (-item[1]["total"], item[0]),
        ):
            upos_total = bucket["total"] or 1
            upos_report[upos_tag] = {
                "total": bucket["total"],
                "accuracy": round(bucket["correct"] / upos_total, 4),
            }

        char_gen_total = stats[lang]["char_gen_total"] or 1
        char_gen_report = {}
        if stats[lang]["char_gen_total"] > 0:
            char_gen_report = {
                "total": stats[lang]["char_gen_total"],
                "accuracy": round(stats[lang]["char_gen_correct"] / char_gen_total, 4),
            }

        edit_tree_total = stats[lang]["edit_tree_total"] or 1

        summary = {
            "lang": lang,
            "total": stats[lang]["total"],
            "upos_accuracy": round(stats[lang]["upos_correct"] / total, 4),
            "lemma_accuracy": round(stats[lang]["lemma_correct"] / lemma_total, 4),
            "lemma_total": stats[lang]["lemma_total"],
            "unknown_rate": round(stats[lang]["unknown"] / lemma_total, 4),
            "raw_wrong_language_rate": round(stats[lang]["raw_wrong_language"] / lemma_total, 4),
            "constrained_wrong_language_rate": round(
                stats[lang]["constrained_wrong_language"] / lemma_total, 4
            ),
            "failed_apply_rate": round(stats[lang]["failed_apply"] / lemma_total, 4),
            "missing_prediction_rate": round(stats[lang]["missing_prediction"] / lemma_total, 4),
            "oov_accuracy": round(stats[lang]["oov_correct"] / oov_total, 4),
            "in_vocab_accuracy": round(stats[lang]["in_vocab_correct"] / in_vocab_total, 4),
            "char_gen": char_gen_report,
            "edit_tree_accuracy": (
                round(stats[lang]["edit_tree_correct"] / edit_tree_total, 4)
                if stats[lang]["edit_tree_total"] > 0
                else None
            ),
            "source_counts": dict(stats[lang]["source_counts"]),
            "upos_by_gold": upos_report,
        }

        report["languages"][lang] = summary
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
