from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from edit_trees import apply_edit_label, make_edit_label
from language_assets import language_assets
from multitask_model import EuroBertForUposLemma, EuroBertUposLemmaConfig
from postprocess_rules import (
    DE_CONTRACTIONS,
    EN_IRREGULAR_PLURALS,
    apply_postprocess_rules,
)

MAX_LENGTH = 256
CEFR_DATA_DIR = Path("data/cefr_eval")
VOCAB_INVENTORY_PATH = Path("artifacts/vocab/canonical_inventory.json")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def env_int(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_cefr_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error"):
                continue
            for sentence in row.get("sentences", []):
                rows.append(
                    {
                        "lang": row["lang"],
                        "level": row["level"],
                        "term": row["term"],
                        "sentence": sentence,
                    }
                )
    return rows


def find_term_index(words: list[str], term: str) -> int | None:
    term_lower = term.lower()
    for idx, word in enumerate(words):
        if word.lower() == term_lower:
            return idx
    return None


DE_IRREGULAR_VERBS = {
    "bringen": {"brachte", "gebracht", "brächte", "bringst", "bringt"},
    "denken": {"dachte", "gedacht", "dächte", "denkst", "denkt"},
    "kennen": {"kannte", "gekannt", "kenne", "kennst", "kennt"},
    "nennen": {"nannte", "genannt", "nenne", "nennst", "nennt"},
    "wissen": {"wusste", "gewusst", "weiß", "weißt", "wisst"},
    "lassen": {"ließ", "gelassen", "lasse", "lässt", "lasst"},
    "laufen": {"lief", "gelaufen", "laufe", "läufst", "läuft"},
    "schlafen": {"schlief", "geschlafen", "schlafe", "schläfst", "schläft"},
    "stoßen": {"stieß", "gestoßen", "stoße", "stößt"},
    "ziehen": {"zog", "gezogen", "ziehe", "ziehst", "zieht"},
    "fahren": {"fuhr", "gefahren", "fahre", "fährst", "fährt"},
    "fallen": {"fiel", "gefallen", "falle", "fällst", "fällt"},
    "halten": {"hielt", "gehalten", "halte", "hältst", "hält"},
    "raten": {"riet", "geraten", "rate", "rätst", "rät"},
    "geben": {"gab", "gegeben", "gebe", "gibst", "gibt"},
    "lesen": {"las", "gelesen", "lese", "liest", "lest"},
    "sehen": {"sah", "gesehen", "sehe", "siehst", "sieht"},
    "nehmen": {"nahm", "genommen", "nehme", "nimmst", "nimmt"},
    "sprechen": {"sprach", "gesprochen", "spreche", "sprichst", "spricht"},
    "treffen": {"traf", "getroffen", "treffe", "triffst", "trifft"},
    "essen": {"aß", "gegessen", "esse", "isst", "esst"},
    "sitzen": {"saß", "gesessen", "sitze", "sitzt"},
    "liegen": {"lag", "gelegen", "liege", "liegst", "liegt"},
    "schreiben": {"schrieb", "geschrieben", "schreibe", "schreibst", "schreibt"},
    "bleiben": {"blieb", "geblieben", "bleibe", "bleibst", "bleibt"},
    "treiben": {"trieb", "getrieben", "treibe", "treibst", "treibt"},
    "schneiden": {"schnitt", "geschnitten", "schneide", "schneidest", "schneidet"},
    "reiten": {"ritt", "geritten", "reite", "reitest", "reitet"},
    "leiden": {"litt", "gelitten", "leide", "leidend", "leidet"},
    "greifen": {"griff", "gegriffen", "greife", "greifst", "greift"},
    "beißen": {"biss", "gebissen", "beiße", "beißt"},
    "finden": {"fand", "gefunden", "finde", "findest", "findet"},
    "binden": {"band", "gebunden", "binde", "bindest", "bindet"},
    "dringen": {"drang", "gedrungen", "dringe", "dringst", "dringt"},
    "ringen": {"rang", "gerungen", "ringe", "ringst", "ringt"},
    "schwingen": {"schwang", "geschwungen", "schwingen", "schwingst", "schwingt"},
    "beginnen": {"begann", "begonnen", "beginne", "beginnst", "beginnt"},
    "gewinnen": {"gewann", "gewonnen", "gewinne", "gewinnst", "gewinnt"},
    "schwimmen": {"schwamm", "geschwommen", "schwimme", "schwimmst", "schwimmt"},
}

DE_SEPARABLE_PREFIXES = {
    "ab", "an", "auf", "aus", "bei", "durch", "ein", "entgegen", "fort",
    "her", "hin", "hinterher", "los", "mit", "nach", "nieder", "vor",
    "weg", "zu", "zurück", "zusammen", "empor", "gegenüber", "hinaus",
    "heraus", "herein", "hinein", "herunter", "hinunter", "herauf",
    "hinauf", "herüber", "hinüber", "entlang", "fehl", "kennen", "miss",
    "über", "um", "unter", "voll", "wieder",
}

DE_COMPARATIVES = {
    "besser": "gut",
    "beste": "gut",
    "besten": "gut",
    "besseren": "gut",
    "besseres": "gut",
    "besserer": "gut",
    "meist": "viel",
    "meiste": "viel",
    "meisten": "viel",
    "größer": "groß",
    "größte": "groß",
    "größten": "groß",
    "weniger": "wenig",
    "wenigste": "wenig",
    "wenigsten": "wenig",
    "näher": "nah",
    "nächste": "nah",
    "nächsten": "nah",
    "höher": "hoch",
    "höchste": "hoch",
    "höchsten": "hoch",
    "tiefer": "tief",
    "tiefste": "tief",
    "tiefsten": "tief",
    "älter": "alt",
    "älteste": "alt",
    "ältesten": "alt",
    "jünger": "jung",
    "jüngste": "jung",
    "jüngsten": "jung",
    "länger": "lang",
    "längste": "lang",
    "längsten": "lang",
    "kürzer": "kurz",
    "kürzeste": "kurz",
    "kürzesten": "kurz",
    "stärker": "stark",
    "stärkste": "stark",
    "stärksten": "stark",
    "schwerer": "schwer",
    "schwerste": "schwer",
    "schwersten": "schwer",
    "einfacher": "einfach",
    "einfachste": "einfach",
    "einfachsten": "einfach",
    "früher": "früh",
    "früheste": "früh",
    "frühesten": "früh",
    "später": "spät",
    "späteste": "spät",
    "spätesten": "spät",
}

DE_UMLAUT_NOUNS = {
    "Mütter": "Mutter", "Väter": "Vater", "Brüder": "Bruder",
    "Töchter": "Tochter", "Söhne": "Sohn", "Bäume": "Baum",
    "Häuser": "Haus", "Mäuse": "Maus", "Bücher": "Buch",
    "Wände": "Wand", "Kräfte": "Kraft", "Länder": "Land",
    "Schlüssel": "Schlüssel", "Städte": "Stadt", "Kälte": "kalt",
    "Wärme": "warm", "Länge": "lang", "Stärke": "stark",
    "Größe": "groß", "Höhe": "hoch", "Tiefe": "tief",
    "Kürze": "kurz", "Breite": "breit", "Weite": "weit",
    "Nähe": "nah", "Ferne": "fern", "Alter": "alt",
    "Jugend": "jung",
}

EN_IRREGULAR_PAST = {
    "was": "be", "were": "be", "am": "be", "is": "be", "are": "be",
    "had": "have", "has": "have", "did": "do", "went": "go",
    "got": "get", "made": "make", "came": "come", "took": "take",
    "saw": "see", "knew": "know", "thought": "think", "found": "find",
    "gave": "give", "told": "tell", "became": "become", "left": "leave",
    "felt": "feel", "brought": "bring", "wrote": "write", "sat": "sit",
    "stood": "stand", "lost": "lose", "heard": "hear", "let": "let",
    "paid": "pay", "met": "meet", "ran": "run", "kept": "keep",
    "held": "hold", "spoke": "speak", "read": "read", "grew": "grow",
    "led": "lead", "broke": "break", "sent": "send", "fell": "fall",
    "cut": "cut", "hit": "hit", "put": "put", "set": "set",
    "shut": "shut", "spread": "spread", "bet": "bet", "cast": "cast",
    "cost": "cost", "burst": "burst", "crept": "creep", "dealt": "deal",
    "dug": "dig", "fed": "feed", "fled": "flee", "hung": "hang",
    "spun": "spin", "stuck": "stick", "swung": "swing", "wept": "weep",
    "born": "bear", "eaten": "eat", "given": "give", "taken": "take",
    "written": "write", "spoken": "speak", "broken": "break",
    "chosen": "choose", "driven": "drive", "frozen": "freeze",
    "risen": "rise", "stolen": "steal", "worn": "wear",
    "torn": "tear", "sworn": "swear", "drawn": "draw",
    "grown": "grow", "known": "know", "thrown": "throw",
    "blown": "blow", "shown": "show", "flown": "fly",
    "done": "do", "gone": "go", "seen": "see",
}

ES_IRREGULAR_VERBS = {
    "es": "ser", "fue": "ser", "era": "ser", "fueron": "ser",
    "soy": "ser", "eres": "ser", "somos": "ser", "son": "ser",
    "está": "estar", "están": "estar", "estuvo": "estar",
    "estaba": "estar", "estoy": "estar", "estás": "estar",
    "estamos": "estar",
    "tiene": "tener", "tienen": "tener", "tuvo": "tener",
    "tenía": "tener", "tengo": "tener", "tienes": "tener",
    "hace": "hacer", "hacen": "hacer", "hizo": "hacer",
    "hacía": "hacer", "hago": "hacer", "haces": "hacer",
    "dice": "decir", "dicen": "decir", "dijo": "decir",
    "decía": "decir", "digo": "decir", "dices": "decir",
    "puede": "poder", "pueden": "poder", "pudo": "poder",
    "podía": "poder", "puedo": "poder", "puedes": "poder",
    "va": "ir", "van": "ir", "iba": "ir",
    "voy": "ir", "vas": "ir", "vamos": "ir",
    "sabe": "saber", "saben": "saber", "supo": "saber",
    "sabía": "saber", "sé": "saber", "sabes": "saber",
    "quiere": "querer", "quieren": "querer", "quiso": "querer",
    "quería": "querer", "quiero": "querer", "quieres": "querer",
    "viene": "venir", "vienen": "venir", "vino": "venir",
    "venía": "venir", "vengo": "venir", "vienes": "venir",
    "da": "dar", "dan": "dar", "dio": "dar", "daba": "dar",
    "doy": "dar", "das": "dar", "damos": "dar",
}


def is_irregular_verb(word: str, gold_lemma: str, lang: str) -> bool:
    w = word.lower()
    if lang == "de":
        for _lemma, forms in DE_IRREGULAR_VERBS.items():
            if w in forms or (w.startswith("ge") and w[2:] in forms):
                return True
        for prefix in DE_SEPARABLE_PREFIXES:
            if w.startswith(prefix) and w[len(prefix):] in {
                f for forms in DE_IRREGULAR_VERBS.values() for f in forms
            }:
                return True
    elif lang == "en":
        if w in EN_IRREGULAR_PAST:
            return True
        if w in EN_IRREGULAR_PLURALS:
            return True
    elif lang == "es":
        if w in ES_IRREGULAR_VERBS:
            return True
    return False


def is_separable_verb(word: str, gold_lemma: str, lang: str) -> bool:
    if lang != "de":
        return False
    w = word.lower()
    for prefix in DE_SEPARABLE_PREFIXES:
        if w.startswith(prefix) and len(w) > len(prefix) + 2:
            return True
    return False


def is_umlaut_change(word: str, gold_lemma: str, lang: str) -> bool:
    if lang != "de":
        return False
    w = word.lower()
    if w in {k.lower() for k in DE_UMLAUT_NOUNS}:
        return True
    umlaut_chars = "äöüÄÖÜ"
    base_chars = "aouAOU"
    has_umlaut = any(c in word for c in umlaut_chars)
    has_base = any(c in gold_lemma for c in base_chars)
    return has_umlaut and has_base


def is_comparative(word: str, gold_lemma: str, lang: str) -> bool:
    w = word.lower()
    if lang == "de":
        return w in DE_COMPARATIVES
    elif lang == "en":
        if w.endswith("er") or w.endswith("est"):
            return True
    elif lang == "es":
        if w.endswith("mente") and len(w) > 5:
            return True
    return False


def is_participle_adjective(word: str, upos: str, gold_lemma: str, lang: str) -> bool:
    if lang == "de":
        w = word.lower()
        if upos == "ADJ" and (w.startswith("ge") and w.endswith("t")):
            return True
        if upos == "ADJ" and (w.startswith("be") and w.endswith("t")):
            return True
    elif lang == "en":
        w = word.lower()
        if upos == "ADJ" and w.endswith(("ed", "en", "ing")):
            return True
    elif lang == "es":
        w = word.lower()
        if upos == "ADJ" and (w.endswith("ado") or w.endswith("ido")):
            return True
    return False


def is_contraction(word: str, lang: str) -> bool:
    if lang == "de":
        return word.lower() in DE_CONTRACTIONS
    return False


def is_case_or_number_inflection(word: str, gold_lemma: str, lang: str) -> bool:
    if lang == "de":
        w = word.lower()
        stripped_gold = gold_lemma.lower().rstrip("enrsmte")  # noqa: B005
        stripped_w = w.rstrip("enrsmte")  # noqa: B005
        if w.endswith(("e", "en", "er", "es", "em")) and stripped_gold != stripped_w:
            return True
    elif lang == "es":
        w = word.lower()
        stripped_gold_es = gold_lemma.lower().rstrip("oas")
        stripped_w_es = w.rstrip("oas")
        if w.endswith(("o", "a", "os", "as")) and stripped_gold_es != stripped_w_es:
            return True
    return False


def categorize_error(
    word: str, gold_lemma: str, pred_label: str, pred_lemma: str | None, upos: str, lang: str
) -> list[str]:
    categories = []

    if pred_lemma is None:
        categories.append("edit_tree_failure")

    if pred_lemma is not None and pred_lemma.lower() != gold_lemma.lower():
        if is_irregular_verb(word, gold_lemma, lang):
            categories.append("irregular_verb")
        if is_separable_verb(word, gold_lemma, lang):
            categories.append("separable_verb")
        if is_umlaut_change(word, gold_lemma, lang):
            categories.append("umlaut_change")
        if is_comparative(word, gold_lemma, lang):
            categories.append("comparative")
        if is_participle_adjective(word, upos, gold_lemma, lang):
            categories.append("participle_adjective")
        if is_contraction(word, lang):
            categories.append("contraction")
        if is_case_or_number_inflection(word, gold_lemma, lang):
            categories.append("case_number_inflection")

        if pred_lemma is not None:
            gold_label = make_edit_label(word, gold_lemma)
            if gold_label != pred_label:
                if gold_label == "IDENTITY" and pred_label != "IDENTITY":
                    categories.append("spurious_edit")
                elif gold_label != "IDENTITY" and pred_label == "IDENTITY":
                    categories.append("missed_edit")
                else:
                    categories.append("wrong_edit_label")

    if not categories and pred_lemma is not None and pred_lemma.lower() != gold_lemma.lower():
        categories.append("other")

    return categories


def resolve_base_model_source(model_dir: str) -> str:
    adapter_config = Path(model_dir) / "adapter_config.json"
    if adapter_config.exists():
        data = load_json(adapter_config)
        base_model = data.get("base_model_name_or_path")
        if isinstance(base_model, str) and base_model.strip():
            return base_model
    return "EuroBERT/EuroBERT-210m"


def run_analysis(lang: str) -> dict:
    assets = language_assets()
    if assets.lang != lang:
        os.environ["LEMMA_LANG"] = lang
        assets = language_assets()

    model_dir = os.getenv("MODEL_DIR", str(assets.output_dir))
    data_dir = Path(os.getenv("CEFR_EVAL_DIR", str(CEFR_DATA_DIR)))
    path = data_dir / f"{lang}.jsonl"
    batch_size = max(1, env_int("EVAL_BATCH_SIZE", 8))

    tokenizer = AutoTokenizer.from_pretrained(str(assets.tokenizer_dir), trust_remote_code=True)
    device = get_device()

    base_model_source = resolve_base_model_source(model_dir)
    label2id = load_json(assets.label2id_path)
    id2label = load_json(assets.id2label_path)
    upos_label2id = load_json(assets.upos_label2id_path)
    upos_id2label = load_json(assets.upos_id2label_path)

    config = EuroBertUposLemmaConfig(
        base_model_name_or_path=base_model_source,
        upos_label2id=upos_label2id,
        lemma_label2id=label2id,
    )

    base_model = EuroBertForUposLemma.from_pretrained(
        base_model_source,
        config=config,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    if Path(model_dir, "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        model = EuroBertForUposLemma.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    model.eval()

    rows = load_cefr_rows(path)

    error_counts: Counter = Counter()
    level_error_counts: dict[str, Counter] = defaultdict(Counter)
    level_total: Counter = Counter()
    level_correct: Counter = Counter()
    all_errors: list[dict] = []
    label_confusion: dict[str, Counter] = defaultdict(Counter)
    pred_label_counts: Counter = Counter()
    gold_label_counts: Counter = Counter()

    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            words_batch = []
            metadata = []
            for row in batch:
                words = row["sentence"].split()
                idx = find_term_index(words, row["term"])
                if idx is None:
                    continue
                words_batch.append(words)
                metadata.append((row, idx))

            if not words_batch:
                continue

            encoded = tokenizer(
                words_batch,
                is_split_into_words=True,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            outputs = model(**{k: v.to(device) for k, v in encoded.items()})
            upos_logits = outputs.logits[0].detach().cpu().numpy()
            lemma_logits = outputs.logits[1].detach().cpu().numpy()

            for batch_index, (row, idx) in enumerate(metadata):
                word_ids = encoded.word_ids(batch_index=batch_index)
                words = row["sentence"].split()
                word = words[idx]
                level = row["level"]
                level_total[level] += 1

                pred_label = "UNKNOWN"
                pred_upos = "X"
                for token_idx, word_id in enumerate(word_ids):
                    if word_id == idx:
                        label_id = int(np.argmax(lemma_logits[batch_index][token_idx]))
                        pred_label = id2label.get(str(label_id), "UNKNOWN")
                        upos_id = int(np.argmax(upos_logits[batch_index][token_idx]))
                        pred_upos = upos_id2label.get(str(upos_id), "X")
                        break

                gold_lemma = row["term"]
                gold_label = make_edit_label(word, gold_lemma)
                pred_lemma_raw = apply_edit_label(word, pred_label)
                pred_lemma = (
                    apply_postprocess_rules(word, lang, pred_lemma_raw, pred_upos)
                    if pred_lemma_raw is not None
                    else None
                )

                if pred_lemma is not None and pred_lemma.lower() == gold_lemma.lower():
                    level_correct[level] += 1
                    continue

                gold_label_counts[gold_label] += 1
                pred_label_counts[pred_label] += 1
                if pred_label != gold_label:
                    label_confusion[gold_label][pred_label] += 1

                cats = categorize_error(word, gold_lemma, pred_label, pred_lemma, pred_upos, lang)
                for cat in cats:
                    error_counts[cat] += 1
                    level_error_counts[level][cat] += 1

                all_errors.append({
                    "word": word,
                    "gold_lemma": gold_lemma,
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "pred_lemma": pred_lemma,
                    "pred_upos": pred_upos,
                    "level": level,
                    "categories": cats,
                    "sentence": row["sentence"],
                })

    return {
        "lang": lang,
        "total_evaluated": sum(level_total.values()),
        "total_errors": len(all_errors),
        "error_categories": dict(error_counts.most_common()),
        "level_errors": {lvl: dict(cnt.most_common()) for lvl, cnt in level_error_counts.items()},
        "level_accuracy": {
            lvl: round(level_correct[lvl] / level_total[lvl], 4) if level_total[lvl] else 0.0
            for lvl in level_total
        },
        "top_confused_labels": [
            {"gold": gold, "pred": pred, "count": cnt}
            for gold, preds in sorted(
                label_confusion.items(), key=lambda x: -sum(x[1].values()),
            )[:20]
            for pred, cnt in preds.most_common(3)
        ],
        "rare_gold_labels": [
            {"label": lbl, "count": cnt}
            for lbl, cnt in gold_label_counts.most_common()[::-1]
            if cnt <= 5
        ][:30],
        "sample_errors_by_category": {
            cat: [e for e in all_errors if cat in e["categories"]][:5]
            for cat in error_counts
        },
    }


def analyze_label_frequencies(lang: str) -> dict:
    from datasets import load_from_disk

    assets = language_assets()
    if assets.lang != lang:
        os.environ["LEMMA_LANG"] = lang
        assets = language_assets()

    ds_path = assets.dataset_path
    if not Path(ds_path).exists():
        return {"lang": lang, "error": "dataset not found"}

    ds = load_from_disk(str(ds_path))
    label2id = load_json(assets.label2id_path)

    train_labels: Counter = Counter()
    for row in ds["train"]:
        for lbl in row["labels"]:
            if lbl != -100:
                train_labels[lbl] += 1

    val_labels: Counter = Counter()
    for row in ds["validation"]:
        for lbl in row["labels"]:
            if lbl != -100:
                val_labels[lbl] += 1

    id2label = {int(v): k for k, v in label2id.items()}
    total = sum(train_labels.values())

    label_stats = []
    for label_id, count in train_labels.most_common():
        label_name = id2label.get(label_id, f"LABEL_{label_id}")
        val_count = val_labels.get(label_id, 0)
        label_stats.append({
            "label_id": label_id,
            "label_name": label_name,
            "train_count": count,
            "val_count": val_count,
            "pct_of_total": round(count / total * 100, 4),
        })

    rare_labels = [s for s in label_stats if s["train_count"] < 50]
    missing_val = [s for s in label_stats if s["val_count"] == 0]

    return {
        "lang": lang,
        "total_labels": len(label2id),
        "total_train_tokens": total,
        "rare_labels_count": len(rare_labels),
        "missing_in_val_count": len(missing_val),
        "rare_labels": rare_labels[:30],
        "missing_in_val": missing_val[:30],
        "top_20_labels": label_stats[:20],
        "bottom_20_labels": label_stats[-20:],
    }


def main():
    lang = os.getenv("LEMMA_LANG", "de")
    report = run_analysis(lang)
    freq_report = analyze_label_frequencies(lang)
    report["label_frequencies"] = freq_report

    out_path = Path(f"logs/error_analysis_{lang}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
