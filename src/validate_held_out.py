import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import AutoTokenizer

from evaluate import collect_word_predictions, is_clean_vocab_oov, resolve_prediction
from language_assets import SPACY_MODELS
from multitask_model import EuroBertForUposLemma

MERGED_DIRS = {
    "de": "models/eurobert-lemma-de-210m-merged",
    "en": "models/eurobert-lemma-en-210m-merged",
    "es": "models/eurobert-lemma-es-210m-merged",
}

NOVEL_SENTENCES = {
    "de": [
        "Der ehemalige Bürgermeister hat gestern eine überraschende Rede gehalten.",
        "Meine Großeltern wuchsen in einem kleinen Dorf in der Nähe von München auf.",
        "Die Wissenschaftlerinnen veröffentlichten ihre bahnbrechenden Ergebnisse im Fachjournal.",
        "Wir hätten den Zug genommen, wenn wir früher aufgestanden wären.",
        "Die älteren Schülerinnen halfen den jüngeren bei den Hausaufgaben.",
        "Nachdem der Sturm vorbeigezogen war, reparierten die Handwerker das beschädigte Dach.",
        "In diesem Restaurant werden ausschließlich regionale Zutaten verwendet.",
        "Die verschlossene Tür ließ sich mit dem neuen Schlüssel öffnen.",
        "Hättest du gewusst, dass dieser Fluss ins Meer mündet?",
        "Die angestrebten Verbesserungen wurden leider nicht rechtzeitig umgesetzt.",
        "Trotz der schwierigen Verhältnisse blieben die Anwohner freundlich und hilfsbereit.",
        "Ein aufmerksamer Beobachter hätte die Unterschiede sofort bemerkt.",
        "Die zusammengefassten Ergebnisse wurden dem Ausschuss vorgelegt.",
        "Wir sind montags immer besonders beschäftigt.",
        "Das neu renovierte Museum öffnet seine Türen nächstes Wochenende.",
        "Die ungeduldig wartenden Fahrgäste beschwerten sich beim Fahrer.",
        "Während des Unwetters fielen mehrere alte Bäume um.",
        "Die vorsichtige Fahrerin vermied einen drohenden Unfall.",
        "Wir hätten uns mehr Mühe geben müssen.",
        "Der gelegentlich auftretende Fehler ließ sich schwer reproduzieren.",
    ],
    "en": [
        "The researchers published groundbreaking findings in an obscure journal last Tuesday.",
        "Had we known about the delay, we would have taken an earlier train.",
        "Several elderly neighbors helped shovel the unexpectedly heavy snowfall.",
        "The newly appointed manager restructured the entire department within weeks.",
        "Children should not have to endure such unnecessarily harsh punishments.",
        "The crumbling infrastructure has been overlooked by successive administrations.",
        "She painstakingly reconstructed the shattered vase using specialized adhesive.",
        "Whichever candidate wins will face unprecedented challenges ahead.",
        "The thoroughly revised manuscript was finally accepted for publication.",
        "Misunderstood instructions led to unnecessarily complicated procedures.",
        "The breathtaking landscape reminded us why we moved to the countryside.",
        "Unprecedented rainfall caused widespread flooding throughout the region.",
        "The overqualified applicant reluctantly accepted the entry-level position.",
        "Nobody could have predicted such devastating consequences.",
        "The astonishingly talented musician performed an unrehearsed composition.",
        "The hastily written report contained several glaring inaccuracies.",
        "Unbeknownst to the committee, the proposal had been altered significantly.",
        "The uncharacteristically quiet classroom surprised the substitute teacher.",
        "The painstakingly restored painting fetched an unexpectedly high price.",
        "She unfailingly remembered every single colleague's birthday.",
    ],
    "es": [
        "Los científicos publicaron sus revolucionarios hallazgos en una revista especializada.",
        "Habríamos tomado el tren si hubiéramos llegado más temprano a la estación.",
        "Las alumnas mayores ayudaron a las más jóvenes con sus deberes.",
        "A pesar de las difíciles circunstancias, los residentes permanecieron tranquilos.",
        "El recién nombrado director reorganizó todo el departamento en pocas semanas.",
        "La puerta cerrada se pudo abrir con la nueva llave que nos dieron.",
        "En este restaurante se utilizan exclusivamente ingredientes de la región.",
        "Habrías sabido la respuesta si hubieras estudiado más a fondo.",
        "Los resultados resumidos fueron presentados al comité la semana pasada.",
        "Las mejoras propuestas desafortunadamente no se implementaron a tiempo.",
        "Un observador atento habría notado las diferencias inmediatamente.",
        "Estamos especialmente ocupados los lunes por la mañana.",
        "El museo recién renovado abrirá sus puertas el próximo fin de semana.",
        "Nadie podría haber predicho consecuencias tan devastadoras.",
        "La lluvia sin precedentes causó inundaciones generalizadas en toda la comarca.",
        "Los impacientes pasajeros se quejaron al conductor del retraso.",
        "Durante la tormenta cayeron varios árboles antiguos.",
        "La conductora precavida evitó un accidente inminente.",
        "Habríamos debido esforzarnos más para terminar a tiempo.",
        "El error ocasional resultaba difícil de reproducir sistemáticamente.",
    ],
}

MANUAL_CORRECTIONS = {
    "de": {
        "Großeltern": "Großelter",
        "Wissenschaftlerinnen": "Wissenschaftlerin",
        "im": "in",
        "früher": "früh",
        "ins": "in",
        ".": ".",
        ",": ",",
        "?": "?",
        "!": "!",
        ":": ":",
        ";": ";",
        "-": "-",
    },
    "en": {},
    "es": {
        "Habríamos": "haber",
        "Habrías": "haber",
        "hubieras": "haber",
        "esforzarnos": "esforzarse",
    },
}


def get_gold(lang, sentences):
    import spacy

    nlp = spacy.load(SPACY_MODELS[lang])
    corrections = MANUAL_CORRECTIONS.get(lang, {})
    results = []
    for sent in sentences:
        doc = nlp(sent)
        tokens = [tok.text for tok in doc]
        gold_lemmas = []
        for tok in doc:
            lemma = corrections.get(tok.text, tok.lemma_)
            gold_lemmas.append(lemma)
        gold_upos = [tok.pos_ for tok in doc]
        results.append(
            {
                "words": tokens,
                "lemmas": gold_lemmas,
                "upos": gold_upos,
                "lang": lang,
            }
        )
    return results


def load_artifacts(lang):
    merged_dir = MERGED_DIRS[lang]
    assets_dir = Path(f"artifacts/lemma_{lang}")
    tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
    model = EuroBertForUposLemma.from_pretrained(merged_dir, trust_remote_code=True)
    model.eval()
    id2label = json.loads((assets_dir / "label2id.json").read_text())
    id2label = {str(v): k for k, v in id2label.items()}
    upos_label2id = json.loads((assets_dir / "upos_label2id.json").read_text())
    upos_id2label = {str(v): k for k, v in upos_label2id.items()}
    lexicon = json.loads((assets_dir / "lexicon.json").read_text())
    canonical_vocab = set(lexicon.values())
    return tokenizer, model, id2label, upos_id2label, lexicon, canonical_vocab


def predict_batch(tokenizer, model, id2label, upos_id2label, rows, lang):
    encoded = tokenizer(
        [r["words"] for r in rows],
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.inference_mode():
        outputs = model(**{k: v for k, v in encoded.items()})

    upos_logits_all = outputs.logits[0].detach().cpu().numpy()
    lemma_logits_all = outputs.logits[1].detach().cpu().numpy()

    all_preds = []
    for batch_idx, row in enumerate(rows):
        word_ids = encoded.word_ids(batch_index=batch_idx)
        raw_labels, _, predicted_upos = collect_word_predictions(
            upos_logits_all[batch_idx],
            lemma_logits_all[batch_idx],
            word_ids,
            id2label,
            upos_id2label,
        )
        word_preds = []
        for word_idx, word in enumerate(row["words"], start=1):
            base_label = raw_labels.get(word_idx)
            upos = predicted_upos.get(word_idx, "X")
            lemma, source, edit_failed = resolve_prediction(
                word, upos, base_label, row.get("_lexicon", {}), lang=lang
            )
            if lemma is None:
                lemma = word
            word_preds.append(
                {
                    "word": word,
                    "upos": upos,
                    "lemma": lemma,
                    "source": source,
                    "edit_failed": edit_failed,
                }
            )
        all_preds.append(word_preds)
    return all_preds


def evaluate_lang(lang):
    print(f"\n{'=' * 60}")
    print(f"  {lang.upper()} Held-Out Validation (manually corrected gold)")
    print(f"{'=' * 60}")

    tokenizer, model, id2label, upos_id2label, lexicon, canonical_vocab = load_artifacts(lang)
    rows = get_gold(lang, NOVEL_SENTENCES[lang])
    for r in rows:
        r["_lexicon"] = lexicon

    all_preds = predict_batch(tokenizer, model, id2label, upos_id2label, rows, lang)

    total = 0
    lemma_correct = 0
    upos_correct = 0
    lemma_total = 0
    oov_total = 0
    oov_correct = 0
    in_vocab_total = 0
    in_vocab_correct = 0
    canonical_oov_total = 0
    canonical_oov_correct = 0
    edit_failed_count = 0
    errors = []

    for row, preds in zip(rows, all_preds, strict=False):
        for i, pred in enumerate(preds):
            gold_lemma = row["lemmas"][i]
            gold_upos = row["upos"][i]
            total += 1

            if pred["upos"] == gold_upos:
                upos_correct += 1

            if gold_upos == "PROPN":
                continue

            lemma_total += 1
            lemma_match = pred["lemma"].lower() == gold_lemma.lower()

            if lemma_match:
                lemma_correct += 1
            else:
                errors.append(
                    {
                        "word": pred["word"],
                        "gold": gold_lemma.lower(),
                        "pred": pred["lemma"].lower(),
                        "gold_upos": gold_upos,
                        "pred_upos": pred["upos"],
                        "source": pred["source"],
                        "edit_failed": pred["edit_failed"],
                    }
                )

            if pred["edit_failed"]:
                edit_failed_count += 1

            word = pred["word"]
            in_lex = word in lexicon
            lemma_key = gold_lemma.lower()
            in_canon = lemma_key in canonical_vocab

            if not in_lex:
                oov_total += 1
                if lemma_match:
                    oov_correct += 1
            else:
                in_vocab_total += 1
                if lemma_match:
                    in_vocab_correct += 1

            if not in_canon:
                canonical_oov_total += 1
                if is_clean_vocab_oov(word, gold_lemma, gold_upos):
                    if lemma_match:
                        canonical_oov_correct += 1

    lemma_acc = lemma_correct / lemma_total if lemma_total else 0
    upos_acc = upos_correct / total if total else 0
    oov_acc = oov_correct / oov_total if oov_total else 0
    in_vocab_acc = in_vocab_correct / in_vocab_total if in_vocab_total else 0
    canon_oov_acc = canonical_oov_correct / canonical_oov_total if canonical_oov_total else 0

    print(f"\nTotal tokens:       {total}")
    print(f"Lemma accuracy:     {lemma_acc:.4f} ({lemma_correct}/{lemma_total})")
    print(f"UPOS accuracy:      {upos_acc:.4f} ({upos_correct}/{total})")
    print(f"In-vocab accuracy:  {in_vocab_acc:.4f} ({in_vocab_correct}/{in_vocab_total})")
    print(f"OOV accuracy:       {oov_acc:.4f} ({oov_correct}/{oov_total})")
    print(
        f"Canonical OOV acc:  {canon_oov_acc:.4f} ({canonical_oov_correct}/{canonical_oov_total})"
    )
    print(f"Edit-failed count:  {edit_failed_count}")

    if errors:
        print(f"\nLemma errors ({len(errors)}):")
        for e in errors[:30]:
            flag = " [EDIT-FAIL]" if e["edit_failed"] else ""
            print(
                f"  {e['word']:22s} -> gold={e['gold']:22s}  "
                f"pred={e['pred']:22s}  [{e['gold_upos']}/{e['pred_upos']}]{flag}"
            )

    return {
        "lang": lang,
        "total": total,
        "lemma_acc": round(lemma_acc, 4),
        "upos_acc": round(upos_acc, 4),
        "in_vocab_acc": round(in_vocab_acc, 4),
        "oov_acc": round(oov_acc, 4),
        "canonical_oov_acc": round(canon_oov_acc, 4),
        "edit_failed": edit_failed_count,
        "errors": len(errors),
    }


def main():
    results = []
    for lang in ["de", "en", "es"]:
        results.append(evaluate_lang(lang))

    print(f"\n{'=' * 60}")
    print("  HELD-OUT VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    hdr = (
        f"{'Lang':<6} {'Tokens':<8} {'Lemma':<10} {'UPOS':<10} "
        f"{'InVocab':<10} {'OOV':<10} {'CanOOV':<10} {'Errs':<6}"
    )
    print(hdr)
    for r in results:
        print(
            f"{r['lang']:<6} {r['total']:<8} {r['lemma_acc']:<10.4f} "
            f"{r['upos_acc']:<10.4f} {r['in_vocab_acc']:<10.4f} "
            f"{r['oov_acc']:<10.4f} {r['canonical_oov_acc']:<10.4f} {r['errors']:<6}"
        )

    out_path = Path("artifacts/held_out_validation_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
