from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from language_assets import normalize_lang
from lmstudio_client import LMStudioClient, parse_sentence_lines

DEFAULT_OUTPUT_DIR = Path("data/silver/targeted")
SYSTEM_PROMPT = (
    "You generate concise natural language training sentences. "
    "Return only sentence lines, no commentary, no numbering, no markdown."
)

DE_PROMPTS = [
    (
        "inflected_adj",
        "Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        "Jeder Satz muss ein flektiertes Adjektiv enthalten (Endungen -e, -en, -er, -es, -em). "
        "Beispiele: 'der schnelle Zug', 'eine schöne Blume', "
        "'mit großem Erfolg', 'aufmerksamere Schüler'. "
        "Gib nur die Sätze, genau eine Zeile pro Satz.",
    ),
    (
        "plural_nouns",
        "Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        "Jeder Satz muss ein Pluralnomen enthalten. "
        "Beispiele: 'die Kinder', 'viele Bücher', 'alle Studenten', 'die Ergebnisse'. "
        "Gib nur die Sätze, genau eine Zeile pro Satz.",
    ),
    (
        "contractions",
        "Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        "Jeder Satz muss eine Präposition-Kontraktion enthalten "
        "(ins, beim, zum, zur, vom, im, am, ans, aufs, hinters, übers, unters, durchs, fürs). "
        "Beispiele: 'Ich gehe ins Kino', 'Er ist beim Arzt', 'Wir fahren zum Strand'. "
        "Gib nur die Sätze, genau eine Zeile pro Satz.",
    ),
    (
        "verb_participles",
        "Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        "Jeder Satz muss ein Partizip II oder eine Vergangenheitsform enthalten. "
        "Beispiele: 'hat gelesen', 'sind gegangen', 'wurde veröffentlicht', 'hatten geholfen'. "
        "Gib nur die Sätze, genau eine Zeile pro Satz.",
    ),
    (
        "comparative_superlative",
        "Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        "Jeder Satz muss einen Komparativ oder Superlativ enthalten, "
        "besonders unregelmäßige Formen (besser, beste, länger, näher, größer, mehr, weniger, "
        "älter, jüngste, weitere, kürzeste). "
        "Beispiele: 'schneller als', 'am schönsten', 'das größte Haus', 'bessere Noten'. "
        "Gib nur die Sätze, genau eine Zeile pro Satz.",
    ),
    (
        "separable_verbs",
        "Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        "Jeder Satz muss ein trennbares Verb oder eine Form eines trennbaren Verbs enthalten. "
        "Beispiele: 'Er kommt an', 'Sie schließt ab', 'Wir rufen an', 'Er fährt mit', "
        "'Das Projekt ist abgeschlossen', 'Sie ist angekommen', 'Ich stehe auf', "
        "'Er nimmt teil', 'Wir brechen zusammen'. "
        "Gib nur die Sätze, genau eine Zeile pro Satz.",
    ),
    (
        "irregular_verbs",
        "Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        "Jeder Satz muss ein unregelmäßiges Verb in einer konjugierten Form enthalten "
        "(Präteritum oder Präsens: fand, gab, wurde, ging, sah, kam, schrieb, sprach, "
        "nahm, las, lief, fiel, half, wusste, begann, blieb, bekam). "
        "Gib nur die Sätze, genau eine Zeile pro Satz.",
    ),
    (
        "determiner_paradigms",
        "Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        "Jeder Satz muss flektierte Determinanten oder Pronomina enthalten "
        "(einen, einem, einer, allem, alles, andere, anderes, dieser, diese, dieses, "
        "solche, keine, meine, seine, ihre, unsere, welche). "
        "Gib nur die Sätze, genau eine Zeile pro Satz.",
    ),
    (
        "noun_umlaut_plurals",
        "Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        "Jeder Satz muss ein Nomen im Plural mit Umlaut enthalten "
        "(Häuser, Bücher, Männer, Städte, Wände, Gärten, Töchter, Mütter, Väter, "
        "Kräfte, Ärzte, Bänke, Hände, Länder, Länder, Flüsse). "
        "Gib nur die Sätze, genau eine Zeile pro Satz.",
    ),
    (
        "participle_adjectives",
        "Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        "Jeder Satz muss ein Partizip als Adjektiv enthalten "
        "(beschäftigt, geöffnet, geschlossen, verbunden, bekannt, verwandt, "
        "erfahren, geschätzt, verheiratet, interessiert). "
        "Gib nur die Sätze, genau eine Zeile pro Satz.",
    ),
    (
        "identity_nouns",
        "Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        "Jeder Satz muss ein Nomen enthalten, dessen Form gleich seinem Lemma ist "
        "(Namen, Güte, Beamte, Brocken, Fetzen, Galeonen, Ventures, Milliarden). "
        "Diese Nomen werden nicht flektiert, ihr Lemma ist identisch zur Form. "
        "Gib nur die Sätze, genau eine Zeile pro Satz.",
    ),
    (
        "identity_adj_adv",
        "Schreibe {count} natürliche deutsche Sätze auf CEFR-{level}-Niveau. "
        "Jeder Satz muss ein Adjektiv oder Adverb enthalten, dessen Form gleich seinem Lemma ist "
        "(weiter, gehobener, gut, bekannt, allem). "
        "Diese Wörter behalten ihre Form als Lemma. "
        "Gib nur die Sätze, genau eine Zeile pro Satz.",
    ),
]

EN_PROMPTS = [
    (
        "irregular_plurals",
        "Write {count} natural English sentences at CEFR {level} level. "
        "Each sentence must contain an irregular plural noun "
        "(children, men, women, people, mice, geese, teeth, feet, "
        "phenomena, criteria, alumni, stimuli). "
        "Return only sentences, one per line.",
    ),
    (
        "past_participles_adj",
        "Write {count} natural English sentences at CEFR {level} level. "
        "Each sentence must contain a past participle used as an adjective "
        "(broken, shattered, complicated, revised, unprecedented, misunderstood). "
        "Return only sentences, one per line.",
    ),
    (
        "comparative_superlative",
        "Write {count} natural English sentences at CEFR {level} level. "
        "Each sentence must contain a comparative or superlative adjective, "
        "especially irregular forms (better, best, worse, worst, more, most, "
        "less, least, farther, farthest, further, furthest). "
        "Return only sentences, one per line.",
    ),
    (
        "gerund_participle",
        "Write {count} natural English sentences at CEFR {level} level. "
        "Each sentence must contain a gerund or present participle "
        "(crumbling, staggering, breathtaking, painstakingly, unfailingly). "
        "Return only sentences, one per line.",
    ),
    (
        "det_context",
        "Write {count} natural English sentences at CEFR {level} level. "
        "Each sentence must use the article 'an' before a vowel-sound word. "
        "Examples: 'an apple', 'an hour', 'an unusual event'. "
        "Return only sentences, one per line.",
    ),
    (
        "irregular_past",
        "Write {count} natural English sentences at CEFR {level} level. "
        "Each sentence must contain an irregular verb in past tense or participle "
        "(was, were, had, did, went, came, saw, knew, thought, took, "
        "gave, told, found, left, stood, heard, brought, wrote, ran, sat, "
        "fell, grew, spoke, broke, drove, chose, drew, threw, bore, born). "
        "Return only sentences, one per line.",
    ),
    (
        "be_forms",
        "Write {count} natural English sentences at CEFR {level} level. "
        "Each sentence must contain a form of the verb 'be' "
        "(am, is, are, was, were, been, being). "
        "Return only sentences, one per line.",
    ),
    (
        "have_forms",
        "Write {count} natural English sentences at CEFR {level} level. "
        "Each sentence must contain a form of the verb 'have' "
        "(have, has, had, having) or a contraction (I've, he's, they've). "
        "Return only sentences, one per line.",
    ),
    (
        "do_forms",
        "Write {count} natural English sentences at CEFR {level} level. "
        "Each sentence must contain a form of the verb 'do' "
        "(do, does, did, done, doing) or a contraction (don't, doesn't, didn't). "
        "Return only sentences, one per line.",
    ),
    (
        "identity_adj_participle",
        "Write {count} natural English sentences at CEFR {level} level. "
        "Each sentence must contain an adjective or participle whose lemma equals its surface form "
        "(her, finished, retarded, involved, means, amazed, nuts, done, soaked, "
        "disappointed, opening, pairing, dying, looking, watching, insulting). "
        "These words are NOT lemmatized to a different base form — "
        "their lemma IS the surface form. "
        "Return only sentences, one per line.",
    ),
]

ES_PROMPTS = [
    (
        "conditional_compound",
        "Escribe {count} oraciones naturales en español a nivel CEFR {level}. "
        "Cada oración debe contener un verbo en condicional compuesto "
        "(habríamos, habrías, habrían, habría + participio). "
        "Ejemplos: 'habríamos ido', 'habrías sabido', 'habrían llegado'. "
        "Devuelve solo las oraciones, una por línea.",
    ),
    (
        "reflexive_verbs",
        "Escribe {count} oraciones naturales en español a nivel CEFR {level}. "
        "Cada oración debe contener un verbo reflexivo conjugado "
        "(nos esforzamos, se levantaron, me desperté, se arrepintieron). "
        "Devuelve solo las oraciones, una por línea.",
    ),
    (
        "plural_nouns",
        "Escribe {count} oraciones naturales en español a nivel CEFR {level}. "
        "Cada oración debe contener un sustantivo en plural. "
        "Ejemplos: 'los niños', 'muchas casas', 'revolucionarios', 'hallazgos'. "
        "Devuelve solo las oraciones, una por línea.",
    ),
    (
        "subjunctive",
        "Escribe {count} oraciones naturales en español a nivel CEFR {level}. "
        "Cada oración debe contener un verbo en subjuntivo "
        "(quiera, pueda, sepamos, hicieras, tuviéramos). "
        "Devuelve solo las oraciones, una por línea.",
    ),
    (
        "comparative_superlative",
        "Escribe {count} oraciones naturales en español a nivel CEFR {level}. "
        "Cada oración debe contener un comparativo o superlativo "
        "(más grande, el mejor, la más importante, menores, mayores). "
        "Devuelve solo las oraciones, una por línea.",
    ),
    (
        "pronoun_forms",
        "Escribe {count} oraciones naturales en español a nivel CEFR {level}. "
        "Cada oración debe contener pronombres personales en varias formas "
        "(ella, ellas, ellos, nosotros, vosotros, estos, estas, esos, esas, "
        "aquella, aquellas, aquel). "
        "Devuelve solo las oraciones, una por línea.",
    ),
    (
        "determiner_identity",
        "Escribe {count} oraciones naturales en español a nivel CEFR {level}. "
        "Cada oración debe contener determinantes o adjetivos donde la forma "
        "flextionada es igual al lema (cualquier, bueno, grande, primero, "
        "tercero, nuevo, viejo, principal, general). "
        "Devuelve solo las oraciones, una por línea.",
    ),
    (
        "ser_estar_forms",
        "Escribe {count} oraciones naturales en español a nivel CEFR {level}. "
        "Cada oración debe contener formas conjugadas de 'ser' o 'estar' "
        "(es, son, era, eran, fue, fueron, estoy, estás, está, estamos, "
        "están, estaba, estaban, estuve, estuvieron). "
        "Devuelve solo las oraciones, una por línea.",
    ),
    (
        "identity_nouns_adj",
        "Escribe {count} oraciones naturales en español a nivel CEFR {level}. "
        "Cada oración debe contener un sustantivo o adjetivo donde la forma "
        "flextionada es igual al lema (tan, bolas, finales, horas, sí, mecenas, "
        "herido, fría, dado, vecina, homicida, rodillas, medias, llegado, "
        "centrista, adversaria, espaldas, saben, gritos, falta, buenas). "
        "Estas palabras NO se lematizan a una forma base diferente — "
        "el lema ES la forma superficial. "
        "Devuelve solo las oraciones, una por línea.",
    ),
]

ALL_PROMPTS = {"de": DE_PROMPTS, "en": EN_PROMPTS, "es": ES_PROMPTS}


@dataclass(frozen=True)
class TargetedJob:
    lang: str
    level: str
    category: str
    prompt: str


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def generate_job(client: LMStudioClient, job: TargetedJob) -> dict[str, object]:
    text = client.chat(job.prompt, system_prompt=SYSTEM_PROMPT)
    return {
        "lang": job.lang,
        "level": job.level,
        "category": job.category,
        "sentences": parse_sentence_lines(text),
        "raw_output": text,
    }


def main():
    lang = normalize_lang()
    lmstudio_url = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")
    model = os.getenv("LMSTUDIO_MODEL", "gemma-4-e2b-it-qat")
    sentences_per_prompt = env_int("TARGETED_SENTENCES_PER_PROMPT", 5)
    batches_per_category = env_int("TARGETED_BATCHES_PER_CATEGORY", 60)
    parallel = max(1, env_int("TARGETED_PARALLEL", 4))
    output_dir = Path(os.getenv("TARGETED_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"targeted_{lang}_raw.jsonl"

    prompts = ALL_PROMPTS.get(lang, [])
    if not prompts:
        print(f"No targeted prompts defined for language '{lang}'")
        return

    client = LMStudioClient(base_url=lmstudio_url, model=model)
    write_lock = threading.Lock()
    existing_counts: dict[tuple[str, str], int] = {}
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (row.get("level", ""), row.get("category", ""))
                existing_counts[key] = existing_counts.get(key, 0) + 1

    jobs: list[TargetedJob] = []
    for category, template in prompts:
        for level in ["A1", "A2", "B1", "B2", "C1"]:
            existing = existing_counts.get((level, category), 0)
            needed = max(0, batches_per_category - existing)
            for _ in range(needed):
                prompt_text = template.format(count=sentences_per_prompt, level=level)
                jobs.append(
                    TargetedJob(lang=lang, level=level, category=category, prompt=prompt_text)
                )

    print(
        f"Generating {len(jobs)} {lang} targeted batches "
        f"({len(prompts)} categories x 5 levels x {batches_per_category} batches)"
    )

    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(generate_job, client, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "lang": job.lang,
                        "level": job.level,
                        "category": job.category,
                        "sentences": [],
                        "raw_output": "",
                        "error": str(exc),
                    }

                with write_lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()

    print(f"Wrote targeted silver sentences to {output_path}")


if __name__ == "__main__":
    main()
