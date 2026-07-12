#!/usr/bin/env python3
"""Generate handcrafted German A1 validation CoNLL-U data (de_a1_val_016–100)."""

from __future__ import annotations

from pathlib import Path

OUT = Path("data/handcraft/de/val/a1_new_001.conllu")

# Each sentence: (sent_id_num, text, [(form, lemma, upos), ...])
SENTENCES: list[tuple[int, str, list[tuple[str, str, str]]]] = [
    # Batch 1: 016–040
    (16, "Der Vogel sitzt auf dem Baum.", [
        ("Der", "der", "DET"), ("Vogel", "Vogel", "NOUN"), ("sitzt", "sitzen", "VERB"),
        ("auf", "auf", "ADP"), ("dem", "der", "DET"), ("Baum", "Baum", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (17, "Es regnet heute.", [
        ("Es", "er", "PRON"), ("regnet", "regnen", "VERB"), ("heute", "heute", "ADV"), (".", ".", "PUNCT"),
    ]),
    (18, "Ich trage eine blaue Jacke.", [
        ("Ich", "ich", "PRON"), ("trage", "tragen", "VERB"), ("eine", "ein", "DET"),
        ("blaue", "blau", "ADJ"), ("Jacke", "Jacke", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (19, "Das Wasser ist kalt.", [
        ("Das", "der", "DET"), ("Wasser", "Wasser", "NOUN"), ("ist", "sein", "AUX"),
        ("kalt", "kalt", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (20, "Wir fahren mit dem Zug.", [
        ("Wir", "wir", "PRON"), ("fahren", "fahren", "VERB"), ("mit", "mit", "ADP"),
        ("dem", "der", "DET"), ("Zug", "Zug", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (21, "Der Fisch schwimmt im See.", [
        ("Der", "der", "DET"), ("Fisch", "Fisch", "NOUN"), ("schwimmt", "schwimmen", "VERB"),
        ("im", "in", "ADP"), ("See", "See", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (22, "Mein Opa liest die Zeitung.", [
        ("Mein", "mein", "DET"), ("Opa", "Opa", "NOUN"), ("liest", "lesen", "VERB"),
        ("die", "der", "DET"), ("Zeitung", "Zeitung", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (23, "Sie tanzt sehr gut.", [
        ("Sie", "sie", "PRON"), ("tanzt", "tanzen", "VERB"), ("sehr", "sehr", "ADV"),
        ("gut", "gut", "ADV"), (".", ".", "PUNCT"),
    ]),
    (24, "Drei Eier liegen auf dem Tisch.", [
        ("Drei", "drei", "NUM"), ("Eier", "Ei", "NOUN"), ("liegen", "liegen", "VERB"),
        ("auf", "auf", "ADP"), ("dem", "der", "DET"), ("Tisch", "Tisch", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (25, "Ich habe Hunger.", [
        ("Ich", "ich", "PRON"), ("habe", "haben", "VERB"), ("Hunger", "Hunger", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (26, "Die Tür ist offen.", [
        ("Die", "der", "DET"), ("Tür", "Tür", "NOUN"), ("ist", "sein", "AUX"),
        ("offen", "offen", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (27, "Er malt ein Bild.", [
        ("Er", "er", "PRON"), ("malt", "malen", "VERB"), ("ein", "ein", "DET"),
        ("Bild", "Bild", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (28, "Wo ist die Apotheke?", [
        ("Wo", "wo", "ADV"), ("ist", "sein", "AUX"), ("die", "der", "DET"),
        ("Apotheke", "Apotheke", "NOUN"), ("?", "?", "PUNCT"),
    ]),
    (29, "Der Hase ist schnell.", [
        ("Der", "der", "DET"), ("Hase", "Hase", "NOUN"), ("ist", "sein", "AUX"),
        ("schnell", "schnell", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (30, "Wir singen ein Lied.", [
        ("Wir", "wir", "PRON"), ("singen", "singen", "VERB"), ("ein", "ein", "DET"),
        ("Lied", "Lied", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (31, "Das Eis schmilzt schnell.", [
        ("Das", "der", "DET"), ("Eis", "Eis", "NOUN"), ("schmilzt", "schmelzen", "VERB"),
        ("schnell", "schnell", "ADV"), (".", ".", "PUNCT"),
    ]),
    (32, "Ich kaufe Bananen und Orangen.", [
        ("Ich", "ich", "PRON"), ("kaufe", "kaufen", "VERB"), ("Bananen", "Banane", "NOUN"),
        ("und", "und", "CCONJ"), ("Orangen", "Orange", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (33, "Der Mantel ist zu groß.", [
        ("Der", "der", "DET"), ("Mantel", "Mantel", "NOUN"), ("ist", "sein", "AUX"),
        ("zu", "zu", "ADV"), ("groß", "groß", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (34, "Sie schläft am Nachmittag.", [
        ("Sie", "sie", "PRON"), ("schläft", "schlafen", "VERB"), ("am", "an", "ADP"),
        ("Nachmittag", "Nachmittag", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (35, "Fährst du mit dem Fahrrad?", [
        ("Fährst", "fahren", "VERB"), ("du", "du", "PRON"), ("mit", "mit", "ADP"),
        ("dem", "der", "DET"), ("Fahrrad", "Fahrrad", "NOUN"), ("?", "?", "PUNCT"),
    ]),
    (36, "Der Wind ist stark.", [
        ("Der", "der", "DET"), ("Wind", "Wind", "NOUN"), ("ist", "sein", "AUX"),
        ("stark", "stark", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (37, "Ich wasche meine Hände.", [
        ("Ich", "ich", "PRON"), ("wasche", "waschen", "VERB"), ("meine", "mein", "DET"),
        ("Hände", "Hand", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (38, "Das Pferd läuft auf der Wiese.", [
        ("Das", "der", "DET"), ("Pferd", "Pferd", "NOUN"), ("läuft", "laufen", "VERB"),
        ("auf", "auf", "ADP"), ("der", "der", "DET"), ("Wiese", "Wiese", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (39, "Wir essen Salat heute.", [
        ("Wir", "wir", "PRON"), ("essen", "essen", "VERB"), ("Salat", "Salat", "NOUN"),
        ("heute", "heute", "ADV"), (".", ".", "PUNCT"),
    ]),
    (40, "Meine Tante wohnt in Hamburg.", [
        ("Meine", "mein", "DET"), ("Tante", "Tante", "NOUN"), ("wohnt", "wohnen", "VERB"),
        ("in", "in", "ADP"), ("Hamburg", "Hamburg", "PROPN"), (".", ".", "PUNCT"),
    ]),
    # Batch 2: 041–065
    (41, "Der Regen fällt leise.", [
        ("Der", "der", "DET"), ("Regen", "Regen", "NOUN"), ("fällt", "fallen", "VERB"),
        ("leise", "leise", "ADV"), (".", ".", "PUNCT"),
    ]),
    (42, "Ich sehe einen Film im Kino.", [
        ("Ich", "ich", "PRON"), ("sehe", "sehen", "VERB"), ("einen", "ein", "DET"),
        ("Film", "Film", "NOUN"), ("im", "in", "ADP"), ("Kino", "Kino", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (43, "Die Banane ist gelb.", [
        ("Die", "der", "DET"), ("Banane", "Banane", "NOUN"), ("ist", "sein", "AUX"),
        ("gelb", "gelb", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (44, "Er trinkt warmen Tee.", [
        ("Er", "er", "PRON"), ("trinkt", "trinken", "VERB"), ("warmen", "warm", "ADJ"),
        ("Tee", "Tee", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (45, "Zwei Vögel fliegen hoch.", [
        ("Zwei", "zwei", "NUM"), ("Vögel", "Vogel", "NOUN"), ("fliegen", "fliegen", "VERB"),
        ("hoch", "hoch", "ADV"), (".", ".", "PUNCT"),
    ]),
    (46, "Das Fenster ist sauber.", [
        ("Das", "der", "DET"), ("Fenster", "Fenster", "NOUN"), ("ist", "sein", "AUX"),
        ("sauber", "sauber", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (47, "Ich schreibe einen Brief.", [
        ("Ich", "ich", "PRON"), ("schreibe", "schreiben", "VERB"), ("einen", "ein", "DET"),
        ("Brief", "Brief", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (48, "Der Schuh ist neu.", [
        ("Der", "der", "DET"), ("Schuh", "Schuh", "NOUN"), ("ist", "sein", "AUX"),
        ("neu", "neu", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (49, "Wir gehen zum Bahnhof.", [
        ("Wir", "wir", "PRON"), ("gehen", "gehen", "VERB"), ("zum", "zu", "ADP"),
        ("Bahnhof", "Bahnhof", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (50, "Sie hat lange Haare.", [
        ("Sie", "sie", "PRON"), ("hat", "haben", "VERB"), ("lange", "lang", "ADJ"),
        ("Haare", "Haar", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (51, "Der Käse schmeckt gut.", [
        ("Der", "der", "DET"), ("Käse", "Käse", "NOUN"), ("schmeckt", "schmecken", "VERB"),
        ("gut", "gut", "ADV"), (".", ".", "PUNCT"),
    ]),
    (52, "Ich höre Musik.", [
        ("Ich", "ich", "PRON"), ("höre", "hören", "VERB"), ("Musik", "Musik", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (53, "Das Radio ist laut.", [
        ("Das", "der", "DET"), ("Radio", "Radio", "NOUN"), ("ist", "sein", "AUX"),
        ("laut", "laut", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (54, "Er putzt die Zähne.", [
        ("Er", "er", "PRON"), ("putzt", "putzen", "VERB"), ("die", "der", "DET"),
        ("Zähne", "Zahn", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (55, "Eine Maus läuft schnell.", [
        ("Eine", "ein", "DET"), ("Maus", "Maus", "NOUN"), ("läuft", "laufen", "VERB"),
        ("schnell", "schnell", "ADV"), (".", ".", "PUNCT"),
    ]),
    (56, "Wir sitzen im Park.", [
        ("Wir", "wir", "PRON"), ("sitzen", "sitzen", "VERB"), ("im", "in", "ADP"),
        ("Park", "Park", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (57, "Der See ist tief.", [
        ("Der", "der", "DET"), ("See", "See", "NOUN"), ("ist", "sein", "AUX"),
        ("tief", "tief", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (58, "Ich finde den Schlüssel nicht.", [
        ("Ich", "ich", "PRON"), ("finde", "finden", "VERB"), ("den", "der", "DET"),
        ("Schlüssel", "Schlüssel", "NOUN"), ("nicht", "nicht", "PART"), (".", ".", "PUNCT"),
    ]),
    (59, "Das Hemd ist weiß.", [
        ("Das", "der", "DET"), ("Hemd", "Hemd", "NOUN"), ("ist", "sein", "AUX"),
        ("weiß", "weiß", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (60, "Sie kommt aus Spanien.", [
        ("Sie", "sie", "PRON"), ("kommt", "kommen", "VERB"), ("aus", "aus", "ADP"),
        ("Spanien", "Spanien", "PROPN"), (".", ".", "PUNCT"),
    ]),
    (61, "Der Berg ist sehr hoch.", [
        ("Der", "der", "DET"), ("Berg", "Berg", "NOUN"), ("ist", "sein", "AUX"),
        ("sehr", "sehr", "ADV"), ("hoch", "hoch", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (62, "Ich esse Reis mit Gemüse.", [
        ("Ich", "ich", "PRON"), ("esse", "essen", "VERB"), ("Reis", "Reis", "NOUN"),
        ("mit", "mit", "ADP"), ("Gemüse", "Gemüse", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (63, "Die Uhr zeigt zehn Uhr.", [
        ("Die", "der", "DET"), ("Uhr", "Uhr", "NOUN"), ("zeigt", "zeigen", "VERB"),
        ("zehn", "zehn", "NUM"), ("Uhr", "Uhr", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (64, "Er sitzt auf dem Stuhl.", [
        ("Er", "er", "PRON"), ("sitzt", "sitzen", "VERB"), ("auf", "auf", "ADP"),
        ("dem", "der", "DET"), ("Stuhl", "Stuhl", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (65, "Wir treffen uns am Park.", [
        ("Wir", "wir", "PRON"), ("treffen", "treffen", "VERB"), ("uns", "sich", "PRON"),
        ("am", "an", "ADP"), ("Park", "Park", "NOUN"), (".", ".", "PUNCT"),
    ]),
    # Batch 3: 066–090
    (66, "Das Wetter ist schön heute.", [
        ("Das", "der", "DET"), ("Wetter", "Wetter", "NOUN"), ("ist", "sein", "AUX"),
        ("schön", "schön", "ADJ"), ("heute", "heute", "ADV"), (".", ".", "PUNCT"),
    ]),
    (67, "Ich trinke Wasser, weil ich Durst habe.", [
        ("Ich", "ich", "PRON"), ("trinke", "trinken", "VERB"), ("Wasser", "Wasser", "NOUN"),
        (",", ",", "PUNCT"), ("weil", "weil", "SCONJ"), ("ich", "ich", "PRON"),
        ("Durst", "Durst", "NOUN"), ("habe", "haben", "VERB"), (".", ".", "PUNCT"),
    ]),
    (68, "Der Himmel ist blau.", [
        ("Der", "der", "DET"), ("Himmel", "Himmel", "NOUN"), ("ist", "sein", "AUX"),
        ("blau", "blau", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (69, "Sie kocht Nudeln.", [
        ("Sie", "sie", "PRON"), ("kocht", "kochen", "VERB"), ("Nudeln", "Nudel", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (70, "Fünf Kinder spielen Fußball.", [
        ("Fünf", "fünf", "NUM"), ("Kinder", "Kind", "NOUN"), ("spielen", "spielen", "VERB"),
        ("Fußball", "Fußball", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (71, "Das Museum ist geschlossen.", [
        ("Das", "der", "DET"), ("Museum", "Museum", "NOUN"), ("ist", "sein", "AUX"),
        ("geschlossen", "geschlossen", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (72, "Ich putze das Zimmer.", [
        ("Ich", "ich", "PRON"), ("putze", "putzen", "VERB"), ("das", "der", "DET"),
        ("Zimmer", "Zimmer", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (73, "Der Rock ist rot.", [
        ("Der", "der", "DET"), ("Rock", "Rock", "NOUN"), ("ist", "sein", "AUX"),
        ("rot", "rot", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (74, "Wir besuchen unsere Freunde.", [
        ("Wir", "wir", "PRON"), ("besuchen", "besuchen", "VERB"), ("unsere", "unser", "DET"),
        ("Freunde", "Freund", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (75, "Er trägt eine Mütze.", [
        ("Er", "er", "PRON"), ("trägt", "tragen", "VERB"), ("eine", "ein", "DET"),
        ("Mütze", "Mütze", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (76, "Die Straße ist lang.", [
        ("Die", "der", "DET"), ("Straße", "Straße", "NOUN"), ("ist", "sein", "AUX"),
        ("lang", "lang", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (77, "Ich gehe spät ins Bett.", [
        ("Ich", "ich", "PRON"), ("gehe", "gehen", "VERB"), ("spät", "spät", "ADV"),
        ("ins", "in", "ADP"), ("Bett", "Bett", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (78, "Der Brief kommt morgen.", [
        ("Der", "der", "DET"), ("Brief", "Brief", "NOUN"), ("kommt", "kommen", "VERB"),
        ("morgen", "morgen", "ADV"), (".", ".", "PUNCT"),
    ]),
    (79, "Sie liest eine Zeitung.", [
        ("Sie", "sie", "PRON"), ("liest", "lesen", "VERB"), ("eine", "ein", "DET"),
        ("Zeitung", "Zeitung", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (80, "Wo wohnst du?", [
        ("Wo", "wo", "ADV"), ("wohnst", "wohnen", "VERB"), ("du", "du", "PRON"), ("?", "?", "PUNCT"),
    ]),
    (81, "Das Frühstück ist fertig.", [
        ("Das", "der", "DET"), ("Frühstück", "Frühstück", "NOUN"), ("ist", "sein", "AUX"),
        ("fertig", "fertig", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (82, "Ich gehe in die Bäckerei.", [
        ("Ich", "ich", "PRON"), ("gehe", "gehen", "VERB"), ("in", "in", "ADP"),
        ("die", "der", "DET"), ("Bäckerei", "Bäckerei", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (83, "Der Nebel ist dick.", [
        ("Der", "der", "DET"), ("Nebel", "Nebel", "NOUN"), ("ist", "sein", "AUX"),
        ("dick", "dick", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (84, "Wir lernen Spanisch zusammen.", [
        ("Wir", "wir", "PRON"), ("lernen", "lernen", "VERB"), ("Spanisch", "Spanisch", "NOUN"),
        ("zusammen", "zusammen", "ADV"), (".", ".", "PUNCT"),
    ]),
    (85, "Sie hat eine schöne Stimme.", [
        ("Sie", "sie", "PRON"), ("hat", "haben", "VERB"), ("eine", "ein", "DET"),
        ("schöne", "schön", "ADJ"), ("Stimme", "Stimme", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (86, "Der Schnee ist weiß.", [
        ("Der", "der", "DET"), ("Schnee", "Schnee", "NOUN"), ("ist", "sein", "AUX"),
        ("weiß", "weiß", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (87, "Ich brauche einen Regenschirm.", [
        ("Ich", "ich", "PRON"), ("brauche", "brauchen", "VERB"), ("einen", "ein", "DET"),
        ("Regenschirm", "Regenschirm", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (88, "Das Pferd frisst Gras.", [
        ("Das", "der", "DET"), ("Pferd", "Pferd", "NOUN"), ("frisst", "fressen", "VERB"),
        ("Gras", "Gras", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (89, "Er wartet vor der Tür.", [
        ("Er", "er", "PRON"), ("wartet", "warten", "VERB"), ("vor", "vor", "ADP"),
        ("der", "der", "DET"), ("Tür", "Tür", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (90, "Wir machen einen Spaziergang.", [
        ("Wir", "wir", "PRON"), ("machen", "machen", "VERB"), ("einen", "ein", "DET"),
        ("Spaziergang", "Spaziergang", "NOUN"), (".", ".", "PUNCT"),
    ]),
    # Batch 4: 091–100
    (91, "Die Nacht ist dunkel.", [
        ("Die", "der", "DET"), ("Nacht", "Nacht", "NOUN"), ("ist", "sein", "AUX"),
        ("dunkel", "dunkel", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (92, "Ich esse Honig auf dem Brot.", [
        ("Ich", "ich", "PRON"), ("esse", "essen", "VERB"), ("Honig", "Honig", "NOUN"),
        ("auf", "auf", "ADP"), ("dem", "der", "DET"), ("Brot", "Brot", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (93, "Der Fluss ist breit.", [
        ("Der", "der", "DET"), ("Fluss", "Fluss", "NOUN"), ("ist", "sein", "AUX"),
        ("breit", "breit", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (94, "Sie trägt grüne Schuhe.", [
        ("Sie", "sie", "PRON"), ("trägt", "tragen", "VERB"), ("grüne", "grün", "ADJ"),
        ("Schuhe", "Schuh", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (95, "Das Café ist voll.", [
        ("Das", "der", "DET"), ("Café", "Café", "NOUN"), ("ist", "sein", "AUX"),
        ("voll", "voll", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (96, "Ich rufe meine Oma an.", [
        ("Ich", "ich", "PRON"), ("rufe", "anrufen", "VERB"), ("meine", "mein", "DET"),
        ("Oma", "Oma", "NOUN"), ("an", "an", "ADV"), (".", ".", "PUNCT"),
    ]),
    (97, "Der Laptop ist klein.", [
        ("Der", "der", "DET"), ("Laptop", "Laptop", "NOUN"), ("ist", "sein", "AUX"),
        ("klein", "klein", "ADJ"), (".", ".", "PUNCT"),
    ]),
    (98, "Wir trinken Tee, weil es kalt ist.", [
        ("Wir", "wir", "PRON"), ("trinken", "trinken", "VERB"), ("Tee", "Tee", "NOUN"),
        (",", ",", "PUNCT"), ("weil", "weil", "SCONJ"), ("es", "er", "PRON"),
        ("kalt", "kalt", "ADJ"), ("ist", "sein", "AUX"), (".", ".", "PUNCT"),
    ]),
    (99, "Sie wäscht das Geschirr.", [
        ("Sie", "sie", "PRON"), ("wäscht", "waschen", "VERB"), ("das", "der", "DET"),
        ("Geschirr", "Geschirr", "NOUN"), (".", ".", "PUNCT"),
    ]),
    (100, "Tschüss, wir sehen uns morgen.", [
        ("Tschüss", "tschüss", "INTJ"), (",", ",", "PUNCT"), ("wir", "wir", "PRON"),
        ("sehen", "sehen", "VERB"), ("uns", "sich", "PRON"), ("morgen", "morgen", "ADV"),
        (".", ".", "PUNCT"),
    ]),
]


def format_sentence(sent_num: int, text: str, tokens: list[tuple[str, str, str]]) -> str:
    lines = [
        f"# sent_id = de_a1_val_{sent_num:03d}",
        f"# text = {text}",
    ]
    for i, (form, lemma, upos) in enumerate(tokens, 1):
        lines.append(f"{i}\t{form}\t{lemma}\t{upos}\t_\t_\t_\t_\t_\t_")
    lines.append("")
    return "\n".join(lines)


def write_batches(batch_size: int = 25) -> None:
    assert len(SENTENCES) == 85, f"Expected 85 sentences, got {len(SENTENCES)}"
    assert SENTENCES[0][0] == 16 and SENTENCES[-1][0] == 100

    OUT.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    for start in range(0, len(SENTENCES), batch_size):
        batch = SENTENCES[start : start + batch_size]
        batch_text = "\n".join(format_sentence(n, text, toks) for n, text, toks in batch)
        if batch_text:
            chunks.append(batch_text)
        print(f"Batch {start // batch_size + 1}: sentences {batch[0][0]:03d}–{batch[-1][0]:03d} ({len(batch)} sents)")

    OUT.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print(f"Wrote {len(SENTENCES)} sentences to {OUT}")


if __name__ == "__main__":
    write_batches()