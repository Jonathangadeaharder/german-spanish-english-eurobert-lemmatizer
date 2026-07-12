"""Generate 90 handcrafted German C2 validation CoNLL-U sentences (de_c2_val_011–100)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.duplicate_detector import check_cross_file, check_text as check_dup_text
from lemmatizer.data.lemma_checker import check_text

BATCH_1 = [
    "Obschon die Metaphysik das Absolute zu fassen sucht, verbleibt sie, gleichwohl sie das Seiende kategorisiert, dem Abgrund verhaftet, der jede positive Bestimmung des Seins grundlegend untergräbt.",
    "Vermöge der Einkehr in die eigene Endlichkeit gelangt das Subjekt an jene Schwelle, jenseits deren die Reflexion in die Schwere des Schweigens übergeht, die kein Begriff mehr durchdringt.",
    "Wenngleich die Hermeneutik das Verstehen als Seinsmodus des Daseins begreift, entzieht sich ihr doch der Grund, aus dem heraus jede Auslegung je erst möglich wird.",
    "Unbeschadet der Überzeugung, dass die Vernunft alles durchdringen könne, hält die Phänomenologie an der Einsicht fest, dass das Sein sich der totalen Ergriffenheit entzieht.",
    "Dass die Zeitlichkeit des Daseins in seiner Endlichkeit gründet, obschon das Denken das Absolute zeitlos zu denken beansprucht, zeugt von der Tragweite ontologischer Untersuchung.",
    "Die Kritik der reinen Vernunft, die die Grenzen des Erkenntnisvermögens absteckt, vermag, gleichwohl sie die Kategorien durchmustert, das Ding an sich niemals unmittelbar zu erfassen.",
    "Obgleich das Bewusstsein die Welt als seine Vorstellung begreift, entzieht sich ihm das Seiende, sobald es dieses zum Gegenstand reflexiver Analyse macht.",
    "Der Abgrund der Befindlichkeit, der in der Angst aufbricht, macht das Dasein zum Ort, an dem es sein eigenes Sein allererst gewahren lässt, obgleich es nie ganz versteht.",
    "Dass die Geschichte des Seins in den Werken der Metaphysiker aufgeschrieben liegt, gehört zu den Einsichten, die ein Denken des Seins zu bedenken hat.",
    "Gleichwohl vermag die Dichtung, das Unaussprechliche zur Erscheinung zu bringen, ohne es in Begriffe zu zwingen, was die Ontologie vor jene Grenze stellt.",
]

BATCH_2 = [
    "Obschon die Erkenntnistheorie den Gegenstand der Wahrnehmung zu sichern sucht, bleibt sie doch, gleichwohl sie die Sinnesdaten ordnet, dem Rätsel verhaftet, das das Subjekt selbst ausmacht.",
    "Vermöge der Rückkehr zum Ursprung des Denkens gelangt der Philosoph an jene Stätte, jenseits deren das Fragen in das Staunen umschlägt, das jede begriffliche Festlegung sprengt.",
    "Wenngleich die Dialektik den Widerspruch als Motor des Geistes begreift, zeitigt sie jene Befindlichkeit der Unruhe, die das Denken vergeblich in der Ruhe des Systems zu bergen sucht.",
    "Unbeschadet der Behauptung, dass die Logik die Form des Denkens ausmache, hält die Phänomenologie an der Möglichkeit fest, das Sein vor jeder Prädikation zu fassen.",
    "Dass das Dasein sich seiner Geworfenheit bewusst ist, obschon es den Grund dieser Geworfenheit nie ganz erreicht, gehört zu den Wesenszügen, die es in seiner Faktizität ausmachen.",
    "Die Ontologie, die das Seiende in seinen Kategorien erfasst, vermag, gleichwohl sie das Allgemeine im Besonderen aufzuweisen trachtet, das Sein selbst niemals ganz zu begreifen.",
    "Obgleich die Transzendentalphilosophie die Bedingungen der Möglichkeit erforscht, entzieht sie der Konstitution, die das Bewusstsein beansprucht, indem es die Welt zum Korrelat macht.",
    "Der Wesenszug der Erinnerung, der darin liegt, dass sie das Vergangene gegenwärtigt, macht sie zum Ort, an dem das Dasein seine Vergangenheit allererst zu verstehen bekommt.",
    "Dass das Gedankengut der Aufklärung, das dem Dogmatismus entsprungen war, jene Befindlichkeit der Selbstgewissheit zeitigte, gehört zu den Einsichten, die eine Kritik der Vernunft zu bedenken hat.",
    "Gleichwohl vermag die Musik, das Unbenannte zur Erscheinung zu bringen, ohne es begrifflich zu fixieren, was die Ästhetik vor jene Aufgabe stellt, die sie nie ganz löst.",
]

BATCH_3 = [
    "Obschon die Ethik das Sittliche zu normieren sucht, bleibt sie doch, gleichwohl sie die Maximen prüft, dem Abgrund verhaftet, der das Gewissen in seiner Unhintergehbarkeit umgibt.",
    "Vermöge der Besinnung auf das Eigene gelangt der Mensch, der der Sorge anheimgefallen ist, das Ganze zu ahnen, an jene Grenze, jenseits derer das Denken in die Stille verstummt.",
    "Wenngleich die Geschichtsphilosophie den Sinn der Weltgeschichte zu enträtseln sucht, zeitigt sie jene Befindlichkeit der Entfremdung, die der Mensch vergeblich in Fortschritt zu verbergen sucht.",
    "Unbeschadet der Tatsache, dass jede Wissenschaft an die Bedingungen empirischer Methode gebunden bleibt, hält die Philosophie an der Möglichkeit fest, das Wesen des Seins zu treffen.",
    "Dass die Sprache sich ihrer selbst als Medium vergewissert, obschon sie den Grund dieser Selbstverständigung nie ganz erreicht, gehört zu den Wesenszügen, die das Dasein in seiner Mitseinlichkeit ausmachen.",
    "Die Fundamentalontologie, die das Dasein in seiner Alltäglichkeit trifft, vermag, gleichwohl sie das Existenziale im Konkreten aufzuweisen trachtet, die Seinsfrage niemals ganz zu beantworten.",
    "Obgleich die Subjektphilosophie das Ich zur Gewissheit erhebt, entzieht es sich der Reflexion, die das Denken beansprucht, indem es das Selbst zum Gegenstand einer inneren Wahrnehmung macht.",
    "Der Wesenszug der Stille, der darin liegt, dass sie das Gesagte übersteigt, macht sie zum Ort, an dem das Dasein sein eigenes Schweigen allererst zu hören bekommt, obgleich es nie ganz vernimmt.",
    "Dass das Gedankengut der Romantik, das der Aufklärung entgegentrat, jene Befindlichkeit der Sehnsucht nach dem Unendlichen zeitigte, gehört zu den Einsichten, die eine Phänomenologie der Stimmung zu bedenken hat.",
    "Gleichwohl vermag die Malerei, das Unsichtbare zur Erscheinung zu bringen, ohne es in Worte zu fassen, was die Ontologie des Bildes vor jene Schwierigkeit stellt, die sie nie ganz überwindet.",
]

BATCH_4 = [
    "Obschon die Naturphilosophie das Absolute in der Natur zu erkennen sucht, bleibt sie doch, gleichwohl sie die Erscheinungen ordnet, dem Geheimnis verhaftet, das das Lebendige in seiner Eigenheit umgibt.",
    "Vermöge der Einkehr in die Vergangenheit gelangt das Gedächtnis an jene Schwelle, jenseits deren die Erinnerung in das Vergessen umschlägt, das jede bewusste Rekonstruktion sprengt.",
    "Wenngleich die Religionsphilosophie das Heilige zu begreifen sucht, zeitigt sie jene Befindlichkeit der Ehrfurcht, die der Mensch vergeblich in rationaler Argumentation zu erfassen sucht.",
    "Unbeschadet der Behauptung, dass die Mathematik die Sprache der Natur sei, hält die Metaphysik an der Möglichkeit fest, das Sein jenseits quantitativer Bestimmung zu denken.",
    "Dass der Tod das Dasein in seiner Endlichkeit bestimmt, obschon das Denken die Vergänglichkeit zu verleugnen sucht, gehört zu den Einsichten, die eine existenziale Analytik zu bedenken hat.",
    "Die Phänomenologie der Lebenswelt, die das Alltägliche in seiner Selbstverständlichkeit trifft, vermag, gleichwohl sie das Vertraute im Fremden aufzuweisen trachtet, die Welt niemals ganz zu durchdringen.",
    "Obgleich die Erkenntnis des Menschen je die seine ist, entzieht sie der Verfügung, die die Wissenschaft beansprucht, indem sie sie zum Gegenstand einer objektivierenden Untersuchung macht.",
    "Der Wesenszug der Nähe, der darin liegt, dass sie das Ferne in sich birgt, macht sie zum Ort, an dem das Dasein sein eigenes Da allererst zu erfahren bekommt, obgleich es nie ganz ankommt.",
    "Dass das Gedankengut des Idealismus, das dem Sensualismus entgegentrat, jene Befindlichkeit der Innerlichkeit zeitigte, gehört zu den Einsichten, die eine Phänomenologie des Geistes zu bedenken hat.",
    "Gleichwohl vermag die Tragödie, das Unabwendbare zur Erscheinung zu bringen, ohne es moralisch zu deuten, was die Ontologie des Schicksals vor jene Aufgabe stellt, die sie nie ganz löst.",
]

BATCH_5 = [
    "Obschon die Sozialphilosophie das Gemeinschaftliche zu fassen sucht, bleibt sie doch, gleichwohl sie die Institutionen analysiert, dem Rätsel verhaftet, das das Mitsein in seiner Grundlage umgibt.",
    "Vermöge der Hinwendung zum Anderen gelangt die Ethik an jene Grenze, jenseits deren das Verstehen in die Anerkennung umschlägt, die kein Konzept mehr erschöpft.",
    "Wenngleich die Ästhetik das Schöne zu definieren sucht, zeitigt sie jene Befindlichkeit der Rührung, die der Betrachter vergeblich in ästhetischen Kategorien zu fassen sucht.",
    "Unbeschadet der Tatsache, dass jede Erkenntnis an die Bedingungen sprachlicher Ausdrückbarkeit gebunden bleibt, hält die Philosophie an der Möglichkeit fest, das Unsagbare zu erfahren.",
    "Dass die Freiheit des Willens sich ihrer selbst vergewissert, obschon sie den Grund dieser Selbstbestimmung nie ganz erreicht, gehört zu den Wesenszügen, die das Dasein in seiner Möglichkeit ausmachen.",
    "Die Ontologie des Raumes, die das Seiende in seinen räumlichen Bezügen erfasst, vermag, gleichwohl sie das Örtliche im Weiten aufzuweisen trachtet, den Raum selbst niemals ganz zu begreifen.",
    "Obgleich die Intentionalität des Bewusstseins je auf etwas gerichtet ist, entzieht sich ihr das Noema, sobald die Phänomenologie es zum Gegenstand einer reduktiven Analyse macht.",
    "Der Wesenszug der Ferne, der darin liegt, dass sie die Nähe erst ermöglicht, macht ihn zum Ort, an dem das Dasein seine Entfernung allererst zu messen bekommt, obgleich es nie ganz misst.",
    "Dass das Gedankengut der Moderne, das der Tradition entsprungen war, jene Befindlichkeit der Zerrissenheit zeitigte, gehört zu den Einsichten, die eine Diagnose der Zeit zu bedenken hat.",
    "Gleichwohl vermag die Architektur, das Unbewohnte zur Erscheinung zu bringen, ohne es funktional zu reduzieren, was die Ontologie des Bauens vor jene Frage stellt, die sie nie ganz beantwortet.",
]

BATCH_6 = [
    "Obschon die Wissenschaftstheorie die Objektivität zu sichern sucht, bleibt sie doch, gleichwohl sie die Methoden prüft, dem Abgrund verhaftet, der die Paradigmen in ihrer Geschichtlichkeit umgibt.",
    "Vermöge der Rückbesinnung auf die Quellen des Denkens gelangt der Forscher an jene Stätte, jenseits deren die Erkenntnis in die Skepsis umschlägt, die jede Gewissheit erschüttert.",
    "Wenngleich die Geschichtsontologie das Sein in der Zeit zu fassen sucht, zeitigt sie jene Befindlichkeit der Vergänglichkeit, die der Mensch vergeblich in bleibende Werke zu überwinden sucht.",
    "Unbeschadet der Überzeugung, dass die Analytische Philosophie die Sprache klären könne, hält die Phänomenologie an der Einsicht fest, dass das Sein der Sprache vorgängig ist.",
    "Dass die Sorge das Dasein in seiner Zukunft bestimmt, obschon das Denken die Zukunft in Gegenwart zu ziehen sucht, gehört zu den Wesenszügen, die es in seiner Vorlaufigkeit ausmachen.",
    "Die Metaphysik des Schönen, die das Ästhetische in seinen Kategorien erfasst, vermag, gleichwohl sie das Erhabene im Anmutigen aufzuweisen trachtet, die Schönheit selbst niemals ganz zu fassen.",
    "Obgleich die Philosophie des Geistes das Bewusstsein zu erklären sucht, entzieht es sich der Naturalisierung, die die Neurowissenschaft beansprucht, indem sie es zum Gegenstand kausaler Erklärung macht.",
    "Der Wesenszug der Dunkelheit, der darin liegt, dass sie das Licht erst sichtbar macht, macht sie zum Ort, an dem das Dasein seine Blindheit allererst zu erkennen bekommt, obgleich es nie ganz erkennt.",
    "Dass das Gedankengut der Postmoderne, das dem Humanismus entgegentrat, jene Befindlichkeit der Dekonstruktion zeitigte, gehört zu den Einsichten, die eine Kritik der Metanarrative zu bedenken hat.",
    "Gleichwohl vermag der Tanz, das Unbewegte zur Erscheinung zu bringen, ohne es in Statik zu erstarren, was die Ontologie der Bewegung vor jene Aufgabe stellt, die sie nie ganz löst.",
]

BATCH_7 = [
    "Obschon die Rechtsphilosophie das Gerechte zu bestimmen sucht, bleibt sie doch, gleichwohl sie die Normen systematisiert, dem Abgrund verhaftet, der die Gerechtigkeit in ihrer Unabschließbarkeit umgibt.",
    "Vermöge der Einkehr in die eigenen Voraussetzungen gelangt die Wissenschaft an jene Grenze, jenseits deren die Theorie in die Paradoxie umschlägt, die kein Axiom mehr löst.",
    "Wenngleich die Philosophie der Technik das Machbare zu begreifen sucht, zeitigt sie jene Befindlichkeit der Entfremdung, die der Mensch vergeblich in technischer Beherrschung zu überwinden sucht.",
    "Unbeschadet der Tatsache, dass jede Ontologie an die Bedingungen endlichen Denkens gebunden bleibt, hält die Metaphysik an der Möglichkeit fest, das Absolute in seiner Unendlichkeit zu denken.",
    "Dass die Angst das Dasein in seiner Geworfenheit offenbart, obschon es den Grund dieser Offenbarung nie ganz erreicht, gehört zu den Wesenszügen, die es in seiner Existenzialität ausmachen.",
    "Die Phänomenologie der Wahrnehmung, die das Wahrgenommene in seiner Gegebenheit trifft, vermag, gleichwohl sie das Noematische im Noetischen aufzuweisen trachtet, die Wahrnehmung niemals ganz zu durchleuchten.",
    "Obgleich die Philosophie der Geschichte den Sinn des Geschehens zu enträtseln sucht, entzieht er sich der Deutung, die das Denken beansprucht, indem es die Vergangenheit zum Objekt teleologischer Konstruktion macht.",
    "Der Wesenszug der Schwelle, der darin liegt, dass sie den Übergang ermöglicht, macht sie zum Ort, an dem das Dasein seine Grenze allererst zu überschreiten bekommt, obgleich es nie ganz überschreitet.",
    "Dass das Gedankengut der Stoa, das dem Epikureismus entgegentrat, jene Befindlichkeit der Gelassenheit zeitigte, gehört zu den Einsichten, die eine Ethik der Tugend zu bedenken hat.",
    "Gleichwohl vermag die Skulptur, das Unberührbare zur Erscheinung zu bringen, ohne es taktil zu vereinnahmen, was die Ontologie des Körpers vor jene Schwierigkeit stellt, die sie nie ganz überwindet.",
]

BATCH_8 = [
    "Obschon die Erkenntnisphilosophie die Wahrheit zu definieren sucht, bleibt sie doch, gleichwohl sie die Korrespondenztheorien prüft, dem Abgrund verhaftet, der die Wahrheit in ihrer Unverborgenheit umgibt.",
    "Vermöge der Hinwendung zum Sein des Seienden gelangt der Denker an jene Stätte, jenseits deren das Fragen in die Verstummung umschlägt, die jede ontologische Antwort verweigert.",
    "Wenngleich die Philosophie der Religion das Transzendente zu fassen sucht, zeitigt sie jene Befindlichkeit der Demut, die der Gläubige vergeblich in dogmatischer Gewissheit zu bergen sucht.",
    "Unbeschadet der Behauptung, dass die Phänomenologie die Sachen selbst zu beschreiben vermag, hält die Ontologie an der Möglichkeit fest, das Sein vor jeder Beschreibung zu erfahren.",
    "Dass die Verfallenheit das Dasein in seiner Alltäglichkeit bestimmt, obschon es die Eigentlichkeit zu erreichen sucht, gehört zu den Einsichten, die eine existenziale Analytik zu bedenken hat.",
    "Die Ontologie der Zeit, die das Seiende in seiner Zeitlichkeit erfasst, vermag, gleichwohl sie das Vergangene im Gegenwärtigen aufzuweisen trachtet, die Zeit selbst niemals ganz zu begreifen.",
    "Obgleich die Philosophie des Absurden die Sinnlosigkeit zu konstatieren sucht, entzieht sich ihr das Erleben, sobald sie es zum Gegenstand einer literarischen Darstellung macht.",
    "Der Wesenszug der Distanz, der darin liegt, dass sie die Beziehung erst ermöglicht, macht sie zum Ort, an dem das Dasein seine Entfremdung allererst zu spüren bekommt, obgleich es nie ganz spürt.",
    "Dass das Gedankengut der Scholastik, das der Mystik entsprungen war, jene Befindlichkeit der Kontemplation zeitigte, gehört zu den Einsichten, die eine Theologie der Vernunft zu bedenken hat.",
    "Gleichwohl vermag die Lyrik, das Unausdrückbare zur Erscheinung zu bringen, ohne es prosaisch zu entstellen, was die Ontologie des Wortes vor jene Aufgabe stellt, die sie nie ganz löst.",
]

BATCH_9 = [
    "Obschon die Philosophie der Natur das Lebendige zu begreifen sucht, bleibt sie doch, gleichwohl sie die Organismen klassifiziert, dem Geheimnis verhaftet, das das Leben in seiner Autonomie umgibt.",
    "Vermöge der Einkehr in die Sprache gelangt das Denken an jene Schwelle, jenseits deren das Reden in das Verstummen umschlägt, das jede semantische Festlegung sprengt.",
    "Wenngleich die Philosophie der Geschichte die Vergangenheit zu bewältigen sucht, zeitigt sie jene Befindlichkeit der Melancholie, die der Mensch vergeblich in historischem Wissen zu überwinden sucht.",
    "Unbeschadet der Tatsache, dass jede Metaphysik an die Bedingungen sprachlicher Artikulation gebunden bleibt, hält die Philosophie an der Möglichkeit fest, das Schweigen des Seins zu vernehmen.",
    "Dass die Eigentlichkeit das Dasein in seiner Möglichkeit bestimmt, obschon es die Verfallenheit zu überwinden sucht, gehört zu den Wesenszügen, die es in seiner Seinsweise ausmachen.",
    "Die Phänomenologie der Stimmung, die das Befindliche in seiner Unbestimmtheit trifft, vermag, gleichwohl sie das Stimmungsmäßige im Seienden aufzuweisen trachtet, die Stimmung niemals ganz zu fassen.",
    "Obgleich die Philosophie der Kunst das Schönerische zu erklären sucht, entzieht es sich der Ästhetik, die das Denken beansprucht, indem es das Kunstwerk zum Gegenstand theoretischer Deutung macht.",
    "Der Wesenszug der Heimat, der darin liegt, dass sie die Fremde in sich birgt, macht sie zum Ort, an dem das Dasein seine Zugehörigkeit allererst zu erfahren bekommt, obgleich es nie ganz ankommt.",
    "Dass das Gedankengut des Pragmatismus, das dem Rationalismus entgegentrat, jene Befindlichkeit der Handlungsorientierung zeitigte, gehört zu den Einsichten, die eine Philosophie der Praxis zu bedenken hat.",
    "Gleichwohl vermag die Oper, das Unhörbare zur Erscheinung zu bringen, ohne es in Noten zu fixieren, was die Ontologie des Klanges vor jene Grenze stellt, die sie, obschon sie es anheischig macht, nie ganz überwindet.",
]

BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5, BATCH_6, BATCH_7, BATCH_8, BATCH_9]
SENTENCES = [s for batch in BATCHES for s in batch]

assert len(SENTENCES) == 90, f"Expected 90 sentences, got {len(SENTENCES)}"

import stanza

print("Loading Stanza...")
nlp = stanza.Pipeline(lang="de", processors="tokenize,mwt,pos,lemma", use_gpu=False)

MODALS = {"müssen", "können", "wollen", "sollen", "dürfen", "mögen", "möchte"}
AUX_LEMMAS = {"sein", "haben", "werden"}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "mich": ("ich", "PRON"),
    "dich": ("du", "PRON"),
    "sich": ("sich", "PRON"),
    "uns": ("wir", "PRON"),
    "euch": ("ihr", "PRON"),
    "mir": ("ich", "PRON"),
    "dir": ("du", "PRON"),
    "ihm": ("er", "PRON"),
    "ihnen": ("sie", "PRON"),
    "Ihnen": ("Sie", "PRON"),
    "Sie": ("Sie", "PRON"),
    "mein": ("mein", "DET"),
    "meine": ("mein", "DET"),
    "meinen": ("mein", "DET"),
    "meinem": ("mein", "DET"),
    "meiner": ("mein", "DET"),
    "meines": ("mein", "DET"),
    "dein": ("dein", "DET"),
    "deine": ("dein", "DET"),
    "deinen": ("dein", "DET"),
    "deinem": ("dein", "DET"),
    "deiner": ("dein", "DET"),
    "deines": ("dein", "DET"),
    "sein": ("sein", "DET"),
    "seine": ("sein", "DET"),
    "seinen": ("sein", "DET"),
    "seinem": ("sein", "DET"),
    "seiner": ("sein", "DET"),
    "seines": ("sein", "DET"),
    "ihr": ("ihr", "DET"),
    "ihre": ("ihr", "DET"),
    "ihren": ("ihr", "DET"),
    "ihrem": ("ihr", "DET"),
    "ihrer": ("ihr", "DET"),
    "ihres": ("ihr", "DET"),
    "unser": ("unser", "DET"),
    "unsere": ("unser", "DET"),
    "unseren": ("unser", "DET"),
    "unserem": ("unser", "DET"),
    "unserer": ("unser", "DET"),
    "unseres": ("unser", "DET"),
    "der": ("der", "DET"),
    "die": ("der", "DET"),
    "das": ("der", "DET"),
    "den": ("der", "DET"),
    "dem": ("der", "DET"),
    "des": ("der", "DET"),
    "ein": ("ein", "DET"),
    "eine": ("ein", "DET"),
    "einen": ("ein", "DET"),
    "einem": ("ein", "DET"),
    "einer": ("ein", "DET"),
    "eines": ("ein", "DET"),
    "jegliche": ("jeglicher", "DET"),
    "jeglicher": ("jeglicher", "DET"),
    "jegliches": ("jeglicher", "DET"),
    "jeglichen": ("jeglicher", "DET"),
    "jeglichem": ("jeglicher", "DET"),
    "dieser": ("dieser", "DET"),
    "diese": ("dieser", "DET"),
    "dieses": ("dieser", "DET"),
    "diesen": ("dieser", "DET"),
    "diesem": ("dieser", "DET"),
    "jener": ("jener", "DET"),
    "jene": ("jener", "DET"),
    "jenes": ("jener", "DET"),
    "jenen": ("jener", "DET"),
    "jenem": ("jener", "DET"),
    "Dessen": ("der", "PRON"),
    "dessen": ("der", "PRON"),
    "deren": ("der", "PRON"),
    "ins": ("in", "ADP"),
    "im": ("in", "ADP"),
    "am": ("an", "ADP"),
    "zum": ("zu", "ADP"),
    "zur": ("zu", "ADP"),
    "beim": ("bei", "ADP"),
    "vom": ("von", "ADP"),
    "Obschon": ("obschon", "SCONJ"),
    "obschon": ("obschon", "SCONJ"),
    "Obgleich": ("obgleich", "SCONJ"),
    "obgleich": ("obgleich", "SCONJ"),
    "Wenngleich": ("wenngleich", "SCONJ"),
    "wenngleich": ("wenngleich", "SCONJ"),
    "Gleichwohl": ("gleichwohl", "ADV"),
    "gleichwohl": ("gleichwohl", "ADV"),
    "Wiewohl": ("wiewohl", "SCONJ"),
    "wiewohl": ("wiewohl", "SCONJ"),
    "Ungeachtet": ("ungeachtet", "ADP"),
    "ungeachtet": ("ungeachtet", "ADP"),
    "Unbeschadet": ("unbeschadet", "ADP"),
    "unbeschadet": ("unbeschadet", "ADP"),
    "Vermöge": ("vermöge", "ADP"),
    "vermöge": ("vermöge", "ADP"),
    "jenseits": ("jenseits", "ADP"),
}

AUX_FORMS = {
    "ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen", "sei", "seien",
    "habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt", "hätt",
    "wird", "wurde", "werden", "würde", "würden", "geworden", "worden",
}

SCONJ_FORMS = {
    "weil", "dass", "obwohl", "wenn", "ob", "da", "als", "wie", "während", "sofern", "falls",
    "bevor", "nachdem", "sodass", "damit", "indem", "bevor",
}

CCONJ_FORMS = {"und", "oder", "aber", "sondern", "sowie"}
ADP_FORMS = {
    "um", "für", "mit", "von", "bei", "nach", "aus", "über", "vor", "unter", "durch",
    "ohne", "gegen", "wegen", "in", "an", "auf", "trotz", "gegenüber", "entlang",
}
PART_FORMS = {"nicht", "ja", "nein", "doch", "nur", "auch", "schon", "noch", "bloß", "allein", "dennoch"}
PRON_FORMS = {"ich", "du", "er", "sie", "es", "wir", "man", "wer", "was", "etwas", "nichts"}

CONTRACTION_PAIRS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("in", "dem"): ("im", "in", "ADP"),
    ("an", "dem"): ("am", "an", "ADP"),
    ("zu", "dem"): ("zum", "zu", "ADP"),
    ("zu", "der"): ("zur", "zu", "ADP"),
    ("von", "dem"): ("vom", "von", "ADP"),
    ("bei", "dem"): ("beim", "bei", "ADP"),
    ("in", "das"): ("ins", "in", "ADP"),
    ("an", "das"): ("ans", "an", "ADP"),
}


def simple_tokenize(sentence: str) -> list[str]:
    forms: list[str] = []
    for word in sentence.split():
        match = re.match(r"^(.+?)([.,;:!?]+)$", word)
        if match:
            forms.append(match.group(1))
            forms.extend(list(match.group(2)))
        else:
            forms.append(word)
    return forms


def align_stanza_to_text(
    sentence: str, stanza_words: list
) -> list[tuple[str, str, str]]:
    expected = simple_tokenize(sentence)
    aligned: list[tuple[str, str, str]] = []
    si = 0
    for exp in expected:
        if si >= len(stanza_words):
            raise ValueError(f"Stanza tokens exhausted for '{exp}' in: {sentence}")

        sw = stanza_words[si]
        if sw.text == exp:
            aligned.append((exp, sw.upos or "X", sw.lemma or exp))
            si += 1
            continue

        if si + 1 < len(stanza_words):
            pair = (stanza_words[si].text.lower(), stanza_words[si + 1].text.lower())
            if pair in CONTRACTION_PAIRS:
                form, lemma, upos = CONTRACTION_PAIRS[pair]
                if form == exp:
                    aligned.append((form, upos, lemma))
                    si += 2
                    continue

        aligned.append((exp, sw.upos or "X", sw.lemma or exp))
        si += 1

    if si != len(stanza_words):
        raise ValueError(
            f"Unconsumed Stanza tokens {si}/{len(stanza_words)} in: {sentence}"
        )
    return aligned


VERB_IRREGULAR: dict[str, str] = {
    "vermag": "vermögen",
    "vermögen": "vermögen",
    "vermochte": "vermögen",
    "vermochten": "vermögen",
    "bedarf": "bedürfen",
    "bedürfen": "bedürfen",
    "gilt": "gelten",
    "gelten": "gelten",
    "galt": "gelten",
    "galten": "gelten",
    "entspricht": "entsprechen",
    "entsprechen": "entsprechen",
    "entsprach": "entsprechen",
    "stieß": "stoßen",
    "stießen": "stoßen",
    "traf": "treffen",
    "trafen": "treffen",
    "gab": "geben",
    "gaben": "geben",
    "blieb": "bleiben",
    "blieben": "bleiben",
    "zog": "ziehen",
    "zogen": "ziehen",
    "hielt": "halten",
    "hielten": "halten",
    "ließ": "lassen",
    "ließen": "lassen",
    "wies": "weisen",
    "wiesen": "weisen",
    "führt": "führen",
    "führte": "führen",
    "führten": "führen",
    "steht": "stehen",
    "stand": "stehen",
    "standen": "stehen",
    "fällt": "fallen",
    "fiel": "fallen",
    "fielen": "fallen",
    "lässt": "lassen",
    "läßt": "lassen",
    "zieht": "ziehen",
    "ziehen": "ziehen",
    "schließt": "schließen",
    "schloss": "schließen",
    "schlossen": "schließen",
    "spricht": "sprechen",
    "sprach": "sprechen",
    "sprachen": "sprechen",
    "bricht": "brechen",
    "brach": "brechen",
    "brachen": "brechen",
    "hilft": "helfen",
    "half": "helfen",
    "halfen": "helfen",
    "nimmt": "nehmen",
    "nahm": "nehmen",
    "nahmen": "nehmen",
    "kommt": "kommen",
    "kam": "kommen",
    "kamen": "kommen",
    "findet": "finden",
    "fand": "finden",
    "fanden": "finden",
    "zeitigt": "zeitigen",
    "zeitigen": "zeitigen",
}


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    if form in SPECIAL_LEMMAS:
        return SPECIAL_LEMMAS[form]

    if form.lower() == "das" and upos == "PRON":
        return "der", "PRON"
    if form.lower() in {"die", "der", "den", "dem", "des"} and upos == "PRON":
        return "der", "PRON"

    if upos == "NOUN":
        lemma = lemma[0].upper() + lemma[1:] if lemma else lemma

    if upos == "VERB" or lemma in MODALS or form.lower() in MODALS:
        upos = "VERB"
        lemma = lemma.lower() if lemma else lemma
        if form in VERB_IRREGULAR:
            lemma = VERB_IRREGULAR[form]
        elif lemma in VERB_IRREGULAR:
            lemma = VERB_IRREGULAR[lemma]

    if form.lower() in AUX_FORMS:
        upos = "AUX"
        if form.lower() in {"ist", "war", "sind", "wäre", "wären", "bin", "bist", "seid", "gewesen", "sei", "seien"}:
            lemma = "sein"
        elif form.lower() in {"habe", "hat", "hatte", "hätte", "hätten", "haben", "habt", "gehabt"}:
            lemma = "haben"
        else:
            lemma = "werden"
    elif upos == "AUX" and lemma not in AUX_LEMMAS:
        upos = "VERB"

    if upos == "ADJ":
        lemma = lemma.lower() if lemma else lemma

    if upos == "PROPN":
        lemma = lemma[0].upper() + lemma[1:] if lemma else lemma

    if upos == "PUNCT":
        lemma = form

    low = form.lower()
    if low in SCONJ_FORMS:
        upos = "SCONJ"
        lemma = low
    elif low in CCONJ_FORMS:
        upos = "CCONJ"
        lemma = low
    elif low in ADP_FORMS:
        upos = "ADP"
        lemma = low
    elif low in PART_FORMS:
        upos = "PART"
        lemma = low
    elif low in PRON_FORMS:
        upos = "PRON"
        lemma = "Sie" if form == "Sie" else low

    if form in SPECIAL_LEMMAS:
        return SPECIAL_LEMMAS[form]

    return lemma, upos


def build_conllu(sentences: list[str], start_id: int, prefix: str = "de_c2_val") -> str:
    output_lines: list[str] = []

    for idx, sent in enumerate(sentences):
        sent_id = f"{prefix}_{start_id + idx:03d}"
        doc = nlp(sent)
        output_lines.append(f"# sent_id = {sent_id}")
        output_lines.append(f"# text = {sent}")

        stanza_words = [
            w for s in doc.sentences for w in s.words if isinstance(w.id, int)
        ]
        aligned = align_stanza_to_text(sent, stanza_words)

        token_counter = 1
        for form, upos, lemma in aligned:
            lemma, upos = normalize_token(form, upos, lemma)
            cols = [
                str(token_counter),
                form,
                lemma,
                upos,
                "_", "_", "_", "_", "_", "_",
            ]
            output_lines.append("\t".join(cols))
            token_counter += 1
        output_lines.append("")

    output_lines.append("")
    return "\n".join(output_lines)


def main() -> None:
    start_id = 11
    conllu_text = build_conllu(SENTENCES, start_id)

    target_path = project_root / "data/handcraft/de/val/c2_new_001.conllu"
    target_path.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {len(SENTENCES)} sentences to {target_path}")

    validation_res = validate_text(conllu_text)
    print(f"Validate: count={validation_res.sentence_count}, passed={validation_res.passed}")
    if not validation_res.passed:
        for err in validation_res.errors[:50]:
            print("  ", err)
        sys.exit(1)

    lemma_res = check_text(conllu_text, lang="de")
    print(f"Lemma check: passed={lemma_res.passed}")
    if not lemma_res.passed:
        for err in lemma_res.errors[:50]:
            print("  ", err)
        sys.exit(1)

    dup_res = check_dup_text(conllu_text)
    print(f"Duplicate check: passed={dup_res.passed}")
    if not dup_res.passed:
        for err in dup_res.duplicates[:20]:
            print("  ", err)
        sys.exit(1)

    existing_val = (project_root / "data/handcraft/de/val/c2.conllu").read_text(encoding="utf-8")
    cross_res = check_cross_file("", existing_val + "\n" + conllu_text)
    # check only new vs existing c2.conllu
    cross_res2 = check_cross_file(existing_val, conllu_text)
    print(f"Cross-file duplicate check (vs c2.conllu): passed={cross_res2.passed}")
    if not cross_res2.passed:
        for err in cross_res2.duplicates[:20]:
            print("  ", err)
        sys.exit(1)

    blocks = conllu_text.strip().split("\n\n")
    bad: list[tuple[str, int]] = []
    for block in blocks:
        if not block.startswith("#"):
            continue
        sid_m = re.search(r"sent_id = (\S+)", block)
        if not sid_m:
            continue
        sid = sid_m.group(1)
        n = sum(1 for line in block.split("\n") if line and not line.startswith("#"))
        if not (25 <= n <= 45):
            bad.append((sid, n))
    if bad:
        print(f"Token count violations: {len(bad)}")
        for sid, cnt in bad[:20]:
            print(f"  {sid}: {cnt} tokens")
        sys.exit(1)
    print("Token counts: all 25-45")

    print("Sent_ids: de_c2_val_011 – de_c2_val_100")
    print("Status: OK")


if __name__ == "__main__":
    main()