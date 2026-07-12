"""Generate a2_new_002.conllu — es_a2_train_201 through es_a2_train_400."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import es_reflexive_lemma

# 8 batches × 25 sentences = 200 (A2: 5–12 tokens, sports / festivals / phone / bank / comparisons / reflexive / past / mixed)
SENTENCE_BATCHES: list[list[str]] = [
    # 201–225: Sports
    [
        "Yo juego al tenis todos los sábados por la mañana.",
        "Ella corre cinco kilómetros en el parque.",
        "Vemos el partido de fútbol en la tele.",
        "Nado en la piscina tres veces por semana.",
        "Entrenan duro antes del partido importante.",
        "Él marca un gol en el segundo tiempo.",
        "Ella se une a una clase de yoga.",
        "Animamos a nuestro equipo en el estadio.",
        "Monto en bicicleta por el camino del río.",
        "Gana la carrera por dos segundos.",
        "Ella estira las piernas antes de correr.",
        "Pasamos la pelota a nuestro compañero.",
        "Pierdo el partido pero me divierto mucho.",
        "Él chuta la pelota dentro de la red.",
        "Ella lleva zapatillas nuevas para la carrera.",
        "Practicamos baloncesto en el gimnasio del colegio.",
        "Atrapo la pelota con las dos manos.",
        "Él se lastima el tobillo durante el partido.",
        "Ella lanza la pelota a su amiga.",
        "Calentamos antes de empezar a correr.",
        "Yo salto a la comba en el jardín a veces.",
        "Él levanta pesas en el centro deportivo.",
        "Ella corre más rápido que los otros corredores.",
        "Descansamos en el banco después del partido.",
        "Me apunto al club de natación este otoño.",
    ],
    # 226–250: Festivals
    [
        "Celebramos la Navidad con toda la familia.",
        "Ella lleva un vestido colorido en el festival.",
        "Bailan en la calle durante el desfile.",
        "Como comida tradicional en la feria de verano.",
        "Él enciende las velas en la tarta de cumpleaños.",
        "Vemos fuegos artificiales junto al río esta noche.",
        "Ella aprende canciones con el coro del colegio.",
        "Decoramos la casa con luces brillantes.",
        "Yo regalo presentes a mis amigos en Pascua.",
        "Bebemos chocolate caliente en el mercado de invierno.",
        "Él toca música en el festival del pueblo.",
        "Ella compra regalos en el mercado navideño.",
        "Llevan máscaras en la fiesta de carnaval.",
        "Hago fotos del desfile en la calle.",
        "Aplaudimos cuando termina de tocar la banda.",
        "Él trae flores al festival de primavera.",
        "Ella hace farolillos para el evento de otoño.",
        "Compartimos tarta en la fiesta de bodas.",
        "Veo bailarines en la plaza del pueblo.",
        "Quedamos con amigos en el festival de música.",
        "Él cuenta chistes en la reunión familiar.",
        "Ella se pinta la cara para el desfile.",
        "Comen dulces en la fiesta de Halloween.",
        "Oigo tambores desde el escenario del festival.",
        "Nos quedamos despiertos hasta tarde en Nochevieja.",
    ],
    # 251–275: Phone calls
    [
        "Llamo a mi madre todos los domingos por la noche.",
        "Ella contesta el teléfono en el trabajo rápido.",
        "Él cuelga después de una conversación corta.",
        "Hablamos con el gerente por teléfono esta tarde.",
        "Dejo un mensaje en su buzón de voz.",
        "Ella marca el número de su lista de contactos.",
        "Él habla con el médico sobre su tos.",
        "Llamamos al hotel para reservar una habitación.",
        "Espero en espera durante cinco minutos largos.",
        "Ella devuelve la llamada después de leer el mensaje.",
        "Él pide ayuda en la línea de atención.",
        "Oímos tono de ocupado y volvemos a intentar.",
        "Cojo el teléfono en la cocina de casa.",
        "Ella habla más alto porque la línea es mala.",
        "Él olvida llamar a su amigo a tiempo.",
        "Hablamos del plan por teléfono esta noche.",
        "Guardo su nuevo número en mi móvil.",
        "Ella dice adiós y termina la llamada.",
        "Él pierde una llamada importante a la hora de comer.",
        "Charlamos con la abuela durante una hora.",
        "Le escribo un mensaje cuando no contesta.",
        "Ella rechaza la llamada durante la cena.",
        "Él explica el problema por teléfono con claridad.",
        "Acordamos una hora de reunión por teléfono.",
        "Yo cargo el móvil antes del viaje largo.",
    ],
    # 276–300: Bank
    [
        "Abro una cuenta bancaria en la sucursal local.",
        "Ella paga sus facturas en el banco hoy.",
        "Él saca dinero del cajero automático de fuera.",
        "Transferimos dinero a nuestra cuenta de ahorros.",
        "Consulto mi saldo en la aplicación del banco.",
        "Ella deposita su sueldo cada mes.",
        "Él firma el formulario en el mostrador del banco.",
        "Necesitamos un código PIN para la tarjeta.",
        "Pierdo mi tarjeta bancaria en el autobús.",
        "Ella pregunta por el préstamo en el mostrador.",
        "Él ahorra cincuenta euros cada semana.",
        "Pagamos el alquiler por transferencia bancaria mensual.",
        "Muestro mi documento en la oficina del banco.",
        "Ella cambia su dirección en el banco.",
        "Él recibe un recibo después del pago.",
        "Comparamos tipos en dos bancos diferentes.",
        "Pido dinero prestado para mi portátil nuevo.",
        "Ella cierra su cuenta antigua esta semana.",
        "Él cuenta las monedas en su cartera.",
        "Esperamos en la cola del banco concurrido.",
        "Escribo mi contraseña en la pantalla.",
        "Ella guarda su tarjeta en un lugar seguro.",
        "Él paga intereses por su préstamo pequeño.",
        "Recibimos un extracto por correo cada mes.",
        "Denuncio una tarjeta perdida en el banco.",
    ],
    # 301–325: Comparisons
    [
        "Este gimnasio es más grande que el antiguo.",
        "Ella juega al tenis mejor que su hermana.",
        "El festival era más ruidoso que el concierto.",
        "Él está tan en forma como su entrenador ahora.",
        "Hoy hace más frío que ayer por la tarde.",
        "Mi móvil es más nuevo que el tuyo.",
        "Ella habla más claro que él.",
        "El banco está más cerca que la oficina de correos.",
        "Él corre tan lejos como el puente cada día.",
        "Este equipo es más fuerte que el otro equipo.",
        "Duermo menos que mi compañero de piso.",
        "El parque es tan tranquilo como la biblioteca.",
        "Ella gana más dinero que su compañera de trabajo.",
        "Los deportes de invierno son más difíciles que los de verano.",
        "Él es más joven que su hermano mayor.",
        "La tarta es más dulce que el pan.",
        "Llegamos antes que ellos esta noche.",
        "Esta calle es más estrecha que la carretera principal.",
        "Ella baila tan bien como su profesora.",
        "El partido fue más emocionante que la película.",
        "Me siento más feliz que la semana pasada.",
        "Su coche es más rápido que mi bicicleta vieja.",
        "El mercado es tan lleno como el centro comercial.",
        "Ella lee más rápido que la mayoría de sus compañeros.",
        "Esta chaqueta es más cálida que esa camisa fina.",
    ],
    # 326–350: Reflexive verbs
    [
        "Me hago daño jugando al fútbol en el parque.",
        "Ella se mira en el espejo antes del espectáculo.",
        "Él se culpa por haber perdido el partido.",
        "Nos divertimos mucho en el festival de música.",
        "Aprendo palabras nuevas yo solo cada semana.",
        "Ella se compra un par de zapatos nuevos.",
        "Se sirven comida en la fiesta sin pedir.",
        "Él se recuerda que debe llamar al banco.",
        "Me lavo rápido antes de la reunión.",
        "Ella se prepara para la gran carrera.",
        "Nos presentamos a los vecinos nuevos del piso.",
        "Él se esfuerza más en cada entrenamiento.",
        "Me pregunto por qué falló el plan.",
        "Ella se mira en el espejo del baño.",
        "Se sientan cerca del escenario del festival.",
        "Él se encuentra solo en la parada de autobús.",
        "Me digo que debo mantener la calma.",
        "Ella se regala un café después de correr.",
        "Nos preparamos para el espectáculo de danza.",
        "Él se corta mientras corta las verduras.",
        "Me enorgullezco de mi buena letra.",
        "Ella se aleja de la multitud ruidosa.",
        "Se organizan en dos equipos pequeños.",
        "Él se supera con ejercicios más difíciles.",
        "Me recuerdo que debo ahorrar dinero cada mes.",
    ],
    # 351–375: Past tense
    [
        "Me apunté al club de tenis el mes pasado.",
        "Ella bailó toda la noche en el festival.",
        "Llamamos al banco para resolver el problema.",
        "Él ganó su primer partido de fútbol ayer.",
        "Visité el mercado navideño con mis amigos.",
        "Ella depositó el dinero en su cuenta nueva.",
        "Celebramos el cumpleaños de Ana el sábado.",
        "Él colgó el teléfono sin decir adiós.",
        "Compré entradas para el concierto del parque.",
        "Ella corrió cinco kilómetros en veinticinco minutos.",
        "Nos quedamos en casa durante la tormenta.",
        "Él sacó dinero del cajero antes de cenar.",
        "Perdí mi móvil en el estadio del partido.",
        "Ella decoró la casa para la fiesta navideña.",
        "Jugamos al baloncesto en el parque municipal.",
        "Él firmó el contrato en la oficina del banco.",
        "Recibí una llamada importante durante el almuerzo.",
        "Ella preparó una tarta para la fiesta familiar.",
        "Viajamos en tren al festival de música.",
        "Él olvidó su tarjeta en el mostrador del banco.",
        "Nadé en la piscina después del entrenamiento.",
        "Ella envió un mensaje a todos sus amigos.",
        "Comimos paella en la feria del pueblo.",
        "Él aprendió a tocar la guitarra el año pasado.",
        "Recordé la fecha del partido a última hora.",
    ],
    # 376–400: Mixed A2 topics
    [
        "Me uno a mis amigos para un picnic en mayo.",
        "Ella envía un correo al gerente del banco.",
        "Alquilamos bicicletas y pedaleamos por la costa.",
        "Él repara su raqueta antes del partido de tenis.",
        "Preparo bocadillos para el largo día del festival.",
        "Ella comparte su asiento en el autobús lleno.",
        "Planeamos un viaje corto para las vacaciones de primavera.",
        "Él escribe una nota corta para su profesor.",
        "Pido prestado un libro de deportes de la biblioteca.",
        "Ella encuentra un lugar tranquilo junto al río.",
        "Nos reímos juntos con las máscaras del carnaval.",
        "Él pide té y se sienta junto a la ventana.",
        "Devuelvo el material deportivo después de la clase.",
        "Ella aprende un baile nuevo para la feria del pueblo.",
        "Agradecemos al personal por su amable ayuda.",
        "Él guarda las entradas del festival en su cartera.",
        "Quedo con mi entrenador en el campo deportivo.",
        "Ella escucha con atención las instrucciones por teléfono.",
        "Elegimos un regalo en el puesto del mercado.",
        "Él sonríe cuando ve ganar a su equipo.",
        "Leo el programa del festival antes de salir.",
        "Ella espera con paciencia en la cola del banco.",
        "Volvemos a casa despacio después del concierto.",
        "Él sube el volumen de la radio un poco.",
        "Me siento orgulloso después de mi primera medalla.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 201
BATCH_SIZE = 25
MIN_TOKENS = 5
MAX_TOKENS = 12

SER_FORMS = {
    "es", "son", "era", "eran", "fue", "fui", "fuiste", "fueron", "sería", "serían",
    "será", "serán", "sido", "siendo", "sea", "seas", "sean", "fuese", "fuesen",
    "eres", "somos", "sois", "fui", "fuimos", "fuisteis", "soy",
}
ESTAR_FORMS = {
    "está", "están", "estaba", "estaban", "estuvo", "estuve", "estará", "estarán",
    "estén", "estemos", "estoy", "estás", "estamos", "estáis", "estado", "estando",
}
HABER_FORMS = {
    "ha", "han", "había", "habían", "hubo", "hubiera", "hubieran", "haya", "hayan",
    "he", "has", "hemos", "habéis", "habido", "habiendo",
}
HAY_FORMS = {"hay", "habrá", "habría", "habrían"}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "yo": ("yo", "PRON"), "Yo": ("yo", "PRON"),
    "tú": ("tú", "PRON"), "Tú": ("tú", "PRON"),
    "él": ("él", "PRON"), "Él": ("él", "PRON"),
    "ella": ("él", "PRON"), "Ella": ("él", "PRON"),
    "ellos": ("él", "PRON"), "ellas": ("él", "PRON"),
    "me": ("yo", "PRON"), "Me": ("yo", "PRON"),
    "te": ("tú", "PRON"),
    "ti": ("tú", "PRON"),
    "se": ("él", "PRON"),
    "nos": ("nosotros", "PRON"), "Nos": ("nosotros", "PRON"),
    "os": ("vosotros", "PRON"),
    "le": ("él", "PRON"), "les": ("él", "PRON"),
    "lo": ("él", "PRON"), "la": ("él", "PRON"), "los": ("él", "PRON"), "las": ("él", "PRON"),
    "nosotros": ("nosotros", "PRON"), "Nosotros": ("nosotros", "PRON"),
    "vosotros": ("vosotros", "PRON"),
    "usted": ("él", "PRON"), "ustedes": ("él", "PRON"),
    "quien": ("quien", "PRON"), "Quién": ("quien", "PRON"), "quién": ("quien", "PRON"),
    "cual": ("cual", "PRON"), "Cuál": ("cual", "PRON"), "cuál": ("cual", "PRON"),
    "que": ("que", "PRON"), "Qué": ("que", "PRON"), "qué": ("que", "PRON"),
    "algo": ("algo", "PRON"), "nada": ("nada", "PRON"), "alguien": ("alguien", "PRON"),
    "nadie": ("nadie", "PRON"),
    "el": ("el", "DET"), "la": ("el", "DET"), "los": ("el", "DET"), "las": ("el", "DET"),
    "El": ("el", "DET"), "La": ("el", "DET"), "Los": ("el", "DET"), "Las": ("el", "DET"),
    "un": ("uno", "DET"), "una": ("uno", "DET"), "unos": ("uno", "DET"), "unas": ("uno", "DET"),
    "Un": ("uno", "DET"), "Una": ("uno", "DET"),
    "mi": ("mío", "DET"), "mis": ("mío", "DET"), "Mi": ("mío", "DET"), "Mis": ("mío", "DET"),
    "tu": ("tuyo", "DET"), "tus": ("tuyo", "DET"),
    "su": ("suyo", "DET"), "sus": ("suyo", "DET"),
    "nuestro": ("nuestro", "DET"), "nuestra": ("nuestro", "DET"),
    "nuestros": ("nuestro", "DET"), "nuestras": ("nuestro", "DET"),
    "vuestro": ("vuestro", "DET"), "vuestra": ("vuestro", "DET"),
    "este": ("este", "DET"), "esta": ("este", "DET"), "estos": ("este", "DET"), "estas": ("este", "DET"),
    "ese": ("ese", "DET"), "esa": ("ese", "DET"), "esos": ("ese", "DET"), "esas": ("ese", "DET"),
    "aquel": ("aquel", "DET"), "aquella": ("aquel", "DET"),
    "mucho": ("mucho", "DET"), "muchos": ("mucho", "DET"), "mucha": ("mucho", "DET"), "muchas": ("mucho", "DET"),
    "poco": ("poco", "DET"), "pocos": ("poco", "DET"), "poca": ("poco", "DET"), "pocas": ("poco", "DET"),
    "todo": ("todo", "DET"), "toda": ("todo", "DET"), "todos": ("todo", "DET"), "todas": ("todo", "DET"),
    "cada": ("cada", "DET"), "Cada": ("cada", "DET"),
    "otro": ("otro", "DET"), "otra": ("otro", "DET"), "otros": ("otro", "DET"), "otras": ("otro", "DET"),
    "mismo": ("mismo", "DET"), "misma": ("mismo", "DET"), "mismos": ("mismo", "DET"), "mismas": ("mismo", "DET"),
    "del": ("del", "ADP"), "al": ("al", "ADP"),
    "de": ("de", "ADP"), "en": ("en", "ADP"), "a": ("a", "ADP"), "con": ("con", "ADP"),
    "por": ("por", "ADP"), "para": ("para", "ADP"), "sin": ("sin", "ADP"), "sobre": ("sobre", "ADP"),
    "entre": ("entre", "ADP"), "hasta": ("hasta", "ADP"), "desde": ("desde", "ADP"), "según": ("según", "ADP"),
    "ante": ("ante", "ADP"), "bajo": ("bajo", "ADP"), "tras": ("tras", "ADP"),
    "y": ("y", "CCONJ"), "o": ("o", "CCONJ"), "ni": ("ni", "CCONJ"), "pero": ("pero", "CCONJ"),
    "aunque": ("aunque", "SCONJ"), "Aunque": ("aunque", "SCONJ"),
    "si": ("si", "SCONJ"), "Si": ("si", "SCONJ"),
    "que": ("que", "SCONJ"),
    "como": ("como", "SCONJ"), "cuando": ("cuando", "SCONJ"), "mientras": ("mientras", "SCONJ"),
    "porque": ("porque", "SCONJ"),
    "donde": ("donde", "ADV"), "dónde": ("donde", "ADV"), "Dónde": ("donde", "ADV"),
    "adonde": ("adonde", "ADV"), "Adónde": ("adonde", "ADV"),
    "no": ("no", "ADV"), "sí": ("sí", "ADV"),
    "muy": ("muy", "ADV"), "más": ("más", "ADV"), "menos": ("menos", "ADV"),
    "ya": ("ya", "ADV"), "aún": ("aún", "ADV"), "aun": ("aun", "ADV"),
    "solo": ("solo", "ADV"), "sólo": ("solo", "ADV"),
    "también": ("también", "ADV"), "además": ("además", "ADV"),
    "hoy": ("hoy", "ADV"), "ayer": ("ayer", "ADV"), "mañana": ("mañana", "ADV"),
    "siempre": ("siempre", "ADV"), "nunca": ("nunca", "ADV"), "jamás": ("jamás", "ADV"),
    "aquí": ("aquí", "ADV"), "allí": ("allí", "ADV"), "ahora": ("ahora", "ADV"),
    "antes": ("antes", "ADV"), "después": ("después", "ADV"),
    "bien": ("bien", "ADV"), "mal": ("mal", "ADV"),
    "cómo": ("cómo", "ADV"), "por qué": ("por qué", "ADV"),
    "Lo": ("él", "PRON"),
}

SCONJ_WORDS = {
    "que", "aunque", "si", "como", "cuando", "mientras", "porque", "pues",
    "donde", "según", "aun", "sino",
}
CCONJ_WORDS = {"y", "o", "ni", "pero", "sino"}
ADP_WORDS = {
    "de", "en", "a", "con", "por", "para", "sin", "sobre", "entre", "hasta",
    "desde", "según", "ante", "bajo", "tras", "del", "al", "hacia", "mediante",
}

NOUN_OVERRIDES = {
    "tenis": "tenis",
}

VERB_OVERRIDES = {
    "ducha": "duchar",
    "envío": "enviar",
    "envio": "enviar",
    "verte": "ver",
    "gana": "ganar",
    "salto": "saltar",
    "regalo": "regalar",
    "cargo": "cargar",
    "oigo": "oir",
    "oímos": "oir",
    "oír": "oir",
    "cojo": "coger",
    "reímos": "reir",
    "reír": "reir",
    "sonríe": "sonreir",
    "sonreír": "sonreir",
}


def _aux_lemma(form: str) -> tuple[str, str]:
    fl = form.lower()
    if fl in SER_FORMS:
        return "ser", "AUX"
    if fl in ESTAR_FORMS:
        return "estar", "AUX"
    if fl in HABER_FORMS:
        return "haber", "AUX"
    if fl in HAY_FORMS:
        return "haber", "VERB"
    return fl, "AUX"


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    """Apply handcraft lemma/UPOS conventions per es_test.conllu."""
    if form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    fl = form.lower()

    aux_lemma, aux_upos = _aux_lemma(form)
    if fl in SER_FORMS | ESTAR_FORMS | HABER_FORMS:
        lemma, upos = aux_lemma, aux_upos
    elif fl in HAY_FORMS:
        lemma, upos = "haber", "VERB"

    if upos == "VERB":
        lemma = lemma.lower() if lemma else form.lower()
        reflexive = es_reflexive_lemma(lemma)
        if reflexive is not None:
            lemma = reflexive
        if lemma in VERB_OVERRIDES:
            lemma = VERB_OVERRIDES[lemma]
        elif fl in VERB_OVERRIDES:
            lemma = VERB_OVERRIDES[fl]
        if not any(lemma.endswith(e) for e in ("ar", "er", "ir", "se", "ír")):
            if fl.endswith(("ar", "er", "ir", "arse", "erse", "irse")):
                lemma = fl

    if upos == "NOUN" and lemma:
        if fl in NOUN_OVERRIDES:
            lemma = NOUN_OVERRIDES[fl]
        else:
            lemma = lemma.lower()
            if lemma.endswith("es") and fl.endswith("es") and len(lemma) > 3:
                lemma = lemma[:-2] if lemma.endswith("iones") else lemma[:-1]
            elif lemma.endswith("s") and fl.endswith("s") and not lemma.endswith("ss"):
                lemma = lemma[:-1]

    if upos == "ADJ" and lemma:
        lemma = lemma.lower()
        if lemma.endswith("mente"):
            pass
        elif lemma.endswith("os") and fl.endswith("os"):
            lemma = lemma[:-1] + "o" if lemma.endswith("ivos") else lemma[:-1]
        elif lemma.endswith("as") and fl.endswith("as"):
            lemma = lemma[:-1] + "o"
        elif lemma.endswith("a") and fl.endswith("a") and not lemma.endswith("ía"):
            lemma = lemma[:-1] + "o"

    if upos == "PROPN" and lemma:
        lemma = lemma[0].upper() + lemma[1:] if lemma else lemma

    if upos == "PUNCT":
        lemma = form

    if fl in {"del", "al"}:
        upos, lemma = "ADP", fl

    if fl in SCONJ_WORDS and not (fl == "como" and upos == "ADP"):
        if fl != "donde" or upos != "ADV":
            upos = "SCONJ"
            lemma = fl
    elif fl in CCONJ_WORDS:
        upos, lemma = "CCONJ", fl
    elif fl in ADP_WORDS:
        upos, lemma = "ADP", fl if fl in {"del", "al"} else fl

    if fl == "se" and upos in {"PRON", "PART"}:
        upos, lemma = "PRON", "él"

    if form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    return lemma, upos


def _reconstruct_text(forms: list[str]) -> str:
    punct_prefixes = ".,;:!?\"')"
    parts: list[str] = []
    for form in forms:
        if form and form[0] in punct_prefixes and parts:
            parts[-1] = parts[-1] + form
        else:
            parts.append(form)
    return " ".join(parts)


def _word_for_token(token, words_by_id: dict[int, object]):
    ids = token.id if isinstance(token.id, tuple) else (token.id,)
    return words_by_id.get(ids[0])


def count_tokens(sentence: str) -> int:
    return len(re.findall(r"[\w']+|[^\w\s]", sentence))


def main() -> None:
    import stanza

    for i, sent in enumerate(SENTENCES, START_ID):
        tc = count_tokens(sent)
        if tc < MIN_TOKENS or tc > MAX_TOKENS:
            print(f"PRE-CHECK es_a2_train_{i:03d}: {tc} tokens — {sent}")
            sys.exit(1)

    print("Loading Stanza...")
    stanza.download("es", processors="tokenize,mwt,pos,lemma", verbose=False)
    nlp = stanza.Pipeline(
        lang="es",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
    )

    target_path = project_root / "data/handcraft/es/train/a2_new_002.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES):
        assert len(batch) == BATCH_SIZE, f"Batch {batch_num} has {len(batch)} sentences"
        print(
            f"Processing batch {batch_num + 1}/8 "
            f"(sentences {START_ID + global_idx}–{START_ID + global_idx + BATCH_SIZE - 1})"
        )
        for sent in batch:
            sent_id = f"es_a2_train_{START_ID + global_idx:03d}"
            doc = nlp(sent)

            sent_forms: list[str] = []
            sent_rows: list[str] = []
            token_counter = 1

            for stanza_sent in doc.sentences:
                words_by_id = {
                    w.id: w for w in stanza_sent.words if isinstance(w.id, int)
                }
                for token in stanza_sent.tokens:
                    form = token.text
                    word = _word_for_token(token, words_by_id)
                    upos = word.upos if word and word.upos else "X"
                    lemma = word.lemma if word and word.lemma else form
                    lemma, upos = normalize_token(form, upos, lemma)

                    cols = [
                        str(token_counter),
                        form,
                        lemma,
                        upos,
                        "_", "_", "_", "_", "_", "_",
                    ]
                    sent_rows.append("\t".join(cols))
                    sent_forms.append(form)
                    token_counter += 1

            all_lines.append(f"# sent_id = {sent_id}")
            all_lines.append(f"# text = {_reconstruct_text(sent_forms)}")
            all_lines.extend(sent_rows)
            all_lines.append("")
            global_idx += 1

    conllu_text = "\n".join(all_lines) + "\n"
    target_path.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {global_idx} sentences to {target_path}")

    validation_res = validate_text(conllu_text)
    print(
        f"Validate: count={validation_res.sentence_count}, "
        f"tokens={validation_res.token_count}, passed={validation_res.passed}"
    )
    if not validation_res.passed:
        for err in validation_res.errors:
            print(f"  {err}")
        sys.exit(1)

    lemma_res = check_text(conllu_text, lang="es")
    print(f"Lemma check: passed={lemma_res.passed}")
    if not lemma_res.passed:
        for err in lemma_res.errors:
            print(f"  {err}")
        sys.exit(1)

    print("STATUS: OK — 200 sentences, es_a2_train_201–es_a2_train_400")


if __name__ == "__main__":
    main()