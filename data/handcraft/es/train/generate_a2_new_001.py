"""Generate a2_new_001.conllu — es_a2_train_001 through es_a2_train_200."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import es_reflexive_lemma

# 8 batches × 25 sentences = 200 (A2: 5–12 tokens, daily life / travel / health / past / modals)
SENTENCE_BATCHES: list[list[str]] = [
    # 001–025: Daily life — morning & home
    [
        "Me levanto a las siete cada mañana.",
        "Ella se cepilla los dientes antes del desayuno.",
        "Comemos tostadas con mermelada en el desayuno.",
        "Él se ducha después de hacer ejercicio.",
        "Mi madre hace café en la cocina.",
        "Los niños se visten rápido para la escuela.",
        "Siempre reviso mi correo por la mañana.",
        "Ella riega las plantas del balcón.",
        "Limpiamos la casa cada sábado por la mañana.",
        "Él alimenta al gato antes de salir.",
        "Pongo las llaves en mi bolso cada día.",
        "Ella cuelga el abrigo en el armario.",
        "Vemos las noticias después de cenar juntos.",
        "Él apaga las luces antes de dormir.",
        "Pongo la alarma para las seis de mañana.",
        "Ella dobla la ropa limpia en la cama.",
        "Abrimos las ventanas cuando hace calor.",
        "Él pasa la aspiradora en el salón los domingos.",
        "Lavamos los platos después de cada comida.",
        "Ella hace la cama antes de salir de casa.",
        "Sacamos la basura cada noche.",
        "Él arregla la silla rota de la cocina.",
        "Cierro la puerta con llave al salir.",
        "Ella escucha música mientras cocina la cena.",
        "Nos sentamos en el sofá y leemos libros.",
    ],
    # 026–050: Daily life — shopping & food
    [
        "Voy al supermercado los viernes.",
        "Ella compra fruta fresca en el mercado.",
        "Necesitamos leche, pan y huevos hoy.",
        "Él paga la compra con su tarjeta.",
        "Cocino pasta con salsa de tomate para cenar.",
        "Ella corta las verduras para la ensalada.",
        "Bebemos té con miel por la noche.",
        "Él hornea un pastel para el cumpleaños de su hermana.",
        "Pido una pizza para cenar esta noche.",
        "Ella prueba la sopa y añade más sal.",
        "Compartimos una pizza grande en el restaurante.",
        "Él corta el pan en trozos pequeños.",
        "Hierve los huevos durante diez minutos.",
        "Ella sirve la comida en platos azules.",
        "Terminamos la comida y pagamos la cuenta.",
        "Él pone la leche en la nevera rápidamente.",
        "Caliento la sopa en el microondas.",
        "Ella añade azúcar a su taza de café.",
        "Probamos una receta nueva del libro de cocina.",
        "Él come una manzana después del almuerzo.",
        "Preparo un bocadillo para el trabajo mañana.",
        "Ella lava la fruta antes de comerla.",
        "Compramos queso y mantequilla en la tienda.",
        "Él asa pollo para la cena familiar.",
        "Remuevo el arroz mientras se cocina.",
    ],
    # 051–075: Travel
    [
        "Llegamos al aeropuerto al mediodía.",
        "Hago la maleta para el viaje de mañana.",
        "Ella muestra su pasaporte en la frontera.",
        "Él compra un billete para el tren de la mañana.",
        "Esperamos en la parada de autobús cerca del hotel.",
        "Hago el check-in en la recepción del hotel.",
        "Ella pide indicaciones para llegar al museo.",
        "Él toma un taxi al centro de la ciudad.",
        "Visitamos el castillo antiguo en la colina.",
        "Reservo una habitación con vistas al mar.",
        "Ella hace fotos del puente famoso.",
        "Él pierde el autobús y va andando al trabajo.",
        "Esperamos en el semáforo antes de cruzar.",
        "Miro el mapa en mi teléfono móvil.",
        "Ella coge el tren a Madrid hoy.",
        "Él alquila un coche para el viaje del fin de semana.",
        "Exploramos el mercado del casco antiguo.",
        "Dejo mi bolsa en la recepción del hotel.",
        "Ella cambia dinero en el banco cercano.",
        "Él encuentra un hostal barato cerca de la estación.",
        "Tomamos el ferry a la isla pequeña.",
        "Envío una postal a mis padres.",
        "Ella embarca en el avión en la puerta doce.",
        "Él devuelve el coche de alquiler el lunes.",
        "Disfrutamos de la vista desde el balcón del hotel.",
    ],
    # 076–100: Health
    [
        "Me siento mal y me quedo en casa hoy.",
        "Ella tiene dolor de cabeza después de un día largo.",
        "Llamamos al médico para pedir una cita.",
        "Él toma medicina para el dolor de garganta.",
        "Descanso en la cama cuando tengo fiebre.",
        "Ella bebe té caliente para su resfriado.",
        "Vamos a la farmacia después de comer.",
        "Él se hace daño en la rodilla jugando al fútbol.",
        "Necesito ir al dentista esta semana.",
        "Ella lleva una venda en el dedo cortado.",
        "Comemos comida sana y bebemos mucha agua.",
        "Él duerme ocho horas cada noche.",
        "Hago ejercicio tres veces por semana en casa.",
        "Ella se siente mejor después de descansar bien.",
        "Nos lavamos las manos antes de cada comida.",
        "Él tose mucho durante la noche.",
        "Evito el azúcar cuando no me siento bien.",
        "Ella se toma la temperatura con un termómetro.",
        "Paseamos por el parque para tomar aire fresco.",
        "Él visita a la enfermera en la clínica del colegio.",
        "Estiro la espalda después de estar sentado mucho tiempo.",
        "Ella pone crema en su piel seca.",
        "Bebemos sopa cuando estamos enfermos.",
        "Él se recupera rápido después de la operación pequeña.",
        "Me siento cansado pero voy al trabajo.",
    ],
    # 101–125: Past tense
    [
        "Visité a mis abuelos el fin de semana pasado.",
        "Ella cocinó la cena para toda la familia.",
        "Caminamos al parque después del colegio.",
        "Él jugó al fútbol con sus amigos ayer.",
        "Vi una película en el cine.",
        "Ella limpió su habitación el sábado por la mañana.",
        "Nos quedamos en un hotel pequeño en Roma.",
        "Él llamó a su madre el domingo por la noche.",
        "Terminé los deberes antes de cenar.",
        "Ella compró un vestido nuevo en la tienda.",
        "Nos encontramos con amigos en el bar.",
        "Él aprendió diez palabras nuevas en clase.",
        "Perdí mi cartera en el autobús ayer.",
        "Ella escribió una carta a su tía.",
        "Bailamos en la fiesta toda la noche.",
        "Él ayudó a su padre a arreglar la bicicleta.",
        "Olvidé mi paraguas en el restaurante.",
        "Ella abrió la ventana porque hacía calor.",
        "Disfrutamos del concierto en el parque.",
        "Él cerró la tienda a las ocho.",
        "Recibí un paquete de mi hermana.",
        "Ella pintó la pared de su dormitorio.",
        "Viajamos en tren por todo el país.",
        "Él respondió todas las preguntas del examen.",
        "Recordé su nombre después de un momento.",
    ],
    # 126–150: Modals
    [
        "Puedo nadar pero no puedo bucear bien.",
        "Ella debe terminar su trabajo antes de las cinco.",
        "Deberíamos salir ahora o llegaremos tarde.",
        "Él visitará a su tío el mes que viene.",
        "Querría un vaso de agua, por favor.",
        "Puedes sentarte aquí si quieres.",
        "Ella debe tomar esta medicina dos veces al día.",
        "Podemos quedar en el café a las tres.",
        "Él debería llamar al médico por su tos.",
        "Te ayudaré a llevar esas bolsas pesadas.",
        "No pueden venir a la fiesta esta noche.",
        "Ella puede hablar inglés y un poco de francés.",
        "Debemos mostrar los billetes en la puerta.",
        "Él no estará en casa esta tarde.",
        "Puedo cocinar la cena si compras la comida.",
        "Deberías llevar abrigo porque hace frío.",
        "Ella empezará su nuevo trabajo el lunes.",
        "Podemos tomar el autobús al aeropuerto.",
        "Él debe devolver el libro a la biblioteca.",
        "Querría visitar España el próximo verano.",
        "Ella puede ayudarte a encontrar la calle correcta.",
        "Deberíamos reservar los billetes con antelación.",
        "Él traerá el postre a la cena.",
        "Puedo quedar contigo después del trabajo el jueves.",
        "Debes apagar el móvil en clase.",
    ],
    # 151–175: Comparisons
    [
        "Mi habitación es más pequeña que la tuya.",
        "Ella corre más rápido que su hermano mayor.",
        "Este hotel es más barato que el otro.",
        "Él es más alto que yo por dos centímetros.",
        "El tren es más rápido que el autobús hoy.",
        "Soy tan alto como mi hermana pequeña.",
        "Esta sopa es mejor que la ensalada.",
        "Ella baila tan bien como su profesora de música.",
        "La camisa azul es más bonita que la verde.",
        "Él trabaja más que su compañero de oficina.",
        "El invierno es más frío que el otoño aquí.",
        "Como menos azúcar que mi hermano.",
        "El mercado está más lleno que la tienda cercana.",
        "Ella es más joven de lo que pensé al principio.",
        "Esta calle es más larga que la siguiente.",
        "Él conduce con más cuidado que su amigo.",
        "El pastel sabe más dulce que las galletas.",
        "Estamos tan cansados como tú esta noche.",
        "Su bolso es mucho más pesado que el mío.",
        "El parque nuevo es más grande que el antiguo.",
        "Hablo inglés mejor que mi primo.",
        "Hoy hace más calor que ayer.",
        "Él es tan fuerte como su padre ahora.",
        "La película fue más aburrida que el libro.",
        "Ella camina tan rápido como su mejor amiga.",
    ],
    # 176–200: Mixed A2 topics
    [
        "Trabajo en una oficina cerca de la estación.",
        "Ella estudia inglés en la escuela de idiomas.",
        "Vivimos en un piso encima de la panadería.",
        "Él va en bicicleta al colegio cada día.",
        "Me gusta leer libros sobre viajes y comida.",
        "Ella trabaja a tiempo parcial en el café local.",
        "Planeamos un picnic en el parque el domingo.",
        "Él escucha la radio por la mañana.",
        "Envío un mensaje a mi amigo cada día.",
        "Ella lleva gafas cuando lee texto pequeño.",
        "Celebramos su cumpleaños con una fiesta pequeña.",
        "Él ahorra dinero para sus vacaciones en Grecia.",
        "Pido libros prestados de la biblioteca pública.",
        "Ella enseña matemáticas en el instituto.",
        "Invitamos a los vecinos a cenar el viernes.",
        "Él arregla su teléfono con ayuda de la tienda.",
        "Prefiero el té pero mi esposa prefiere el café.",
        "Ella cultiva flores en macetas en la terraza.",
        "Esperamos a que pare la lluvia fuera.",
        "Él toca la guitarra en una banda pequeña.",
        "Necesito una chaqueta nueva para el frío.",
        "Ella encuentra sus gafas bajo el periódico.",
        "Elegimos una mesa cerca de la ventana.",
        "Él vuelve a casa tarde después del partido.",
        "Espero verte de nuevo muy pronto.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 1
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

VERB_OVERRIDES = {
    "ducha": "duchar",
    "envío": "enviar",
    "envio": "enviar",
    "verte": "ver",
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

    target_path = project_root / "data/handcraft/es/train/a2_new_001.conllu"
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

    print("STATUS: OK — 200 sentences, es_a2_train_001–es_a2_train_200")


if __name__ == "__main__":
    main()