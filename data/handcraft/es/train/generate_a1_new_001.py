"""Generate a1_new_001.conllu — es_a1_train_001 through es_a1_train_200."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text, _reconstruct_text
from lemmatizer.data.lemma_checker import check_text

START_ID = 1
BATCH_SIZE = 25
STANZA_DIR = project_root / ".stanza_resources"

# 8 batches × 25 = 200 A1 sentences (3–8 tokens, present tense)
SENTENCE_BATCHES: list[list[str]] = [
    # 001–025: Greetings
    [
        "Hola, amigo mío.",
        "Buenos días, mamá.",
        "Buenas noches, papá.",
        "¿Cómo estás tú hoy?",
        "Estoy bien, gracias.",
        "Un placer conocer.",
        "Hasta luego, amigo.",
        "Adiós, nos vemos.",
        "¿Cómo está Tom hoy?",
        "Estoy feliz hoy.",
        "Bienvenido a casa.",
        "Que tengas buen día.",
        "Buenas tardes, señor.",
        "Hola, yo soy Ana.",
        "Hola, yo soy Ben.",
        "Buenas tardes, clase.",
        "Digo hola ahora.",
        "Ella dice adiós ahora.",
        "Decimos buenos días.",
        "Él es muy amable.",
        "Son mis amigos.",
        "Es agradable aquí.",
        "Di hola, por favor.",
        "Muchas gracias a ti.",
        "De nada, amigo.",
    ],
    # 026–050: Family
    [
        "Esta es mi madre.",
        "Ese es mi padre.",
        "Amo a mi hermana.",
        "Él ama a su hermano.",
        "Ella tiene un bebé.",
        "Tenemos dos hijos.",
        "Mi abuela es amable.",
        "Mi abuelo es alto.",
        "Su tía vive aquí.",
        "Su tío trabaja aquí.",
        "Llamo a mi mamá.",
        "Ella abraza a papá.",
        "El bebé es pequeño.",
        "Mi familia es grande.",
        "Tu hermano es joven.",
        "Nuestros padres están aquí.",
        "Su hijo tiene cinco años.",
        "Visito a mi abuela.",
        "Ella ayuda a mamá.",
        "Él juega con hermana.",
        "Comemos con familia.",
        "Mi primo está aquí.",
        "Su esposo es bueno.",
        "Su esposa cocina bien.",
        "Amo a mi familia.",
    ],
    # 051–075: Food
    [
        "Me gusta el té.",
        "A ella le gusta leche.",
        "Comemos pan fresco.",
        "Él come manzanas rojas.",
        "Bebo agua limpia.",
        "Ella bebe zumo naranja.",
        "La sopa está caliente.",
        "El arroz es blanco.",
        "Quiero un poco de pastel.",
        "¿Te gusta el pescado?",
        "No me gustan huevos.",
        "Ella cocina buena comida.",
        "Él compra fruta fresca.",
        "Tenemos pizza grande.",
        "Como una ensalada verde.",
        "El queso es amarillo.",
        "Mi almuerzo está listo.",
        "Ella hace café caliente.",
        "Quiero más agua.",
        "A él le gusta caramelo.",
        "Compartimos el pastel.",
        "La carne está buena.",
        "Pruebo la sopa hoy.",
        "Ella corta el pan.",
        "Comen arroz hoy.",
    ],
    # 076–100: Colors
    [
        "El cielo es azul.",
        "La hierba es verde.",
        "El sol es amarillo.",
        "El coche es rojo.",
        "El gato es negro.",
        "El perro es marrón.",
        "La nieve es blanca.",
        "La rosa es roja.",
        "Su vestido es morado.",
        "Su camisa es naranja.",
        "Me gustan ojos azules.",
        "Ella tiene zapatos verdes.",
        "Vemos un bus rojo.",
        "Él lleva sombrero negro.",
        "La pared es blanca.",
        "Mi bolsa es marrón.",
        "Tu abrigo es gris.",
        "El libro es azul.",
        "Lo pinto de verde.",
        "A ella le gustan flores rosas.",
        "La noche es oscura.",
        "El día es claro.",
        "Este color es bonito.",
        "¿Qué color es?",
        "Veo muchos colores.",
    ],
    # 101–125: Numbers
    [
        "Tengo un gato.",
        "Ella tiene dos perros.",
        "Vemos tres pájaros.",
        "Él tiene cuatro libros.",
        "Quiero cinco manzanas.",
        "Ella compra seis huevos.",
        "Queremos siete sillas.",
        "Él ve ocho coches.",
        "Cuento nueve estrellas.",
        "Ella tiene diez bolígrafos.",
        "Uno más uno son dos.",
        "Tengo diez años hoy.",
        "Él tiene seis años.",
        "Tenemos tres habitaciones.",
        "Ella lee cinco páginas.",
        "Veo dos coches rojos.",
        "Él come una manzana.",
        "Compramos cuatro plátanos.",
        "Ella encuentra tres llaves.",
        "Escribo dos palabras.",
        "Él tiene una hermana.",
        "Queremos dos tazas.",
        "Ella ve cuatro gatos.",
        "Me gusta el número siete.",
        "¿Cuántos gatos hay?",
    ],
    # 126–150: Daily routines
    [
        "Me levanto temprano.",
        "Ella se levanta tarde.",
        "Él se lava los dientes.",
        "Me lavo la cara.",
        "Desayunamos ahora.",
        "Ella va a la escuela.",
        "Él camina al trabajo.",
        "Leo mi libro hoy.",
        "Ella escribe una carta.",
        "Vemos la tele hoy.",
        "Él escucha música.",
        "Limpio mi cuarto.",
        "Ella hace la cama.",
        "Cocinamos la cena.",
        "Él toma una ducha.",
        "Me pongo los zapatos.",
        "Ella abre la puerta.",
        "Cerramos la ventana.",
        "Él se sienta aquí.",
        "Me paro en la puerta.",
        "Ella corre en parque.",
        "Jugamos en el jardín.",
        "Él duerme por noche.",
        "Trabajo cada día.",
        "Ella estudia en casa.",
    ],
    # 151–175: Simple statements
    [
        "El tiempo es cálido.",
        "Llueve mucho hoy.",
        "El viento es fuerte.",
        "Vivo en Madrid.",
        "Ella vive en París.",
        "Vamos al parque.",
        "Él juega con pelota.",
        "Me siento al sol.",
        "La tienda está abierta.",
        "La tienda está cerrada.",
        "Mi teléfono es nuevo.",
        "Su bolsa es vieja.",
        "El bus está aquí.",
        "El tren es rápido.",
        "Espero el bus hoy.",
        "Ella compra un billete.",
        "Nos vemos en escuela.",
        "Él habla con maestro.",
        "Aprendo palabras nuevas.",
        "Ella habla buen inglés.",
        "La clase empieza ahora.",
        "El examen es fácil.",
        "Mi cuarto es pequeño.",
        "Su casa es grande.",
        "Me siento muy bien.",
    ],
    # 176–200: Simple questions
    [
        "¿Dónde está mi bolsa?",
        "¿Cómo te llamas tú?",
        "¿Quién es tu maestro?",
        "¿Cómo estás hoy?",
        "¿Cuándo es tu cumpleaños?",
        "¿Por qué estás triste?",
        "¿Dónde vives tú?",
        "¿Qué quieres tú?",
        "¿Quién está en puerta?",
        "¿Cuántos años tienes?",
        "¿Qué es esta cosa?",
        "¿Dónde está el gato?",
        "¿Cuándo empieza clase?",
        "¿Por qué llega tarde?",
        "¿Quién es tu amigo?",
        "¿Cómo está el tiempo?",
        "¿Qué color es coche?",
        "¿Dónde están llaves?",
        "¿Cuándo comes tú?",
        "¿Por qué está feliz?",
        "¿Quién tiene libro?",
        "¿Cuántos perros hay?",
        "¿Qué hora es?",
        "¿Dónde está la tienda?",
        "¿Te gusta el café?",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

AUX_SER = {"soy", "eres", "es", "somos", "sois", "son", "sea", "seas", "seamos", "seáis", "sean"}
AUX_ESTAR = {"estoy", "estás", "está", "estamos", "estáis", "están"}
AUX_HABER = {"he", "has", "ha", "hemos", "habéis", "han", "hay", "había", "habían"}
AUX_PODER = {"puedo", "puedes", "puede", "podemos", "podéis", "pueden"}
AUX_FORMS = AUX_SER | AUX_ESTAR | AUX_HABER | AUX_PODER

PROPN_NAMES = {
    "Ana", "Ben", "Tom", "Madrid", "París", "Mamá", "Papá", "mamá", "papá",
}

CONTRACTIONS: dict[str, tuple[str, str]] = {
    "al": ("al", "ADP"),
    "del": ("del", "ADP"),
}

CONTRACTION_EXPANSIONS: dict[str, list[str]] = {
    "al": ["a", "el"],
    "del": ["de", "el"],
}

DET_FORMS = {"el", "la", "los", "las", "un", "una", "unos", "unas"}
DET_LEMMA_EL = {"el", "la", "los", "las", "El", "La", "Los", "Las"}
DET_LEMMA_UNO = {"un", "una", "unos", "unas", "Un", "Una", "Unos", "Unas"}
DET_LEMMA_MIO = {"mi", "mis", "Mi", "Mis"}
DET_LEMMA_TUYO = {"tu", "tus", "Tu", "Tus"}
DET_LEMMA_SUYO = {"su", "sus", "Su", "Sus"}
DET_LEMMA_NUESTRO = {"nuestro", "nuestra", "nuestros", "nuestras", "Nuestro", "Nuestra", "Nuestros", "Nuestras"}
DET_LEMMA_ESTE = {"este", "esta", "estos", "estas", "Este", "Esta", "Estos", "Estas"}
DET_LEMMA_ESE = {"ese", "esa", "esos", "esas", "Ese", "Esa", "Esos", "Esas"}
DET_LEMMA_TODO = {"todo", "toda", "todos", "todas", "Todo", "Toda", "Todos", "Todas"}
DET_LEMMA_MUCHO = {"mucho", "mucha", "muchos", "muchas", "Mucho", "Mucha", "Muchos", "Muchas"}

PRON_FORMS = {
    "yo", "Yo", "tú", "Tú", "él", "Él", "ella", "Ella", "nosotros", "Nosotros",
    "nosotras", "Nosotras", "vosotros", "vosotras", "ellos", "Ellos", "ellas", "Ellas",
    "me", "Me", "te", "Te", "nos", "Nos", "os", "se", "Se", "lo", "la", "le", "les",
    "mí", "ti", "usted", "ustedes",
}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "yo": ("yo", "PRON"),
    "Yo": ("yo", "PRON"),
    "tú": ("tú", "PRON"),
    "Tú": ("tú", "PRON"),
    "él": ("él", "PRON"),
    "Él": ("él", "PRON"),
    "ella": ("él", "PRON"),
    "Ella": ("él", "PRON"),
    "nosotros": ("nosotros", "PRON"),
    "Nosotros": ("nosotros", "PRON"),
    "nosotras": ("nosotros", "PRON"),
    "Nosotras": ("nosotros", "PRON"),
    "ellos": ("él", "PRON"),
    "Ellos": ("él", "PRON"),
    "ellas": ("él", "PRON"),
    "Ellas": ("él", "PRON"),
    "me": ("yo", "PRON"),
    "Me": ("yo", "PRON"),
    "te": ("tú", "PRON"),
    "Te": ("tú", "PRON"),
    "nos": ("nosotros", "PRON"),
    "Nos": ("nosotros", "PRON"),
    "se": ("él", "PRON"),
    "Se": ("él", "PRON"),
    "ti": ("tú", "PRON"),
    "mí": ("yo", "PRON"),
    "el": ("el", "DET"),
    "la": ("el", "DET"),
    "los": ("el", "DET"),
    "Las": ("el", "DET"),
    "las": ("el", "DET"),
    "El": ("el", "DET"),
    "La": ("el", "DET"),
    "Los": ("el", "DET"),
    "un": ("uno", "DET"),
    "una": ("uno", "DET"),
    "unos": ("uno", "DET"),
    "unas": ("uno", "DET"),
    "Un": ("uno", "DET"),
    "Una": ("uno", "DET"),
    "mi": ("mío", "DET"),
    "mis": ("mío", "DET"),
    "Mi": ("mío", "DET"),
    "Mis": ("mío", "DET"),
    "tu": ("tuyo", "DET"),
    "tus": ("tuyo", "DET"),
    "Tu": ("tuyo", "DET"),
    "Tus": ("tuyo", "DET"),
    "su": ("suyo", "DET"),
    "sus": ("suyo", "DET"),
    "Su": ("suyo", "DET"),
    "Sus": ("suyo", "DET"),
    "nuestro": ("nuestro", "DET"),
    "nuestra": ("nuestro", "DET"),
    "nuestros": ("nuestro", "DET"),
    "nuestras": ("nuestro", "DET"),
    "Nuestros": ("nuestro", "DET"),
    "este": ("este", "DET"),
    "esta": ("este", "DET"),
    "estos": ("este", "DET"),
    "estas": ("este", "DET"),
    "Este": ("este", "DET"),
    "Esta": ("este", "DET"),
    "ese": ("ese", "DET"),
    "esa": ("ese", "DET"),
    "Ese": ("ese", "DET"),
    "todo": ("todo", "DET"),
    "toda": ("todo", "DET"),
    "todos": ("todo", "DET"),
    "todas": ("todo", "DET"),
    "mucho": ("mucho", "DET"),
    "mucha": ("mucho", "DET"),
    "muchos": ("mucho", "DET"),
    "muchas": ("mucho", "DET"),
    "Muchas": ("mucho", "DET"),
    "más": ("más", "ADV"),
    "Más": ("más", "ADV"),
    "muy": ("muy", "ADV"),
    "Muy": ("muy", "ADV"),
    "hoy": ("hoy", "ADV"),
    "Hoy": ("hoy", "ADV"),
    "ahora": ("ahora", "ADV"),
    "Ahora": ("ahora", "ADV"),
    "aquí": ("aquí", "ADV"),
    "Aquí": ("aquí", "ADV"),
    "bien": ("bien", "ADV"),
    "Bien": ("bien", "ADV"),
    "tarde": ("tarde", "ADV"),
    "Tarde": ("tarde", "ADV"),
    "temprano": ("temprano", "ADV"),
    "Temprano": ("temprano", "ADV"),
    "siempre": ("siempre", "ADV"),
    "cada": ("cada", "DET"),
    "Cada": ("cada", "DET"),
    "no": ("no", "ADV"),
    "No": ("no", "ADV"),
    "donde": ("donde", "ADV"),
    "Dónde": ("donde", "ADV"),
    "cómo": ("cómo", "ADV"),
    "Cómo": ("cómo", "ADV"),
    "mío": ("mío", "DET"),
    "Mío": ("mío", "DET"),
    "cuando": ("cuando", "ADV"),
    "Cuándo": ("cuando", "ADV"),
    "quien": ("quien", "PRON"),
    "Quién": ("quien", "PRON"),
    "que": ("que", "PRON"),
    "Qué": ("que", "PRON"),
    "cuanto": ("cuanto", "DET"),
    "Cuántos": ("cuanto", "DET"),
    "por": ("por", "ADP"),
    "Por": ("por", "ADP"),
    "qué": ("que", "PRON"),
    "a": ("a", "ADP"),
    "A": ("a", "ADP"),
    "de": ("de", "ADP"),
    "De": ("de", "ADP"),
    "en": ("en", "ADP"),
    "En": ("en", "ADP"),
    "con": ("con", "ADP"),
    "Con": ("con", "ADP"),
    "al": ("al", "ADP"),
    "del": ("del", "ADP"),
    "y": ("y", "CCONJ"),
    "Y": ("y", "CCONJ"),
    "o": ("o", "CCONJ"),
    "es": ("ser", "AUX"),
    "Es": ("ser", "AUX"),
    "son": ("ser", "AUX"),
    "Son": ("ser", "AUX"),
    "soy": ("ser", "AUX"),
    "eres": ("ser", "AUX"),
    "estoy": ("estar", "AUX"),
    "Estoy": ("estar", "AUX"),
    "estás": ("estar", "AUX"),
    "está": ("estar", "AUX"),
    "están": ("estar", "AUX"),
    "hay": ("haber", "VERB"),
    "Hay": ("haber", "VERB"),
    "ha": ("haber", "AUX"),
    "han": ("haber", "AUX"),
    "bienvenido": ("bienvenido", "ADJ"),
    "Bienvenido": ("bienvenido", "ADJ"),
    "hola": ("hola", "INTJ"),
    "Hola": ("hola", "INTJ"),
    "adiós": ("adiós", "INTJ"),
    "Adiós": ("adiós", "INTJ"),
    "gracias": ("gracias", "NOUN"),
    "gracias.": ("gracias", "NOUN"),
    "¿": ("¿", "PUNCT"),
    "?": ("?", "PUNCT"),
    ".": (".", "PUNCT"),
    ",": (",", "PUNCT"),
}

NOUN_LEMMA_MAP = {
    "niños": "niño",
    "niñas": "niño",
    "niña": "niño",
    "días": "día",
    "día": "día",
    "años": "año",
    "hijos": "hijo",
    "hermanos": "hermano",
    "hermanas": "hermana",
    "abuelos": "abuelo",
    "manzanas": "manzana",
    "huevos": "huevo",
    "pájaros": "pájaro",
    "libros": "libro",
    "coches": "coche",
    "estrellas": "estrella",
    "bolígrafos": "bolígrafo",
    "habitaciones": "habitación",
    "páginas": "página",
    "plátanos": "plátano",
    "llaves": "llave",
    "palabras": "palabra",
    "tazas": "taza",
    "gatos": "gato",
    "perros": "perro",
    "flores": "flor",
    "ojos": "ojo",
    "zapatos": "zapato",
    "colores": "color",
    "dientes": "diente",
    "zapatos": "zapato",
    "amigos": "amigo",
}

ADJ_LEMMA_MAP = {
    "buenos": "bueno",
    "buenas": "bueno",
    "buen": "bueno",
    "bonita": "bonito",
    "bonitas": "bonito",
    "bonitos": "bonito",
    "roja": "rojo",
    "rojas": "rojo",
    "rojos": "rojo",
    "rojas": "rojo",
    "blanca": "blanco",
    "blancas": "blanco",
    "verdes": "verde",
    "azules": "azul",
    "amarillo": "amarillo",
    "naranja": "naranja",
    "morado": "morado",
    "marrón": "marrón",
    "gris": "gris",
    "oscura": "oscuro",
    "claro": "claro",
    "fresca": "fresco",
    "fresco": "fresco",
    "limpia": "limpio",
    "caliente": "caliente",
    "verde": "verde",
    "rojas": "rojo",
    "rosas": "rosa",
    "nuevas": "nuevo",
    "nuevo": "nuevo",
    "vieja": "viejo",
    "pequeño": "pequeño",
    "grandes": "grande",
    "grande": "grande",
    "amable": "amable",
    "agradable": "agradable",
    "feliz": "feliz",
    "joven": "joven",
    "alto": "alto",
    "cálido": "cálido",
    "fuerte": "fuerte",
    "abierta": "abierto",
    "cerrada": "cerrado",
    "rápido": "rápido",
    "fácil": "fácil",
    "triste": "triste",
}

VERB_LEMMA_MAP = {
    "estoy": "estar",
    "estás": "estar",
    "está": "estar",
    "están": "estar",
    "soy": "ser",
    "eres": "ser",
    "es": "ser",
    "son": "ser",
    "digo": "decir",
    "dice": "decir",
    "decimos": "decir",
    "di": "decir",
    "vemos": "ver",
    "vives": "vivir",
    "vivo": "vivir",
    "vive": "vivir",
    "vamos": "ir",
    "va": "ir",
    "vas": "ir",
    "tengo": "tener",
    "tienes": "tener",
    "tiene": "tener",
    "tenemos": "tener",
    "tengas": "tener",
    "gusta": "gustar",
    "gustan": "gustar",
    "llueve": "llover",
    "hace": "hacer",
    "hago": "hacer",
    "pongo": "poner",
    "siento": "sentar",
    "llamo": "llamar",
    "llamas": "llamar",
    "llama": "llamar",
    "puedo": "poder",
    "puedes": "poder",
    "quiero": "querer",
    "quieres": "querer",
    "como": "comer",
    "come": "comer",
    "comemos": "comer",
    "comen": "comer",
    "bebo": "beber",
    "bebe": "beber",
    "cocina": "cocinar",
    "compra": "comprar",
    "compramos": "comprar",
    "quiero": "querer",
    "pruebo": "probar",
    "corta": "cortar",
    "veo": "ver",
    "ve": "ver",
    "cuento": "contar",
    "lee": "leer",
    "escribo": "escribir",
    "escribe": "escribir",
    "encuentra": "encontrar",
    "levanto": "levantar",
    "levanta": "levantar",
    "lavo": "lavar",
    "lava": "lavar",
    "desayunamos": "desayunar",
    "camina": "caminar",
    "leo": "leer",
    "escucha": "escuchar",
    "limpio": "limpiar",
    "cocinamos": "cocinar",
    "toma": "tomar",
    "abre": "abrir",
    "cerramos": "cerrar",
    "sienta": "sentar",
    "paro": "parar",
    "corre": "correr",
    "jugamos": "jugar",
    "juega": "jugar",
    "duerme": "dormir",
    "trabajo": "trabajar",
    "trabaja": "trabajar",
    "estudia": "estudiar",
    "espero": "esperar",
    "habla": "hablar",
    "aprendo": "aprender",
    "empieza": "empezar",
    "amo": "amar",
    "ama": "amar",
    "abraza": "abrazar",
    "visito": "visitar",
    "ayuda": "ayudar",
    "lleva": "llevar",
    "pinto": "pintar",
    "compartimos": "compartir",
    "llamas": "llamar",
    "llega": "llegar",
    "comes": "comer",
    "hay": "haber",
}


def _strip_trailing_punct(text: str) -> str:
    return re.sub(r"[.,;:!?]+$", "", text)


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    if lemma and _strip_trailing_punct(lemma) != lemma and _strip_trailing_punct(form) == form:
        lemma = _strip_trailing_punct(lemma)

    if form in CONTRACTIONS:
        lemma, upos = CONTRACTIONS[form]
    elif form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    if form in PROPN_NAMES:
        upos = "PROPN"
        lemma = form[0].upper() + form[1:] if form[0].islower() else form

    lower = form.lower()

    if lower in AUX_FORMS:
        upos = "AUX" if lower not in {"hay"} else "VERB"
        if lower in AUX_SER:
            lemma = "ser"
            upos = "AUX"
        elif lower in AUX_ESTAR:
            lemma = "estar"
            upos = "AUX"
        elif lower in AUX_HABER:
            if lower == "hay":
                upos = "VERB"
            else:
                upos = "AUX"
            lemma = "haber"
        elif lower in AUX_PODER:
            lemma = "poder"
            upos = "VERB"
    elif upos in {"VERB", "AUX"}:
        upos = "VERB"
        lemma = lemma.lower() if lemma else lower
        if form in VERB_LEMMA_MAP:
            lemma = VERB_LEMMA_MAP[form]
        elif lower in VERB_LEMMA_MAP:
            lemma = VERB_LEMMA_MAP[lower]
        elif lemma in VERB_LEMMA_MAP:
            lemma = VERB_LEMMA_MAP[lemma]

    if upos == "NOUN":
        if form in NOUN_LEMMA_MAP:
            lemma = NOUN_LEMMA_MAP[form]
        elif lower in NOUN_LEMMA_MAP:
            lemma = NOUN_LEMMA_MAP[lower]
        elif lemma:
            lemma = lemma.lower()

    if upos == "ADJ":
        if form in ADJ_LEMMA_MAP:
            lemma = ADJ_LEMMA_MAP[form]
        elif lower in ADJ_LEMMA_MAP:
            lemma = ADJ_LEMMA_MAP[lower]
        elif lemma:
            lemma = lemma.lower()
            if lemma in ADJ_LEMMA_MAP:
                lemma = ADJ_LEMMA_MAP[lemma]

    if upos == "DET":
        if form in DET_LEMMA_EL or lower in {"el", "la", "los", "las"}:
            lemma = "el"
            upos = "DET"
        elif form in DET_LEMMA_UNO:
            lemma = "uno"
            upos = "DET"
        elif form in DET_LEMMA_MIO:
            lemma = "mío"
            upos = "DET"
        elif form in DET_LEMMA_TUYO:
            lemma = "tuyo"
            upos = "DET"
        elif form in DET_LEMMA_SUYO:
            lemma = "suyo"
            upos = "DET"
        elif form in DET_LEMMA_NUESTRO:
            lemma = "nuestro"
            upos = "DET"
        elif form in DET_LEMMA_ESTE:
            lemma = "este"
            upos = "DET"
        elif form in DET_LEMMA_ESE:
            lemma = "ese"
            upos = "DET"
        elif form in DET_LEMMA_TODO:
            lemma = "todo"
            upos = "DET"
        elif form in DET_LEMMA_MUCHO:
            lemma = "mucho"
            upos = "DET"
        elif lower in {"qué", "que"} and form[0].isupper():
            lemma = "que"
            upos = "PRON" if lower == "qué" and upos == "PRON" else upos

    if upos == "PRON" or form in PRON_FORMS:
        if form in {"Ella", "ella", "Ellas", "ellas"}:
            lemma = "él"
            upos = "PRON"
        elif form in {"Ellos", "ellos"}:
            lemma = "él"
            upos = "PRON"
        elif form in {"Nosotras", "nosotras"}:
            lemma = "nosotros"
            upos = "PRON"
        elif form in {"Me", "me"}:
            lemma = "yo"
            upos = "PRON"
        elif form in {"Te", "te"}:
            lemma = "tú"
            upos = "PRON"
        elif form in {"Nos", "nos"}:
            lemma = "nosotros"
            upos = "PRON"
        elif form in {"Se", "se"}:
            lemma = "él"
            upos = "PRON"

    if upos == "PROPN" and lemma:
        lemma = lemma[0].upper() + lemma[1:] if lemma else form

    if upos == "NUM":
        lemma = form.lower() if form.isdigit() else form

    if upos == "PUNCT":
        lemma = form

    if lower in {"por", "para", "con", "sin", "sobre", "entre", "desde", "hasta", "hacia"}:
        upos = "ADP"
        lemma = lower

    if lower in {"y", "o", "pero"}:
        upos = "CCONJ"
        lemma = lower

    if lower in {"no"}:
        upos = "ADV"
        lemma = "no"

    if form in {"Dónde", "dónde"}:
        upos = "ADV"
        lemma = "donde"
    if form in {"Cómo", "cómo"}:
        upos = "ADV"
        lemma = "cómo"
    if form in {"Cuándo", "cuándo"}:
        upos = "ADV"
        lemma = "cuando"
    if form in {"Quién", "quién"}:
        upos = "PRON"
        lemma = "quien"
    if form in {"Qué", "qué"} and upos in {"PRON", "DET", "X"}:
        upos = "PRON" if form == "Qué" and "es" not in form else upos
    if form in {"Cuántos", "cuántos", "Cuántas", "cuántas"}:
        upos = "DET"
        lemma = "cuanto"

    if form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    return lemma, upos


def tokenize_text(sentence: str) -> list[str]:
    tokens: list[str] = []
    for word in sentence.split():
        word_work = word
        while word_work and word_work[0] in "¿¡":
            tokens.append(word_work[0])
            word_work = word_work[1:]
        if not word_work:
            continue
        match = re.match(r"^(.+?)([.,;:!?]+)$", word_work)
        if match:
            tokens.append(match.group(1))
            tokens.extend(list(match.group(2)))
        else:
            tokens.append(word_work)
    return tokens


def stanza_words_flat(doc) -> list:
    words = []
    for stanza_sent in doc.sentences:
        for word in stanza_sent.words:
            if isinstance(word.id, int):
                words.append(word)
    return words


def match_expansion(form: str, words: list, start: int) -> int | None:
    key = form if form in CONTRACTION_EXPANSIONS else form.lower()
    if key not in CONTRACTION_EXPANSIONS:
        return None
    expansion = CONTRACTION_EXPANSIONS[key]
    if start + len(expansion) > len(words):
        return None
    for i, piece in enumerate(expansion):
        if _strip_trailing_punct(words[start + i].text).lower() != piece.lower():
            return None
    return start + len(expansion)


def align_tokens(sentence: str, words: list) -> list[tuple[str, str, str]]:
    aligned: list[tuple[str, str, str]] = []
    text_tokens = tokenize_text(sentence)
    wi = 0
    punct_chars = ".,;:!?¿¡"

    for form in text_tokens:
        if form in punct_chars:
            aligned.append((form, form, "PUNCT"))
            if wi < len(words) and words[wi].upos == "PUNCT" and words[wi].text == form:
                wi += 1
            continue

        if wi >= len(words):
            lemma, upos = normalize_token(form, "X", form)
            aligned.append((form, lemma, upos))
            continue

        end = match_expansion(form, words, wi)
        if end is not None:
            head = words[wi]
            lemma, upos = normalize_token(form, head.upos or "X", head.lemma or form)
            aligned.append((form, lemma, upos))
            wi = end
            continue

        head = words[wi]
        stanza_form = _strip_trailing_punct(head.text)
        stanza_lemma = head.lemma or stanza_form
        if stanza_form.lower() == form.lower() or form in PROPN_NAMES:
            lemma, upos = normalize_token(form, head.upos or "X", stanza_lemma)
        else:
            lemma, upos = normalize_token(form, "X", form)
        aligned.append((form, lemma, upos))
        wi += 1

    return aligned


def count_tokens(sentence: str) -> int:
    return len(tokenize_text(sentence))


for i, sent in enumerate(SENTENCES, start=1):
    n = count_tokens(sent)
    if n < 3 or n > 8:
        raise ValueError(f"Sentence {i} has {n} tokens (need 3–8): {sent}")


def main() -> None:
    import stanza

    os.environ["STANZA_RESOURCES_DIR"] = str(STANZA_DIR)
    STANZA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Stanza (es)...")
    stanza.download(
        "es",
        package="default_fast",
        processors="tokenize,mwt,pos,lemma",
        verbose=False,
    )
    nlp = stanza.Pipeline(
        lang="es",
        package="default_fast",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
        dir=str(STANZA_DIR),
    )

    target_path = project_root / "data/handcraft/es/train/a1_new_001.conllu"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    output_lines: list[str] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES):
        assert len(batch) == BATCH_SIZE, f"Batch {batch_num} has {len(batch)} sentences"
        print(
            f"Processing batch {batch_num + 1}/8 "
            f"(sentences {START_ID + global_idx}–{START_ID + global_idx + BATCH_SIZE - 1})"
        )
        for sent in batch:
            sent_id = f"es_a1_train_{START_ID + global_idx:03d}"
            doc = nlp(sent)
            words = stanza_words_flat(doc)
            aligned = align_tokens(sent, words)
            display_text = _reconstruct_text([form for form, _, _ in aligned])

            output_lines.append(f"# sent_id = {sent_id}")
            output_lines.append(f"# text = {display_text}")

            token_counter = 1
            for form, lemma, upos in aligned:
                cols = [
                    str(token_counter),
                    form,
                    lemma,
                    upos,
                    "_",
                    "_",
                    "_",
                    "_",
                    "_",
                    "_",
                ]
                output_lines.append("\t".join(cols))
                token_counter += 1

            output_lines.append("")
            global_idx += 1

    conllu_text = "\n".join(output_lines) + "\n"
    target_path.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {global_idx} sentences to {target_path}")

    validation_res = validate_text(conllu_text)
    print(f"Validate: count={validation_res.sentence_count}, passed={validation_res.passed}")
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


if __name__ == "__main__":
    main()