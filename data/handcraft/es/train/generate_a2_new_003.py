"""Generate a2_new_003.conllu — es_a2_train_401 through es_a2_train_600."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import es_reflexive_lemma

# 8 batches × 25 sentences = 200 (A2: 5–12 tokens, weather / school / restaurant / hobbies / future / directions / tech / mixed)
SENTENCE_BATCHES: list[list[str]] = [
    # 401–425: Weather & seasons
    [
        "Llueve mucho en otoño aquí en mi ciudad.",
        "El cielo está despejado y azul hoy.",
        "Llevamos abrigos calientes en el invierno.",
        "Ella coge un paraguas cuando llueve mucho.",
        "El sol brilla mucho en verano aquí.",
        "Me gusta pasear bajo la lluvia de primavera.",
        "Hace viento en la costa esta tarde.",
        "Juegan fuera en los días soleados del fin de semana.",
        "La nieve cubre las montañas cada enero aquí.",
        "El pronóstico del tiempo anuncia lluvia mañana.",
        "Él tiene frío sin su bufanda gruesa hoy.",
        "Disfrutamos de los días calurosos en la playa.",
        "Las hojas se ponen marrones en octubre.",
        "Me quedo en casa cuando viene la tormenta.",
        "Ella abre la ventana en las noches calurosas.",
        "La niebla dificulta conducir por la mañana.",
        "Recogemos fruta del jardín en verano.",
        "La temperatura baja rápido por la noche.",
        "Él revisa la app del tiempo cada mañana.",
        "El hielo aparece en el lago en invierno.",
        "Llevo gafas de sol en los días soleados.",
        "Hacen un muñeco de nieve después de nevar.",
        "Aparece un arcoíris después de la lluvia fuerte.",
        "Secamos la ropa al sol en el balcón.",
        "Las flores de primavera florecen pronto en el parque.",
    ],
    # 426–450: School & learning
    [
        "Ella estudia matemáticas en el colegio cada día.",
        "Tenemos un examen el viernes por la mañana.",
        "Él escribe apuntes en su cuaderno de clase.",
        "Pregunto al profesor sobre los deberes hoy.",
        "Se sientan en la primera fila esta tarde.",
        "Ella aprende palabras nuevas en la clase de inglés.",
        "Leemos un cuento en la biblioteca del colegio.",
        "Él olvida su estuche de lápices en casa.",
        "Termino el proyecto de ciencias esta semana.",
        "Ella termina el examen con buena nota.",
        "Dibujamos mapas en la clase de geografía.",
        "Él escucha con atención la charla de historia.",
        "Practico la ortografía con mi compañero de clase.",
        "Ella se une al coro del colegio después de clase.",
        "Copiamos las respuestas de la pizarra del aula.",
        "Él prepara su mochila antes de sonar el timbre.",
        "Repaso para el examen cada tarde en casa.",
        "Ella explica la regla a su amiga del banco.",
        "Visitamos el museo con nuestra clase de arte.",
        "Él encuentra su aula en el segundo piso.",
        "Pido prestada una regla al profesor de matemáticas.",
        "Ella escribe su nombre en el papel del examen.",
        "Aplaudimos cuando termina la obra de teatro.",
        "Él pierde el autobús después del colegio hoy.",
        "Aprendo sobre animales en la clase de biología.",
    ],
    # 451–475: Restaurant & café
    [
        "Pedimos sopa y ensalada en el café del centro.",
        "Ella pide la cuenta después de cenar juntos.",
        "Reservo una mesa para cuatro personas esta noche.",
        "Él prueba el plato especial del día.",
        "Sirven pan fresco con la comida caliente.",
        "Bebo agua con limón en el almuerzo de hoy.",
        "Ella deja propina al camarero después de comer.",
        "Esperamos una mesa cerca de la ventana.",
        "Él se queja del café frío en el bar.",
        "Elijo pollo en lugar de pescado esta noche.",
        "Ella come postre después del plato principal.",
        "Compartimos una pizza en el restaurante italiano.",
        "Él lee el menú antes de pedir la comida.",
        "Recomiendo esta pasta a mi mejor amigo.",
        "Ella añade pimienta a su sopa de tomate.",
        "Dejamos una buena reseña para el chef del local.",
        "Él paga la cuenta con su tarjeta de crédito.",
        "Pruebo la salsa antes de servirla a los invitados.",
        "Ella pide té sin azúcar esta vez en el café.",
        "Nos sentamos fuera en la terraza soleada.",
        "Él devuelve el plato equivocado al camarero.",
        "Reservo el almuerzo en el nuevo restaurante italiano.",
        "Ella se seca las manos con una servilleta limpia.",
        "Celebramos su cumpleaños en el restaurante del centro.",
        "Él disfruta de la vista desde el café en la azotea.",
    ],
    # 476–500: Hobbies & free time
    [
        "Pinto cuadros en mi tiempo libre los domingos.",
        "Ella colecciona sellos de diferentes países del mundo.",
        "Jugamos juegos de mesa en las noches de lluvia.",
        "Él construye aviones de modelismo en el garaje.",
        "Leo cómics antes de dormir cada noche.",
        "Ella teje bufandas para los miembros de su familia.",
        "Vamos a pescar al lago los domingos por la mañana.",
        "Él hace fotos de pájaros en el bosque cercano.",
        "Aprendo canciones de guitarra con vídeos en internet.",
        "Ella dibuja cómics en su cuaderno de dibujo.",
        "Horneamos galletas con los niños esta tarde.",
        "Él ve programas de comedia en la tele esta noche.",
        "Me uno al club de ajedrez del centro cultural.",
        "Ella cultiva hierbas en el alféizar de la cocina.",
        "Pedaleamos junto al río el sábado por la mañana.",
        "Él repara bicis viejas como pasatiempo los fines de semana.",
        "Escribo relatos cortos en un cuaderno pequeño.",
        "Ella baila salsa con sus amigos cada semana.",
        "Acampamos en el bosque durante las vacaciones de verano.",
        "Él hace vídeos para su pequeño canal en línea.",
        "Visito galerías de arte en el centro de la ciudad.",
        "Ella cose botones en su chaqueta favorita rota.",
        "Escuchamos podcasts en los viajes largos en tren.",
        "Él resuelve crucigramas cada mañana con su café.",
        "Juego videojuegos con mi hermano pequeño los viernes.",
    ],
    # 501–525: Future (ir a + infinitive)
    [
        "Voy a visitar a mi tía mañana por la tarde.",
        "Ella va a estudiar en el extranjero el año que viene.",
        "Vamos a mudarnos de casa en el mes de junio.",
        "Él va a arreglar la estantería rota del salón.",
        "Van a abrir un pequeño café en la plaza.",
        "Voy a llamarte esta tarde después del trabajo.",
        "Ella va a aprender francés en el instituto el próximo curso.",
        "Vamos a cocinar pasta para la cena de hoy.",
        "Él va a comprar un portátil nuevo muy pronto.",
        "Voy a empezar a correr el próximo lunes por la mañana.",
        "Ella va a llevar un vestido azul esta noche.",
        "Vamos a coger el tren temprano de la mañana.",
        "Él va a pintar las paredes del dormitorio nuevo.",
        "Voy a ahorrar dinero para el viaje de verano.",
        "Ella va a quedar con su amiga al mediodía hoy.",
        "Vamos a plantar árboles en el jardín del colegio.",
        "Él va a limpiar el coche este fin de semana.",
        "Voy a enviar el paquete por correo esta semana.",
        "Ella va a practicar piano cada tarde en casa.",
        "Vamos a ver el partido en la tele esta noche.",
        "Él va a pedir ayuda a su jefe en la oficina.",
        "Voy a llegar antes de las ocho de la mañana.",
        "Ella va a cambiar su número de teléfono muy pronto.",
        "Vamos a invitarlos a comer el domingo al mediodía.",
        "Él va a terminar el informe antes del viernes.",
    ],
    # 526–550: Directions & transport
    [
        "Gira a la izquierda en el semáforo de delante.",
        "La parada de autobús está frente al banco.",
        "Cruzamos el puente a pie hasta la estación.",
        "Ella pregunta dónde está la oficina de correos.",
        "Sigue recto hasta que veas la iglesia del pueblo.",
        "La farmacia está al lado del supermercado grande.",
        "Cojo el metro hasta el centro de la ciudad.",
        "Él se baja tarde porque pierde su parada.",
        "Seguimos las señales hacia la salida del aeropuerto.",
        "Ella le muestra el camino en el mapa del móvil.",
        "El museo está detrás del ayuntamiento antiguo.",
        "Compro el billete en la máquina de fuera.",
        "Él aparca el coche cerca de la plaza del mercado.",
        "Bajamos por esta calle hasta el puerto del pueblo.",
        "Ella coge el tranvía en la plaza principal.",
        "La entrada del hotel está a tu derecha.",
        "Pido un taxi desde la estación de tren.",
        "Él lee el horario en la pared del andén.",
        "Montamos en patinetes por las calles estrechas del centro.",
        "Ella gira a la derecha después de la segunda esquina.",
        "El cine está al final de esta calle larga.",
        "Valido mi billete antes de subir al tren.",
        "Él espera en la esquina a que ponga verde.",
        "Pasamos por la biblioteca de camino a casa hoy.",
        "Ella encuentra la puerta correcta en el aeropuerto.",
    ],
    # 551–575: Technology & social media
    [
        "Descargo música en mi móvil esta noche.",
        "Ella publica fotos en su página de redes sociales.",
        "Él conecta su portátil al Wi-Fi del hotel.",
        "Vemos vídeos en línea durante el descanso del trabajo.",
        "Actualizo mis aplicaciones antes de empezar el viaje.",
        "Ella borra archivos viejos de su ordenador de casa.",
        "Él comparte un enlace con su equipo de trabajo.",
        "Charlamos en línea con amigos del extranjero.",
        "Hago una copia de mis fotos en la nube.",
        "Ella inicia sesión en su correo cada mañana.",
        "Él olvida la contraseña y la restablece rápido.",
        "Buscamos vuelos baratos en la página web.",
        "Respondo a los mensajes antes de salir de casa.",
        "Ella ve películas en su tableta por la noche.",
        "Él instala un juego nuevo en su consola.",
        "Seguimos las noticias en el móvil cada día.",
        "Imprimo los billetes desde la impresora de casa.",
        "Ella bloquea las llamadas spam en su móvil.",
        "Él escanea el código QR en la entrada del museo.",
        "Grabamos un vídeo corto para el blog de viajes.",
        "Cargo mi tableta antes del vuelo largo de mañana.",
        "Ella copia la dirección en la app de navegación.",
        "Él silencia el micrófono durante la llamada en línea.",
        "Subimos fotos de las vacaciones al álbum familiar.",
        "Apago las notificaciones cuando estudio en casa.",
    ],
    # 576–600: Mixed A2 topics
    [
        "Visito al dentista dos veces al año.",
        "Ella adopta un perro pequeño del refugio local.",
        "Alquilamos un piso cerca del campus universitario.",
        "Él arregla el grifo que gotea en el baño.",
        "Voto en las elecciones locales esta primavera.",
        "Ella dona ropa a la tienda benéfica del barrio.",
        "Reciclamos botellas de plástico cada semana en casa.",
        "Él planta flores a lo largo del camino del jardín.",
        "Hago voluntariado en el centro de animales los sábados.",
        "Ella solicita un trabajo de verano en el hotel.",
        "Asistimos a una boda en la iglesia del pueblo.",
        "Él alquila herramientas en la ferretería del barrio.",
        "Firmo el contrato en la oficina de la inmobiliaria.",
        "Ella llama a la puerta y espera fuera un rato.",
        "Bloqueamos las bicis fuera del pabellón deportivo.",
        "Él trae semillas para los pájaros del parque.",
        "Planco mi camisa antes de la entrevista de trabajo.",
        "Ella dobla el mapa y lo guarda en su bolso.",
        "Colgamos cuadros en la pared del salón.",
        "Él recoge las entradas en la taquilla del cine.",
        "Envuelvo el regalo en papel de colores brillantes.",
        "Ella riega el jardín después del día caluroso de hoy.",
        "Hacemos cola en silencio en el mostrador de correos.",
        "Él cede su asiento a un pasajero mayor en el autobús.",
        "Me siento agradecido por mis vecinos tan serviciales.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 401
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
    "Wi-Fi": "wi-fi",
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
    "plano": "planchar",
    "valido": "validar",
    "valida": "validar",
    "silencia": "silenciar",
    "restablece": "restablecer",
    "inicia": "iniciar",
    "llueve": "llover",
    "llueva": "llover",
    "nevar": "nevar",
    "neva": "nevar",
    "hace": "hacer",
    "gira": "girar",
    "sigue": "seguir",
    "apago": "apagar",
    "voto": "votar",
    "seca": "secar",
    "secamos": "secar",
    "secamar": "secar",
    "imprimo": "imprimir",
    "Imprimo": "imprimir",
    "copia": "copiar",
    "dona": "donar",
    "seco": "secar",
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

    if upos == "PROPN" and lemma and lemma.lower() in VERB_OVERRIDES:
        upos = "VERB"
        lemma = VERB_OVERRIDES[lemma.lower()]
    elif upos == "PROPN" and fl in VERB_OVERRIDES:
        upos = "VERB"
        lemma = VERB_OVERRIDES[fl]

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

    target_path = project_root / "data/handcraft/es/train/a2_new_003.conllu"
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

    print("STATUS: OK — 200 sentences, es_a2_train_401–es_a2_train_600")


if __name__ == "__main__":
    main()