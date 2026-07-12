"""Generate 200 handcrafted Spanish C1 CoNLL-U sentences (es_c1_train_201–400)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import es_reflexive_lemma

BATCH_1 = [
    # Derecho constitucional y derechos fundamentales 201-225
    "El tribunal constitucional examinó si la delegación legislativa excedió los límites fijados por la ley de habilitación parlamentaria.",
    "El control de constitucionalidad garantiza que decretos ejecutivos respeten derechos fundamentales consagrados en la carta magna nacional.",
    "Aunque el federalismo distribuye competencias entre niveles de gobierno, surgen conflictos frecuentes sobre jurisdicción regulatoria concurrente en comercio.",
    "La separación de poderes limita el exceso legislativo al facultar tribunales independientes para invalidar normas inconstitucionales.",
    "Las reformas constitucionales requieren aprobación cualificada y deliberación prolongada para evitar revisiones apresuradas que debiliten la estabilidad institucional.",
    "Sigue siendo controvertido si los poderes de emergencia concedidos en crisis deben caducar automáticamente tras umbrales temporales predeterminados.",
    "La declaración de derechos garantiza equidad procesal mientras obliga al Estado a justificar proporcionalmente limitaciones a libertades civiles.",
    "Tras ratificar el convenio, el Estado incorporó normas internacionales de derechos humanos en marcos de adjudicación constitucional exigible.",
    "Académicos debatieron si la interpretación originalista aborda adecuadamente desarrollos tecnológicos imprevistos por redactores constitucionales del siglo XVIII.",
    "El defensor del pueblo impugnó órdenes de detención administrativa sin autorización judicial individual dentro de plazos constitucionalmente exigidos.",
    "Las cuestiones prejudiciales permiten que tribunales nacionales soliciten orientación autoritativa del tribunal supranacional sobre compatibilidad con tratados.",
    "No obstante la soberanía parlamentaria, ciertas normas jurídicas pueden vincular legislaturas futuras mediante garantías constitucionales reforzadas.",
    "El tribunal constitucional declaró que la demarcación electoral vulneró principios de sufragio igualitario consagrados en textos constitucionales fundacionales.",
    "Las cláusulas de supremacía federal resuelven conflictos entre legislación estatal y estatutos nacionales en dominios de competencia enumerada.",
    "El fiscal general defendió la constitucionalidad de la ley demostrando objetivos estrechamente ajustados servidos por medidas regulatorias impugnadas.",
    "Las convenciones constitucionales complementan disposiciones escritas orientando conducta ejecutiva cuando mandatos textuales formales permanecen deliberadamente ambiguos.",
    "El voto disidente advirtió que escrutinio deferente de leyes de seguridad erosionaba protección judicial de comunidades marginadas.",
    "Los gobiernos subnacionales invocaron garantías de autonomía constitucional para resistir estandarización centralizada de currículos educativos a nivel nacional.",
    "El recurso de amparo constitucional permite a particulares impugnar leyes directamente ante el máximo adjudicador constitucional del país.",
    "El análisis de proporcionalidad exige evaluar idoneidad, necesidad y gravedad equilibrada antes de restringir libertades expresivas constitucionalmente protegidas.",
    "La gran sala reexaminó el caso tras decisiones contradictorias que crearon incertidumbre doctrinal sobre derechos de privacidad en línea.",
    "Las cláusulas de identidad constitucional permiten a Estados miembros resistir armonización percibida como amenaza a tradiciones jurídicas fundacionales domésticas.",
    "El juicio político examinó si la conducta ejecutiva constituía delitos graves justificando destitución conforme a estándares constitucionales de remoción.",
    "Las comisiones de nombramiento judicial buscan despolitizar selección preservando rendición de cuentas democrática mediante audiencias de nominación transparentes.",
    "El acuerdo constitucional equilibró mecanismos de veto minoritario con reglas mayoritarias para evitar parálisis legislativa en sociedades pluralistas.",
]

BATCH_2 = [
    # Macroeconomía y finanzas 226-250
    "Los responsables monetarios señalaron disposición a mantener tipos restrictivos hasta que expectativas inflacionarias converjan sosteniblemente hacia objetivos oficiales.",
    "El multiplicador fiscal de inversión en infraestructura parece mayor en recesiones cuando capacidad ociosa y condiciones monetarias acomodaticias coexisten.",
    "Aunque el desempleo descendió, efectos de trabajadores desalentados y caída de participación complicaron interpretación de indicadores laborales principales.",
    "La independencia del banco central protege tipos de interés de presiones políticas que alimentarían expansiones insostenibles de demanda agregada.",
    "La curva de Phillips se debilitó cuando choques de oferta alteraron compensaciones históricas entre inflación y desempleo.",
    "Sigue debatiéndose si la flexibilización cuantitativa estimuló principalmente actividad real o infló valoraciones de activos favoreciendo desproporcionadamente hogares adinerados.",
    "Los déficits gemelos presupuestarios y de cuenta corriente reflejan desequilibrios estructurales que requieren programas coordinados de ajuste fiscal.",
    "La inversión de la curva de rendimientos precede históricamente recesiones cuando primas a plazo fluctúan impredeciblemente ante incertidumbre política.",
    "Tras acumular reservas precautorias, economías emergentes resistieron presiones de salida de capital mejor que en crisis financieras anteriores.",
    "Los reguladores macroprudenciales endurecieron ratios de crédito para frenar boom inmobiliario financiado con préstamos que amenazaba estabilidad del sistema.",
    "El ratio de deuda pública se estabilizó tras superávits fiscales primarios y crecimiento nominal favorable del producto interior.",
    "Las depreciaciones cambiarias se transmiten a precios al consumo según dinámicas de traspaso del tipo de cambio.",
    "La inflación subyacente de servicios persistió por crecimiento salarial en sectores con restricciones laborales pese a endurecimiento monetario agresivo.",
    "La revisión de cuentas nacionales incorporó medición mejorada de inversión intangible y aportaciones digitales al producto interior bruto.",
    "Los estabilizadores automáticos atenuaron severidad de la recesión expandiendo transferencias y reduciendo impuestos al contraerse ingreso agregado cíclicamente.",
    "La orientación forward guía expectativas comunicando trayectorias condicionales de política según evolución de datos de inflación y empleo publicados.",
    "La estimación de brecha sugirió holgura persistente pese a bajo desempleo, reflejando errores en medición del potencial productivo.",
    "Las primas de riesgo soberano se ampliaron ante temores de que pasivos estatales desestabilizaran sostenibilidad fiscal de mediano plazo.",
    "La moneda de reserva confería privilegio exorbitante, permitiendo endeudamiento barato pese a déficits crónicos de cuenta corriente.",
    "Escenarios de estanflación complican política cuando respuestas monetarias contractivas arriesgan amplificar desempleo sin contener inflación impulsada por oferta.",
    "La descomposición contable del crecimiento atribuyó desaceleración a menor productividad factorial total más que solo a vientos demográficos adversos.",
    "La fuga de capitales se aceleró tras anticiparse devaluación inminente e imposición de controles cambiarios por autoridades monetarias desesperadas.",
    "La regla de Taylor implicaba tipos elevados dado exceso inflacionario y producción superior al potencial estimado consistentemente.",
    "Los efectos de histeresis sugieren que recesiones prolongadas reducen producción potencial mediante atrofia de habilidades y menor creación empresarial.",
    "La restricción presupuestaria intertemporal exige deuda actual coherente con compromisos futuros creíbles de balance primario políticamente.",
]

BATCH_3 = [
    # Filosofía y ética aplicada 251-275
    "Los filósofos analíticos examinan si las proposiciones morales expresan actitudes o describen hechos objetivos verificables empíricamente.",
    "El dilema del tranvía ilustra tensiones entre consecuencialismo utilitario y deontología que prohíbe usar personas meramente como medios.",
    "Aunque el relativismo cultural reconoce diversidad normativa, críticos advierten que dificulta condenar prácticas gravemente lesivas universalmente.",
    "La ética de la virtud enfatiza disposiciones habituales cultivadas mediante práctica reflexiva más que cumplimiento de reglas abstractas.",
    "Es ampliamente debatido si el libre albedrío es compatible con determinismo causal sugerido por neurociencia cognitiva contemporánea.",
    "El imperativo categórico kantiano exige actuar solo según máximas que podrían universalizarse sin contradicción lógica interna.",
    "Tras revisar experimentos mentales, el comité reevaluó criterios de personalidad para atribuir derechos morales a entidades artificiales avanzadas.",
    "La paradoja del mentiroso socava teorías semánticas que equiparan significado con condiciones de verdad en lenguaje natural ordinario.",
    "Filósofos feministas cuestionan supuestos androcéntricos que invisibilizan experiencias de cuidado y trabajo reproductivo no remunerado doméstico.",
    "No obstante avances biomédicos, comités de ética debaten límites aceptables de mejora cognitiva y manipulación genética germinal humana.",
    "La distinción entre hecho y valor impide deducir prescripciones normativas directamente desde descripciones empíricas científicas aisladas.",
    "El contractualismo busca principios de justicia que ciudadanos racionales aceptarían en posición hipotética de equidad original imparcial.",
    "Sigue siendo polémico si animales no humanos poseen intereses morales fundamentales que imponen deberes directos de no causación.",
    "La fenomenología describe estructuras de experiencia vivida desafiando reduccionismos que ignoran cualidades subjetivas irreductibles de conciencia.",
    "Argumentos de superinteligencia artificial plantean riesgos existenciales si objetivos instrumentalmente racionales divergen de valores humanos.",
    "El principio de igual consideración de intereses rechaza discriminación arbitraria basada en especie al evaluar sufrimiento evitable comparable.",
    "Los escépticos morales desafían pretensiones de fundamentación objetiva sosteniendo que juicios éticos reflejan convenciones sociales contingentes históricamente.",
    "Tras confrontar objeciones, el autor refinó teoría de justicia incorporando prioridad para peores situados y equidad de oportunidades.",
    "La ética del cuidado prioriza relaciones contextualizadas y responsabilidades interdependientes frente a abstracciones racionales universalistas descontextualizadas.",
    "Es reconocido que sesgos cognitivos sistemáticos distorsionan intuiciones morales en escenarios de riesgo, aversión a pérdidas y disponibilidad.",
    "El problema difícil de la conciencia pregunta por qué procesamiento neural genera experiencia subjetiva cualitativa aparentemente irreducible físicamente.",
    "Deontólogos insisten que consecuencias agregadas no justifican violar derechos individuales inviolables aunque beneficien mayorías numéricamente superiores.",
    "La bioética clínica integra autonomía del paciente, beneficencia, no maleficencia y justicia distributiva en protocolos de consentimiento.",
    "No obstante pluralismo moral, sociedades democráticas requieren marcos institucionales para resolver desacuerdos razonables sobre bienes públicos comunes.",
    "El argumento ontológico afirma existencia divina deducida conceptualmente, aunque críticos detectan fallos lógicos en inferencias modales involucradas.",
]

BATCH_4 = [
    # Urbanismo y patrimonio cultural 276-300
    "Los planificadores urbanos integraron corredores verdes para mitigar islas de calor y mejorar conectividad peatonal entre barrios fragmentados históricamente.",
    "La regeneración del frente fluvial equilibró conservación industrial patrimonial con vivienda mixta y espacios culturales de acceso público.",
    "Aunque la densificación redujo expansión periférica, residentes exigieron salvaguardas contra gentrificación que desplace comunidades de renta baja establecidas.",
    "Los códigos de edificación histórica exigen materiales compatibles y restauración reversible respetando capas arquitectónicas acumuladas secularmente.",
    "Es evidente que peatonalizar centros comerciales mejora calidad del aire sin sacrificar necesariamente accesibilidad para personas con movilidad reducida.",
    "El inventario patrimonial catalogó edificios modernistas amenazados por presiones inmobiliarias y falta de mantenimiento preventivo sistemático adecuado.",
    "Tras consultas vecinales, el ayuntamiento aprobó límites de altura preservando vistas panorámicas desde fortalezas declaradas monumento nacional protegido.",
    "La movilidad activa prioriza ciclovías protegidas conectadas a transporte público frecuente en corredores de alta demanda intermodal diaria.",
    "Arquitectos críticos cuestionaron iconos espectaculares que priorizan imagen global sobre habitabilidad cotidiana y eficiencia energética de largo plazo.",
    "No obstante inversión en metro, barrios periféricos mantienen dependencia automovilística por carencias persistentes en servicios proximidad esenciales.",
    "La rehabilitación energética de cascos antiguos combina aislamiento térmico discreto con ventilación natural compatible con fachadas protegidas legalmente.",
    "El plan director zonificó usos mixtos reduciendo desplazamientos obligatorios entre vivienda, empleo y equipamientos educativos comunitarios cercanos.",
    "Sigue siendo complejo conciliar turismo masivo con conservación de espacios sagrados que requieren solemnidad y límites estrictos de aforo.",
    "Los paisajistas diseñaron humedales urbanos que filtran escorrentía pluvial mientras albergan biodiversidad y espacios recreativos educativos intergeneracionales.",
    "El catastro cultural documentó técnicas constructivas vernáculas amenazadas por estandarización global de materiales industriales importados económicamente.",
    "Tras incendios recurrentes, autoridades reforzaron normas contra materiales combustibles en rehabilitaciones de edificios patrimoniales de madera histórica.",
    "La participación ciudadana en masterplans incorporó criterios de accesibilidad universal y espacios lúdicos seguros para población infantil diversa.",
    "Urbanistas estudian cómo morfologías de cuadrícula y calles radiales moldean patrones de segregación socioespacial en metrópolis latinoamericanas contemporáneas.",
    "El premio reconoció intervención mínima que reveló estructuras romanas durante excavaciones arqueológicas integradas en plaza contemporánea renovada cuidadosamente.",
    "No obstante certificaciones verdes, edificios inteligentes plantean riesgos de vigilancia doméstica mediante sensores conectados sin consentimiento explícito informado.",
    "La política de vivienda vacía grava propiedades desocupadas incentivando alquiler asequible en distritos con escasez crónica de oferta residencial.",
    "Cooperativas de vivienda autogestionada demostraron modelos alternativos de propiedad colectiva con decisiones asamblearias sobre gastos comunes compartidos.",
    "Sigue incierto si ciudades de quince minutos reducen desigualdades territoriales sin inversión redistributiva en periferias históricamente infrafinanciadas crónicamente.",
    "El museo reinterpretó colecciones coloniales mediante diálogo con comunidades originarias sobre restitución de piezas ceremonialmente significativas culturalmente.",
    "Los ingenieros preservaron patrimonio ferroviario industrial reutilizando hangares como mercados gastronómicos y espacios culturales con economía circular local.",
]

BATCH_5 = [
    # Lingüística y sociolingüística 301-325
    "Los sociolingüistas examinan cómo variantes dialectales marcan pertenencia social y generan estigmatización en contextos educativos institucionalizados formales.",
    "La adquisición del subjuntivo presenta asimetrías temporales entre comprensión pasiva y producción activa en aprendices adultos avanzados hispanohablantes.",
    "Aunque la estandarización unificó ortografía, persisten tensiones entre normas académicas prescriptivas y prácticas comunicativas digitales juveniles emergentes.",
    "El análisis contrastivo revela interferencias prosódicas cuando hablantes bilingües transfieren patrones acentuales entre lenguas románicas e inglesas.",
    "Es aceptado que contacto prolongado entre lenguas produce fenómenos de code-switching estratégico en comunidades fronterizas comercialmente interdependientes.",
    "La pragmática intercultural estudia actos de habla corteses que varían según jerarquías sociales y expectativas de formalidad interlocutoria contextualizada.",
    "Tras documentar corpus orales, investigadores codificaron variantes fonéticas asociadas a procesos de relajación consonántica en habla espontánea urbana.",
    "Los modelos generativos representan estructuras sintácticas mediante operaciones de movimiento que explican orden de palabras permutado topicalización discursiva.",
    "Politólogos lingüísticos analizan framing mediático mediante metáforas conceptuales que moldean percepción pública de crisis migratorias complejas internacionales.",
    "No obstante políticas de inmersión, revitalizar lenguas minorizadas requiere dominios institucionales sostenidos más allá de enseñanza escolar.",
    "La tipología clasifica lenguas según orden básico de constituyente y estrategias morfológicas para expresar relaciones gramaticales de caso.",
    "Sigue debatiéndose si universalidad de categorías sintácticas refleja propiedades cognitivas innatas o generalizaciones inductivas desde input lingüístico limitado.",
    "La variación intrahablante muestra estilización consciente cuando entrevistados alternan registros según imagen social proyectada ante interlocutores institucionales prestigiosos.",
    "El procesamiento psicolingüístico mide tiempos de lectura revelando costos cognitivos en ambigüedades sintácticas temporalmente irresolubles inicialmente on-line.",
    "Tras entrevistar generaciones, el estudio documentó cambio generacional acelerado en vocales posteriores entre jóvenes urbanos cosmopolitas educados.",
    "Los hablantes nativos juzgan menos prestigiosas variantes que neutralizan distinciones fonológicas históricas conservadas en dialectos rurales periféricos tradicionales.",
    "La semántica léxica representa redes de polisemia vinculando significados relacionados por metonimia y metáfora mediante principios de economía.",
    "Es innegable que algoritmos de traducción automática aún fallan capturando matices irónicos y referencias culturales implícitas contextualmente situadas.",
    "La política lingüística promueve diglosia funcional equilibrando lengua nacional con derechos educativos en lenguas cooficiales históricamente marginadas regionalmente.",
    "Corpus paralelos multilingües alimentan modelos neuronales que aprenden correspondencias morfosintácticas sin anotación manual exhaustiva palabra por palabra.",
    "No obstante globalización anglófona, academias hispanas negocian incorporación controlada de anglicismos técnicos mediante equivalencias neológicas científicas propuestas.",
    "La fonología generativa formaliza reglas subyacentes que derivan alternancias alomórficas en conjugaciones verbales con diptongación e hiatos vocálicos.",
    "Sociólogos del lenguaje examinan cómo políticas de exclusión lingüística reproducen desigualdades en mercados laborales altamente competitivos globalizados actuales.",
    "El análisis del discurso crítico desvela presuposiciones ideológicas en textos periodísticos que naturalizan jerarquías de poder económico financiero.",
    "Tras seminarios internacionales, lingüistas acordaron protocolos para documentar lenguas amenazadas con metadatos éticos sobre propiedad intelectual comunitaria indígena.",
]

BATCH_6 = [
    # Arte, cultura y crítica estética 326-350
    "Los críticos de arte examinaron cómo instituciones museísticas legitimaron narrativas canónicas excluyendo producciones femeninas y no occidentales sistemáticamente históricamente.",
    "La estética analítica pregunta si juicios de gusto poseen validez intersubjetiva o expresan meras preferencias subjetivas individualmente contingentes culturalmente.",
    "Aunque el mercado de subastas creció, artistas emergentes cuestionaron concentración de valor en obras especulativas promovidas por galerías.",
    "El conservador restauró frescos renacentistas empleando capas reversibles que preservan pigmentos originales sin ocultar huellas del deterioro temporal visible.",
    "Es reconocido que performances interdisciplinares desafían categorías tradicionales separando artes plásticas de prácticas escénicas contemporáneas experimentales híbridas.",
    "La historiografía del arte revisó iconografías coloniales reinterpretando símbolos impositivos desde perspectivas descolonizadoras y memoria histórica crítica situada.",
    "Tras la polémica curatorial, el museo reordenó salas priorizando diálogos entre piezas antiguas e instalaciones conceptuales contemporáneas provocadoras.",
    "Los compositores incorporaron microtonalidad y texturas electroacústicas expandiendo paletas sonoras más allá de tonalidad funcional clásica occidental tradicional.",
    "Críticos literarios debaten si autoficción borra límites éticos entre experiencia autobiográfica y construcción narrativa deliberadamente ficticia literaria.",
    "No obstante digitalización masiva, bibliotecas defienden preservación material de manuscritos frágiles cuya textura informa prácticas de lectura histórica.",
    "La danza contemporánea explora corporeidad política cuestionando normas de género mediante coreografías colectivas participativas en espacios públicos urbanos.",
    "El festival programó cine de autor latinoamericano destacando estéticas documentales que tensionan fronteras entre observación etnográfica y reenactment escénico.",
    "Sigue siendo discutido si inteligencia artificial genera obras con autoría estética reconocible o reproduce promedios de corpora artísticos existentes.",
    "Los curadores integraron piezas arqueológicas con arte sonoro invitando reflexión sobre temporalidades múltiples y memoria sensible del territorio.",
    "La crítica cultural analiza industrias creativas midiendo impacto económico sin reducir valor simbólico de prácticas artísticas subversivas comunitarias periféricas.",
    "Tras debates académicos, la facultad reformó planes incorporando estudios visuales decoloniales y teoría queer en análisis iconográfico riguroso.",
    "Los fotógrafos documentalistas negociaron consentimiento informado equilibrando testimonio social con riesgos de exposición revictimizante en comunidades vulnerables fotografiadas.",
    "El patrocinio corporativo de exposiciones plantea conflictos cuando marcas controvertidas buscan reputación cultural mediante mecenazgo estratégicamente calculado publicitariamente.",
    "No obstante presupuestos recortados, orquestas sinfónicas innovaron formatos inmersivos atrayendo audiencias jóvenes habituadas a experiencias multimedia interactivas.",
    "La poética barroca emplea conceptismo y culteranismo elevando densidad retórica mediante hipérboles, antítesis y metaforización sistemática del lenguaje.",
    "Historiadores del teatro estudian genealogías performativas que vinculan máscaras antiguas con dramaturgias políticas contemporáneas de protesta callejera organizada.",
    "Es evidente que mercados NFT fragmentaron coleccionismo digital generando discusiones sobre autenticidad, sostenibilidad energética y especulación financiera descontrolada.",
    "El comité de adquisiciones rechazó donaciones con procedencia dudosa exigiendo diligencia debida sobre tráfico ilícito de patrimonio cultural transnacional.",
    "Sigue incierto si políticas culturales europeas armonizan diversidad lingüística artística sin homogeneizar financiación de proyectos periféricos experimentalmente radicales.",
    "La arquitectura expositiva diseñó recorridos narrativos que contextualizan instalaciones inmersivas mediante iluminación escenográfica y textos interpretativos accesibles multilingüemente.",
]

BATCH_7 = [
    # Agricultura y seguridad alimentaria 351-375
    "Los agrónomos evaluaron rotaciones de cultivos que restauran fertilidad del suelo reduciendo dependencia de fertilizantes nitrogenados sintéticos importados costosos.",
    "La seguridad alimentaria exige cadenas de frío ininterrumpidas para perecederos distribuidos en mercados urbanos con infraestructura logística precaria intermitente.",
    "Aunque los rendimientos aumentaron, la erosión hídrica amenaza productividad de laderas cultivadas intensivamente sin terrazas ni cobertura vegetal.",
    "Los pequeños productores adoptaron semillas criollas resilientes ante sequías prolongadas preservando agrobiodiversidad local frente a monocultivos comerciales homogéneos.",
    "Es ampliamente reconocido que deforestación para pastoreo intensivo libera carbono almacenado y reduce precipitaciones regionales de manera acumulativa medible.",
    "La certificación orgánica verifica prácticas sin pesticidas sintéticos mediante auditorías periódicas que documentan trazabilidad desde parcela hasta consumidor final.",
    "Tras negociar acuerdos comerciales, cooperativas accedieron a exportación directa eliminando intermediarios que capturaban márgenes desproporcionados históricamente consolidados.",
    "Los modelos agroecológicos integran polinizadores nativos y barreras cortavientos mejorando estabilidad productiva sin insumos externos energéticamente intensivos contaminantes.",
    "Economistas rurales analizaron si subsidios a biocombustibles desvían cultivos alimentarios hacia commodities energéticos elevando precios internacionales de granos básicos.",
    "No obstante innovaciones biotecnológicas, consumidores exigen etiquetado transparente sobre organismos modificados y posibles alergenos ocultos no declarados explícitamente.",
    "La irrigación por goteo optimiza consumo hídrico en zonas áridas donde acuíferos se sobreexplotan por políticas de acceso gratuito.",
    "Sigue siendo urgente diversificar dietas dependientes del trigo cuando conflictos geopolíticos interrumpen exportaciones concentradas en pocas regiones productoras dominantes.",
    "Los veterinarios implementaron planes sanitarios integrados controlando zoonosis emergentes vinculadas a granjas intensivas y comercio ilegal de fauna silvestre.",
    "El seguro agrícola indexado al clima indemniza pérdidas por sequía mediante indicadores satelitales reduciendo costos de verificación presencial tradicional.",
    "Tras evaluar suelos degradados, técnicos promovieron agroforestería combinando árboles frutales con cultivos anuales diversificando ingresos familiares rurales estacionales.",
    "Las mujeres rurales coordinaron mercados solidarios distribuyendo excedentes locales evitando desperdicio alimentario y fortaleciendo redes de mutuo apoyo comunitario.",
    "La pesca artesanal reclamó cuotas sostenibles ante flotas industriales extranjeras que agotan stocks migratorios en aguas económicas exclusivas nacionales.",
    "Es innegable que precios volátiles de fertilizantes presionan márgenes de cultivadores sin cobertura financiera ante shocks de mercado globales.",
    "Los laboratorios analizaron micotoxinas en cereales almacenados deficientemente alertando sobre riesgos hepáticos asociados a humedad inadecuada prolongada postcosecha.",
    "No obstante metas de cero hambre, distribución inequitativa deja millones sin acceso a dietas nutritivas en territorios exportadores.",
    "La ganadería regenerativa rotaciona pastoreo móvil restaurando pastizales y secuestrando carbono mientras mejora bienestar animal frente a confinamiento industrial.",
    "Investigadores cuantificaron huella hídrica virtual exportada mediante productos agrícolas revelando dependencias ocultas de recursos hídricos extranjeros no visibles contablemente.",
    "Tras sequías extremas, gobiernos reorganizaron reservas estratégicas de granos equilibrando precios con incentivos a producción doméstica competitiva sostenible.",
    "Los contratos agrarios justos fijan precios mínimos protegiendo campesinos de subidas en costos de insumos sin compensación comercial posterior.",
    "Sigue debatiéndose si verticalización tecnológica en invernaderos automatizados concentra propiedad agraria desplazando economías campesinas tradicionalmente autosuficientes regionalmente arraigadas.",
]

BATCH_8 = [
    # Energía y transición tecnológica 376-400
    "Los ingenieros dimensionaron redes inteligentes capaces de integrar generación distribuida variable compensando fluctuaciones mediante almacenamiento electroquímico escalable modularmente.",
    "La transición energética requiere modernizar redes obsoletas sin interrumpir suministro crítico a hospitales y tratamiento de agua potable urbana.",
    "Aunque la eólica marina creció, comunidades pesqueras exigieron compensaciones por restricciones de acceso a caladeros tradicionalmente explotados generacionalmente.",
    "Los analistas proyectaron brechas de capacidad si electrificación industrial acelera sin licitaciones oportunas de infraestructura de transmisión troncal nacional.",
    "Es reconocido que dependencia energética importada expone economías a volatilidad geopolítica y riesgos de corte abrupto en conflictos regionales.",
    "El operador implementó mercados intradiarios equilibrando oferta y demanda mediante precios dinámicos que reflejan escasez temporal regional horaria.",
    "Tras auditorías técnicas, autoridades suspendieron permisos a centrales con emisiones superiores a límites de planes nacionales de descarbonización.",
    "Los fabricantes desarrollaron turbinas de mayor capacidad reduciendo costo nivelado de energía en parques eólicos terrestres de altitud ventosa.",
    "Expertos debaten si hidrógeno verde competirá sin subvenciones cuando electrificación directa cubra mayoría de usos residenciales e industriales ligeros.",
    "No obstante apagones recurrentes, municipios instalaron microrredes resilientes combinando solar fotovoltaica con baterías comunitarias gestionadas cooperativamente localmente.",
    "La nuclear de cuarta generación promete reactores modulares seguros, aunque regulatorios retrasan despliegue comercial previsto a corto plazo.",
    "Los fondos canalizan inversión hacia modernización de redes en regiones mineras dependientes de carbón buscando reconversión laboral justa planificada.",
    "Sigue siendo crítico armonizar estándares transfronterizos para comercio de electricidad evitando cuellos de botella en interconexiones binacionales subutilizadas crónicamente.",
    "La eficiencia energética en edificios reduce picos de demanda más rentablemente que ampliar generación en centrales térmicas de respaldo.",
    "Tras consultas regulatorias, el mercado incorporó señales de carbono internas incentivando inversión en tecnologías bajas en emisiones sectorialmente.",
    "Los operadores monitorean transformadores críticos mediante sensores predictivos que anticipan fallos reduciendo interrupciones en temporada de máxima demanda estival.",
    "Litigios por expropiación de terrenos retrasan líneas de transmisión necesarias para evacuar energía renovable desde regiones con recurso abundante.",
    "La geopolítica del gas natural reconfiguró contratos obligando inversiones aceleradas en terminales de gas licuado y almacenamiento subterráneo estratégico.",
    "No obstante objetivos ambiciosos, combustibles fósiles siguen subsidiados indirectamente mediante externalidades no internalizadas en precios al consumidor final.",
    "Los centros de datos exigen refrigeración eficiente y contratos renovables ante crecimiento exponencial del consumo eléctrico de inteligencia artificial.",
    "Tras apagón nacional, comisión investigó fallas de coordinación entre operadores que amplificaron cortes evitables mediante protocolos de aislamiento.",
    "La captura de carbono en industria pesada requiere infraestructura de transporte y almacenamiento geológico insuficientemente desplegada a escala industrial.",
    "Sigue incierto si tarifas dinámicas domésticas modificarán hábitos de consumo sin afectar desproporcionadamente hogares térmicamente ineficientes de bajos ingresos.",
    "Los planes sectoriales priorizan electrificación del transporte pesado mediante recarga en corredores logísticos interprovinciales de alto flujo mercantil.",
    "Académicos modelaron abandono anticipado de activos fósiles alertando sobre riesgos financieros sistémicos para instituciones con carteras energéticas concentradas.",
]

SENTENCE_BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5, BATCH_6, BATCH_7, BATCH_8]
SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"
for i, batch in enumerate(SENTENCE_BATCHES, 1):
    assert len(batch) == 25, f"BATCH_{i} has {len(batch)} sentences"

START_ID = 201
BATCH_SIZE = 25
TARGET_PATH = project_root / "data/handcraft/es/train/c1_new_002.conllu"

SER_FORMS = {
    "soy", "eres", "es", "somos", "sois", "son",
    "era", "eras", "éramos", "eran", "fue", "fueron",
    "sea", "seas", "sean", "sido", "siendo", "ser",
}
ESTAR_FORMS = {
    "estoy", "estás", "está", "estamos", "estáis", "están",
    "estaba", "estabas", "estábamos", "estaban",
    "estuvo", "estuvieron", "esté", "estén", "estado", "estando", "estar",
}
HABER_FORMS = {
    "he", "has", "ha", "hemos", "habéis", "han", "hay", "había", "habías",
    "habíamos", "habían", "hubo", "hubieron", "haya", "hayan",
    "habría", "habrían", "habrá", "habrán", "habiendo", "habido", "haber",
}
PODER_FORMS = {
    "puedo", "puedes", "puede", "podemos", "podéis", "pueden",
    "podía", "podían", "pudo", "pudieron", "pueda", "puedan",
    "podría", "podrían", "podrá", "podrán", "poder",
}
DEBER_FORMS = {
    "debo", "debes", "debe", "debemos", "debéis", "deben",
    "debía", "debían", "deba", "deban", "debería", "deberían", "deber",
}
AUX_LEMMAS = {"ser", "estar", "haber", "poder", "deber"}

EL_DET = {"el", "la", "los", "las", "lo"}
UNO_DET = {"un", "una", "unos", "unas"}
MIO_DET = {"mi", "mis", "mío", "mía", "míos", "mías"}
SU_DET = {"su", "sus", "suyo", "suya", "suyos", "suyas"}
ESTE_DET = {"este", "esta", "estos", "estas"}
ESE_DET = {"ese", "esa", "esos", "esas"}
AQUEL_DET = {"aquel", "aquella", "aquellos", "aquellas"}
OTRO_DET = {"otro", "otra", "otros", "otras"}
TODO_DET = {"todo", "toda", "todos", "todas"}
CUANTO_DET = {"cuanto", "cuanta", "cuantos", "cuantas", "cuánto", "cuánta", "cuántos", "cuántas"}
QUE_DET = {"qué", "que"}

SCONJ_WORDS = {
    "que", "aunque", "si", "como", "cuando", "donde", "mientras", "porque",
    "pues", "ya", "antes", "después", "hasta", "desde", "según", "salvo",
}
CCONJ_WORDS = {"y", "o", "u", "e", "ni", "pero", "sino"}
ADP_WORDS = {
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "del", "desde",
    "durante", "en", "entre", "hacia", "hasta", "mediante", "para", "por",
    "según", "sin", "so", "sobre", "tras", "al", "versus", "vía",
}
PART_WORDS = {"no", "sí"}
PRON_WORDS = {
    "yo", "tú", "él", "ella", "ello", "nosotros", "nosotras", "vosotros",
    "vosotras", "ellos", "ellas", "usted", "ustedes", "me", "te", "se", "le",
    "lo", "la", "los", "las", "nos", "os", "les", "mí", "ti", "sí", "consigo",
}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "NFT": ("NFT", "PROPN"),
    "STEM": ("STEM", "PROPN"),
    "on-line": ("on-line", "ADV"),
    "del": ("del", "ADP"),
    "al": ("al", "ADP"),
    "Pendiente": ("pendiente", "ADP"),
    "pendiente": ("pendiente", "ADP"),
    "No": ("no", "PART"),
    "no": ("no", "PART"),
    "Sí": ("sí", "PART"),
    "sí": ("sí", "PART"),
    "cada": ("cada", "DET"),
    "mismo": ("mismo", "DET"),
    "misma": ("mismo", "DET"),
    "mismos": ("mismo", "DET"),
    "mismas": ("mismo", "DET"),
    "cualquier": ("cualquier", "DET"),
    "ningún": ("ninguno", "DET"),
    "ninguna": ("ninguno", "DET"),
    "ningunos": ("ninguno", "DET"),
    "ningunas": ("ninguno", "DET"),
    "varios": ("vario", "DET"),
    "varias": ("vario", "DET"),
    "mucho": ("mucho", "ADV"),
    "mucha": ("mucho", "ADV"),
    "muchos": ("mucho", "ADV"),
    "muchas": ("mucho", "ADV"),
    "poco": ("poco", "ADV"),
    "poca": ("poco", "ADV"),
    "pocos": ("poco", "ADV"),
    "pocas": ("poco", "ADV"),
    "más": ("más", "ADV"),
    "menos": ("menos", "ADV"),
    "muy": ("muy", "ADV"),
    "tan": ("tan", "ADV"),
    "tanto": ("tanto", "ADV"),
    "tanta": ("tanto", "ADV"),
    "tantos": ("tanto", "ADV"),
    "tantas": ("tanto", "ADV"),
    "bien": ("bien", "ADV"),
    "mal": ("mal", "ADV"),
    "cómo": ("cómo", "ADV"),
    "cuándo": ("cuándo", "ADV"),
    "dónde": ("dónde", "ADV"),
    "porqué": ("porqué", "NOUN"),
    "por": ("por", "ADP"),
    "para": ("para", "ADP"),
    "con": ("con", "ADP"),
    "sin": ("sin", "ADP"),
    "sobre": ("sobre", "ADP"),
    "entre": ("entre", "ADP"),
    "hacia": ("hacia", "ADP"),
    "desde": ("desde", "ADP"),
    "hasta": ("hasta", "ADP"),
    "durante": ("durante", "ADP"),
    "mediante": ("mediante", "ADP"),
    "según": ("según", "ADP"),
    "contra": ("contra", "ADP"),
    "ante": ("ante", "ADP"),
    "bajo": ("bajo", "ADP"),
    "tras": ("tras", "ADP"),
    "y": ("y", "CCONJ"),
    "o": ("o", "CCONJ"),
    "u": ("o", "CCONJ"),
    "e": ("y", "CCONJ"),
    "ni": ("ni", "CCONJ"),
    "pero": ("pero", "CCONJ"),
    "sino": ("sino", "CCONJ"),
    "aunque": ("aunque", "SCONJ"),
    "si": ("si", "SCONJ"),
    "que": ("que", "SCONJ"),
    "como": ("como", "SCONJ"),
    "cuando": ("cuando", "SCONJ"),
    "donde": ("donde", "ADV"),
    "mientras": ("mientras", "SCONJ"),
    "porque": ("porque", "SCONJ"),
    "pues": ("pues", "SCONJ"),
    "ya": ("ya", "ADV"),
    "a": ("a", "ADP"),
    "de": ("de", "ADP"),
    "en": ("en", "ADP"),
}

VERB_OVERRIDES: dict[str, str] = {
    "monitorearon": "monitorear",
    "monitorea": "monitorear",
    "monitoreó": "monitorear",
    "monitorear": "monitorear",
    "monitor": "monitorear",
    "monitorean": "monitorear",
}

PREDICATE_ADJ: frozenset[str] = frozenset({
    "adecuado", "adecuada", "adecuados", "adecuadas",
    "suficiente", "suficientes", "necesario", "necesaria", "necesarios", "necesarias",
})


def simple_tokenize(sentence: str) -> list[str]:
    forms: list[str] = []
    for word in sentence.split():
        match = re.match(r"^(.+?)([.,;:!?¿¡]+)$", word)
        if match:
            forms.append(match.group(1))
            forms.extend(list(match.group(2)))
        else:
            forms.append(word)
    return forms


def count_tokens(sentence: str) -> int:
    return len(simple_tokenize(sentence))


def _reconstruct_text(forms: list[str]) -> str:
    punct_prefixes = ".,;:!?¿¡\"')"
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


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    """Apply handcrafted Spanish lemma/UPOS rules (es_test conventions)."""
    if form in SPECIAL_LEMMAS:
        return SPECIAL_LEMMAS[form]

    low = form.lower()

    if upos == "PUNCT":
        return form, "PUNCT"

    if low in SER_FORMS:
        return "ser", "AUX"
    if low in ESTAR_FORMS:
        return "estar", "AUX"
    if low in HABER_FORMS:
        return "haber", "AUX"
    if low in PODER_FORMS:
        return "poder", "AUX"
    if low in DEBER_FORMS:
        return "deber", "AUX"

    if form in VERB_OVERRIDES:
        return VERB_OVERRIDES[form], "VERB"
    if lemma in VERB_OVERRIDES:
        return VERB_OVERRIDES[lemma], "VERB"

    if low in SCONJ_WORDS and low not in {"como", "donde"}:
        if low == "que" and upos in {"PRON", "DET"}:
            pass
        else:
            return low, "SCONJ"
    if low in CCONJ_WORDS:
        return low if low != "e" else "y", "CCONJ"
    if low in PART_WORDS:
        return low, "PART"
    if low in ADP_WORDS or (upos == "ADP" and low not in PRON_WORDS):
        if low == "del":
            return "del", "ADP"
        if low == "al":
            return "al", "ADP"
        return low, "ADP"

    if low in EL_DET or form in {"El", "La", "Los", "Las", "Lo"}:
        return "el", "DET"
    if low in UNO_DET:
        return "uno", "DET"
    if low in MIO_DET:
        return "mío", "DET"
    if low in SU_DET:
        return "suyo", "DET"
    if low in ESTE_DET:
        return "este", "DET"
    if low in ESE_DET:
        return "ese", "DET"
    if low in AQUEL_DET:
        return "aquel", "DET"
    if low in OTRO_DET:
        return "otro", "DET"
    if low in TODO_DET:
        return "todo", "DET"
    if low in CUANTO_DET:
        return "cuanto", "DET"
    if low in QUE_DET and upos in {"DET", "PRON"}:
        return "que", "DET" if upos == "DET" else "que"

    if form in {"Ella", "Ellos", "Ellas"} or low in {"ella", "ellos", "ellas"}:
        return "él", "PRON"
    if low == "se":
        return "él", "PRON"
    if low == "le" or low == "les":
        return "él", "PRON"
    if low in {"yo", "tú", "nosotros", "nosotras", "vosotros", "vosotras", "usted", "ustedes"}:
        return low, "PRON"
    if form == "Él":
        return "él", "PRON"
    if low == "él":
        return "él", "PRON"

    if low in PREDICATE_ADJ and upos == "VERB":
        adj_lemma = {
            "adecuado": "adecuado", "adecuada": "adecuado",
            "adecuados": "adecuado", "adecuadas": "adecuado",
            "suficiente": "suficiente", "suficientes": "suficiente",
            "necesario": "necesario", "necesaria": "necesario",
            "necesarios": "necesario", "necesarias": "necesario",
        }[low]
        return adj_lemma, "ADJ"

    if upos == "VERB":
        lem = (lemma or form).lower()
        reflexive = es_reflexive_lemma(low)
        if reflexive is not None:
            lem = reflexive
        if form in VERB_OVERRIDES:
            lem = VERB_OVERRIDES[form]
        elif lemma in VERB_OVERRIDES:
            lem = VERB_OVERRIDES[lemma]
        return lem, "VERB"

    if upos == "ADJ" and lemma:
        return lemma.lower(), "ADJ"

    if upos == "NOUN" and lemma:
        return lemma.lower(), "NOUN"

    if upos == "PROPN":
        lem = lemma if lemma else form
        if lem and lem[0].islower():
            lem = lem[0].upper() + lem[1:]
        return lem, "PROPN"

    if upos == "ADV":
        return (lemma or form).lower(), "ADV"

    if upos == "AUX":
        if lemma in AUX_LEMMAS:
            return lemma, "AUX"
        return (lemma or form).lower(), "VERB"

    if upos == "DET":
        if low in EL_DET:
            return "el", "DET"
        if low in UNO_DET:
            return "uno", "DET"

    return lemma or form, upos


def process_sentence(sent: str, sent_id: str, nlp) -> tuple[list[str], int]:
    doc = nlp(sent)
    sent_forms: list[str] = []
    sent_rows: list[str] = []
    token_counter = 1

    for stanza_sent in doc.sentences:
        words_by_id = {w.id: w for w in stanza_sent.words if isinstance(w.id, int)}
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
                "_",
                "_",
                "_",
                "_",
                "_",
                "_",
            ]
            sent_rows.append("\t".join(cols))
            sent_forms.append(form)
            token_counter += 1

    block = [f"# sent_id = {sent_id}", f"# text = {_reconstruct_text(sent_forms)}"]
    block.extend(sent_rows)
    block.append("")
    return block, len(sent_forms)


def main() -> None:
    import stanza

    bad_lengths = [(i + 1, count_tokens(s)) for i, s in enumerate(SENTENCES)
                   if count_tokens(s) < 12 or count_tokens(s) > 20]
    if bad_lengths:
        print(f"PRE-RUN token length violations ({len(bad_lengths)}):")
        for num, count in bad_lengths[:30]:
            print(f"  sentence {num}: {count} tokens — {SENTENCES[num - 1]}")
        sys.exit(1)

    print("Loading Stanza Spanish pipeline...")
    nlp = stanza.Pipeline(
        lang="es",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
    )

    output_lines: list[str] = []
    token_counts: list[tuple[str, int]] = []
    global_idx = 0

    for batch_num, batch in enumerate(SENTENCE_BATCHES, 1):
        assert len(batch) == BATCH_SIZE
        start = START_ID + global_idx
        end = start + BATCH_SIZE - 1
        print(f"Processing batch {batch_num}/8 (es_c1_train_{start:03d}–{end:03d})...")

        for sent in batch:
            sent_id = f"es_c1_train_{START_ID + global_idx:03d}"
            block, n_tokens = process_sentence(sent, sent_id, nlp)
            output_lines.extend(block)
            token_counts.append((sent_id, n_tokens))
            global_idx += 1

    conllu_text = "\n".join(output_lines) + "\n"
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    TARGET_PATH.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {global_idx} sentences to {TARGET_PATH}")

    bad_lengths = [(sid, n) for sid, n in token_counts if n < 12 or n > 20]
    if bad_lengths:
        print(f"Token length violations ({len(bad_lengths)}):")
        for sid, count in bad_lengths[:20]:
            print(f"  {sid}: {count} tokens")
        sys.exit(1)

    val = validate_text(conllu_text)
    lem = check_text(conllu_text, lang="es")
    print(
        f"COUNT={val.sentence_count} TOKENS={val.token_count} "
        f"VAL={val.passed} LEM={lem.passed}"
    )

    if val.errors:
        print("VAL ERRORS:")
        for err in val.errors[:30]:
            print(f"  {err}")
    if lem.errors:
        print("LEM ERRORS:")
        for err in lem.errors[:30]:
            print(f"  {err}")

    if not val.passed or not lem.passed:
        sys.exit(1)

    print("STATUS: OK — es_c1_train_201 through es_c1_train_400")


if __name__ == "__main__":
    main()