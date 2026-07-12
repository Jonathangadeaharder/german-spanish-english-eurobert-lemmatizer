"""Generate 200 handcrafted Spanish C1 CoNLL-U sentences (es_c1_train_001–200)."""

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
    # Política pública y gobernanza 001-025
    "El marco regulatorio que rige las transferencias transfronterizas de datos exige mecanismos de rendición de cuentas sólidos.",
    "Los responsables políticos deben equilibrar la consolidación fiscal con inversiones focalizadas en infraestructuras y recualificación.",
    "Aunque la propuesta legislativa obtuvo apoyo bipartidista, varios constitucionalistas cuestionaron su compatibilidad con precedentes judiciales.",
    "El panel independiente recomendó reforzar la protección de denunciantes sin comprometer la confidencialidad de deliberaciones ejecutivas.",
    "La evidencia recopilada en múltiples jurisdicciones sugiere que el presupuesto participativo mejora la transparencia del gasto municipal.",
    "Sigue siendo incierto si las reformas administrativas reducirán retrasos procesales sin debilitar garantías del debido proceso.",
    "El ministerio publicó una evaluación de impacto exhaustiva sobre efectos en pequeñas empresas y consumidores vulnerables.",
    "Los actores de la sociedad civil instaron a legisladores a incorporar criterios ambientales vinculantes en licitaciones públicas.",
    "Tras consultar a autoridades regionales, la comisión finalizó directrices para implementar servicios públicos descentralizados.",
    "La comisión parlamentaria examinó si los poderes de emergencia excedieron los límites temporalmente constitucionales durante la crisis.",
    "Un análisis longitudinal reveló disparidades persistentes en el acceso a asistencia jurídica subsidiada en distritos rurales.",
    "El libro blanco describe una transición gradual hacia modelos de financiación basados en resultados en programas sociales.",
    "Los críticos argumentaron que el entorno regulatorio experimental protege insuficientemente a consumidores financieros vulnerables.",
    "El grupo interinstitucional coordinó respuestas a riesgos sistémicos derivados de mandatos regulatorios superpuestos y fragmentados.",
    "Pendiente de clarificación judicial, los municipios podrán aplicar estándares locales negociados para vivienda asequible.",
    "El marco de rendición de cuentas exige divulgación periódica de indicadores vinculados a objetivos del plan nacional.",
    "Varios peritos testificaron que la fragmentación institucional socava políticas de adaptación climática intersectoriales coherentes.",
    "El gobierno se comprometió a simplificar trámites de licencias sin debilitar inspecciones de seguridad en sectores de alto riesgo.",
    "No obstante severas limitaciones presupuestarias, la agencia mantuvo funciones esenciales de supervisión durante la reestructuración.",
    "El defensor del pueblo documentó quejas recurrentes sobre demoras en recursos administrativos y documentación incompleta.",
    "Un estudio comparado examinó cómo distintos sistemas electorales moldean incentivos legislativos para cooperación en políticas.",
    "El borrador introduce pruebas de proporcionalidad destinadas a limitar la discrecionalidad de funcionarios de aplicación normativa.",
    "Las consultas públicas revelaron preocupación por perfiles algorítmicos en determinaciones automatizadas de elegibilidad social.",
    "El gabinete aprobó un plan de contingencia ante interrupciones del suministro de equipamiento médico crítico a nivel nacional.",
    "Académicos abogaron por reestructurar consultas para incluir sistemáticamente comunidades históricamente subrepresentadas.",
]

BATCH_2 = [
    # Metodología de investigación y política científica 026-050
    "El estudio de cohorte longitudinal siguió biomarcadores entre participantes expuestos a concentraciones variables de material particulado.",
    "Los investigadores prerregistraron hipótesis y planes de análisis para minimizar sesgo de publicación selectiva en resultados experimentales.",
    "Aunque el tamaño muestral fue adecuado, los autores reconocieron limitaciones por deserción no aleatoria en oleadas posteriores.",
    "El metanálisis sintetizó tamaños del efecto de ensayos aleatorizados sobre intervenciones cognitivo-conductuales para dolor crónico.",
    "Los revisores solicitaron comprobaciones de robustez sobre variables confusoras omitidas en la regresión inicial.",
    "El equipo de replicación reprodujo hallazgos centrales con conjuntos de datos independientes y flujos computacionales transparentes.",
    "Los cálculos de potencia indicaron que detectar efectos pequeños requeriría reclutamiento colaborativo multisitio considerablemente mayor.",
    "El comité de ética aprobó el protocolo condicionado a consentimiento informado previo a la aleatorización de participantes.",
    "La triangulación de métodos mixtos combinó encuestas con entrevistas semiestructuradas para esclarecer mecanismos de cambio conductual.",
    "El investigador principal subrayó que análisis exploratorios deben distinguirse de contrastes confirmatorios de hipótesis.",
    "Los planes de gestión de datos especificaron conservación, controles de acceso y procedimientos para desidentificar registros sensibles.",
    "Las agencias de financiación exigen publicación abierta y depósito de materiales de investigación en repositorios certificados.",
    "La revisión sistemática siguió estándares que documentan búsqueda, criterios de inclusión y evaluación de calidad metodológica.",
    "Los modelos bayesianos jerárquicos incorporaron agrupamientos anidados y cuantificaron incertidumbre paramétrica de forma explícita.",
    "Tras validar el instrumento en distintos contextos, el equipo prosiguió con encuestas comparativas transnacionales de campo.",
    "El capítulo metodológico detalla evaluación cegada de resultados y procedimientos contra efectos de expectativa del observador.",
    "Las preocupaciones por p-hacking impulsaron umbrales más estrictos para declarar significación estadística en manuscritos enviados.",
    "El consorcio coordinó imágenes multiespectrales con investigación archivística para reconstruir transiciones históricas del suelo.",
    "Los análisis de sensibilidad demostraron conclusiones estables bajo especificaciones alternativas y supuestos de imputación diversos.",
    "El panel priorizó propuestas que integran participación comunitaria con diseños cuasiexperimentales de evaluación rigurosos.",
    "Los procedimientos de calibración garantizaron comparabilidad de mediciones fisiológicas entre centros clínicos participantes.",
    "El análisis secundario prerregistrado examinó efectos diferenciales del tratamiento en subgrupos demográficos preespecificados.",
    "Los marcos cualitativos se refinaron mediante contraste entre investigadores y verificación con representantes participantes.",
    "El comité asesor recomendó supervisión estadística independiente para salvaguardar integridad del ensayo durante el reclutamiento.",
    "Los estándares de reproducibilidad fomentan compartir código, datos y metadatos de procedencia con publicaciones académicas.",
]

BATCH_3 = [
    # Clima y política ambiental 051-075
    "La estrategia nacional de adaptación prioriza infraestructuras resilientes en comunidades costeras ante el aumento del nivel marino.",
    "Los precios al carbono deben considerar impactos distributivos en hogares pobres y regiones industriales intensivas en energía.",
    "Aunque la capacidad renovable creció rápidamente, persisten desafíos de integración entre generación intermitente y demanda variable.",
    "El panel intergubernamental vinculó emisiones antropogénicas con mayor frecuencia de eventos extremos de precipitación.",
    "Los planes municipales incorporan soluciones basadas en la naturaleza para mitigar calor urbano y gestionar aguas pluviales.",
    "Sigue siendo controvertido si los mercados de compensación generan reducciones reales o desplazan cargas ambientales geográficamente.",
    "La evaluación ambiental analizó riesgos para biodiversidad del desarrollo hidroeléctrico en corredores fluviales sensibles.",
    "La transición industrial depende de hidrógeno bajo en carbono escalable y despliegue oportuno de captura de carbono.",
    "Representantes pesqueros abogaron por aplicación estricta de áreas marinas protegidas amenazadas por arrastre ilegal.",
    "Tras ratificar el acuerdo, los Estados miembros revisarán contribuciones nacionales conforme a objetivos globales de temperatura.",
    "El informe examina compensaciones entre productividad agrícola y extracción sostenible de aguas subterráneas en zonas áridas.",
    "El monitoreo satelital detecta casi en tiempo real focos de deforestación en paisajes de conservación tropical.",
    "Los críticos advirtieron que exportar gas licuado podría prolongar dependencia fósil pese a metas de descarbonización.",
    "La hoja de ruta circular fija metas de residuos mediante responsabilidad ampliada del productor e infraestructura de reciclaje.",
    "La financiación climática canaliza préstamos concesionales hacia adaptación en países vulnerables a choques climáticos.",
    "Pendiente de aprobación legislativa, la agencia impondrá límites estrictos de emisiones en centrales de carbón envejecidas.",
    "Las iniciativas de restauración reconectan hábitats fragmentados apoyando medios de vida forestales sostenibles.",
    "El tribunal determinó que divulgar insuficientemente riesgos climáticos violó normativas bursátiles sobre deberes fiduciarios.",
    "Los urbanistas integraron diseño de enfriamiento pasivo en códigos para reducir dependencia de aire acondicionado intensivo.",
    "No obstante metas ambiciosas, la electrificación del transporte público avanza desigualmente entre regiones metropolitanas.",
    "El consorcio modeló efectos en cascada del retroceso glaciar sobre seguridad hídrica y producción agrícola aguas abajo.",
    "Portadores de conocimiento indígena enriquecieron evaluaciones científicas sobre dinámicas cambiantes del deshielo permafrost.",
    "El ministerio lanzó compras verdes que favorecen proveedores con reducciones verificadas de emisiones de efecto invernadero.",
    "Defensores ambientales exigieron consulta significativa antes de ubicar residuos peligrosos cerca de barrios marginados.",
    "Académicos debaten si reportes voluntarios de sostenibilidad limitan el greenwashing en industrias extractivas intensivas.",
]

BATCH_4 = [
    # Sanidad y salud pública 076-100
    "El ensayo aleatorizado evaluó protocolos combinados farmacológicos y psicoterapéuticos para depresión resistente al tratamiento.",
    "La vigilancia poblacional detectó señales tempranas de brotes respiratorios en distritos urbanos densamente poblados.",
    "Aunque la cobertura vacunal mejoró, persistieron disparidades rurales con acceso limitado a atención primaria.",
    "La guía clínica recomienda terapias individualizadas informadas por biomarcadores validados y perfiles de comorbilidad.",
    "Los economistas evaluaron si terapias innovadoras ofrecen valor aceptable frente a impactos presupuestarios incrementales.",
    "Los registros transparentes de ensayos ayudan a identificar reporte selectivo y sesgo de publicación farmacéutica.",
    "La reforma integra salud conductual en equipos de atención primaria que atienden comunidades crónicamente desatendidas.",
    "Tras implementar detección, las autoridades monitorearon incidencia oncológica por estadio en cohortes demográficas.",
    "Los protocolos nosocomiales enfatizan higiene de manos, administración prudente de antimicrobianos y descontaminación ambiental.",
    "La revisión sintetizó preferencias de pacientes con eficacia clínica al formular recomendaciones de decisión compartida.",
    "Es reconocido que determinantes sociales influyen en resultados de pacientes con múltiples condiciones crónicas simultáneas.",
    "La farmacovigilancia señaló eventos adversos inesperados que exigieron reevaluar el perfil riesgo-beneficio de la terapia.",
    "Muchos clínicos reportan que documentación administrativa resta tiempo de la atención directa al paciente diariamente.",
    "El metanálisis reportó efectos moderados cuya relevancia clínica requiere interpretación contextual por profesionales.",
    "Los comités éticos examinaron si placebos siguen justificados dada disponibilidad de tratamientos activos estándar.",
    "La estrategia de salud combina prevención, detección temprana y vías coordinadas para enfermedades cardiometabólicas.",
    "La multimorbilidad exige reembolsos interdisciplinarios que respalden atención geriátrica integrada y cuidados paliativos.",
    "El organismo asesor subrayó que datos genómicos requieren protección reforzada contra usos discriminatorios laborales.",
    "Sigue incierto si telesalud mejora durablemente acceso a especialistas en condados rurales geográficamente aislados.",
    "Los protocolos predefinieron criterios secundarios con umbrales estadísticos y clínicos antes de análisis sin cegamiento.",
    "No obstante altos costos, el acceso equitativo a terapias eficaces sigue central en debates de política farmacéutica.",
    "La evaluación vacunal incorporó inmunogenicidad y requisitos logísticos para redes nacionales con cadena de frío.",
    "Los algoritmos clínicos deben validarse prospectivamente y auditarse por sesgo algorítmico y deriva de calibración.",
    "Los datos regionales revelan disparidades persistentes en tiempos de espera para procedimientos ambulatorios especializados.",
    "La medicina de precisión personaliza diagnósticos mediante integración multimodal de datos y estratificación con aprendizaje automático.",
]

BATCH_5 = [
    # Economía y mercados laborales 101-125
    "Las instituciones laborales deben amortiguar desplazamiento estructural mediante subsidios de recualificación y diversificación regional.",
    "El trabajo mediado por plataformas crea desafíos novedosos para cobertura de seguros sociales y negociación colectiva.",
    "Aunque el empleo creció, la desigualdad salarial entre tecnología calificada y manufactura heredada persistió obstinadamente.",
    "El banco central señaló cautela ante inflación moderada y consumo desigual entre quintiles de ingreso.",
    "Los ajustes del salario mínimo equilibran poder adquisitivo con posibles efectos sobre empleo en servicios intensivos.",
    "No puede negarse que teletrabajo afecta diferencialmente productividad, colaboración y gestión de límites ocupacionales.",
    "La comisión recomendó activación focalizada en lugar de desregulación amplia de protección del empleo.",
    "El empleo parcial correlaciona con brechas de género en liderazgo corporativo y en instituciones públicas.",
    "Los pronósticos incorporan normalización de cadenas de suministro y primas de riesgo geopolítico en importaciones.",
    "El acuerdo comercial busca armonizar estándares técnicos preservando margen para regulaciones ambientales domésticas.",
    "Tras negociar acuerdos marco, sindicatos buscaron salvaguardas contra subcontratación que eluda salarios negociados.",
    "Los multiplicadores fiscales varían según holgura económica, condiciones monetarias y capacidad de implementar proyectos.",
    "La autoridad investigó si plataformas dominantes excluyeron injustamente competidores emergentes del mercado digital.",
    "La reforma pensional desató debate sobre equidad intergeneracional y jubilación adecuada para trabajadores precarios.",
    "La política industrial canaliza subsidios hacia semiconductores y resiliencia de manufactura de energía limpia.",
    "No obstante desinflación reciente, vivienda e inflación de servicios moldean negativamente expectativas de ingreso real.",
    "Estrategias econométricas explotaron discontinuidades para estimar efectos causales del empleo en aprendizajes ampliados.",
    "El análisis de deuda soberana destacó mejoras del balance primario y reformas estructurales creíbles de crecimiento.",
    "Los capítulos digitales regulan flujos transfronterizos de datos, código fuente y derechos en comercio electrónico.",
    "La participación de mayores refleja incentivos pensionales, salud y demanda selectiva de habilidades experienciales.",
    "La mesa salarial recomendó ajustes sectoriales sin erosionar derechos de consulta de comités de empresa negociados.",
    "Sigue debatible si migración temporal resuelve sosteniblemente escasez estructural en salud e ingeniería.",
    "La evaluación de competencia modeló precios post-fusión con demanda diferenciada y mercados locales definidos.",
    "La inversión en educación infantil genera retornos fiscales mediante mayores ingresos y menor remediación escolar.",
    "El ministro enfatizó gestión prudente de deuda financiando transición verde y ampliación de protección social.",
]

BATCH_6 = [
    # Relaciones internacionales y seguridad 126-150
    "La estrategia reconoce amenazas híbridas que combinan desinformación, coerción económica y operaciones cibernéticas encubiertas.",
    "Medidas de confianza facilitaron diálogo incremental entre partes en disputas territoriales prolongadas.",
    "Aunque las sanciones se intensificaron, exenciones humanitarias para alimentos y medicinas resultaron difíciles de aplicar.",
    "El régimen de inspección exige hitos verificados de desmantelamiento con supervisión intrusiva de arsenales declarados.",
    "Diplomáticos propusieron ceses al fuego monitoreados en corredores con evacuación civil y distribución urgente de ayuda.",
    "Es aceptado que transferencias de armas escalan inestabilidad regional sin supervisión adecuada del usuario final.",
    "El mandato de paz prioriza proteger no combatientes respetando soberanía anfitriona y gobernanza constitucional.",
    "Tras conversaciones en proximidad, mediadores instaron a intercambios de prisioneros como pasos de confianza iniciales.",
    "Analistas advirtieron que ataques submarinos podrían interrumpir suministros energéticos y conectividad de telecomunicaciones.",
    "El consejo humanitario documentó violaciones sistemáticas del derecho internacional que protege civiles en conflicto activo.",
    "No obstante ganancias tácticas, un marco político duradero para reconstrucción posconflicto sigue profundamente controvertido.",
    "La resiliencia cibernética integra inteligencia de amenazas con simulacros intersectoriales y reporte regulatorio obligatorio.",
    "El canciller vinculó asistencia al desarrollo con prevención de conflictos que aborda déficits de gobernanza frágil.",
    "Organizaciones regionales debatieron condicionar normalización económica a reformas democráticas medibles y verificables.",
    "La conferencia exploró verificación de precursores químicos de doble uso y licencias controladas de exportación.",
    "La seguridad marítima coordina patrullas, control de pesca ilegal y ejercicios de libertad de navegación disputada.",
    "Sigue incierto si alianzas en el Indo-Pacífico limitarán proliferación de capacidades avanzadas de misiles.",
    "Instrumentos de refugio exigen asilo justo desalentando migración irregular peligrosa facilitada por redes criminales.",
    "El comité exigió salvaguardas legales sobre vigilancia y órdenes judiciales para solicitudes transfronterizas de datos.",
    "La diplomacia preventiva despliega mediación técnica antes de que disputas localizadas escalen regionalmente.",
    "El debate nuclear sopesa escalada, costos de modernización y estabilidad estratégica entre grandes potencias.",
    "La diplomacia migratoria intersecta desplazamiento climático, movilidad laboral y fortalecimiento de fronteras.",
    "La misión facilitó verificación de prisioneros y reunificación familiar en estructuras de mando fragmentadas.",
    "Estudiosos analizan cómo reparto de poder reduce recurrencia tras terminación negociada de guerras civiles.",
    "El informe vinculó desplazamiento por sequía con competencia por agua transfronteriza y rutas pastoriles.",
]

BATCH_7 = [
    # Educación y política social 151-175
    "La reforma curricular integra alfabetización computacional con razonamiento crítico en humanidades y ciencias naturales.",
    "Fórmulas equitativas dirigen recursos complementarios a escuelas con alta concentración de estudiantes desfavorecidos.",
    "Aunque la alfabetización mejoró nacionalmente, escasez docente rural obstaculizó prestación equitativa de calidad educativa.",
    "La evaluación longitudinal examinó si tutorías tempranas producen ganancias duraderas en matemáticas secundarias.",
    "La acreditación superior enfatiza resultados de aprendizaje, integridad académica y empleabilidad de egresados.",
    "Es evidente que inseguridad alimentaria infantil afecta desarrollo cognitivo y participación en aula de forma medible.",
    "El marco de inclusión exige adaptaciones razonables para estudiantes con discapacidad en entornos educativos ordinarios.",
    "Tras pilotar competencias, el distrito amplió credenciales modulares vinculadas al mercado laboral regional.",
    "Protección social redujo pobreza extrema introduciendo incentivos al trabajo en transferencias condicionadas nacionales.",
    "La estrategia de vivienda combina zonificación inclusiva, vales de alquiler y terrenos públicos para cooperativas.",
    "No obstante mayor matrícula, orientación subfinanciada dejó adolescentes sin apoyo oportuno en salud mental.",
    "La campaña desplegó voluntarios y materiales en lengua materna para mejorar lectura fundacional en aldeas remotas.",
    "Estudiosos examinan cómo educación parental y vecindario transmiten ventajas generacionales de manera sistemática.",
    "El aprendizaje profesional alinea certificaciones industriales con créditos apilables y práctica supervisada laboral.",
    "Agencias infantiles reforzaron acogida tras informes que documentaron fallas sistémicas en gestión de casos.",
    "La equidad de género aborda brechas STEM mediante mentoría y contratación docente consciente del sesgo institucional.",
    "Sigue incierto si dispositivos en aula mejoran aprendizaje sin inversión complementaria en desarrollo docente.",
    "Preescolar universal desató debates fiscales sobre beneficios de largo plazo frente a costos inmediatos de expansión.",
    "La ley antidiscriminación fortalece remedios contra acoso y exige auditorías proactivas de equidad salarial.",
    "Institutos comunitarios facilitan reincorporación laboral de adultos desplazados en recualificación técnica acreditada.",
    "El programa cultural documenta historias orales apoyando revitalización de lenguas indígenas con alianzas escolares.",
    "Políticas de inclusión coordinan vivienda, transporte y acceso digital para reducir segregación metropolitana.",
    "El panel recomendó revisar evaluaciones para reducir estrés de exámenes de alto riesgo sin perder rigor académico.",
    "Intervenciones juveniles combinan pasantías subsidiadas, asesoría empresarial y orientación en educación vocacional.",
    "Un consorcio filantrópico financió servicios integrales para personas sin hogar, adicciones y desempleo crónico.",
]

BATCH_8 = [
    # Tecnología, IA y gobernanza digital 176-200
    "El marco de inteligencia artificial exige evaluaciones de riesgo en sistemas automatizados de alto impacto sobre derechos.",
    "Autoridades de datos investigaron si consentimientos para publicidad conductual cumplieron principios de limitación de finalidad.",
    "Aunque la banda ancha creció, brechas rurales persistieron por asequibilidad e inversión insuficiente en redes.",
    "La directiva de ciberseguridad exige plazos de incidentes, divulgación de vulnerabilidades y auditorías de cadena de suministro.",
    "Estándares de interoperabilidad buscan reducir bloqueo del consumidor preservando innovación en ecosistemas digitales.",
    "Se reconoce que datos sesgados pueden perpetuar discriminación en algoritmos de contratación y calificación crediticia.",
    "El organismo publicó especificaciones para auditar modelos respecto a equidad, robustez y explicabilidad algorítmica.",
    "Tras adoptar confianza cero, agencias segmentaron redes y aplicaron autenticación multifactor en cuentas privilegiadas.",
    "Políticas de código abierto equilibran revisión de seguridad con contribución comunitaria en adquisiciones públicas.",
    "Antimonopolio examinó si autopreferencia en mercados perjudicó injustamente a comerciantes independientes competidores.",
    "No obstante debates sobre cifrado, agencias solicitaron acceso lícito sujeto a autorización judicial y supervisión.",
    "La cartera digital permite divulgación selectiva minimizando almacenamiento centralizado de credenciales personales sensibles.",
    "Investigadores demostraron que perturbaciones adversarias podrían engañar percepción autónoma, planteando retos de certificación.",
    "La estrategia de semiconductores incentiva fabricación nacional mediante créditos fiscales y formación con universidades.",
    "Transparencia algorítmica exige documentar procedencia de entrenamiento y disparidades entre subgrupos demográficos protegidos.",
    "Adquisición en nube evalúa residencia de datos, cifrado y continuidad operativa para cargas críticas gubernamentales.",
    "Sigue disputado si moderación de contenidos equilibra expresión libre con reducción de daños en plataformas sociales.",
    "Directrices robóticas abordan responsabilidad, desplazamiento laboral y supervisión humana en operaciones autónomas críticas.",
    "Operadores de infraestructura crítica deben monitorear continuamente e intercambiar información bajo directivas revisadas.",
    "El sandbox regulatorio permite pruebas fintech controladas salvaguardando depósitos y resolución de disputas de consumidores.",
    "Tras actualizar conservación, el archivo preservó registros mediante migración de formatos y verificación criptográfica.",
    "Académicos advierten que policía predictiva refuerza vigilancia excesiva en barrios urbanos históricamente marginados.",
    "La hoja de ruta cuántica prioriza migración criptográfica poscuántica para redes financieras y de telecomunicaciones.",
    "Normas digitales prohíben a plataformas guardián imponer vinculaciones injustas a usuarios empresariales esenciales.",
    "Parlamento examinó adquisición de vigilancia y compatibilidad con salvaguardas constitucionales de privacidad doméstica.",
]

SENTENCE_BATCHES = [BATCH_1, BATCH_2, BATCH_3, BATCH_4, BATCH_5, BATCH_6, BATCH_7, BATCH_8]
SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"
for i, batch in enumerate(SENTENCE_BATCHES, 1):
    assert len(batch) == 25, f"BATCH_{i} has {len(batch)} sentences"

START_ID = 1
BATCH_SIZE = 25
TARGET_PATH = project_root / "data/handcraft/es/train/c1_new_001.conllu"

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
    "Indo-Pacífico": ("Indo-Pacífico", "PROPN"),
    "STEM": ("STEM", "PROPN"),
    "p-hacking": ("p-hacking", "NOUN"),
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

    print("STATUS: OK — es_c1_train_001 through es_c1_train_200")


if __name__ == "__main__":
    main()