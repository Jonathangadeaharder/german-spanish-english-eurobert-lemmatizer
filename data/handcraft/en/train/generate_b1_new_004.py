"""Generate b1_new_004.conllu (en_b1_train_601–800) in sub-batches of 25."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from lemmatizer.data.conllu_validator import validate_text
from lemmatizer.data.lemma_checker import check_text
from lemmatizer.inference.postprocess import EN_DET_LEMMAS, EN_IRREGULAR_PLURALS

# 200 B1 sentences: family, law, science, weather, international affairs, retail, animals, mixed
SENTENCE_BATCHES: list[list[str]] = [
    # 601–625: family and relationships
    [
        "The wedding was postponed because several relatives could not travel during the winter holidays.",
        "If couples communicate openly, they are more likely to resolve conflicts before they grow serious.",
        "Many parents believe that childcare costs should be shared more fairly between partners.",
        "The family reunion was organized while relatives prepared traditional dishes from their home region.",
        "Although the apartment is crowded, siblings said they enjoyed living together during university.",
        "New support groups were introduced to help single parents manage work and household responsibilities.",
        "I think grandparents should receive more recognition for the care they provide to young children.",
        "The counselor explained that mediation sessions would begin after both parties signed the agreement.",
        "If trust is broken, relationships may take years to rebuild even with professional help.",
        "The adoption process was reviewed because officials wanted to reduce unnecessary delays for applicants.",
        "Many teenagers feel that family rules should be explained rather than imposed without discussion.",
        "The invitation was sent after the couple confirmed the date for their engagement celebration.",
        "She said she would move closer to her parents if her employer allowed remote work permanently.",
        "Baby showers were arranged by friends who wanted to support the expecting mother practically.",
        "We talked about whether adult children should contribute to household expenses while living at home.",
        "The divorce papers were filed when both partners agreed that separation was the best option.",
        "If relatives visit unexpectedly, hosts may require extra time to prepare meals and sleeping arrangements.",
        "I believe families should teach children how to manage money before they leave school.",
        "The anniversary dinner was canceled because the restaurant closed due to a kitchen fire.",
        "Many neighbors worry that rising rents will force extended families to live in distant suburbs.",
        "The custody arrangement was changed so that children could spend holidays with both parents.",
        "If communication apps fail, relatives abroad may struggle to stay in touch during emergencies.",
        "He explained that the family photo album had been restored after water damaged several pages.",
        "Although the household is busy, most members said they valued shared meals at the weekend.",
        "The community center promised that parenting workshops would be offered in several local languages.",
    ],
    # 626–650: law, justice, and civic safety
    [
        "The trial was delayed because key witnesses were unable to attend the first scheduled hearing.",
        "If citizens report fraud early, investigators can often prevent larger financial losses.",
        "Many residents believe that community policing should be strengthened in neighborhoods with high turnover.",
        "The fine was reduced after the driver proved that road signs had been covered by construction barriers.",
        "Although the case was complex, the jury reached a unanimous decision within three days.",
        "New guidelines were introduced to help courts handle minor offenses more quickly and fairly.",
        "I think legal documents should be written in plain language that ordinary people can understand.",
        "The lawyer explained that the appeal must be submitted before the deadline at the end of the month.",
        "If evidence is mishandled, convictions may be overturned on technical grounds during review.",
        "The neighborhood watch was formed by residents who wanted to improve safety around the school.",
        "Many citizens feel that whistleblowers should be protected when they report serious misconduct.",
        "The complaint was investigated after several customers accused the firm of misleading advertising.",
        "She said she would testify in court if her statement could help protect other vulnerable tenants.",
        "Security cameras were installed because shop owners reported repeated thefts after closing time.",
        "We talked about whether stricter penalties would reduce reckless driving in residential streets.",
        "The settlement was negotiated while both sides sought to avoid a lengthy and costly court battle.",
        "If patrols are reduced, some communities may experience longer response times during night shifts.",
        "I believe restorative justice programs should be expanded for non violent first time offenders.",
        "The license was suspended after inspectors found that safety equipment had not been maintained.",
        "Many voters worry that complex regulations make it difficult for small firms to comply fully.",
        "The public defender was assigned when the defendant could not afford private legal representation.",
        "If court interpreters are unavailable, hearings may be postponed despite urgent family matters.",
        "He explained that the complaint had been filed through the official online reporting portal.",
        "Although crime statistics improved, residents still asked for better lighting near the station.",
        "The ministry promised that victim support services would receive additional funding next year.",
    ],
    # 651–675: science and research
    [
        "The experiment was repeated because the initial results did not match the expected measurements.",
        "If researchers share data openly, other teams can verify findings more quickly and accurately.",
        "Many scientists believe that public funding should support long term studies on climate adaptation.",
        "The laboratory was expanded while engineers installed new equipment for genetic analysis.",
        "Although the hypothesis was bold, reviewers praised the methodology used in the pilot study.",
        "New grants were introduced to help young researchers publish their first peer reviewed articles.",
        "I think science education should include practical experiments rather than only textbook theory.",
        "The professor explained that fieldwork would begin after students completed safety training.",
        "If samples are contaminated, months of careful research may have to be discarded entirely.",
        "The conference was organized by institutes that wanted to promote cooperation across borders.",
        "Many students feel that research internships should be paid to reduce inequality in access.",
        "The dataset was published after experts confirmed that personal information had been removed.",
        "She said she would join the research team if the project concentrated on renewable materials.",
        "Volunteer participants were recruited because the study required responses from diverse age groups.",
        "We talked about whether artificial intelligence can accelerate discovery without replacing human judgment.",
        "The telescope was upgraded while astronomers calibrated instruments for the next observation season.",
        "If budgets are cut, several promising experiments may be canceled despite early positive signals.",
        "I believe scientific advice should inform policy decisions on health and environmental protection.",
        "The patent application was withdrawn after collaborators disagreed about commercial use of the invention.",
        "Many readers worry that sensational headlines misrepresent complex research on nutrition and medicine.",
        "The survey was designed by statisticians who specialize in sampling methods for national studies.",
        "If equipment fails during a mission, space agencies may lose valuable data from planned experiments.",
        "He explained that the manuscript had been revised after reviewers requested clearer statistical tables.",
        "Although progress is gradual, the team said their vaccine trial showed encouraging immune responses.",
        "The university promised that open access publications would be supported through a new library fund.",
    ],
    # 676–700: weather and natural events
    [
        "The festival was canceled because forecasters predicted dangerous winds along the coastal path.",
        "If temperatures rise sharply, elderly residents may require additional support during heatwaves.",
        "Many farmers believe that early warning systems should be improved for drought prone regions.",
        "The bridge was closed while engineers inspected damage caused by flooding after the storm.",
        "Although the snow was heavy, rescue teams reached isolated villages before supplies ran out.",
        "New shelters were introduced after several families lost their homes during the spring floods.",
        "I think weather reports should be communicated more clearly to tourists visiting mountain areas.",
        "The meteorologist explained that alerts would be updated hourly until the hurricane moved inland.",
        "If rivers overflow, downstream communities may be evacuated with very little advance notice.",
        "The harvest was delayed because continuous rain made fields too muddy for heavy machinery.",
        "Many residents feel that urban drainage systems should be upgraded before the next rainy season.",
        "The power supply was restored after crews repaired lines knocked down by falling trees.",
        "She said she would postpone the hike if lightning was expected near the ridge after midday.",
        "Emergency kits were distributed by volunteers who wanted to prepare households for winter storms.",
        "We talked about whether climate adaptation plans receive enough attention in local budgets.",
        "The ski resort was reopened while staff cleared ice from access roads and chairlift stations.",
        "If frost arrives early, fruit growers may lose a significant share of their annual crop.",
        "I believe schools should teach students how to respond safely during earthquakes and floods.",
        "The wildfire was contained after firefighters created barriers using controlled burns and water drops.",
        "Many insurers worry that extreme weather events will increase claims in coastal property markets.",
        "The dam was inspected when authorities received reports of unusual cracks in the concrete wall.",
        "If hail damages roofs, homeowners may wait weeks for contractors during busy repair seasons.",
        "He explained that the evacuation order had been issued through text messages and local radio.",
        "Although the storm subsided quickly, fallen branches blocked several roads into the national park.",
        "The agency promised that flood maps would be updated using data from recent satellite surveys.",
    ],
    # 701–725: international cooperation and diplomacy
    [
        "The summit was postponed because several delegations could not obtain visas in time for travel.",
        "If countries cooperate on trade standards, businesses may face fewer barriers when exporting goods.",
        "Many diplomats believe that humanitarian aid should reach conflict zones more quickly and safely.",
        "The treaty was signed while observers monitored compliance with previously agreed ceasefire terms.",
        "Although negotiations were tense, both sides announced progress on border crossing procedures.",
        "New exchange programs were introduced to help students study abroad and improve language skills.",
        "I think international organizations should coordinate responses when pandemics spread across regions.",
        "The ambassador explained that talks would resume after officials reviewed the latest proposal.",
        "If sanctions are tightened, importers may struggle to secure essential medical supplies legally.",
        "The peacekeeping mission was extended because local authorities requested continued security support.",
        "Many citizens feel that refugee resettlement should be planned fairly among willing host countries.",
        "The aid shipment was delayed after customs officials inspected containers at the port for several days.",
        "She said she would represent the NGO at the forum if funding covered travel and accommodation.",
        "Cultural partnerships were formed by cities that wanted to promote tourism and artistic exchange.",
        "We talked about whether free trade agreements benefit workers as much as large multinational firms.",
        "The embassy was renovated while consular services moved temporarily to another government building.",
        "If communication channels break down, misunderstandings between allies may escalate into public disputes.",
        "I believe development projects should involve local communities in planning from the earliest stage.",
        "The ceasefire was challenged after reports of attacks near the disputed agricultural zone.",
        "Many analysts worry that rising nationalism will weaken cooperation on global environmental targets.",
        "The delegation was welcomed when host officials organized a reception at the historic town hall.",
        "If interpreters make errors, diplomatic statements may be misreported by international news agencies.",
        "He explained that the cooperation agreement had been drafted during months of quiet negotiation.",
        "Although relations improved, both governments said further trust building measures were still required.",
        "The ministry promised that overseas voting procedures would be simplified before the next election.",
    ],
    # 726–750: fashion, retail, and consumer shopping
    [
        "The boutique was renovated because the owner wanted to attract younger customers to the high street.",
        "If online returns are simplified, shoppers may feel more confident buying clothes without trying them on.",
        "Many consumers believe that product labels should state clearly where garments were manufactured.",
        "The winter collection was launched while designers promoted sustainable fabrics at a trade fair.",
        "Although the dress was expensive, customers praised the quality of stitching and lining materials.",
        "New sizing guides were introduced after buyers complained that measurements varied between brands.",
        "I think retailers should reduce excess packaging when shipping small fashion items to online customers.",
        "The manager explained that loyalty points would be credited after purchases were scanned at checkout.",
        "If supply chains are disrupted, popular items may sell out before restocking trucks arrive.",
        "The market stall was closed because health inspectors found refrigeration problems in the food section.",
        "Many shoppers feel that discount events should not encourage unnecessary spending on unused products.",
        "The refund was approved after the customer proved that the shoes had a manufacturing defect.",
        "She said she would switch brands if the company did not improve working conditions in its factories.",
        "Gift cards were promoted because the store wanted to increase sales during the holiday season.",
        "We talked about whether fast fashion harms the environment more than traditional seasonal collections.",
        "The fitting rooms were expanded while staff installed mirrors and better lighting for customers.",
        "If prices are marked incorrectly, cashiers are expected to honor the lower displayed amount.",
        "I believe second hand shops should receive tax incentives to extend the life of quality clothing.",
        "The product recall was announced after testers found that dye chemicals exceeded safety limits.",
        "Many parents worry that children are influenced too strongly by advertising on social media platforms.",
        "The warehouse sale was organized by a charity that wanted to fund training programs for young designers.",
        "If payment systems fail, customers may abandon their baskets despite waiting in long checkout lines.",
        "He explained that the order had been shipped through a courier service with tracked delivery.",
        "Although the shop is small, regular clients said staff remember their preferences and sizes reliably.",
        "The retailer promised that repair services for expensive coats would be available throughout the winter.",
    ],
    # 751–775: pets and animal welfare
    [
        "The shelter was expanded because the number of abandoned animals had increased during the economic downturn.",
        "If owners microchip their pets, lost dogs and cats can often be returned home more quickly.",
        "Many veterinarians believe that preventive care should be affordable for households with limited income.",
        "The clinic was renovated while staff treated emergency cases in temporary rooms next door.",
        "Although the puppy was nervous, trainers said socialization classes helped it adapt to city noise.",
        "New adoption rules were introduced to ensure animals are placed with responsible long term owners.",
        "I think pet food labels should list ingredients clearly for owners managing allergies and diets.",
        "The vet explained that vaccinations must be updated before animals can travel abroad legally.",
        "If grooming salons are overbooked, owners may wait weeks for appointments during shedding season.",
        "The wildlife rescue was founded by volunteers who wanted to care for injured birds and small mammals.",
        "Many residents feel that urban parks should include designated areas where dogs can run freely.",
        "The inspection was conducted after neighbors reported unsanitary conditions at a local breeding facility.",
        "She said she would foster the cat if the shelter provided food and medical support temporarily.",
        "Donations were collected because the charity could not afford surgery for several rescued horses.",
        "We talked about whether exotic pets should be restricted to protect both animals and public safety.",
        "The aquarium was upgraded while biologists monitored water quality for sensitive marine species.",
        "If antibiotics are misused, resistant infections may spread among animals and humans over time.",
        "I believe animal transport regulations should be enforced more strictly during long international journeys.",
        "The adoption event was canceled because an outbreak of kennel cough affected several shelter dogs.",
        "Many farmers worry that predator attacks will increase if habitat loss continues near grazing land.",
        "The training course was offered by experts who specialize in positive reinforcement for family pets.",
        "If owners neglect dental care, cats may develop painful infections that affect their eating habits.",
        "He explained that the rescue rabbit had been found abandoned near a motorway service area.",
        "Although the horse recovered, the owner said rehabilitation required months of careful exercise.",
        "The council promised that stray animal programs would receive additional funding from the next budget.",
    ],
    # 776–800: mixed B1 structures across topics
    [
        "The family mediation service was contacted because parents could not agree on holiday schedules.",
        "If scientists publish clear summaries, policymakers may act faster on urgent environmental warnings.",
        "Many shoppers complained that online sizing tools were inaccurate for shoes and winter coats.",
        "The diplomatic visit was supported by translators who specialized in trade and cultural agreements.",
        "Although the storm damaged crops, farmers said insurance would cover part of their financial losses.",
        "Animal shelters were staffed by volunteers while the city prepared a new adoption awareness campaign.",
        "I think weather alerts and legal advice should be available in languages spoken by local residents.",
        "The research fellow explained that grant applications would be reviewed before the spring committee meeting.",
        "If retail prices rise sharply, families may postpone purchases of clothing and household appliances.",
        "The court hearing was rescheduled after the defendant's lawyer requested additional time to review evidence.",
        "We talked about whether international aid and local entrepreneurship can strengthen rural communities together.",
        "The vet clinic announced that evening appointments would be offered for working pet owners.",
        "She said she would adopt a rescue dog if her landlord approved pets in the rental contract.",
        "Fashion workshops were organized because the college wanted to teach sustainable design to students.",
        "If flood barriers fail, emergency teams may require boats to reach residents in low lying districts.",
        "The treaty discussion was monitored by journalists who reported on progress throughout the week.",
        "Many parents feel that science museums should offer affordable family tickets during school holidays.",
        "The neighborhood lawyer explained that tenant rights would be explained at a public information session.",
        "Although the boutique is new, customers praised the staff for honest advice about fit and fabric.",
        "Wildlife cameras were installed because researchers wanted to study animal movement near the river.",
        "I believe cooperation between cities can improve disaster response when storms affect several regions.",
        "The wedding planner said floral deliveries would be confirmed after the supplier checked storm forecasts.",
        "If animal shelters are full, foster networks may become essential for managing sudden intake surges.",
        "The trade delegation was welcomed by officials who hoped to expand exports of regional food products.",
        "I am convinced that informed citizens, responsible businesses, and open research can strengthen modern societies.",
    ],
]

SENTENCES = [s for batch in SENTENCE_BATCHES for s in batch]
assert len(SENTENCES) == 200, f"Expected 200 sentences, got {len(SENTENCES)}"

START_ID = 601
BATCH_SIZE = 25

AUX_LEMMAS = {"be", "have", "do", "will", "can", "may", "might", "shall", "should", "must"}
BE_FORMS = {
    "am", "is", "are", "was", "were", "be", "been", "being",
}
HAVE_FORMS = {"have", "has", "had", "having"}
DO_FORMS = {"do", "does", "did", "doing"}
WILL_FORMS = {"will", "would", "'ll", "wo"}
CAN_FORMS = {"can", "could", "'d"}
MAY_FORMS = {"may", "might"}
SHALL_FORMS = {"shall", "should"}
MUST_FORMS = {"must"}

SPECIAL_LEMMAS: dict[str, tuple[str, str]] = {
    "I": ("I", "PRON"),
    "me": ("I", "PRON"),
    "my": ("my", "PRON"),
    "mine": ("my", "PRON"),
    "you": ("you", "PRON"),
    "your": ("your", "PRON"),
    "yours": ("your", "PRON"),
    "he": ("he", "PRON"),
    "him": ("he", "PRON"),
    "his": ("his", "PRON"),
    "she": ("she", "PRON"),
    "her": ("her", "PRON"),
    "hers": ("her", "PRON"),
    "it": ("it", "PRON"),
    "its": ("its", "PRON"),
    "we": ("we", "PRON"),
    "us": ("we", "PRON"),
    "our": ("our", "PRON"),
    "ours": ("our", "PRON"),
    "they": ("they", "PRON"),
    "them": ("they", "PRON"),
    "their": ("their", "PRON"),
    "theirs": ("their", "PRON"),
    "the": ("the", "DET"),
    "a": ("a", "DET"),
    "an": ("a", "DET"),
    "this": ("this", "DET"),
    "that": ("that", "DET"),
    "these": ("this", "DET"),
    "those": ("that", "DET"),
    "what": ("what", "DET"),
    "which": ("which", "DET"),
    "whose": ("whose", "DET"),
    "please": ("please", "INTJ"),
    "not": ("not", "PART"),
    "n't": ("not", "PART"),
    "to": ("to", "PART"),
    "o'clock": ("o'clock", "NOUN"),
}

SCONJS = {
    "because", "that", "if", "when", "while", "although", "though", "unless",
    "since", "before", "after", "until", "whether", "as", "than", "so",
}
CCONJS = {"and", "or", "but", "nor", "yet", "so"}
ADPS = {
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about", "into",
    "over", "under", "between", "through", "during", "without", "against",
    "among", "within", "across", "off", "up", "down", "out", "around", "via",
    "per", "near", "behind", "beyond", "despite", "toward", "towards", "upon",
}
PARTS = {"not", "to"}


def _aux_lemma(form: str) -> str:
    lower = form.lower()
    if lower in BE_FORMS:
        return "be"
    if lower in HAVE_FORMS:
        return "have"
    if lower in DO_FORMS:
        return "do"
    if lower in WILL_FORMS:
        return "will"
    if lower in CAN_FORMS:
        return "can"
    if lower in MAY_FORMS:
        return "may"
    if lower in SHALL_FORMS:
        return "should" if lower == "should" else "shall"
    if lower in MUST_FORMS:
        return "must"
    return lower


def normalize_token(form: str, upos: str, lemma: str) -> tuple[str, str]:
    """Apply handcraft lemma/UPOS conventions per en_test.conllu."""
    if form in SPECIAL_LEMMAS:
        lemma, upos = SPECIAL_LEMMAS[form]

    lower = form.lower()

    if lower in BE_FORMS | HAVE_FORMS | DO_FORMS | WILL_FORMS | CAN_FORMS | MAY_FORMS | SHALL_FORMS | MUST_FORMS:
        if lower == "been" and upos == "VERB":
            lemma = "be"
        else:
            upos = "AUX"
            lemma = _aux_lemma(form)
    elif upos == "AUX" and lemma not in AUX_LEMMAS:
        lemma = lemma.lower()

    if upos == "VERB":
        lemma = lemma.lower()
        if lemma.endswith("ing") and len(lemma) > 5:
            base = lemma[:-3]
            if base.endswith(("t", "d")):
                base = base + "e"
            lemma = base
        elif lemma.endswith("ed") and len(lemma) > 4 and form.lower().endswith("ed"):
            base = lemma[:-2]
            if base.endswith(("t", "d")):
                base = base + "e"
            lemma = base

    if upos == "NOUN" and lower in EN_IRREGULAR_PLURALS:
        lemma = EN_IRREGULAR_PLURALS[lower]

    if upos == "NOUN" and lemma:
        lemma = lemma.lower()
        if lemma.endswith("s") and lower.endswith("s") and lower not in EN_IRREGULAR_PLURALS:
            if not lemma.endswith("ss") and len(lemma) > 3:
                lemma = lemma[:-1]
                if lemma.endswith("ie"):
                    lemma = lemma[:-2] + "y"
                elif lemma.endswith("ve"):
                    lemma = lemma[:-2] + "f"

    if upos == "ADJ":
        lemma = lemma.lower()
        if lemma.endswith("er") and len(lemma) > 4:
            base = lemma[:-2]
            if lower.endswith("ier"):
                lemma = base + "y"
            elif lower.endswith(lower[:-2] + "er"):
                lemma = base
        if lemma.endswith("est") and len(lemma) > 5:
            lemma = lemma[:-3]
            if lemma.endswith("i"):
                lemma = lemma[:-1] + "y"

    if upos == "PROPN" and lemma:
        lemma = lemma[0].upper() + lemma[1:]

    if upos == "PUNCT":
        lemma = form

    if upos == "DET" and lower in EN_DET_LEMMAS:
        lemma = EN_DET_LEMMAS[lower]

    if lower in SCONJS:
        if lower not in {"before", "after", "since", "until"} or upos != "ADP":
            upos = "SCONJ"
            lemma = lower
    elif lower in CCONJS:
        upos = "CCONJ"
        lemma = lower
    elif lower in ADPS:
        if not (lower == "to" and upos == "PART"):
            upos = "ADP"
            lemma = lower
    elif lower in PARTS:
        if lower == "to" and upos == "PART":
            pass
        else:
            upos = "PART"
            lemma = lower

    if lower == "there" and upos in {"ADV", "PRON"}:
        upos = "PRON"
        lemma = "there"

    if lower == "more" and upos == "ADJ":
        upos = "ADV"
        lemma = "more"

    if lower == "most" and upos == "ADV":
        lemma = "most"

    return lemma, upos


def sentence_to_conllu(sent_id: str, sent: str, nlp) -> list[str]:
    lines: list[str] = [f"# sent_id = {sent_id}", f"# text = {sent}"]
    doc = nlp(sent)
    token_counter = 1
    for stanza_sent in doc.sentences:
        for word in stanza_sent.words:
            if not isinstance(word.id, int):
                continue
            form = word.text
            upos = word.upos or "X"
            lemma = word.lemma if word.lemma else form
            lemma, upos = normalize_token(form, upos, lemma)
            cols = [
                str(token_counter),
                form,
                lemma,
                upos,
                "_", "_", "_", "_", "_", "_",
            ]
            lines.append("\t".join(cols))
            token_counter += 1
    lines.append("")
    return lines


def main() -> None:
    import stanza

    print("Loading Stanza...")
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        verbose=False,
    )

    target_path = project_root / "data/handcraft/en/train/b1_new_004.conllu"
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
            sent_id = f"en_b1_train_{START_ID + global_idx:03d}"
            all_lines.extend(sentence_to_conllu(sent_id, sent, nlp))
            global_idx += 1

    conllu_text = "\n".join(all_lines) + "\n"
    target_path.write_text(conllu_text, encoding="utf-8")
    print(f"Wrote {global_idx} sentences to {target_path}")

    validation_res = validate_text(conllu_text)
    print(f"Validate: count={validation_res.sentence_count}, passed={validation_res.passed}")
    if not validation_res.passed:
        for err in validation_res.errors:
            print(f"  {err}")
        sys.exit(1)

    lemma_res = check_text(conllu_text, lang="en")
    print(f"Lemma check: passed={lemma_res.passed}")
    if not lemma_res.passed:
        for err in lemma_res.errors:
            print(f"  {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()