from lemmatizer.inference.postprocess import (
    DE_CONTRACTIONS,
    EN_DET_LEMMAS,
    EN_IRREGULAR_PLURALS,
    apply_postprocess_rules,
    es_reflexive_lemma,
)


class TestDEContractions:
    def test_ins_to_in(self):
        assert apply_postprocess_rules("ins", "de", "ins", "ADP") == "in"

    def test_beim_to_bei(self):
        assert apply_postprocess_rules("beim", "de", "beim", "ADP") == "bei"

    def test_zum_to_zu(self):
        assert apply_postprocess_rules("zum", "de", "zum", "ADP") == "zu"

    def test_zur_to_zu(self):
        assert apply_postprocess_rules("zur", "de", "zur", "ADP") == "zu"

    def test_vom_to_von(self):
        assert apply_postprocess_rules("vom", "de", "vom", "ADP") == "von"

    def test_im_to_in(self):
        assert apply_postprocess_rules("im", "de", "im", "ADP") == "in"

    def test_am_to_an(self):
        assert apply_postprocess_rules("am", "de", "am", "ADP") == "an"

    def test_case_insensitive(self):
        assert apply_postprocess_rules("Ins", "de", "Ins", "ADP") == "in"

    def test_no_match_returns_pred(self):
        assert apply_postprocess_rules("Haus", "de", "Haus", "NOUN") == "Haus"

    def test_all_contractions_have_valid_targets(self):
        for word, lemma in DE_CONTRACTIONS.items():
            result = apply_postprocess_rules(word, "de", word, "ADP")
            assert result == lemma, f"Contraction {word} → {lemma} failed, got {result}"


class TestENIrregularPlurals:
    def test_children_to_child(self):
        assert apply_postprocess_rules("children", "en", "children", "NOUN") == "child"

    def test_men_to_man(self):
        assert apply_postprocess_rules("men", "en", "men", "NOUN") == "man"

    def test_women_to_woman(self):
        assert apply_postprocess_rules("women", "en", "women", "NOUN") == "woman"

    def test_phenomena_to_phenomenon(self):
        assert apply_postprocess_rules("phenomena", "en", "phenomena", "NOUN") == "phenomenon"

    def test_not_applied_for_verb(self):
        assert apply_postprocess_rules("men", "en", "men", "VERB") == "men"

    def test_case_insensitive(self):
        assert apply_postprocess_rules("Children", "en", "Children", "NOUN") == "child"

    def test_no_match_returns_pred(self):
        assert apply_postprocess_rules("dogs", "en", "dog", "NOUN") == "dog"

    def test_all_plurals_have_valid_targets(self):
        for word, lemma in EN_IRREGULAR_PLURALS.items():
            result = apply_postprocess_rules(word, "en", word, "NOUN")
            assert result == lemma, f"Irregular {word} → {lemma} failed, got {result}"


class TestENDetLemmas:
    def test_an_to_a(self):
        assert apply_postprocess_rules("an", "en", "an", "DET") == "a"

    def test_case_insensitive(self):
        assert apply_postprocess_rules("An", "en", "An", "DET") == "a"

    def test_not_applied_for_noun(self):
        assert apply_postprocess_rules("an", "en", "an", "NOUN") == "an"

    def test_all_det_lemmas_valid(self):
        for word, lemma in EN_DET_LEMMAS.items():
            result = apply_postprocess_rules(word, "en", word, "DET")
            assert result == lemma, f"DET {word} → {lemma} failed, got {result}"


class TestESReflexives:
    def test_esforzarnos_to_esforzarse(self):
        assert apply_postprocess_rules("esforzarnos", "es", "esforzarnos", "VERB") == "esforzarse"

    def test_prepararnos_to_prepararse(self):
        assert apply_postprocess_rules("prepararnos", "es", "prepararnos", "VERB") == "prepararse"

    def test_preparaos_not_handled(self):
        assert apply_postprocess_rules("preparaos", "es", "preparaos", "VERB") == "preparaos"

    def test_not_applied_for_noun(self):
        assert apply_postprocess_rules("esforzarnos", "es", "esforzarnos", "NOUN") == "esforzarnos"

    def test_short_word_unchanged(self):
        assert apply_postprocess_rules("dormir", "es", "dormir", "VERB") == "dormir"

    def test_venos_unchanged(self):
        assert apply_postprocess_rules("venos", "es", "venos", "VERB") == "venos"

    def test_reflexive_lemma_helper(self):
        assert es_reflexive_lemma("esforzarnos") == "esforzarse"
        assert es_reflexive_lemma("hablar") is None
        assert es_reflexive_lemma("levantarse") is None


class TestCrossLanguage:
    def test_de_rules_not_applied_to_en(self):
        assert apply_postprocess_rules("im", "en", "im", "ADP") == "im"

    def test_en_rules_not_applied_to_de(self):
        assert apply_postprocess_rules("children", "de", "children", "NOUN") == "children"

    def test_es_rules_not_applied_to_de(self):
        assert apply_postprocess_rules("esforzarnos", "de", "esforzarnos", "VERB") == "esforzarnos"

    def test_unknown_lang_returns_pred(self):
        assert apply_postprocess_rules("test", "fr", "test", "NOUN") == "test"
