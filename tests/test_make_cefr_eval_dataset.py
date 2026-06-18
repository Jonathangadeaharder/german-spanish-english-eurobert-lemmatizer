from make_cefr_eval_dataset import EvalJob, sentence_prompt


def test_sentence_prompt_contains_level():
    result = sentence_prompt("de", "B1", "Haus", 2)
    assert "B1" in result
    assert "Haus" in result
    assert "german" in result
    assert "2" in result


def test_sentence_prompt_english():
    result = sentence_prompt("en", "A1", "house", 1)
    assert "english" in result
    assert "A1" in result
    assert "house" in result


def test_sentence_prompt_spanish():
    result = sentence_prompt("es", "C1", "casa", 3)
    assert "spanish" in result
    assert "C1" in result
    assert "casa" in result
    assert "3" in result


def test_eval_job_frozen():
    job = EvalJob(lang="de", level="A1", term="Haus", prompt="test")
    try:
        job.lang = "en"
        raise AssertionError("Should raise FrozenInstanceError")
    except AttributeError:
        pass


def test_eval_job_fields():
    job = EvalJob(lang="es", level="B2", term="casa", prompt="prompt text")
    assert job.lang == "es"
    assert job.level == "B2"
    assert job.term == "casa"
    assert job.prompt == "prompt text"
