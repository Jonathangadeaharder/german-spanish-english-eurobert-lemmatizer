from lmstudio_client import extract_chat_text, parse_sentence_lines


def test_extract_chat_text_supports_lmstudio_output_list():
    data = {
        "output": [
            {"type": "message", "content": "One."},
            {"type": "message", "content": "Two."},
        ]
    }

    assert extract_chat_text(data) == "One.\nTwo."


def test_parse_sentence_lines_strips_bullets_and_numbering():
    text = """
    1. Das ist ein Satz.
    - Noch ein Satz.

    Final sentence.
    """

    assert parse_sentence_lines(text) == [
        "Das ist ein Satz.",
        "Noch ein Satz.",
        "Final sentence.",
    ]
