from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

import multitask_model
from multitask_model import (
    EuroBertForUposLemma,
    EuroBertUposLemmaConfig,
    compute_multitask_metrics,
    pad_label_sequence,
)


class DummyEvalPrediction:
    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids


def test_pad_label_sequence_extends_with_ignore_index():
    assert pad_label_sequence([3, 4], 5) == [3, 4, -100, -100, -100]


def test_compute_multitask_metrics_handles_tuple_outputs():
    upos_logits = np.array(
        [
            [
                [0.1, 0.9],
                [0.8, 0.2],
            ]
        ]
    )
    lemma_logits = np.array(
        [
            [
                [0.2, 0.8],
                [0.7, 0.3],
            ]
        ]
    )
    eval_pred = DummyEvalPrediction(
        predictions=(upos_logits, lemma_logits),
        label_ids=(
            np.array([[1, -100]]),
            np.array([[1, 0]]),
        ),
    )

    metrics = compute_multitask_metrics(eval_pred)

    assert metrics["upos_accuracy"] == 1.0
    assert metrics["lemma_accuracy"] == 1.0
    assert metrics["joint_accuracy"] == 1.0


def test_multitask_model_loads_backbone_config_as_keyword_and_initializes_heads(monkeypatch):
    backbone_config = SimpleNamespace(hidden_size=8, hidden_dropout_prob=0.0)
    calls = {}

    class DummyBackbone(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embeddings = nn.Embedding(32, config.hidden_size)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
            **kwargs,
        ):
            hidden = self.embeddings(input_ids)
            return SimpleNamespace(
                last_hidden_state=hidden,
                hidden_states=None,
                attentions=None,
            )

        def get_input_embeddings(self):
            return self.embeddings

        def set_input_embeddings(self, value):
            self.embeddings = value

    def fake_auto_config_from_pretrained(*args, **kwargs):
        return backbone_config

    def fake_auto_model_from_pretrained(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return DummyBackbone(backbone_config)

    monkeypatch.setattr(
        multitask_model.AutoConfig,
        "from_pretrained",
        fake_auto_config_from_pretrained,
    )
    monkeypatch.setattr(
        multitask_model.AutoModel,
        "from_pretrained",
        fake_auto_model_from_pretrained,
    )

    model = EuroBertForUposLemma(
        EuroBertUposLemmaConfig(
            base_model_name_or_path="dummy-backbone",
            upos_label2id={"NOUN": 0, "PROPN": 1},
            lemma_label2id={"IDENTITY": 0, "LOWERCASE": 1},
        )
    )

    assert calls["args"] == ("dummy-backbone",)
    assert calls["kwargs"]["config"] is backbone_config
    assert torch.isfinite(model.upos_classifier.weight).all().item()
    assert torch.isfinite(model.lemma_classifier.weight).all().item()

    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    out = model(**batch)
    upos_logits, lemma_logits, route_logits = out.logits
    assert torch.isfinite(upos_logits).all().item()
    assert torch.isfinite(lemma_logits).all().item()
