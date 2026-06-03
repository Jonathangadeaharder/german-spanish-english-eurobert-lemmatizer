from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import TokenClassifierOutput


def pad_label_sequence(values: list[int], length: int, pad_value: int = -100) -> list[int]:
    padded = list(values[:length])

    if len(padded) < length:
        padded.extend([pad_value] * (length - len(padded)))

    return padded


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    mask = flat_labels.ne(-100)

    if not torch.any(mask):
        return None

    return nn.functional.cross_entropy(flat_logits[mask], flat_labels[mask])


class EuroBertUposLemmaConfig(PretrainedConfig):
    model_type = "eurobert-upos-lemma"

    def __init__(
        self,
        base_model_name_or_path: str | None = None,
        upos_label2id: dict[str, int] | None = None,
        lemma_label2id: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.upos_label2id = upos_label2id or {}
        self.lemma_label2id = lemma_label2id or {}
        self.upos_id2label = {str(index): label for label, index in self.upos_label2id.items()}
        self.lemma_id2label = {str(index): label for label, index in self.lemma_label2id.items()}


class EuroBertForUposLemma(PreTrainedModel):
    config_class = EuroBertUposLemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: EuroBertUposLemmaConfig) -> None:
        super().__init__(config)

        if not config.base_model_name_or_path:
            raise ValueError("config.base_model_name_or_path is required")

        backbone_config = AutoConfig.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            config.base_model_name_or_path,
            config=backbone_config,
            trust_remote_code=True,
        )

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "dim", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from backbone config")

        dropout_prob = getattr(self.model.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.upos_classifier = nn.Linear(hidden_size, len(config.upos_label2id))
        self.lemma_classifier = nn.Linear(hidden_size, len(config.lemma_label2id))
        self._init_task_heads()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.model.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def _init_task_heads(self):
        for head in (self.upos_classifier, self.lemma_classifier):
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        upos_labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> TokenClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        backbone_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        sequence_output = self.dropout(backbone_outputs.last_hidden_state)
        upos_logits = self.upos_classifier(sequence_output)
        lemma_logits = self.lemma_classifier(sequence_output)

        loss = None
        upos_loss = None
        lemma_loss = None

        if upos_labels is not None:
            upos_loss = masked_cross_entropy(upos_logits, upos_labels)

        if labels is not None:
            lemma_loss = masked_cross_entropy(lemma_logits, labels)

        if upos_loss is not None and lemma_loss is not None:
            loss = upos_loss + lemma_loss
        elif upos_loss is not None:
            loss = upos_loss
        elif lemma_loss is not None:
            loss = lemma_loss

        if not return_dict:
            output = (upos_logits, lemma_logits)
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=(upos_logits, lemma_logits),
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
        )


class MultiTaskDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        lemma_labels = [feature.pop("labels") for feature in features]
        upos_labels = [feature.pop("upos_labels") for feature in features]
        model_features = [
            {key: value for key, value in feature.items() if key in self.tokenizer.model_input_names}
            for feature in features
        ]

        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        seq_len = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [pad_label_sequence(values, seq_len) for values in lemma_labels],
            dtype=torch.long,
        )
        batch["upos_labels"] = torch.tensor(
            [pad_label_sequence(values, seq_len) for values in upos_labels],
            dtype=torch.long,
        )

        return batch


def unpack_multitask_logits(predictions):
    if isinstance(predictions, (tuple, list)) and len(predictions) >= 2:
        return predictions[0], predictions[1]

    raise TypeError(f"Expected multitask logits tuple, got {type(predictions)!r}")


def compute_multitask_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids
    upos_logits, lemma_logits = unpack_multitask_logits(predictions)

    if not isinstance(label_ids, (tuple, list)) or len(label_ids) < 2:
        raise TypeError(f"Expected multitask label tuple, got {type(label_ids)!r}")

    lemma_labels = label_ids[0]
    upos_labels = label_ids[1]

    upos_predictions = upos_logits.argmax(axis=-1)
    lemma_predictions = lemma_logits.argmax(axis=-1)

    upos_mask = upos_labels != -100
    lemma_mask = lemma_labels != -100

    upos_total = int(upos_mask.sum())
    lemma_total = int(lemma_mask.sum())

    upos_correct = int((upos_predictions == upos_labels)[upos_mask].sum()) if upos_total else 0
    lemma_correct = int((lemma_predictions == lemma_labels)[lemma_mask].sum()) if lemma_total else 0

    metrics = {
        "upos_accuracy": round(upos_correct / upos_total, 4) if upos_total else 0.0,
        "lemma_accuracy": round(lemma_correct / lemma_total, 4) if lemma_total else 0.0,
    }

    joint_total = upos_total + lemma_total
    joint_correct = upos_correct + lemma_correct
    metrics["joint_accuracy"] = round(joint_correct / joint_total, 4) if joint_total else 0.0

    return metrics
