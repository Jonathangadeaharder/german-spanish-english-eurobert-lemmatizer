from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
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
        use_char_generator: bool = False,
        char_vocab_size: int = 276,
        max_lemma_length: int = 32,
        max_word_length: int = 64,
        char_hidden_size: int = 256,
        char_num_layers: int = 2,
        char_num_heads: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.upos_label2id = upos_label2id or {}
        self.lemma_label2id = lemma_label2id or {}
        self.upos_id2label = {str(index): label for label, index in self.upos_label2id.items()}
        self.lemma_id2label = {str(index): label for label, index in self.lemma_label2id.items()}
        self.use_char_generator = use_char_generator
        self.char_vocab_size = char_vocab_size
        self.max_lemma_length = max_lemma_length
        self.max_word_length = max_word_length
        self.char_hidden_size = char_hidden_size
        self.char_num_layers = char_num_layers
        self.char_num_heads = char_num_heads


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

        saved_vocab_size = getattr(config, "vocab_size", None)
        if saved_vocab_size is not None and saved_vocab_size != self.model.config.vocab_size:
            self.model.resize_token_embeddings(saved_vocab_size)

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "dim", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from backbone config")

        dropout_prob = getattr(self.model.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.upos_classifier = nn.Linear(hidden_size, len(config.upos_label2id))
        self.lemma_classifier = nn.Linear(hidden_size, len(config.lemma_label2id))

        self.lemma_router = nn.Linear(hidden_size, 1)

        self.char_generator = None
        if config.use_char_generator:
            from char_generator import PointerGenerator

            self.char_generator = PointerGenerator(
                encoder_hidden_size=hidden_size,
                char_vocab_size=config.char_vocab_size,
                char_hidden_size=config.char_hidden_size,
                num_layers=config.char_num_layers,
                num_heads=config.char_num_heads,
                max_lemma_length=config.max_lemma_length,
                max_word_length=config.max_word_length,
            )

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
        for head in (self.upos_classifier, self.lemma_classifier, self.lemma_router):
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
        lemma_route: torch.Tensor | None = None,
        lemma_chars: torch.Tensor | None = None,
        word_chars: torch.Tensor | None = None,
        word_char_mask: torch.Tensor | None = None,
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
        route_logits = self.lemma_router(sequence_output).squeeze(-1)

        char_gen_result = None
        sel_tc = None
        if (
            self.char_generator is not None
            and word_chars is not None
            and lemma_route is not None
        ):
            char_gen_mask = lemma_route == 1
            if char_gen_mask.any():
                char_gen = self.char_generator
                if hasattr(char_gen, "modules_to_save"):
                    char_gen = char_gen.modules_to_save[char_gen.active_adapter]
                elif hasattr(char_gen, "original_module"):
                    char_gen = char_gen.original_module

                hidden = backbone_outputs.last_hidden_state
                max_word_len = word_chars.shape[2]
                max_lemma_len = lemma_chars.shape[2] if lemma_chars is not None else 1
                hidden_size = hidden.shape[2]

                selected_hidden = []
                selected_mask = []
                selected_word_chars = []
                selected_word_char_mask = []
                selected_target_chars = []

                for b in range(hidden.shape[0]):
                    for s in range(hidden.shape[1]):
                        if char_gen_mask[b, s]:
                            selected_hidden.append(hidden[b, s])
                            if attention_mask is not None:
                                selected_mask.append(attention_mask[b, s])
                            selected_word_chars.append(
                                word_chars[b, s, :max_word_len]
                            )
                            selected_word_char_mask.append(
                                word_char_mask[b, s, :max_word_len]
                            )
                            if lemma_chars is not None:
                                selected_target_chars.append(
                                    lemma_chars[b, s, :max_lemma_len]
                                )

                if selected_hidden:
                    sel_hidden = torch.stack(selected_hidden).unsqueeze(1)
                    sel_mask = None
                    if attention_mask is not None:
                        sel_mask = torch.stack(selected_mask).unsqueeze(1)
                    sel_wc = torch.stack(selected_word_chars)
                    sel_wcm = torch.stack(selected_word_char_mask)
                    sel_tc = None
                    if selected_target_chars:
                        sel_tc = torch.stack(selected_target_chars)

                    char_gen_result = char_gen(
                        encoder_outputs=sel_hidden,
                        encoder_mask=sel_mask,
                        word_chars=sel_wc,
                        word_char_mask=sel_wcm,
                        target_chars=sel_tc,
                    )

        loss = None
        upos_loss = None
        lemma_loss = None
        route_loss = None
        char_loss = None

        if upos_labels is not None:
            upos_loss = masked_cross_entropy(upos_logits, upos_labels)

        if labels is not None:
            lemma_loss = masked_cross_entropy(lemma_logits, labels)

        if lemma_route is not None:
            valid_route_mask = lemma_route.ne(-100)
            if valid_route_mask.any():
                route_loss = nn.functional.binary_cross_entropy_with_logits(
                    route_logits[valid_route_mask],
                    lemma_route[valid_route_mask].float(),
                )

        if (
            char_gen_result is not None
            and sel_tc is not None
        ):
            char_logits = char_gen_result["char_logits"]
            if char_logits.shape[1] > 0:
                char_logits_shifted = char_logits[:, :-1, :]
                target_shifted = sel_tc[:, 1:]
                seq_len = min(
                    char_logits_shifted.shape[1], target_shifted.shape[1]
                )
                if seq_len > 0:
                    char_loss = nn.functional.cross_entropy(
                        char_logits_shifted[:, :seq_len, :].reshape(
                            -1, self.config.char_vocab_size
                        ),
                        target_shifted[:, :seq_len].reshape(-1),
                        ignore_index=0,
                    )

        components = [v for v in [upos_loss, lemma_loss, route_loss, char_loss] if v is not None]
        if components:
            loss = sum(components)

        if not return_dict:
            output = (upos_logits, lemma_logits, route_logits)
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=(upos_logits, lemma_logits, route_logits),
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
        lemma_route = [feature.pop("lemma_route", None) for feature in features]
        lemma_chars = [feature.pop("lemma_chars", None) for feature in features]
        word_chars = [feature.pop("word_chars", None) for feature in features]

        model_features = [
            {
                key: value
                for key, value in feature.items()
                if key in self.tokenizer.model_input_names
            }
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

        if lemma_route[0] is not None:
            batch["lemma_route"] = torch.tensor(
                [pad_label_sequence(values, seq_len) for values in lemma_route],
                dtype=torch.long,
            )

        if lemma_chars[0] is not None:
            max_lemma_len = len(lemma_chars[0][0])
            padded_chars = []
            for sent_chars in lemma_chars:
                padded = list(sent_chars[:seq_len])
                while len(padded) < seq_len:
                    padded.append([0] * max_lemma_len)
                padded_chars.append(padded[:seq_len])
            batch["lemma_chars"] = torch.tensor(padded_chars, dtype=torch.long)

        if word_chars[0] is not None:
            max_word_len = len(word_chars[0][0])
            padded_wchars = []
            for sent_wchars in word_chars:
                padded = list(sent_wchars[:seq_len])
                while len(padded) < seq_len:
                    padded.append([0] * max_word_len)
                padded_wchars.append(padded[:seq_len])
            batch["word_chars"] = torch.tensor(padded_wchars, dtype=torch.long)
            batch["word_char_mask"] = (batch["word_chars"] != 0).long()

        return batch


def unpack_multitask_logits(predictions):
    if isinstance(predictions, (tuple, list)) and len(predictions) >= 3:
        return predictions[0], predictions[1], predictions[2]

    if isinstance(predictions, (tuple, list)) and len(predictions) >= 2:
        return predictions[0], predictions[1], None

    raise TypeError(f"Expected multitask logits tuple, got {type(predictions)!r}")


def compute_multitask_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    upos_logits, lemma_logits, route_logits = unpack_multitask_logits(predictions)

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

    if route_logits is not None and len(label_ids) >= 3:
        route_labels = label_ids[2]
        route_pred = (route_logits > 0).astype(int)
        route_mask = route_labels != -100
        if route_mask.any():
            route_correct = int((route_pred[route_mask] == route_labels[route_mask]).sum())
            route_total = int(route_mask.sum())
            metrics["route_accuracy"] = (
                round(route_correct / route_total, 4) if route_total else 0.0
            )

    joint_total = upos_total + lemma_total
    joint_correct = upos_correct + lemma_correct
    metrics["joint_accuracy"] = round(joint_correct / joint_total, 4) if joint_total else 0.0

    return metrics
