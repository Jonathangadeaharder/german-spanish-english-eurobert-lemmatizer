from __future__ import annotations

import math

import torch
from torch import nn


class PointerGenerator(nn.Module):
    def __init__(
        self,
        encoder_hidden_size: int,
        char_vocab_size: int,
        char_hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_lemma_length: int = 32,
        max_word_length: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.char_vocab_size = char_vocab_size
        self.char_hidden_size = char_hidden_size
        self.max_lemma_length = max_lemma_length
        self.max_word_length = max_word_length

        self.char_embedding = nn.Embedding(char_vocab_size, char_hidden_size)
        self.encoder_proj = nn.Linear(encoder_hidden_size, char_hidden_size)

        self.pos_encoding = nn.Parameter(
            self._sinusoidal_encoding(max_lemma_length, char_hidden_size),
            requires_grad=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=char_hidden_size,
            nhead=num_heads,
            dim_feedforward=char_hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.char_output = nn.Linear(char_hidden_size, char_vocab_size)

        self.word_char_proj = nn.Linear(char_hidden_size, char_hidden_size)
        self.copy_attn = nn.MultiheadAttention(
            embed_dim=char_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.p_gen_linear = nn.Linear(char_hidden_size * 2, 1)

    @staticmethod
    def _sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(sz, sz, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        word_chars: torch.Tensor | None = None,
        word_char_mask: torch.Tensor | None = None,
        target_chars: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = encoder_outputs.shape[0]
        device = encoder_outputs.device

        memory = self.encoder_proj(encoder_outputs)

        if target_chars is not None:
            target_input = target_chars[:, :-1]
            target_emb = self.char_embedding(target_input)
            seq_len = target_input.shape[1]
            target_emb = target_emb + self.pos_encoding[:seq_len].unsqueeze(0)

            causal_mask = self._generate_causal_mask(seq_len, device)

            memory_key_padding_mask = None
            if encoder_mask is not None:
                memory_key_padding_mask = encoder_mask == 0

            decoder_output = self.decoder(
                tgt=target_emb,
                memory=memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            decoder_output = self._autoregressive_decode(
                memory,
                encoder_mask,
                batch_size,
                device,
            )

        char_logits = self.char_output(decoder_output)

        copy_logits = None
        p_gen = None
        if word_chars is not None:
            word_char_emb = self.char_embedding(word_chars)
            word_char_proj = self.word_char_proj(word_char_emb)

            if word_char_mask is not None:
                word_char_key_mask = word_char_mask == 0
            else:
                word_char_key_mask = None

            copy_attn_output, _ = self.copy_attn(
                query=decoder_output,
                key=word_char_proj,
                value=word_char_emb,
                key_padding_mask=word_char_key_mask,
            )
            copy_logits = torch.bmm(decoder_output, word_char_proj.transpose(1, 2))

            p_gen_input = torch.cat([decoder_output, copy_attn_output], dim=-1)
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))

        return {
            "char_logits": char_logits,
            "copy_logits": copy_logits,
            "p_gen": p_gen,
        }

    def _autoregressive_decode(
        self,
        memory: torch.Tensor,
        encoder_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        bos_id = 1
        current_input = torch.full(
            (batch_size, 1),
            bos_id,
            dtype=torch.long,
            device=device,
        )
        all_outputs = []

        memory_key_padding_mask = None
        if encoder_mask is not None:
            memory_key_padding_mask = encoder_mask == 0

        for _step in range(self.max_lemma_length):
            seq_len = current_input.shape[1]
            emb = self.char_embedding(current_input)
            emb = emb + self.pos_encoding[:seq_len].unsqueeze(0)

            causal_mask = self._generate_causal_mask(seq_len, device)

            dec_out = self.decoder(
                tgt=emb,
                memory=memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            next_logits = self.char_output(dec_out[:, -1:, :])
            next_id = next_logits.argmax(dim=-1)
            current_input = torch.cat([current_input, next_id], dim=1)
            all_outputs.append(dec_out[:, -1:, :])

            if (next_id == 2).all():
                break

        if all_outputs:
            return torch.cat(all_outputs, dim=1)

        return memory.new_zeros(batch_size, 0, self.char_hidden_size)
