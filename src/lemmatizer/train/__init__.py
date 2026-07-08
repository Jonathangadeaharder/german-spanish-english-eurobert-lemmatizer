"""Trainer package: shared options + family dispatch.

Adding a language = one `LanguageSpec` entry in `lemmatizer.languages`. The
dispatch in `train_language()` routes to the family's trainer module by
`spec.family`. Each trainer module exposes `run(spec, opts)` as its canonical
entry; its `main()` is a thin argparse wrapper for `python -m ...` use.

Adding a new *family* (rare): add a `Family` member, a `run()` in a new module
under this package, and one branch in `train_language()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lemmatizer.languages import Family, spec


@dataclass
class TrainOptions:
    """Shared training options. Family-specific extras go in `extra`.

    `checkpoint` is the only required field — the base/pretrained model
    dir or id the trainer loads. Family-specific knobs (e.g. zh
    `prune_layers`) go in `extra`.
    """

    checkpoint: str
    output_dir: str = ""
    epochs: float = 0.0
    batch_size: int = 64
    lr: float = 2e-5
    lora_rank: int = 8
    lora_alpha: float = 16.0
    curriculum: bool = False
    max_train_rows: int = 0
    max_val_rows: int = 0
    unfreeze_encoder: bool = False
    unfreeze_last_n: int = 0
    grad_accum: int = 1
    warmup: float = 0.06
    upos_weight: float = 1.0
    extra: dict = field(default_factory=dict)


def train_language(lang: str, opts: TrainOptions) -> None:
    """Dispatch to the canonical trainer for `lang`'s family.

    Looks up the LanguageSpec, then routes by family. Trainers are imported
    lazily so importing this package never pulls mlx/openmed/torch.
    """
    s = spec(lang)
    # Empty checkpoint auto-loads pretrained weights only for ByT5;
    # MULTITASK/ZH_BIO pass opts.checkpoint to model loading where Path("")
    # collapses to Path(".") and yields cryptic errors. Validate before
    if not opts.checkpoint and s.family != Family.BYT5:
        raise ValueError(
            f"--checkpoint is required for {s.family.value} family "
            f"(lang={lang}). Only ByT5 supports the empty-checkpoint "
            "auto-load fallback."
        )
    if s.family == Family.MULTITASK:
        from lemmatizer.train.mlx_multitask import run
    elif s.family == Family.BYT5:
        from lemmatizer.train.train_byt5 import run
    elif s.family == Family.ZH_BIO:
        from lemmatizer.train.zh_bio import run
    else:  # pragma: no cover — exhaustiveness guard
        raise ValueError(f"Unknown family for {lang}: {s.family}")
    run(s, opts)
