# Character-Level Lemma Generation: Implementation Plan

## Executive Summary

Replace the current edit tree classification approach with a **hybrid system** that uses:
1. **Edit tree classifier** for the top 300 most common patterns (covers 95% of training data)
2. **Character-level generator** for rare/unseen patterns (handles the "hard impossible" 8-11% of OOV words)

Expected improvement: OOV accuracy from 65-74% → 85-90% (+20-25 percentage points)

---

## Problem Analysis

### Current Limitations

**Edit tree representation is too specific:**
- 7,578 unique edit trees in training data
- 58-66% appear ≤5 times (rare patterns)
- 1,025 trees appear exactly 2 times (minimum threshold)
- Edit trees encode exact character sequences: `P4|S0|Dgehängt|Ihängen`

**OOV error breakdown:**
| Category | DE | ES | EN |
|----------|-----|-----|-----|
| Identity (easy) | 52.4% | 36.4% | 50.0% |
| Lowercase (easy) | 1.8% | 3.5% | 18.6% |
| Edit tree in vocab (learnable) | 35.2% | 52.0% | 20.2% |
| Edit tree NOT in vocab (impossible) | 10.6% | 8.1% | 11.1% |

**Key insight:** Only 125 edit trees cover 90% of training data, 330 cover 95%. The long tail of 7,000+ rare trees is poorly learned.

---

## Architectural Design

### Recommended: Hybrid Approach (Option C)

> **Status:** Partially implemented. The routing head (`lemma_router`) and data pipeline
> (`make_char_dataset.py`, `build_char_vocab.py`) exist in `src/`. The `PointerGenerator`
> module is archived at `archive/char_generator.py`. The character generator is currently
> **disabled** in the production build — `use_char_generator=True` raises `ValueError` in
> both `multitask_model.py` and `train.py`. The model trains and evaluates using only the
> edit-tree classifier + lexicon fallback path.

```
Input: word + context (from EuroBERT encoder)
         ↓
    [Routing Head]
         ↓
    ┌────┴────┐
    ↓         ↓
[Edit Tree]  [Character Generator]
 Classifier   (Pointer-Generator)
    ↓         ↓
Apply tree   Generate lemma
    ↓         ↓
    └────┬────┘
         ↓
      Lemma
```

**Components:**

1. **Routing Head** (binary classifier)
   - Predicts: "use edit tree" vs "generate characters"
   - Input: word embedding + context
   - Output: probability score
   - Threshold: 0.5 (tunable)

2. **Edit Tree Classifier** (existing, reduced)
   - Top 300 most common edit trees + UNKNOWN
   - 301 classes instead of 3,981
   - Much easier to learn (higher accuracy on common patterns)

3. **Character Generator** (new — currently archived at `archive/char_generator.py`; disabled in production)
   - Pointer-generator network
   - Can copy characters from input word OR generate new characters
   - Handles arbitrary transformations including unseen patterns
   - Autoregressive: generates one character at a time

### Why Hybrid?

**Pros:**
- Best of both worlds: fast classification for common cases, flexible generation for rare
- Can be trained incrementally (start with edit trees, add generator)
- Routing decision is interpretable
- Inference can be optimized (skip generator for high-confidence edit tree predictions)

**Cons:**
- More complex than pure classification or pure generation
- Two-stage training required
- Need to tune routing threshold

### Alternative Architectures Considered

**Option A: Full seq2seq with character decoder**
- Pros: Most flexible, handles any transformation
- Cons: Slower inference (always autoregressive), more complex training, larger model
- Verdict: Overkill for 95% of cases where simple edit trees work

**Option B: Edit operations as actions**
- Pros: Interpretable, can be parallel
- Cons: Still limited to predefined operations, struggles with reorderings
- Verdict: Less flexible than character generation

**Option D: Character-level edit trees**
- Pros: Simpler than full seq2seq
- Cons: Still limited to suffix/prefix operations, doesn't handle complex cases
- Verdict: Incremental improvement, not worth the effort

---

## Implementation Plan

### Phase 1: Data Preparation (2-3 days)

#### 1.1 Create character-level dataset

**New file: `src/make_char_dataset.py`**

```python
def make_char_dataset():
    """
    Convert existing dataset to character-level format.
    
    For each word-lemma pair:
    - Determine if it's a "common edit tree" (top 300) or "rare/unseen"
    - If common: label = edit tree ID (0-299)
    - If rare: label = character sequence + routing flag
    
    Output format:
    {
        "input_ids": [...],  # from tokenizer
        "attention_mask": [...],
        "word_ids": [...],
        "upos_labels": [...],
        "lemma_route": [...],  # 0 = edit tree, 1 = char gen
        "lemma_tree_ids": [...],  # edit tree ID (if route=0)
        "lemma_chars": [...],  # character IDs (if route=1)
        "lemma_char_mask": [...],  # which positions are valid
    }
    """
```

**Key decisions:**
- Character vocabulary: Unicode characters seen in training (likely 200-500 chars)
- Max lemma length: 32 characters (covers 99.9% of lemmas; config default is 32)
- Special tokens: `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>` (as implemented in `build_char_vocab.py`; no `<COPY>` token)

#### 1.2 Build character vocabulary

**New file: `src/build_char_vocab.py`**

```python
def build_char_vocab():
    """
    Scan all lemmas in training data, build character vocabulary.
    
    Output: artifacts/char_vocab.json
    {
        "char2id": {"a": 4, "b": 5, ...},   # IDs 0-3 reserved for special tokens
        "id2char": {"4": "a", "5": "b", ...},
        "special_tokens": ["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
        "vocab_size": <total count>,
        "max_lemma_length": 32
    }
    """
```

#### 1.3 Identify top edit trees

**Modify: `src/build_labels.py`**

```python
def build_labels():
    # ... existing code ...
    
    # NEW: Identify top 300 edit trees
    top_trees = [label for label, count in counter.most_common(300)]
    
    # Save both full and reduced vocabularies
    (OUT_DIR / "label2id.json").write_text(...)  # full (existing)
    (OUT_DIR / "label2id_top300.json").write_text(...)  # reduced (new)
    (OUT_DIR / "top_edit_trees.json").write_text(...)  # list of top trees
```

### Phase 2: Model Architecture (3-4 days)

#### 2.1 Extend model config

**Modify: `src/multitask_model.py`**

```python
class EuroBertUposLemmaConfig(PretrainedConfig):
    def __init__(
        self,
        base_model_name_or_path: str | None = None,
        upos_label2id: dict[str, int] | None = None,
        lemma_label2id: dict[str, int] | None = None,
        # NEW: character generation config
        char_vocab_size: int = 276,
        max_lemma_length: int = 32,
        max_word_length: int = 64,
        char_hidden_size: int = 256,
        char_num_layers: int = 2,
        char_num_heads: int = 4,
        route_pos_weight: float = 17.5,
        use_char_generator: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # ... existing code ...
        self.char_vocab_size = char_vocab_size
        self.max_lemma_length = max_lemma_length
        self.max_word_length = max_word_length
        self.char_hidden_size = char_hidden_size
        self.char_num_layers = char_num_layers
        self.char_num_heads = char_num_heads
        self.route_pos_weight = route_pos_weight
        self.use_char_generator = use_char_generator
```

#### 2.2 Add character generator module

**New file: `src/char_generator.py`**

```python
class PointerGenerator(nn.Module):
    """
    Pointer-generator network for character-level lemma generation.
    
    Architecture:
    - Encoder: EuroBERT (shared, already provides context)
    - Decoder: Small transformer (2 layers, 256 hidden, 4 heads)
    - Output: Character probabilities + copy probabilities
    
    At each step:
    1. Attend to encoder output (context)
    2. Attend to input word characters (for copying)
    3. Generate character from vocabulary OR copy from input
    """
    
    def __init__(self, config: EuroBertUposLemmaConfig):
        super().__init__()
        self.char_embedding = nn.Embedding(config.char_vocab_size, config.char_hidden_size)
        
        # Decoder transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.char_hidden_size,
            nhead=config.char_num_heads,
            dim_feedforward=config.char_hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.char_num_layers)
        
        # Output heads
        self.char_output = nn.Linear(config.char_hidden_size, config.char_vocab_size)
        self.copy_output = nn.Linear(config.char_hidden_size, 1)  # copy probability
        
        # Projection from encoder hidden size to decoder hidden size
        self.encoder_proj = nn.Linear(768, config.char_hidden_size)  # EuroBERT-210m hidden=768
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,  # [batch, seq_len, 768]
        encoder_mask: torch.Tensor,  # [batch, seq_len]
        word_chars: torch.Tensor,  # [batch, max_word_len]
        word_char_mask: torch.Tensor,  # [batch, max_word_len]
        target_chars: torch.Tensor | None = None,  # [batch, max_lemma_len] (for training)
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            char_logits: [batch, max_lemma_len, char_vocab_size]
            copy_logits: [batch, max_lemma_len, max_word_len]
            p_gen: [batch, max_lemma_len, 1] (probability of generating vs copying)
        """
        # Project encoder outputs to decoder dimension
        memory = self.encoder_proj(encoder_outputs)
        
        if target_chars is not None:
            # Training: teacher forcing
            target_emb = self.char_embedding(target_chars)
            # Add positional encoding
            # ... (standard transformer decoder)
            decoder_output = self.decoder(target_emb, memory, tgt_mask=causal_mask)
        else:
            # Inference: autoregressive generation
            # ... (generate one char at a time)
            pass
        
        # Character generation probabilities
        char_logits = self.char_output(decoder_output)
        
        # Copy probabilities (attention over input word characters)
        word_char_emb = self.char_embedding(word_chars)
        copy_attn = torch.bmm(decoder_output, word_char_emb.transpose(1, 2))
        copy_logits = copy_attn / (self.char_hidden_size ** 0.5)
        
        # Generation vs copy probability
        p_gen = torch.sigmoid(self.copy_output(decoder_output))
        
        return {
            "char_logits": char_logits,
            "copy_logits": copy_logits,
            "p_gen": p_gen,
        }
```

#### 2.3 Add routing head

**Modify: `src/multitask_model.py`**

```python
class EuroBertForUposLemma(PreTrainedModel):
    def __init__(self, config: EuroBertUposLemmaConfig) -> None:
        super().__init__(config)
        # ... existing code ...
        
        # Existing edit tree classifier (uses full label2id vocabulary, not reduced to top 300)
        self.lemma_classifier = nn.Linear(hidden_size, len(config.lemma_label2id))
        
        # Routing head
        self.lemma_router = nn.Linear(hidden_size, 1)
        
        # Character generator (optional — currently raises ValueError if enabled)
        if config.use_char_generator:
            raise ValueError(
                "Character-generator lemma decoding is not supported in this build. "
                "Use the edit-tree lemma classifier path."
            )
        
        self._init_task_heads()
    
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        # ... existing args ...
        # NEW: character generation inputs
        word_chars: torch.Tensor | None = None,
        word_char_mask: torch.Tensor | None = None,
        lemma_route: torch.Tensor | None = None,  # 0=tree, 1=char
        lemma_tree_ids: torch.Tensor | None = None,
        lemma_chars: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> TokenClassifierOutput:
        # ... existing backbone code ...
        
        sequence_output = self.dropout(backbone_outputs.last_hidden_state)
        upos_logits = self.upos_classifier(sequence_output)
        
        # Edit tree logits (full vocabulary, not reduced)
        tree_logits = self.lemma_classifier(sequence_output)
        
        # Routing logits
        route_logits = self.lemma_router(sequence_output).squeeze(-1)
        
        # Character generation (if enabled)
        char_outputs = None
        if self.config.use_char_generator and word_chars is not None:
            char_outputs = self.char_generator(
                encoder_outputs=backbone_outputs.last_hidden_state,
                encoder_mask=attention_mask,
                word_chars=word_chars,
                word_char_mask=word_char_mask,
                target_chars=lemma_chars,
            )
        
        # Compute losses
        loss = None
        if upos_labels is not None:
            upos_loss = masked_cross_entropy(upos_logits, upos_labels)
            loss = upos_loss
        
        if lemma_route is not None:
            # Routing loss (binary cross-entropy)
            route_loss = nn.functional.binary_cross_entropy_with_logits(
                route_logits, lemma_route.float()
            )
            loss = loss + route_loss if loss is not None else route_loss
            
            # Edit tree loss (only where route=0)
            tree_mask = lemma_route == 0
            if tree_mask.any():
                tree_loss = masked_cross_entropy(
                    tree_logits[tree_mask], lemma_tree_ids[tree_mask]
                )
                loss = loss + tree_loss if loss is not None else tree_loss
            
            # Character generation loss (only where route=1)
            if char_outputs is not None and (lemma_route == 1).any():
                char_mask = lemma_route == 1
                char_loss = self._compute_char_loss(
                    char_outputs, lemma_chars[char_mask]
                )
                loss = loss + char_loss if loss is not None else char_loss
        
        # Return outputs (actual code returns TokenClassifierOutput with logits as tuple)
        logits = (upos_logits, lemma_logits, route_logits)
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
        )
    
    def _compute_char_loss(self, char_outputs, target_chars):
        """Compute loss for character generation."""
        char_logits = char_outputs["char_logits"]
        copy_logits = char_outputs["copy_logits"]
        p_gen = char_outputs["p_gen"]
        
        # Shift for autoregressive training
        # char_logits: [batch, seq_len-1, vocab_size]
        # target_chars: [batch, seq_len]
        shift_logits = char_logits[:, :-1, :]
        shift_targets = target_chars[:, 1:]
        
        # Character generation loss
        char_loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, self.config.char_vocab_size),
            shift_targets.reshape(-1),
            ignore_index=-100,
        )
        
        # Copy loss (optional, can be combined with char_loss)
        # ...
        
        return char_loss
```

#### 2.4 Update data collator

**Modify: `src/multitask_model.py`**

```python
class MultiTaskDataCollator:
    def __call__(self, features):
        # ... existing code ...
        
        # NEW: Extract character-level data
        word_chars_list = [feature.pop("word_chars", None) for feature in features]
        lemma_route_list = [feature.pop("lemma_route", None) for feature in features]
        lemma_tree_ids_list = [feature.pop("lemma_tree_ids", None) for feature in features]
        lemma_chars_list = [feature.pop("lemma_chars", None) for feature in features]
        
        # ... existing padding code ...
        
        # NEW: Pad character-level data
        if word_chars_list[0] is not None:
            batch["word_chars"] = self._pad_2d(word_chars_list)
            batch["word_char_mask"] = (batch["word_chars"] != 0).long()
        
        if lemma_route_list[0] is not None:
            batch["lemma_route"] = torch.tensor(
                [pad_label_sequence(values, seq_len) for values in lemma_route_list],
                dtype=torch.long,
            )
        
        if lemma_tree_ids_list[0] is not None:
            batch["lemma_tree_ids"] = torch.tensor(
                [pad_label_sequence(values, seq_len) for values in lemma_tree_ids_list],
                dtype=torch.long,
            )
        
        if lemma_chars_list[0] is not None:
            batch["lemma_chars"] = self._pad_3d(lemma_chars_list)
        
        return batch
    
    def _pad_2d(self, sequences):
        """Pad list of 2D sequences to same length."""
        max_len = max(len(seq) for seq in sequences)
        padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = torch.tensor(seq)
        return padded
    
    def _pad_3d(self, sequences):
        """Pad list of 3D sequences to same shape."""
        max_seq_len = max(len(seq) for seq in sequences)
        max_char_len = max(len(chars) for seq in sequences for chars in seq)
        padded = torch.zeros(len(sequences), max_seq_len, max_char_len, dtype=torch.long)
        for i, seq in enumerate(sequences):
            for j, chars in enumerate(seq):
                padded[i, j, :len(chars)] = torch.tensor(chars)
        return padded
```

### Phase 3: Training Pipeline (2-3 days)

#### 3.1 Two-stage training strategy

**Stage 1: Train edit tree classifier (existing approach, reduced vocabulary)**
- Use top 300 edit trees + UNKNOWN
- Train for 3 epochs with LoRA
- Expected: Higher accuracy on common patterns (95% of data)

**Stage 2: Train character generator (new)**
- Freeze edit tree classifier
- Train character generator on rare/unseen patterns
- Use teacher forcing during training
- Expected: Handle the 5-10% of rare cases

**Stage 3: Train routing head (optional)**
- Fine-tune routing decision
- Learn when to use edit tree vs character generator
- Can be done jointly with Stage 2

#### 3.2 Modify training script

**Modify: `src/train.py`**

```python
def main():
    # ... existing setup code ...
    
    # Load character vocabulary
    char_vocab = load_json("artifacts/char_vocab.json")
    
    # Load top edit trees
    top_trees = load_json("artifacts/top_edit_trees.json")
    
    # Update config
    config = EuroBertUposLemmaConfig(
        base_model_name_or_path=MODEL_ID,
        upos_label2id=upos_label2id,
        lemma_label2id=top_trees,  # reduced vocabulary
        char_vocab_size=len(char_vocab["char2id"]),
        use_char_generator=True,
    )
    
    # ... rest of training code ...
```

#### 3.3 Training hyperparameters

**New config: `configs/mps-char-gen.toml`**

```toml
[env]
OUTPUT_DIR = "runs/eurobert-char-gen"
TRAIN_MAX_STEPS = 7621
TRAIN_EPOCHS = 3
TRAIN_LEARNING_RATE = 0.0002
TRAIN_WARMUP_RATIO = 0.06
TRAIN_SAVE_TOTAL_LIMIT = 3
TRAIN_SAVE_STEPS = 500
TRAIN_BATCH_SIZE = 2
TRAIN_GRADIENT_ACCUMULATION_STEPS = 8
TRAIN_BF16 = true
TRAIN_GROUP_BY_LENGTH = true
TRAIN_GRADIENT_CHECKPOINTING = true

# Character generation specific
TRAIN_CHAR_LEARNING_RATE = 0.0001  # lower LR for generator
TRAIN_CHAR_LOSS_WEIGHT = 1.0
TRAIN_ROUTE_LOSS_WEIGHT = 0.5
```

### Phase 4: Evaluation & Inference (2-3 days)

#### 4.1 Update evaluation script

**Modify: `src/evaluate.py`**

```python
def evaluate():
    # ... existing setup ...
    
    char_vocab = load_json("artifacts/char_vocab.json")
    id2char = char_vocab["id2char"]
    
    for batch in dataloader:
        # Get model outputs
        outputs = model(**batch)
        
        # UPOS predictions (existing)
        upos_preds = outputs.logits[0].argmax(axis=-1)
        
        # Lemma predictions (hybrid approach — not yet implemented;
        # current evaluate.py uses constrained label selection + lexicon fallback)
        route_preds = (outputs.logits[2] > 0).astype(int)
        tree_preds = outputs.logits[1].argmax(axis=-1)
        
        for i, (route_pred, tree_pred) in enumerate(zip(route_preds, tree_preds)):
            if route_pred == 0:
                # Use edit tree
                base_label = strip_prefix(id2label[str(tree_pred)], lang)
                lemma_pred = apply_edit_label(word, base_label)
            else:
                # Use character generator (not yet available)
                lemma_pred = word
            
            # Compare with gold lemma
            # ...
```

#### 4.2 Character decoding

**New function in `archive/char_generator.py`** (archived; not in `src/`)

```python
def decode_characters(
    char_outputs: dict[str, torch.Tensor],
    id2char: dict[str, str],
    beam_size: int = 3,
) -> str:
    """
    Decode character outputs to lemma string.
    
    Args:
        char_outputs: Output from PointerGenerator
        id2char: Character ID to character mapping
        beam_size: Beam search width (1 = greedy)
    
    Returns:
        Decoded lemma string
    """
    char_logits = char_outputs["char_logits"]
    copy_logits = char_outputs["copy_logits"]
    p_gen = char_outputs["p_gen"]
    
    # Greedy decoding (simplest)
    generated_chars = []
    for step in range(char_logits.shape[0]):
        # Decide: generate or copy?
        gen_prob = p_gen[step].item()
        
        if gen_prob > 0.5:
            # Generate from vocabulary
            char_id = char_logits[step].argmax().item()
            char = id2char[str(char_id)]
        else:
            # Copy from input
            copy_id = copy_logits[step].argmax().item()
            char = input_word[copy_id]  # need to pass input_word
        
        if char == "<EOS>":
            break
        
        generated_chars.append(char)
    
    return "".join(generated_chars)
```

#### 4.3 Update ONNX export

**Modify: `src/export_onnx.py`**

> **Note:** The current `export_onnx.py` exports only `upos_logits` and `lemma_logits`
> (via `MultiTaskExportWrapper` which returns `outputs.logits[0], outputs.logits[1]`).
> The route logits are not included in the ONNX export. This section describes the
> future two-model export for when the char generator is enabled.

```python
def export():
    # ... existing code ...
    
    # Export both paths (future: when char generator is enabled)
    # 1. Edit tree path (fast, for common cases)
    # 2. Character generator path (slower, for rare cases)
    
    # For ONNX, we need to handle the conditional logic
    # Option A: Export as two separate models
    # Option B: Export with dynamic control flow (ONNX opset 16+)
    
    # Recommended: Option A (simpler, more compatible)
    
    # Export edit tree model
    torch.onnx.export(
        edit_tree_model,
        sample_inputs,
        "model_edit_tree.onnx",
        # ...
    )
    
    # Export character generator
    torch.onnx.export(
        char_generator,
        sample_inputs,
        "model_char_gen.onnx",
        # ...
    )
    
    # Routing logic in JavaScript (web inference)
    # if (route_prob < threshold) {
    #     use edit tree model
    # } else {
    #     use char generator model
    # }
```

### Phase 5: Integration & Testing (2-3 days)

#### 5.1 Update web inference

**Modify: `web/demo.js`** (actual runtime file; plan originally referenced non-existent `web/src/lemmatizer.js`)

> **Note:** The current `web/demo.js` uses a single ONNX model that outputs `upos_logits`
> and `lemma_logits`. The routing/char-gen two-model architecture below is the target
> design once the char generator is enabled.

```javascript
class Lemmatizer {
    constructor() {
        this.editTreeModel = await ort.InferenceSession.create('model_edit_tree.onnx');
        this.charGenModel = await ort.InferenceSession.create('model_char_gen.onnx');
        this.routingThreshold = 0.5;
    }
    
    async lemmatize(word, context) {
        // Get encoder outputs
        const encoderOut = await this.encoder.run({input_ids, attention_mask});
        
        // Get routing decision
        const routeProb = await this.getRoutingProb(encoderOut);
        
        if (routeProb < this.routingThreshold) {
            // Use edit tree (fast)
            const treeId = await this.editTreeModel.run({encoder: encoderOut});
            return this.applyEditTree(treeId, word);
        } else {
            // Use character generator (slower but flexible)
            const chars = await this.charGenModel.run({
                encoder: encoderOut,
                word_chars: this.tokenizeWord(word)
            });
            return this.decodeCharacters(chars);
        }
    }
}
```

#### 5.2 Benchmark performance

**New file: `src/benchmark_char_gen.py`**

```python
def benchmark():
    """
    Compare performance of:
    1. Current edit tree approach
    2. Hybrid approach (edit tree + char gen)
    
    Metrics:
    - Overall accuracy
    - OOV accuracy
    - Inference speed (ms per word)
    - Model size (MB)
    """
```

#### 5.3 A/B testing

Test on held-out data:
- Measure OOV accuracy improvement
- Measure inference speed degradation
- Tune routing threshold for optimal tradeoff

---

## File Changes Summary

### New Files
1. `src/char_generator.py` → **archived** at `archive/char_generator.py` (Pointer-generator network; disabled in production)
2. `src/make_char_dataset.py` - Convert dataset to character-level format
3. `src/build_char_vocab.py` - Build character vocabulary
4. `src/benchmark_char_gen.py` - Performance benchmarking (not yet created)
5. `configs/mps-char-gen.toml` - Training config for character generation (not yet created)

### Modified Files
1. `src/multitask_model.py` - Add routing head, char generator placeholder (disabled), updated collator
2. `src/build_labels.py` - Identify top 300 edit trees, produce `label2id_top300.json` and `top_edit_trees.json`
3. `src/train.py` - Rejects `TRAIN_USE_CHAR_GENERATOR=True` with `ValueError`
4. `src/evaluate.py` - Constrained label selection, lexicon fallback, PROPN gating
5. `src/export_onnx.py` - Exports only upos_logits and lemma_logits (no route_logits)
6. `web/postprocess.js` - Edit-tree application, language-prefix stripping, lexicon lookup

### New Artifacts
1. `artifacts/char_vocab.json` - Character vocabulary
2. `artifacts/top_edit_trees.json` - List of top 300 edit trees
3. `artifacts/label2id_top300.json` - Reduced edit tree vocabulary

---

## Risk Mitigation

### Risk 1: Character generation is too slow for real-time inference

**Mitigation:**
- Routing head ensures character generator only runs for 5-10% of words
- Optimize character generator (smaller model, quantization)
- Cache common character sequences
- Fallback to edit tree if generation times out

### Risk 2: Character generator produces invalid lemmas

**Mitigation:**
- Add post-processing validation (check if lemma is valid Unicode)
- Use beam search with length penalty
- Add language model scoring to rank candidates
- Fallback to identity if generation fails

### Risk 3: Training instability with two-stage approach

**Mitigation:**
- Train edit tree classifier first (stable baseline)
- Freeze edit tree classifier when training character generator
- Use gradient accumulation to stabilize character generator training
- Monitor routing loss separately

### Risk 4: ONNX export complexity

**Mitigation:**
- Export as two separate models (simpler than dynamic control flow)
- Handle routing logic in JavaScript
- Test ONNX models thoroughly before deployment

---

## Timeline & Milestones

| Phase | Duration | Milestone |
|-------|----------|-----------|
| 1. Data Preparation | 2-3 days | Character dataset ready, vocab built |
| 2. Model Architecture | 3-4 days | Hybrid model implemented, unit tests pass |
| 3. Training Pipeline | 2-3 days | Two-stage training complete, checkpoints saved |
| 4. Evaluation & Inference | 2-3 days | Evaluation script updated, ONNX exported |
| 5. Integration & Testing | 2-3 days | Web inference working, benchmarks complete |

**Total: 11-16 days**

---

## Success Criteria

1. **OOV accuracy improvement:** +20-25 percentage points (65-74% → 85-90%)
2. **Overall accuracy:** Maintain or improve current 94-98%
3. **Inference speed:** <10% degradation (character generator runs for <10% of words)
4. **Model size:** <50 MB increase (character generator is small)
5. **Web compatibility:** ONNX models load and run in browser

---

## Next Steps

1. **Start with Phase 1** (data preparation) - lowest risk, foundational
2. **Prototype character generator** in isolation (small test dataset)
3. **Validate architecture** with unit tests before full training
4. **Iterate on routing threshold** based on validation set performance
5. **A/B test** with current model before full deployment

---

## References

- Pointer-Generator Networks: https://arxiv.org/abs/1704.04368
- Character-level NMT: https://arxiv.org/abs/1508.04368
- Lemmatization as seq2seq: https://aclanthology.org/P16-1057/
