import { AutoTokenizer } from "@huggingface/transformers";
import * as ort from "onnxruntime-web";
import {
  argmax,
  buildLanguageLabelIds,
  languageToken,
  resolveLemma,
  selectBestLabel,
  simpleTokenizer,
  stripLanguagePrefix,
} from "./postprocess.js";

const MODEL_PATH = "./model/";
const MAX_LENGTH = 256;

let tokenizer = null;
let session = null;
let id2label = null;
let uposId2label = null;
let lexicon = null;
let labelIdsByLang = null;

function toArray(values) {
  if (!values) {
    return [];
  }

  if (Array.isArray(values)) {
    return values;
  }

  if (typeof values.tolist === "function") {
    return values.tolist();
  }

  if (ArrayBuffer.isView(values)) {
    return Array.from(values);
  }

  if (values.data) {
    return Array.from(values.data);
  }

  return Array.from(values);
}

function toTensor(values, dims) {
  return new ort.Tensor("int64", BigInt64Array.from(values.map((value) => BigInt(value))), dims);
}

function getOutputRow(tensor, rowIndex) {
  const [batch, sequenceLength, classCount] = tensor.dims;

  if (batch !== 1) {
    throw new Error(`Expected batch size 1, got ${batch}`);
  }

  const offset = rowIndex * classCount;
  return Array.from(tensor.data.slice(offset, offset + classCount));
}

export async function loadLemmatizer() {
  const [loadedTokenizer, id2labelResponse, uposResponse, lexiconResponse] = await Promise.all([
    AutoTokenizer.from_pretrained(MODEL_PATH),
    fetch(`${MODEL_PATH}id2label.json`),
    fetch(`${MODEL_PATH}upos_id2label.json`),
    fetch(`${MODEL_PATH}lexicon.json`),
  ]);

  tokenizer = loadedTokenizer;
  session = await ort.InferenceSession.create(`${MODEL_PATH}model.onnx`);
  id2label = await id2labelResponse.json();
  uposId2label = await uposResponse.json();
  labelIdsByLang = buildLanguageLabelIds(id2label);
  lexicon = lexiconResponse.ok ? await lexiconResponse.json() : { de: {}, es: {}, en: {} };
}

export async function lemmatize(text, lang) {
  if (!["de", "es", "en"].includes(lang)) {
    throw new Error("lang must be one of: de, es, en");
  }

  if (!tokenizer || !session || !id2label || !uposId2label) {
    await loadLemmatizer();
  }

  const originalWords = simpleTokenizer(text);
  const wordsForModel = [languageToken(lang), ...originalWords];

  const encoded = await tokenizer(wordsForModel, {
    is_split_into_words: true,
    padding: true,
    truncation: true,
    max_length: MAX_LENGTH,
  });

  const inputIds = toArray(encoded.input_ids ?? encoded.inputIds);
  const attentionMask = toArray(encoded.attention_mask ?? encoded.attentionMask);

  const feeds = {
    input_ids: toTensor(inputIds, [1, inputIds.length]),
    attention_mask: toTensor(attentionMask, [1, attentionMask.length]),
  };

  const outputs = await session.run(feeds);
  const uposTensor = outputs.upos_logits;
  const lemmaTensor = outputs.lemma_logits;
  const wordIds = typeof encoded.word_ids === "function" ? encoded.word_ids() : encoded.word_ids;
  const result = [];
  const seenWordIds = new Set();

  for (let tokenIndex = 0; tokenIndex < wordIds.length; tokenIndex++) {
    const wordId = wordIds[tokenIndex];

    if (wordId === null || wordId === undefined || wordId === 0 || seenWordIds.has(wordId)) {
      continue;
    }

    seenWordIds.add(wordId);

    const uposRow = getOutputRow(uposTensor, tokenIndex);
    const uposId = argmax(uposRow);
    const upos = uposId2label[String(uposId)] || "X";
    const word = originalWords[wordId - 1];

    let lemma = null;

    if (upos !== "PROPN") {
      const lemmaRow = getOutputRow(lemmaTensor, tokenIndex);
      const labelId = selectBestLabel(lemmaRow, labelIdsByLang[lang]);
      const fullLabel = labelId !== null ? id2label[String(labelId)] || "UNKNOWN" : null;
      const baseLabel = fullLabel !== null ? stripLanguagePrefix(fullLabel, lang) : null;
      const resolved = resolveLemma(word, upos, baseLabel, lexicon[lang]);
      lemma = resolved.lemma;
    }

    result.push({
      word,
      upos,
      lemma,
      lang,
    });
  }

  return result;
}
