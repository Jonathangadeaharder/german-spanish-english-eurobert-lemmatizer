import { AutoTokenizer } from "@huggingface/transformers";
import * as ort from "onnxruntime-web";
import {
  argmax,
  buildLanguageLabelIds,
  languageToken,
  resolveLemma,
  selectValidLabel,
  simpleTokenizer,
} from "./postprocess.js";

const MAX_LENGTH = 256;
const HF_RESOLVE_BASE = "https://huggingface.co";
const LEMMATIZER_MODELS = {
  de: "Jonandrop/eurobert-lemma-de-210m",
  en: "Jonandrop/eurobert-lemma-en-210m",
  es: "Jonandrop/eurobert-lemma-es-210m",
};

const lemmatizers = new Map();

function assertSupportedLang(lang) {
  if (!Object.prototype.hasOwnProperty.call(LEMMATIZER_MODELS, lang)) {
    throw new Error("lang must be one of: de, es, en");
  }
}

function modelFileUrl(lang, file) {
  return `${HF_RESOLVE_BASE}/${LEMMATIZER_MODELS[lang]}/resolve/main/${file}`;
}

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

export async function loadLemmatizer(lang) {
  assertSupportedLang(lang);

  if (lemmatizers.has(lang)) {
    return lemmatizers.get(lang);
  }

  const repoId = LEMMATIZER_MODELS[lang];
  const [loadedTokenizer, id2labelResponse, uposResponse, lexiconResponse] = await Promise.all([
    AutoTokenizer.from_pretrained(repoId),
    fetch(modelFileUrl(lang, "id2label.json")),
    fetch(modelFileUrl(lang, "upos_id2label.json")),
    fetch(modelFileUrl(lang, "lexicon.json")),
  ]);

  if (!id2labelResponse.ok) {
    throw new Error(`Failed to load ${repoId}/id2label.json`);
  }

  if (!uposResponse.ok) {
    throw new Error(`Failed to load ${repoId}/upos_id2label.json`);
  }

  const id2label = await id2labelResponse.json();
  const lemmatizer = {
    tokenizer: loadedTokenizer,
    session: await ort.InferenceSession.create(modelFileUrl(lang, "model.int8.onnx")),
    id2label,
    uposId2label: await uposResponse.json(),
    labelIdsByLang: buildLanguageLabelIds(id2label),
    lexicon: lexiconResponse.ok ? await lexiconResponse.json() : {},
  };

  lemmatizers.set(lang, lemmatizer);
  return lemmatizer;
}

export async function lemmatize(text, lang) {
  const { tokenizer, session, id2label, uposId2label, labelIdsByLang, lexicon } =
    await loadLemmatizer(lang);

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
      const baseLabel = selectValidLabel(
        lemmaRow,
        labelIdsByLang[lang],
        id2label,
        lang,
        word,
      );
      const resolved = resolveLemma(word, upos, baseLabel, lexicon);
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
