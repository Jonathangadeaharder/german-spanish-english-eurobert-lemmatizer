export function stripLanguagePrefix(label, lang) {
  const prefix = `${lang}::`;

  if (!label.startsWith(prefix)) {
    return null;
  }

  return label.slice(prefix.length);
}

export function buildLanguageLabelIds(id2label) {
  const labelIds = {
    de: [],
    es: [],
    en: [],
  };

  for (const [idString, label] of Object.entries(id2label)) {
    if (label === "UNKNOWN") {
      continue;
    }

    for (const lang of Object.keys(labelIds)) {
      if (label.startsWith(`${lang}::`)) {
        labelIds[lang].push(Number(idString));
        break;
      }
    }
  }

  for (const lang of Object.keys(labelIds)) {
    labelIds[lang].sort((a, b) => a - b);
  }

  return labelIds;
}

export function selectBestLabel(logitsRow, candidateIds) {
  if (!candidateIds || candidateIds.length === 0) {
    return null;
  }

  let bestId = candidateIds[0];
  let bestValue = logitsRow[bestId];

  for (let i = 1; i < candidateIds.length; i++) {
    const candidateId = candidateIds[i];
    const candidateValue = logitsRow[candidateId];

    if (candidateValue > bestValue) {
      bestValue = candidateValue;
      bestId = candidateId;
    }
  }

  return bestId;
}

export function selectValidLabel(logitsRow, candidateIds, id2label, lang, word, topK = 12) {
  if (!candidateIds || candidateIds.length === 0) {
    return "IDENTITY";
  }

  const ordered = [...candidateIds].sort((a, b) => logitsRow[b] - logitsRow[a]);
  const limit = topK > 0 ? Math.min(topK, ordered.length) : ordered.length;

  for (let i = 0; i < limit; i++) {
    const label = id2label[String(ordered[i])] || "UNKNOWN";

    if (label === "UNKNOWN") {
      continue;
    }

    const baseLabel = stripLanguagePrefix(label, lang);

    if (applyEditLabel(word, baseLabel) !== null) {
      return baseLabel;
    }
  }

  return "IDENTITY";
}

export function argmax(values) {
  if (!values || values.length === 0) {
    return -1;
  }

  let bestIndex = 0;
  let bestValue = values[0];

  for (let index = 1; index < values.length; index++) {
    if (values[index] > bestValue) {
      bestValue = values[index];
      bestIndex = index;
    }
  }

  return bestIndex;
}

export function lookupLexicon(lexicon, word) {
  if (!lexicon) {
    return null;
  }

  return Object.prototype.hasOwnProperty.call(lexicon, word)
    ? lexicon[word]
    : null;
}

export function resolveLemma(word, upos, baseLabel, lexicon) {
  if (upos === "PROPN") {
    return { lemma: null, source: "propn" };
  }

  if (baseLabel !== null) {
    const applied = applyEditLabel(word, baseLabel);

    if (applied !== null) {
      return { lemma: applied, source: "edit" };
    }
  }

  const lexiconLemma = lookupLexicon(lexicon, word);

  if (lexiconLemma !== null) {
    return { lemma: lexiconLemma, source: "lexicon" };
  }

  return { lemma: word, source: "identity" };
}

export function applyEditLabel(word, baseLabel) {
  if (baseLabel === "IDENTITY") {
    return word;
  }

  if (baseLabel === "LOWERCASE") {
    return word.toLowerCase();
  }

  if (!baseLabel.startsWith("P")) {
    return null;
  }

  const parts = baseLabel.split("|");

  if (parts.length !== 4) {
    return null;
  }

  const prefixLen = Number(parts[0].slice(1));
  const suffixLen = Number(parts[1].slice(1));
  const deleteMid = parts[2].slice(1);
  const insertMid = parts[3].slice(1);

  if (!Number.isInteger(prefixLen) || !Number.isInteger(suffixLen)) {
    return null;
  }

  if (prefixLen + suffixLen > word.length) {
    return null;
  }

  const midStart = prefixLen;
  const midEnd = suffixLen > 0 ? word.length - suffixLen : word.length;

  const actualDelete = word.slice(midStart, midEnd);

  if (actualDelete !== deleteMid) {
    return null;
  }

  const prefix = word.slice(0, prefixLen);
  const suffix = suffixLen > 0 ? word.slice(word.length - suffixLen) : "";

  return prefix + insertMid + suffix;
}

export function simpleTokenizer(text) {
  const regex = /[\p{L}\p{M}]+|[\p{N}]+|[^\s]/gu;
  return text.match(regex) || [];
}

export function languageToken(lang) {
  if (lang === "de") return "[LANG_DE]";
  if (lang === "es") return "[LANG_ES]";
  if (lang === "en") return "[LANG_EN]";
  throw new Error(`Unsupported language: ${lang}`);
}
