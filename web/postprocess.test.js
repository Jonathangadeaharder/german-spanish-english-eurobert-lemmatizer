import assert from "node:assert/strict";
import test from "node:test";
import { resolveLemma } from "./postprocess.js";

test("resolveLemma prefers lexicon for German identity predictions", () => {
  assert.deepEqual(resolveLemma("war", "AUX", "IDENTITY", { war: "sein" }), {
    lemma: "sein",
    source: "lexicon",
  });
});

test("resolveLemma keeps case-specific lexicon evidence", () => {
  assert.deepEqual(resolveLemma("War", "AUX", "IDENTITY", { War: "sein" }), {
    lemma: "sein",
    source: "lexicon",
  });
});

test("resolveLemma still applies edit labels when the lexicon has no entry", () => {
  assert.deepEqual(resolveLemma("Hunde", "NOUN", "P4|S0|De|I", {}), {
    lemma: "Hund",
    source: "edit",
  });
});

test("resolveLemma keeps proper nouns out of lemma lookup", () => {
  assert.deepEqual(resolveLemma("War", "PROPN", "IDENTITY", { War: "sein" }), {
    lemma: null,
    source: "propn",
  });
});
