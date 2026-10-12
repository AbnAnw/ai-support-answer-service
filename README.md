# AI Support Answer Service

A small, production-minded command-line service that answers support questions only when a local FAQ knowledge base provides sufficient evidence. It returns exact source citations for supported answers and abstains instead of guessing when evidence is missing, partial, conflicting, or restricted.

The default implementation is deterministic, offline, standard-library-only, and requires no API key.

## Pipeline

```text
LOAD_KB → RETRIEVE → CHECK_SUPPORT → ANSWER | ABSTAIN
```

The stages are separate modules. Answer generation receives an explicit support decision and refuses to run unless support has already been approved.

1. **LOAD_KB** reads and validates `kb.json` from disk.
2. **RETRIEVE** ranks the current records with weighted lexical scoring.
3. **CHECK_SUPPORT** applies numeric thresholds and conservative policy overrides.
4. **ANSWER** extracts source sentences and cites them, or **ABSTAIN** returns a safe fallback.

## Prerequisites

- Python 3.10 or newer
- No package installation, network connection, API key, or external service

## Run

From the project directory:

```bash
python app.py "How do I reset my password?"
```

Use another knowledge-base file:

```bash
python app.py --kb path/to/kb.json "Where can I download an invoice?"
```

The default `kb.json` is resolved relative to `app.py`, so the first command also works when `app.py` is invoked from another working directory.

The application writes one JSON object to standard output. Invalid questions or KB files produce a concise diagnostic on standard error and exit with status `2`.

## Validate and test

Run the representative end-to-end validation:

```bash
python validate.py
```

Run the complete automated suite:

```bash
python -m unittest discover -s tests -v
```

Both commands are offline and use only the Python standard library. Replacement-KB tests create temporary files and never mutate the repository's `kb.json`.

## Example answer

```json
{
  "question": "How do I reset my password?",
  "decision": "answer",
  "answer": "Users can reset their password from the login page by clicking 'Forgot password'.",
  "citations": [
    {
      "id": "doc_1",
      "title": "Password reset",
      "snippet": "Users can reset their password from the login page by clicking 'Forgot password'."
    }
  ],
  "debug": {
    "retrieved_ids": ["doc_1"],
    "support_score": 0.892798
  }
}
```

## Example abstention

```json
{
  "question": "Can support change my email address for me?",
  "decision": "abstain",
  "answer": "I don't have enough supported information in the knowledge base to answer that. Please contact human support for assistance.",
  "citations": [],
  "debug": {
    "retrieved_ids": ["doc_5", "doc_1", "doc_2"],
    "support_score": 0.578221
  }
}
```

Scores are deterministic for a particular question, KB, and implementation, but may change if any of those inputs change.

## Knowledge-base format

The KB must be a non-empty JSON array:

```json
[
  {
    "id": "doc_1",
    "title": "Password reset",
    "text": "Users can reset their password from the login page."
  }
]
```

Every record requires a unique, non-blank string `id` plus non-blank string `title` and `text` fields. Unknown fields are ignored. The loader reports helpful errors for missing or unreadable files, invalid JSON, non-array roots, empty arrays, invalid records, missing/wrong/blank fields, and duplicate IDs.

To replace the KB, either overwrite `kb.json` using the same schema or supply `--kb`. Records are always loaded from disk at request time; sample answers are not embedded in source code.

## Retrieval method

Preprocessing applies Unicode NFKC normalization, case folding, Unicode-aware word extraction, a small English stop-word set, conservative suffix normalization, and adjacent bigrams. Negations, quantities, and time units are retained.

For each request, the retriever builds a small in-memory index from the current KB:

- smoothed IDF: `log((N + 1) / (df + 1)) + 1`;
- sublinear term frequency: `1 + log(count)`;
- title terms weighted `2.0×` and article terms weighted `1.0×`;
- deterministic tie-breaking by document ID;
- up to three positive-score results.

The normalized retrieval score is:

```text
0.70 × TF-IDF cosine similarity
+ 0.20 × IDF-weighted query-term coverage
+ 0.10 × query-bigram coverage
```

This uses more than raw shared-word counts: common function words are removed, rare terms receive more weight, phrase overlap is rewarded, and query coverage is explicit.

## Support rule

The support checker uses only the highest-ranked coherent document. It does not combine unrelated fragments from multiple documents to manufacture an answer.

Its score, clamped to `0.0–1.0`, is:

```text
0.40 × top retrieval score
+ 0.30 × meaningful concept coverage
+ 0.20 × critical-facet coverage
+ 0.10 × phrase consistency
```

`support_score` is therefore lexical confidence that the top evidence covers the complete request; it is not a probability.

An answer requires all of the following:

- support score at least `0.62`;
- concept coverage at least `0.60`;
- critical-facet coverage at least `0.75`;
- coherent evidence with at least two meaningful concepts in one sentence or a matching bigram;
- no missing critical term, requested number/time unit, or polarity conflict.

Conservative overrides apply even when a numeric score is high:

- legal- or tax-advice requests abstain;
- guarantees require explicit guarantee-equivalent evidence for the same outcome;
- requested numbers and time units must match the evidence;
- actor/action/object details must be present rather than merely topically related;
- a multi-part request abstains when a central part is unsupported.

An abstention usually has no citations. A citation may be included when a KB passage explicitly states the legal/tax restriction that caused the abstention.

## Answers and citations

The generator is extractive. It ranks sentences within the approved document and selects the smallest one- or two-sentence evidence set that covers the question. It does not paraphrase procedures, invent details, or add a model-generated claim.

Each citation contains the document ID, exact title, and a snippet copied verbatim from that document's `text`. Runtime invariant checks confirm that every citation references a loaded record, its title matches, and its snippet occurs in the source. Every `answer` result must have at least one citation.

## Chunking choice and tradeoff

Whole FAQ records are ranked because the supplied articles are short and nearby sentences may supply necessary context. Sentences are ranked separately for concise answers and citations.

This design is transparent and effective for small FAQ collections. Very long replacement articles may dilute document-level similarity; sentence- or paragraph-level indexing would be a better extension for long-form corpora.

## Project layout

```text
app.py                         CLI and JSON serialization boundary
kb.json                        Sample runtime knowledge base
validate.py                    Seven representative end-to-end checks
support_service/
  __init__.py                  Public package exports
  models.py                    Immutable stage data contracts
  kb.py                        Disk loading and schema validation
  retrieval.py                 Normalization, TF-IDF ranking, sentence ranking
  support.py                   Explicit support gate and policy overrides
  answering.py                 Extractive answers and citations
  service.py                   Ordered pipeline and output invariants
tests/
  test_kb.py                   Loader and invalid-input coverage
  test_retrieval.py            Ranking and normalization coverage
  test_service.py              Support, grounding, schema, replacement-KB tests
  test_cli.py                  CLI, errors, JSON, and path resolution
outputs/
  AI_SUPPORT_SERVICE_SPEC.md   Pre-implementation specification
```

## Public Python entry point

```python
from support_service import answer_question

result = answer_question("How do I reset my password?", "kb.json")
```

The loader, retriever, support checker, and generator are also importable and independently testable.

## Limitations

- Matching is English-oriented and lexical; unfamiliar synonyms or other languages may cause safe false abstentions.
- The lightweight suffix normalizer is intentionally not a full linguistic stemmer.
- Sentence splitting is punctuation-based and may be imperfect around abbreviations.
- Whole-document retrieval is intended for short FAQ articles, not long manuals.
- The critical-facet checker cannot provide full semantic or contradiction understanding.
- Thresholds may need reevaluation for a substantially different domain or document-length distribution.
- The CLI processes one independent question at a time and has no conversation memory.

## Optional LLM extension

An LLM-backed generator could implement the same generator boundary, but it must receive only support-approved evidence, remain behind the existing support gate, and pass the same citation and output invariants. The deterministic extractive generator must remain the default and offline fallback.

