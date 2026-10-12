# AI Support Answer Service — Specification and Design Notes

**Status:** Pre-implementation specification  
**Target:** Python 3.10+ command-line application  
**Default runtime:** Standard library only; no network access, API key, or external service  
**Source brief:** User-provided implementation requirements dated 2026-10-12

## 1. Purpose

Build a small support-answer service that answers a user's question only when a local FAQ knowledge base contains sufficient evidence. Every answer must be traceable to cited source text. When evidence is weak, incomplete, conflicting, or explicitly restricted, the service must abstain and direct the user to human support.

The required processing invariant is:

```text
LOAD_KB → RETRIEVE → CHECK_SUPPORT → ANSWER | ABSTAIN
```

No answer text may be generated before retrieval and support checking finish successfully.

## 2. Goals and non-goals

### Goals

- Read and validate an arbitrary `kb.json` at runtime.
- Retrieve relevant FAQ documents with a deterministic lexical method.
- make a separate, conservative, explainable support decision.
- Generate concise answers using only sentences from supported evidence.
- Return citations whose snippets occur verbatim in the cited KB record.
- Provide stable JSON output, automated tests, and an offline validation script.
- Run from a clean checkout without installation when a supported Python version is present.

### Non-goals

- Conversational memory or multi-turn dialogue.
- Web search, remote databases, embeddings, model calls, or API-key management.
- Personalized legal, tax, financial, or compliance advice.
- Guessing procedures, guarantees, timelines, eligibility, or product behavior.
- A graphical interface, user accounts, telemetry, or production hosting.

## 3. Repository and delivery constraints

The current workspace contains only `outputs/` and `work/`, has no repository-level instruction file, and is not a Git repository as of this specification. Implementation should therefore create a compact new project rather than modify unrelated material.

Planned implementation files:

```text
app.py
kb.json
validate.py
README.md
.gitignore
support_service/
  __init__.py
  models.py
  kb.py
  retrieval.py
  support.py
  answering.py
  service.py
tests/
  test_kb.py
  test_service.py
  test_cli.py
```

Generated caches, virtual environments, coverage output, secrets, and test artifacts are excluded from delivery.

## 4. User-visible behavior

### Command

```bash
python app.py "How do I reset my password?"
python app.py --kb path/to/kb.json "Question"
```

The default KB path is resolved relative to `app.py`, not the caller's current working directory.

### Success output

The application writes exactly one JSON object to standard output:

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
    "support_score": 0.86
  }
}
```

The exact score above is illustrative; the implemented score is calculated from the current KB.

### Abstention output

```json
{
  "question": "Can support change my email address for me?",
  "decision": "abstain",
  "answer": "I don't have enough supported information in the knowledge base to answer that. Please contact human support for assistance.",
  "citations": [],
  "debug": {
    "retrieved_ids": ["doc_5"],
    "support_score": 0.31
  }
}
```

For a recognized legal/tax restriction, a citation to the restriction passage may be included, but the decision remains `"abstain"`.

### Input and process errors

- An empty or whitespace-only question is a usage error.
- KB loading or validation errors are written to standard error with a concise path-aware message.
- User/input errors return a non-zero exit code (planned: `2`). Unexpected internal failures return a distinct non-zero code (planned: `1`).
- No partial result object is emitted when the pipeline cannot load a valid KB.
- JSON is serialized with Unicode preserved (`ensure_ascii=False`).

## 5. Knowledge-base contract

`kb.json` must contain a JSON array with at least one record. Each record must contain:

| Field | Type | Validation |
|---|---|---|
| `id` | string | Required, trimmed value non-empty, unique across records |
| `title` | string | Required, trimmed value non-empty |
| `text` | string | Required, trimmed value non-empty |

Unknown fields may be ignored for forward compatibility. Invalid input must produce a helpful error identifying the file and, where applicable, the array index or duplicated ID. Validation distinguishes:

- file not found or unreadable;
- malformed JSON, including line/column when available;
- non-array root;
- empty array;
- non-object record;
- missing field;
- field with wrong type or blank value;
- duplicate `id`.

The sample records from the brief are stored only in `kb.json`; answers are never copied into Python constants.

## 6. Internal data contracts

Typed immutable data structures should represent:

- `Document(id, title, text)`;
- `RetrievalResult(document, score, matched_terms, best_sentences)`;
- `SupportDecision(supported, score, reason_code, evidence)`;
- `Citation(id, title, snippet)`;
- `ServiceResult(question, decision, answer, citations, debug)`.

The public service entry point is conceptually:

```python
answer_question(question: str, kb_path: Path, top_k: int = 3) -> dict
```

The loader, retriever, support checker, and answer generator each expose independently testable functions. The answer generator accepts an affirmative `SupportDecision`; it must reject or abstain if called without one.

## 7. Text preprocessing

Preprocessing is deterministic and applied consistently to questions, titles, and article text:

1. Normalize Unicode with NFKC.
2. Apply Unicode-aware case folding.
3. Extract alphanumeric word tokens, retaining numbers.
4. Remove a small documented English stop-word set containing function words only.
5. Apply conservative suffix normalization for common plural and verb endings only when the remaining stem is sufficiently long.
6. Form adjacent bigrams from the remaining tokens.

Negations (`not`, `no`, `cannot`, `without`, `never`), quantities, time units, domain nouns, and action verbs are never treated as stop words because they can change the requested fact.

Articles are ranked as whole documents because the expected FAQ entries are short and context can span adjacent sentences. Sentence splitting is also performed so the answer and citation stage can select the smallest supporting excerpts. This avoids losing local context during retrieval while keeping citations precise. The tradeoff is that unusually long replacement articles may dilute document-level relevance; sentence-level indexing is the intended future upgrade for such KBs.

## 8. Retrieval design

### Index

At request time, build an in-memory index from the loaded KB:

- document frequency is calculated over all records;
- inverse document frequency uses smoothed IDF: `idf(t) = log((N + 1) / (df(t) + 1)) + 1`;
- term frequency uses sublinear scaling: `tf(t) = 1 + log(count(t))`;
- title term weights are multiplied by `2.0` before vector normalization;
- text terms retain weight `1.0`.

### Score

For each document:

1. Calculate TF-IDF cosine similarity between the question and the weighted title-plus-text vector.
2. Calculate meaningful query-term coverage using query IDF weights.
3. Calculate question/document bigram overlap.
4. Combine and clamp to `[0, 1]`:

```text
retrieval_score = 0.70 × cosine
                + 0.20 × weighted_term_coverage
                + 0.10 × bigram_coverage
```

Results are ordered by descending score, then by document ID for deterministic tie-breaking. Return at most `top_k=3` records with a positive score. `retrieved_ids` preserves this actual order.

The use of IDF, stop-word removal, phrase overlap, and query coverage prevents a single generic shared word from dominating retrieval.

## 9. Explicit support gate

Support checking is separate from retrieval and runs before answer construction.

### Evidence signals

For the top result and its candidate evidence sentences, calculate:

- **retrieval confidence:** top retrieval score, normalized to `[0, 1]`;
- **concept coverage:** fraction of query content-term IDF weight present in the evidence;
- **critical-facet coverage:** coverage of numbers, time units, negations, action/permission verbs, and specific requested objects not treated as generic support terms;
- **phrase consistency:** query bigram overlap or equivalent same-sentence co-occurrence;
- **topic consistency:** whether evidence for the question comes from one coherent document rather than unrelated matches across documents.

The proposed support score is:

```text
support_score = clamp(
    0.40 × retrieval_confidence
  + 0.30 × concept_coverage
  + 0.20 × critical_facet_coverage
  + 0.10 × phrase_consistency,
  0, 1
)
```

An answer is allowed only when all of these are true:

- `support_score >= 0.62`;
- `concept_coverage >= 0.60`;
- `critical_facet_coverage >= 0.75`;
- at least one evidence sentence contains two meaningful query concepts, or a matching bigram;
- the evidence does not conflict with the request;
- every independently requested clause is supported by the same coherent evidence set.

Thresholds are implementation constants documented in the README and exercised by boundary tests. They may be tuned only against generic behavioral cases, not by embedding sample answers.

### Mandatory conservative overrides

These rules override a numeric pass:

- Requests for legal advice or tax advice always abstain. If the KB explicitly states the restriction, that passage may be cited.
- Requests for a guarantee abstain unless the evidence explicitly uses guarantee-equivalent language for the same outcome and constraint.
- A requested number, duration, deadline, channel, actor, or object must be present and semantically consistent in the evidence. A general timeline cannot support a different promised timeline.
- Permission/action questions require evidence for the same actor, action, and object. Address-change evidence cannot support an email-change request.
- Contradictory top evidence or materially split evidence causes abstention.
- For multi-part questions, failure of a central clause causes full abstention. The initial version does not emit partially answered composites.

Reason codes such as `SUPPORTED`, `LOW_SCORE`, `MISSING_CRITICAL_FACET`, `RESTRICTED_ADVICE`, `UNSUPPORTED_GUARANTEE`, and `CONFLICTING_EVIDENCE` are internal diagnostics and are not exposed as chain-of-thought.

## 10. Answer and citation generation

The default generator is deterministic and extractive:

1. Receive only the evidence approved by the support gate.
2. Rank sentences within the supported document by covered query concepts, critical-facet coverage, and original order.
3. Select the smallest one or two sentences that cover the supported request.
4. Preserve wording from the KB. A light lead-in such as `According to the knowledge base,` is allowed, but it must not add a factual claim.
5. Deduplicate repeated sentences and cap the response to keep it concise.

Each citation corresponds to a selected source record. Its `snippet` is the exact source sentence or contiguous exact substring from `Document.text`; tests verify substring membership. One citation per document is sufficient, with multiple selected sentences kept contiguous where practical. An answered result must have at least one citation.

Abstention uses the fixed fallback message from the brief. Citations are normally empty; they are included only when an explicit KB restriction directly explains the abstention.

## 11. Pipeline orchestration

`service.py` owns the only normal execution path:

```text
validate question
  → load and validate KB
  → preprocess/index KB
  → retrieve top-k
  → check support
      → if unsupported: construct abstention
      → if supported: generate answer, then citations
  → assert output invariants
  → serialize result
```

Output invariants checked before serialization:

- top-level keys are exactly `question`, `decision`, `answer`, `citations`, `debug`;
- `decision` is exactly `answer` or `abstain`;
- `question` and `answer` are strings;
- `citations` is a list of objects with exactly `id`, `title`, `snippet`;
- `debug` contains exactly `retrieved_ids` and numeric `support_score`;
- `support_score` is finite and within `[0.0, 1.0]`;
- every retrieved/cited ID exists in the loaded KB;
- each citation title matches its document and snippet is found in its text;
- an answer has at least one citation;
- an abstention still contains all required fields.

## 12. Test plan

Use standard-library `unittest`; tests require no network, secret, or external package.

### KB loader tests

- Loads the repository sample from disk.
- Reports a missing file clearly.
- Rejects malformed JSON and non-array roots.
- Rejects an empty array.
- Rejects non-object records and missing, non-string, or blank required fields.
- Rejects duplicate IDs.
- Accepts Unicode content.

### Retrieval tests

- Ranks an exact topical/phrase match above documents sharing only generic words.
- Applies title weighting without making title overlap sufficient by itself.
- Produces deterministic ordering and at most top-k results.
- Preserves document IDs and titles.
- Uses replacement KB content rather than source constants.

### Support tests

- Answers a password-reset question.
- Answers a 2FA-recovery question from supported evidence.
- Answers the documented withdrawal-review timing without promising more.
- Answers account-closure requirements.
- Abstains on unrelated content.
- Abstains on a two-hour withdrawal guarantee.
- Abstains on legal or tax advice.
- Abstains when asked whether support can change an email address.
- Abstains when only one clause of a multi-part question is supported.
- Exercises scores immediately below and above the support boundary.

### Answer/citation tests

- Every answer has at least one citation.
- Citation IDs exist in the loaded KB.
- Citation titles match their records.
- Every snippet is an exact substring of the corresponding record text.
- No answer sentence comes from a non-approved document.
- Abstention wording is clear and recommends human support.

### Service and CLI contract tests

- Output has exactly the required keys and types.
- Both decision values serialize as valid JSON.
- Unicode is preserved.
- Blank questions and invalid KBs return non-zero.
- Running from a different working directory still finds the default KB.
- A temporary KB with novel content changes the answer without editing source files.

Temporary KB tests use `tempfile.TemporaryDirectory`; they never mutate the repository's `kb.json`.

## 13. Validation script

`validate.py` runs the real public pipeline against seven cases:

| Case | Expected decision |
|---|---|
| Password reset procedure | answer |
| Lost second factor / 2FA recovery | answer |
| Withdrawal review timing | answer |
| Account closure requirements | answer |
| Support changes email address | abstain |
| Tax advice for a closed account | abstain |
| Guarantee withdrawal completion in two hours | abstain |

For every case it validates the output schema and, for answers, citation presence and source integrity. It reports per-case failures, decision accuracy, and citation presence rate. Exit status is `0` only if KB loading, all expected decisions, all schema checks, and all citation checks pass.

## 14. Acceptance criteria and verification

Implementation is complete only when all of the following have been observed successfully:

1. `python -m unittest discover -s tests -v`
2. `python validate.py`
3. An answerable CLI query returns valid JSON with `decision: "answer"` and a valid citation.
4. An unsupported CLI query returns valid JSON with `decision: "abstain"`.
5. The application works when launched from outside the project directory.
6. A temporary replacement KB produces behavior based on its novel content.
7. No test, source, or runtime path requires network access or an API key.
8. Only intended project files are present or changed; generated caches and artifacts are absent.

The final implementation report must quote actual observed results and must not claim unexecuted checks.

## 15. Implementation sequence

1. Create typed models and strict KB loader.
2. Add preprocessing and deterministic retrieval, then unit-test ranking.
3. Add support scoring and mandatory overrides, then unit-test abstention boundaries.
4. Add extractive answer/citation generation with invariant checks.
5. Wire the service and CLI with stable JSON/error behavior.
6. Add the sample KB, validation runner, and README documenting actual constants.
7. Run the full verification matrix, repair failures, and inspect the final file set.

## 16. Known limitations and extension points

- Lexical matching is conservative and may abstain on valid paraphrases or synonyms not present in the KB.
- Simple sentence splitting may be imperfect for abbreviations and unusual punctuation.
- Whole-document ranking is best suited to short FAQ records; long articles should later use passage indexing.
- Generic critical-facet detection cannot provide full semantic or contradiction understanding.
- Fixed thresholds need reevaluation if document length, language, or domain changes substantially.
- The initial service handles one question per process and has no conversational context.

An optional future generator interface may accept an LLM implementation, but it must receive only approved evidence, remain behind the same support gate, preserve citation validation, and never replace the offline deterministic default.
