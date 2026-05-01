# Architecture

## Overview

`grounding-toolkit` is a multi-tier, provider-agnostic, domain-agnostic
groundedness verification engine for LLM outputs. It's the 7th sibling
of the Sentinel monorepo.

## Core abstractions

### `Claim`
A single statement to verify. Contains text, optional page reference,
optional `citation_span` (page + char offsets emitted preventively by
the LLM).

### `Source`
The ground-truth corpus to verify against. Contains:
- `text`: plain document text (typically OCR md+kv output)
- `tables`: optional list of structured tables
- `kv_pairs`: optional dict of extracted key-value pairs
- `figures`: optional list of figures
- `signatures`: optional list of signatory metadata
- `page_count`, `doc_id`, `language`

`Source` is fully populated by the consumer. The toolkit does not call
ocr-toolkit, pdf-finder, or any other parser directly — it consumes
already-extracted data.

### `Verdict`
`GROUNDED | UNGROUNDED | UNCERTAIN`

### `EvidencePointer`
Locator into the source: `(doc_id, page, char_start, char_end)`.

### `GroundingResult`
The output of verification:
- `verdict: Verdict`
- `confidence: float` in [0, 1]
- `evidence_pointers: list[EvidencePointer]`
- `conflict_pointers: list[EvidencePointer]` (passages that contradict)
- `tier_results: dict[str, TierVerdict]`
- `merkle_proof: str` (hex-encoded root)
- `reasoning_trace: list[str]`

## Cascade pipeline

```
INPUT      Claim, Source, optional consensus_metadata

STEP 1     Speculative pre-screen
           If claim has citation_span and span deterministically matches → SHORT-CIRCUIT

STEP 2     Tier −1 — consensus prior
           Modulates downstream thresholds based on consensus metadata

STEP 3     Tier 0 — substring exact                ← gratis
STEP 4     Tier 1 — fuzzy lexical (difflib)        ← gratis

STEP 5     Multimodal cross-checks
           Numbers → numerical/number_extraction + multimodal/tables
           Quoted  → multimodal/kv
           Bbox    → spatial/bbox

STEP 6     Numerical derivation (if formula spec injected)

STEP 7     Temporal + definitional consistency

STEP 8     Crossdoc retrieval (if RetrievalFn injected)

STEP 9     Tier 2 — semantic embedding             ← D5 post-Cohere→local
STEP 10    Tier 3 — NLI cross-encoder              ← D5
STEP 11    Tier 4 — LLM-as-judge                   ← D5

STEP 12    Confidence calibration (Bayesian)

STEP 13    Audit (Merkle root over evidence_pointers + reasoning_log)

OUTPUT     GroundingResult
```

## Provider Protocols

Every external dependency is a `Protocol` so the consumer injects the
implementation:

- `EmbeddingFn` — `(texts: Sequence[str]) -> List[List[float]]`
- `NLIFn` — `(claim: str, source: str) -> Dict[label, prob]`
- `LLMJudgeFn` — `(claim: str, source: str) -> Dict[verdict, rationale]`
- `RetrievalFn` — `(query: str, top_k: int) -> List[Passage]`

Stubs for each (`StubEmbeddingFn`, `StubNLIFn`, `StubLLMJudgeFn`,
`StubRetrievalFn`) live in `grounding.testing.stubs`.

## Domain-agnostic design

The toolkit ships zero domain knowledge. Specifically:

- **No financial ratios baked in**. `numerical/derivation_check.py`
  takes a formula spec from the consumer. Sentinel passes its DSCR /
  LTV / ICR formulas; another consumer would pass theirs.
- **No legal-clause libraries**.
- **No regulatory rule sets**. `constitutional/rules.py` is just an
  engine; the rules themselves are YAML files supplied by the consumer.
- **No fixed languages**. `language/multilingual.py` is locale-tag
  driven; consumer supplies the locale + glossary.

## Lockstep with the Sentinel monorepo

CalVer `YYYY.M.D.N` synchronised with the 6 existing sibling toolkits
plus `sentinel-core` itself. Every release wave updates all 7 toolkit
versions in lockstep.
