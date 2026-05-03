---
path: README.md
section: repo-root
doc-type: package-readme
status: stable
last_updated: 2026-05-03
---
# grounding-toolkit

Multi-tier groundedness & hallucination detection for LLM outputs.

The 7th sibling of the Sentinel monorepo, alongside `cognis-toolkit`,
`exchequer-toolkit`, `dspy-toolkit`, `rlm-toolkit`, `ocr-toolkit`, and
`pdf-finder-toolkit`.

## Features

- **Cascade verification**: consensus prior, lexical, semantic, NLI, LLM-judge
- **Preventive grounding**: structured signatures that force `citation_span` emission
- **Multimodal**: tables, KV pairs, figures, signatures
- **Numerical derivation**: generic `DerivationVerifier` that re-computes user-supplied formulas against grounded primitives
- **Temporal & definitional consistency**
- **Cross-document grounding** (multi-doc linker)
- **Multilingual** (locale-tag driven, no hard-coded languages)
- **Explainability**: evidence pointers + conflict spans + reasoning trace
- **Confidence calibration**: Bayesian posterior + uncertainty quantification
- **Audit**: cryptographic Merkle proofs + immutable reasoning logs
- **Adversarial robustness**
- **Calibration framework**: gold-truth schema + Bayesian threshold tuner + feedback loop
- **Eval harness**: RAGAS, DeepEval, TruLens adapters
- **Constitutional rules engine**: pluggable YAML-loaded rules (consumer supplies the rule set; toolkit only ships the matcher)

## Design principles

- **Provider-agnostic**: embedding / NLI / LLM-judge / retrieval are
  Protocols. Inject any backend (LLM, OpenAI, local Sentence-Transformers,
  custom, ...).
- **Domain-agnostic**: zero domain knowledge baked in. The toolkit
  ships a pure verification engine. Domain-specific assets — financial
  ratios, legal-clause libraries, regulatory rule sets — live in the
  consumer and are passed as data.
- **Single-tenant friendly, multi-tenant ready**: per-tenant `Source`
  objects + injectable audit logger.

## Quick start

```python
from grounding import GroundingVerifier, Claim, Source

verifier = GroundingVerifier()
claim = Claim(text='Total debt is EUR 8.4M', citation_span=(5, 1240, 1280))
source = Source.from_text(document_full_text)

result = verifier.verify(claim, source)
print(result.verdict)             # GROUNDED | UNGROUNDED | UNCERTAIN
print(result.confidence)          # 0.0 - 1.0
print(result.evidence_pointers)   # [(doc_id, page, char_start, char_end), ...]
```

## Installation

```bash
pip install grounding-toolkit              # core (lexical + consensus tiers)
pip install grounding-toolkit[all]         # full stack with all integrations
pip install grounding-toolkit[cognis,dspy] # selective
```

## Documentation

See [STRUCTURE.md](STRUCTURE.md) for repository layout and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
