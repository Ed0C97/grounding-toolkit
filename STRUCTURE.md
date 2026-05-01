# Repository structure

```
grounding-toolkit/
├── pyproject.toml                CalVer 2026.5.15.0, MIT, sentinel_provisioning entry-point
├── README.md
├── CHANGELOG.md
├── ROADMAP.md
├── STRUCTURE.md                  this file
├── SECURITY.md
├── SUPPORT.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── AUTHORS.md
├── MAINTAINERS.md
├── LICENSE                       MIT
├── Makefile                      install / install-dev / test / lint / format / check / build / clean / smoke
├── src/grounding/
│   ├── __init__.py               public re-exports (top-level surface)
│   ├── py.typed                  PEP 561 marker
│   │
│   ├── provisioning/             bridge for Sentinel test discovery
│   │   ├── __init__.py           descriptor() + runner() functions
│   │   └── manifest.yaml         test groups exposed to Sentinel
│   │
│   ├── testing/                  public test framework
│   │   ├── __init__.py           re-exports stubs / factories / matchers
│   │   ├── stubs.py              Stub<Backend> implementations of grounding Protocols
│   │   ├── factories.py          canonical dataclass builders
│   │   ├── fixtures.py           pytest plugin
│   │   ├── matchers.py           assert_* helpers
│   │   └── smoke.py              run() -> bool, 10-second self-check
│   │
│   ├── cli/                      grounding CLI
│   │   ├── __init__.py
│   │   ├── main.py               entrypoint for `grounding` command
│   │   └── templates/            scaffolding YAMLs
│   │
│   ├── core/                     cascade orchestrator + types + Protocols
│   │   ├── types.py              Claim, Source, Verdict, EvidencePointer, GroundingResult
│   │   ├── ports.py              EmbeddingFn, NLIFn, LLMJudgeFn, RetrievalFn (Protocols)
│   │   ├── cascade.py            Cascade orchestrator (tier ordering + short-circuit)
│   │   ├── thresholds.py         Threshold modulation by consensus prior
│   │   └── speculative.py        Fast pre-screen (citation-span deterministic match)
│   │
│   ├── tiers/                    verification tiers
│   │   ├── consensus.py          Tier −1: consensus-prior modulation
│   │   ├── lexical.py            Tier 0+1: substring + difflib + similarity utils
│   │   ├── semantic.py           Tier 2: cosine via EmbeddingFn (D5 stub)
│   │   ├── nli.py                Tier 3: cross-encoder via NLIFn (D5 stub)
│   │   └── llm_judge.py          Tier 4: LLM-as-judge via LLMJudgeFn (D5 stub)
│   │
│   ├── citations/                preventive grounding
│   │   ├── span.py               deterministic citation-span verifier
│   │   ├── structured_signature.py   DSPy signature forcing span emission
│   │   ├── provenance.py         Provenance DAG
│   │   └── web_verify.py         (migrated from Sentinel citation_verifier.py)
│   │
│   ├── multimodal/               Source.tables / .kv_pairs / .figures / .signatures
│   │   ├── tables.py             verify numbers against tables
│   │   ├── kv.py                 verify against key-value pairs
│   │   ├── figures.py            stub
│   │   └── signatures.py         stub
│   │
│   ├── numerical/
│   │   ├── number_extraction.py  locale-aware extractor (EU/US/IT formats)
│   │   ├── derivation_check.py   generic DerivationVerifier (formula injected by consumer)
│   │   └── ratio_consistency.py  formula-agnostic ratio coherence
│   │
│   ├── temporal/
│   │   └── date_grounding.py     verify date claims against document timeline
│   │
│   ├── definitional/
│   │   └── consistency.py        term-definition usage consistency
│   │
│   ├── crossdoc/
│   │   ├── linker.py             multi-doc claim linker
│   │   └── retriever.py          supporting evidence retrieval
│   │
│   ├── language/
│   │   └── multilingual.py       locale-tag driven cross-lingua verification
│   │
│   ├── explainability/
│   │   ├── evidence_pointer.py   (doc_id, page, char_start, char_end)
│   │   ├── conflict.py           contradiction span identification
│   │   └── reasoning_trace.py    reproducible decision trace
│   │
│   ├── confidence/
│   │   ├── bayesian.py           posterior probability of grounded-ness
│   │   └── uncertainty.py        Brier / ECE / uncertainty quantification
│   │
│   ├── audit/
│   │   ├── merkle_proof.py       Merkle root over evidence_span list
│   │   └── reasoning_log.py      append-only immutable log
│   │
│   ├── adversarial/
│   │   ├── perturbation.py       synthetic perturbation detection
│   │   └── robustness.py         robustness checks
│   │
│   ├── calibration/
│   │   ├── dataset_schema.py     Pydantic schema for gold-truth dataset
│   │   ├── tuner.py              Bayesian threshold tuning
│   │   ├── feedback_loop.py      online learning from analyst overrides
│   │   └── golden/               placeholder for annotated dataset
│   │
│   ├── eval/
│   │   ├── ragas_metrics.py      Faithfulness, Context Precision/Recall
│   │   ├── deepeval_adapter.py
│   │   ├── trulens_adapter.py
│   │   ├── rag_feedback.py       (migrated from Sentinel feedback_funcs)
│   │   └── benchmark_runner.py
│   │
│   ├── constitutional/
│   │   └── rules.py              YAML-driven rules engine (rules supplied by consumer)
│   │
│   ├── consensus/
│   │   └── quorum.py             (migrated from Sentinel SupremeJudge.aggregate)
│   │
│   ├── tracking/
│   │   └── event_tracker.py      (migrated from Sentinel hallucination_tracker)
│   │
│   ├── spatial/
│   │   └── bbox.py               (migrated from Sentinel bbox_grounding)
│   │
│   └── answer/
│       └── verifier.py           (migrated from Sentinel chat_verifier)
│
├── tests/
│   ├── conftest.py               shared fixtures
│   ├── unit/                     default testpath, zero-skip
│   │   ├── conftest.py
│   │   └── test_*.py
│   ├── integration/              opt-in (live providers)
│   ├── parity/                   cross-toolkit contract mirror
│   ├── e2e/
│   └── benchmarks/
│
├── examples/
│   ├── 01_lexical_cascade.py
│   ├── 02_preventive_grounding.py
│   ├── 03_multimodal_table.py
│   ├── 04_explainability.py
│   └── README.md
│
└── docs/
    ├── ARCHITECTURE.md
    └── PHASE_0_NOTES.md
```

## Sentinel-side wiring (consumer)

```
sentinel/
├── adapters/grounding.py              build Source from Document, Claim from DDFinding,
│                                      call grounding-toolkit, emit events
├── agents/dd_analyzer.py               calls adapter for page / evidence / definition / numerical
├── agents/extractor.py                 calls adapter for clause grounding
├── agents/resolver.py                  calls adapter for definition grounding
├── utils/grounding_audit.py            integrates grounding.audit.merkle_proof
└── tests/provisioning/grounding/       provisioning tests
```
