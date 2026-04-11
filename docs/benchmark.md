# TeleRAG Benchmark Guide

## Goal

This benchmark is designed to make TeleRAG interview-ready for LLM engineering roles by turning the project into a measurable system instead of a UI-only demo.

## Dataset

- Location: `data/eval/beamforming_eval.json`
- Format:
  - `question`
  - `source_doc`
  - `expected_keywords`
- Current scope:
  - definition-style questions
  - bilingual questions
  - document-source verification

## Metrics

- `retrieval_hit_rate`: whether the expected source document appears in retrieved sources
- `answer_keyword_coverage`: fraction of expected keywords present in the final answer
- `avg_total_ms`: average end-to-end response latency

## Experiment Set

Run the benchmark with:

```bash
python scripts/run_benchmark.py
```

Recommended comparisons:

| Variant | Rerank | top_k | Goal |
| --- | --- | --- | --- |
| Fast retrieval | off | 2 | lower latency baseline |
| Balanced pipeline | on | 3 | stronger answer quality |

## Result Snapshot Template

The benchmark script writes a machine-readable summary to `docs/benchmark_results.json`.

Use the latest run to populate a resume-friendly table like this:

| Variant | Retrieval Hit Rate | Answer Keyword Coverage | Avg Total Latency |
| --- | --- | --- | --- |
| Fast retrieval | fill from script | fill from script | fill from script |
| Balanced pipeline | fill from script | fill from script | fill from script |

## How To Present This In Interviews

- Start with the problem: technical-document QA with traceable sources.
- Explain the architecture: chunking -> embedding retrieval -> optional rerank -> answer generation -> answer cleanup.
- Show the tradeoff: rerank improves relevance but increases latency.
- Quote one or two measured metrics from `benchmark_results.json`.
