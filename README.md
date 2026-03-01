# Scientific OCR Benchmark

Benchmark for academic lecture slides using DeepInfra OCR models:

1. `allenai/olmOCR-2-7B-1025`
2. `PaddlePaddle/PaddleOCR-VL-0.9B`
3. `deepseek-ai/DeepSeek-OCR`

## Active pipeline

1. `scripts/run_benchmark.py`
   - uses bundled binary:
     - `pdfocr-linux-x86_64/pdfocr`
   - runs `pdfocr data/documents/benchmark_pages_all.pdf --all-pages` once per model
   - updates `pdfocr-linux-x86_64/config.json` model field per run
   - writes OCR outputs to `results/ocr_text/*.jsonl`

2. `scripts/evaluate_benchmark.py`
   - evaluates OCR outputs against locked gold labels:
     - `data/gold/gold_labels_human.jsonl`
   - writes metrics and recommendations in `results/metrics/`

## Run

```bash
python3 scripts/run_benchmark.py
python3 scripts/evaluate_benchmark.py
```

Or:

```bash
bash scripts/run_all.sh
```

## Current layout

```text
.
|-- config/
|   `-- benchmark_config.json
|-- data/
|   |-- documents/
|   |   `-- benchmark_pages_all.pdf
|   |-- manifests/
|   |   |-- documents.jsonl
|   |   |-- pages.jsonl
|   |   `-- selection_summary.json
|   `-- gold/
|       |-- gold_labels_human.jsonl
|       `-- gold_labels_human.lock.json
|-- pdfocr-linux-x86_64/
|   |-- pdfocr
|   |-- config.json
|   `-- libpdfium.so
|-- scripts/
|   |-- run_benchmark.py
|   |-- evaluate_benchmark.py
|   `-- run_all.sh
`-- results/
    |-- ocr_text/
    |-- raw/
    `-- metrics/
```

## Notes

- `run_benchmark.py` keeps the benchmark output directories (`results/raw`, `results/ocr_text`) clean by recreating them on each run.
- The runner estimates token usage/cost from OCR text length because `pdfocr` JSONL output does not expose provider usage fields.
- The benchmark dataset is fixed in-repo (consolidated PDF + manifests + locked gold labels).

## Current results (gold labels)

| Model | Coverage | CER | WER | ReadingOrderF1 | MathF1 | TokenRecall | CharLCSRecall | Cost (USD) | RobustBalanced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| allenai/olmOCR-2-7B-1025 | 1.000 | 0.4682 | 0.4893 | 0.3252 | 0.6743 | 0.8146 | 0.8496 | 0.0214 | 0.5232 |
| PaddlePaddle/PaddleOCR-VL-0.9B | 1.000 | 0.4634 | 0.4732 | 0.3751 | 0.7248 | 0.7678 | 0.7924 | 0.0420 | 0.4945 |
| deepseek-ai/DeepSeek-OCR | 1.000 | 0.5862 | 0.6512 | 0.1452 | 0.4684 | 0.6857 | 0.7235 | 0.0041 | 0.6540 |

- Accuracy-first (strict): `PaddlePaddle/PaddleOCR-VL-0.9B`
- Accuracy-first (robust): `allenai/olmOCR-2-7B-1025`
- Cost-first: `deepseek-ai/DeepSeek-OCR`
- Balanced (strict): `deepseek-ai/DeepSeek-OCR`
- Balanced (robust): `deepseek-ai/DeepSeek-OCR`

## Interpretation (evidence-based)

This section summarizes what the measured benchmark says, with supporting evidence from the locked gold set.

### 1) Gold-set validity and run completeness

- Gold-set coverage is complete: `68/68` benchmark pages have corresponding entries in `gold_labels_human.jsonl`.
- OCR run coverage is complete for all models: each model returned `68/68` successful pages.

Evidence:
- `data/manifests/pages.jsonl` vs `data/gold/gold_labels_human.jsonl` sample-id coverage check: no missing/extra ids.
- `results/metrics/model_summary.csv`: `n_eval_pages=68`, `n_ok=68`, `coverage=1.0` for all three models.

### 2) Expected vs observed quality ranking

Expected (prior): the larger/more expensive models might be more accurate on strict transcription metrics, while the cheapest model might be weaker but cost-efficient.

Observed:
- Strict accuracy (CER/WER/order/math aggregate) is best for `PaddlePaddle/PaddleOCR-VL-0.9B`.
- Robust content recovery (token/character recall + order/math aggregate) is best for `allenai/olmOCR-2-7B-1025`.
- `deepseek-ai/DeepSeek-OCR` is lowest-cost by a wide margin, but with materially weaker strict and robust accuracy.

Evidence:
- `results/metrics/model_summary.csv`
  - Paddle: best strict profile (`mean_cer=0.4634`, `mean_wer=0.4732`, `mean_reading_order_f1=0.3751`, `mean_math_symbol_f1=0.7248`).
  - OLM: best robust profile (`mean_token_recall=0.8146`, `mean_char_lcs_recall=0.8496`, `robust_accuracy_component=0.6659` vs Paddle `0.6651`).
  - DeepSeek: lowest cost (`total_cost_usd=0.0041`) but lower quality (`mean_wer=0.6512`, `mean_reading_order_f1=0.1452`).

### 3) Where models struggle most

All models degrade most on table-heavy and multi-column pages; this is consistent with OCR reading-order and structure sensitivity.

Evidence:
- `results/metrics/tag_breakdown.csv`
  - Paddle worst tags: `table` (`mean_wer=0.684`, `mean_reading_order_f1=0.248`), `multi_column` (`mean_wer=0.595`).
  - OLM worst tags: `table` (`mean_wer=0.676`, `mean_reading_order_f1=0.159`), `diagram` (`mean_wer=0.601`).
  - DeepSeek worst tags: `table` (`mean_wer=0.828`, `mean_reading_order_f1=0.032`), `multi_column` (`mean_wer=0.746`).

### 4) Why strict metrics can look harsher than expected

A non-trivial fraction of pages hit `CER=1`/`WER=1` even when token recall is high on some of those pages. This indicates heavy penalties from ordering/format differences and long-sequence edit distance effects, not only complete text failure.

Evidence:
- `results/metrics/per_page_scores.csv`
  - Pages with `CER=1`: OLM `20`, Paddle `21`, DeepSeek `28`.
  - Pages with `WER=1`: OLM `20`, Paddle `21`, DeepSeek `29`.
  - Example pages with `CER=1` but high recall:
    - Paddle `ece110__ab42139654__p0367`: `token_recall=0.978`, `reading_order_f1=0.727`.
    - OLM `ece110__ab42139654__p0367`: `token_recall=0.966`, `reading_order_f1=0.667`.
    - DeepSeek `math225__18210815f0__p0347`: `token_recall=0.990`, `reading_order_f1=0.056`.

### 5) Practical conclusions

- If the goal is highest strict transcription fidelity on this benchmark, prefer `PaddlePaddle/PaddleOCR-VL-0.9B`.
- If the goal is strongest content recovery robustness, prefer `allenai/olmOCR-2-7B-1025`.
- If the goal is minimum cost, prefer `deepseek-ai/DeepSeek-OCR`.
- If using a balanced score, interpret it together with absolute quality metrics; DeepSeek wins balanced mostly because of cost dominance rather than best raw OCR fidelity.

### 6) Important caveat

Cost in this benchmark is estimated from prompt-token constant plus output-length heuristic because `pdfocr` JSONL output does not expose provider token usage fields. Relative quality comparisons are direct; cost comparisons are best treated as approximate.
