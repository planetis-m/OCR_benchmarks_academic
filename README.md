# Scientific OCR Benchmark

Benchmark for academic lecture slides using DeepInfra OCR models:

1. `allenai/olmOCR-2-7B-1025`
2. `PaddlePaddle/PaddleOCR-VL-0.9B`
3. `deepseek-ai/DeepSeek-OCR`

## Active pipeline

1. `run_benchmark.py`
   - uses bundled binary:
     - `benchmark/scientific_ocr/pdfocr-linux-x86_64/pdfocr`
   - runs `pdfocr <benchmark_pages_all.pdf> --all-pages` once per model
   - updates `pdfocr-linux-x86_64/config.json` model field per run
   - writes OCR outputs to `results/ocr_text/*.jsonl`

2. `evaluate_benchmark.py`
   - evaluates OCR outputs against locked gold labels:
     - `data/gold/gold_labels_human.jsonl`
   - writes metrics and recommendations in `results/metrics/`

## Run

```bash
python3 benchmark/scientific_ocr/scripts/run_benchmark.py
python3 benchmark/scientific_ocr/scripts/evaluate_benchmark.py
```

Or:

```bash
bash benchmark/scientific_ocr/scripts/run_all.sh
```

## Current layout

```text
benchmark/scientific_ocr/
  config/
    benchmark_config.json
  data/
    documents/
      benchmark_pages_all.pdf
    manifests/
      documents.jsonl
      pages.jsonl
      selection_summary.json
    gold/
      gold_labels_human.jsonl
      gold_labels_human.lock.json
  pdfocr-linux-x86_64/
    pdfocr
    config.json
    libpdfium.so
  scripts/
    run_benchmark.py
    evaluate_benchmark.py
    run_all.sh
  results/
    ocr_text/
    raw/
    metrics/
```

## Notes

- `run_benchmark.py` keeps the benchmark output directories (`results/raw`, `results/ocr_text`) clean by recreating them on each run.
- The runner estimates token usage/cost from OCR text length because `pdfocr` JSONL output does not expose provider usage fields.
- The benchmark dataset is fixed in-repo (consolidated PDF + manifests + locked gold labels).
