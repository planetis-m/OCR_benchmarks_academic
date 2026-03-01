#!/usr/bin/env bash
set -euo pipefail

# Benchmark run using existing consolidated PDF + manifests.
python3 benchmark/scientific_ocr/scripts/run_benchmark.py
python3 benchmark/scientific_ocr/scripts/evaluate_benchmark.py
