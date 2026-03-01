#!/usr/bin/env python3
"""Run OCR benchmark via bundled pdfocr binary on one consolidated PDF."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


class BenchmarkRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pricing:
    model_id: str
    price_per_million_input: float
    price_per_million_output: float


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "model"


def parse_dotenv(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def resolve_api_key(repo_root: Path) -> str:
    key = os.getenv("DEEPINFRA_API_KEY", "").strip()
    if key:
        return key
    key = parse_dotenv(repo_root / ".env").get("DEEPINFRA_API_KEY", "").strip()
    if key:
        return key
    raise BenchmarkRunError("DEEPINFRA_API_KEY missing from environment and .env")


def request_cost_usd(prompt_tokens: int, completion_tokens: int, pricing: Pricing) -> float:
    return (
        (prompt_tokens * pricing.price_per_million_input) / 1_000_000.0
        + (completion_tokens * pricing.price_per_million_output) / 1_000_000.0
    )


def update_pdfocr_config(config_path: Path, model_id: str, api_url: str, prompt: str) -> None:
    cfg = load_json(config_path)
    cfg["model"] = model_id
    cfg["api_url"] = api_url
    cfg["prompt"] = prompt
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_pdfocr_stdout(stdout_text: str) -> Dict[int, Dict]:
    by_page: Dict[int, Dict] = {}
    for idx, line in enumerate(stdout_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkRunError(f"invalid JSON on stdout line {idx}: {exc}") from exc
        page = int(payload.get("page", 0) or 0)
        if page <= 0:
            raise BenchmarkRunError(f"stdout line {idx} missing valid 1-based page number")
        by_page[page] = payload
    return by_page


def run_pdfocr(pdfocr_bin: Path, input_pdf: Path, env: Dict[str, str]) -> Tuple[str, str, int, float]:
    start = datetime.now(timezone.utc)
    proc = subprocess.run(
        [str(pdfocr_bin), str(input_pdf), "--all-pages"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env=env,
    )
    end = datetime.now(timezone.utc)
    elapsed = (end - start).total_seconds()
    return proc.stdout, proc.stderr, int(proc.returncode), elapsed


def run(config_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    cfg = load_json(config_path)
    api_key = resolve_api_key(repo_root)

    paths_cfg = cfg["paths"]
    ocr_cfg = cfg["ocr_request"]
    prompt_tokens_per_page_estimate = int(ocr_cfg.get("prompt_tokens_per_page_estimate", 3000))
    if prompt_tokens_per_page_estimate <= 0:
        raise BenchmarkRunError("prompt_tokens_per_page_estimate must be > 0")

    manifests_dir = repo_root / paths_cfg["manifests_dir"]
    pages_manifest = manifests_dir / "pages.jsonl"
    if not pages_manifest.exists():
        raise BenchmarkRunError(
            f"missing pages manifest at {pages_manifest}; benchmark assets are incomplete"
        )

    page_rows = read_jsonl(pages_manifest)
    if not page_rows:
        raise BenchmarkRunError("pages manifest is empty")

    consolidated_paths = sorted({str(row.get("consolidated_pdf_path", "")) for row in page_rows})
    if len(consolidated_paths) != 1 or not consolidated_paths[0]:
        raise BenchmarkRunError(
            "pages manifest must include one non-empty consolidated_pdf_path for all rows"
        )

    consolidated_pdf = repo_root / consolidated_paths[0]
    if not consolidated_pdf.exists():
        raise BenchmarkRunError(f"missing consolidated PDF at {consolidated_pdf}")

    by_page_no: Dict[int, Dict] = {}
    for row in page_rows:
        page_no = int(row.get("consolidated_page", 0) or 0)
        if page_no <= 0:
            raise BenchmarkRunError(f"invalid consolidated_page for sample {row.get('sample_id')}")
        by_page_no[page_no] = row

    expected_pages = sorted(by_page_no.keys())
    if expected_pages != list(range(1, len(expected_pages) + 1)):
        raise BenchmarkRunError("consolidated_page values must be contiguous from 1..N")

    models = [
        Pricing(
            model_id=m["id"],
            price_per_million_input=float(m["price_per_million_input"]),
            price_per_million_output=float(m["price_per_million_output"]),
        )
        for m in cfg["models"]
    ]

    pdfocr_dir = repo_root / "benchmark/scientific_ocr/pdfocr-linux-x86_64"
    pdfocr_bin = pdfocr_dir / "pdfocr"
    pdfocr_config = pdfocr_dir / "config.json"
    if not pdfocr_bin.exists():
        raise BenchmarkRunError(f"missing pdfocr binary at {pdfocr_bin}")
    if not pdfocr_config.exists():
        raise BenchmarkRunError(f"missing pdfocr config at {pdfocr_config}")

    raw_results_dir = repo_root / paths_cfg["raw_results_dir"]
    ocr_text_dir = repo_root / paths_cfg["ocr_text_dir"]
    metrics_dir = repo_root / paths_cfg["metrics_dir"]

    if raw_results_dir.exists():
        shutil.rmtree(raw_results_dir)
    if ocr_text_dir.exists():
        shutil.rmtree(ocr_text_dir)

    raw_results_dir.mkdir(parents=True, exist_ok=True)
    ocr_text_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["DEEPINFRA_API_KEY"] = api_key
    prev_ld = env.get("LD_LIBRARY_PATH", "").strip()
    env["LD_LIBRARY_PATH"] = str(pdfocr_dir) if not prev_ld else f"{pdfocr_dir}:{prev_ld}"

    all_records: List[Dict] = []
    model_summaries: List[Dict] = []
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for model in models:
        model_slug = safe_slug(model.model_id)
        update_pdfocr_config(
            config_path=pdfocr_config,
            model_id=model.model_id,
            api_url=str(ocr_cfg["api_url"]),
            prompt=str(ocr_cfg["prompt"]),
        )

        stdout_text, stderr_text, return_code, elapsed = run_pdfocr(pdfocr_bin, consolidated_pdf, env)
        if return_code not in {0, 2}:
            raise BenchmarkRunError(
                f"pdfocr failed for model {model.model_id} with exit={return_code}\n{stderr_text}"
            )

        parsed = parse_pdfocr_stdout(stdout_text)

        model_raw_dir = raw_results_dir / model_slug
        model_raw_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            model_raw_dir / "pdfocr_run.json",
            {
                "model": model.model_id,
                "return_code": return_code,
                "elapsed_sec": round(elapsed, 6),
                "stderr_tail": "\n".join(stderr_text.splitlines()[-120:]),
                "line_count": len(stdout_text.splitlines()),
            },
        )

        model_records: List[Dict] = []
        model_prompt_tokens = 0
        model_completion_tokens = 0
        model_cost = 0.0

        for page_no in expected_pages:
            page_meta = by_page_no[page_no]
            sample_id = page_meta["sample_id"]
            out = parsed.get(page_no, {})
            status = str(out.get("status", "missing"))
            text = str(out.get("text", "")) if status == "ok" else ""

            prompt_tokens = prompt_tokens_per_page_estimate
            completion_tokens = max(0, len(text) // 4) if status == "ok" else 0
            cost = request_cost_usd(prompt_tokens, completion_tokens, model)

            if status == "ok":
                model_prompt_tokens += prompt_tokens
                model_completion_tokens += completion_tokens
                model_cost += cost
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_cost += cost
                error_message = ""
                http_status = 0
            else:
                error_message = str(out.get("error_message", "") or "")
                http_status = int(out.get("http_status", 0) or 0)

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model.model_id,
                "model_slug": model_slug,
                "sample_id": sample_id,
                "course": page_meta["course"],
                "page": page_meta["page"],
                "consolidated_page": page_no,
                "status": status,
                "attempts": int(out.get("attempts", 0) or 0),
                "http_status": http_status,
                "error_message": error_message,
                "latency_sec": 0.0,
                "prompt_tokens": prompt_tokens if status == "ok" else 0,
                "completion_tokens": completion_tokens if status == "ok" else 0,
                "total_tokens": (prompt_tokens + completion_tokens) if status == "ok" else 0,
                "usage_estimated": True,
                "cost_usd": round(cost if status == "ok" else 0.0, 10),
                "ocr_text": text,
                "raw_path": str((model_raw_dir / "pdfocr_run.json").relative_to(repo_root)),
            }
            model_records.append(record)
            all_records.append(record)

        write_jsonl(ocr_text_dir / f"{model_slug}.jsonl", model_records)

        model_summaries.append(
            {
                "model": model.model_id,
                "model_slug": model_slug,
                "requests": len(model_records),
                "ok": sum(1 for row in model_records if row["status"] == "ok"),
                "errors": sum(1 for row in model_records if row["status"] == "error"),
                "missing": sum(1 for row in model_records if row["status"] not in {"ok", "error"}),
                "prompt_tokens": model_prompt_tokens,
                "completion_tokens": model_completion_tokens,
                "cost_usd": round(model_cost, 10),
                "mean_latency_sec": round(elapsed / max(1, len(model_records)), 6),
                "pdfocr_exit_code": return_code,
            }
        )

    run_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "runner": "pdfocr_all_pages",
        "pdfocr_bin": str(pdfocr_bin),
        "pdfocr_config": str(pdfocr_config),
        "consolidated_pdf": str(consolidated_pdf),
        "page_count": len(expected_pages),
        "overall_prompt_tokens": total_prompt_tokens,
        "overall_completion_tokens": total_completion_tokens,
        "overall_cost_usd": round(total_cost, 10),
        "models": model_summaries,
        "notes": "Token usage/cost is estimated in this runner because pdfocr stdout does not expose usage.",
    }

    write_json(metrics_dir / "run_summary.json", run_summary)
    write_jsonl(metrics_dir / "all_requests.jsonl", all_records)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR benchmark via bundled pdfocr.")
    parser.add_argument(
        "--config",
        default="benchmark/scientific_ocr/config/benchmark_config.json",
        help="Path to benchmark config JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    try:
        run(config_path=config_path)
    except BenchmarkRunError as exc:
        print(f"run_benchmark error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
