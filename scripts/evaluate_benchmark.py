#!/usr/bin/env python3
"""Evaluate OCR benchmark outputs and produce model comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


MATH_SYMBOLS = set("=+-*/^_<>≤≥≈∑∫√∞πλαβγδθμΩΔ∂∇")
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MARKDOWN_RULE_PATTERN = re.compile(r"^\s*\|?\s*[-:]{2,}[-| :]*\|?\s*$", re.MULTILINE)


class EvaluationError(RuntimeError):
    pass


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


def normalize_text_for_cer(text: str) -> str:
    # Remove common OCR output markup before scoring to reduce format-only penalties.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = MARKDOWN_RULE_PATTERN.sub(" ", text)
    text = text.replace("•", " ").replace("·", " ")
    text = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip().casefold()


def normalize_text_for_tokens(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(normalize_text_for_cer(text))


def token_overlap_scores(ref: str, hyp: str) -> Tuple[float, float, float]:
    ref_tokens = normalize_text_for_tokens(ref)
    hyp_tokens = normalize_text_for_tokens(hyp)

    if not ref_tokens and not hyp_tokens:
        return 1.0, 1.0, 1.0
    if not ref_tokens:
        return 1.0, 0.0, 0.0
    if not hyp_tokens:
        return 0.0, 0.0, 0.0

    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)
    overlap = sum(min(count, hyp_counts.get(tok, 0)) for tok, count in ref_counts.items())

    recall = overlap / max(1, len(ref_tokens))
    precision = overlap / max(1, len(hyp_tokens))
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * recall * precision / (recall + precision)
    return recall, precision, f1


def char_lcs_recall(ref: str, hyp: str) -> float:
    ref_chars = list(normalize_text_for_cer(ref))
    hyp_chars = list(normalize_text_for_cer(hyp))

    if not ref_chars and not hyp_chars:
        return 1.0
    if not ref_chars:
        return 1.0
    if not hyp_chars:
        return 0.0

    lcs = lcs_length(ref_chars, hyp_chars)
    return lcs / max(1, len(ref_chars))


def levenshtein(seq_a: Sequence, seq_b: Sequence) -> int:
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    prev = list(range(len(seq_b) + 1))
    for idx_a, item_a in enumerate(seq_a, start=1):
        curr = [idx_a]
        for idx_b, item_b in enumerate(seq_b, start=1):
            cost = 0 if item_a == item_b else 1
            curr.append(min(curr[-1] + 1, prev[idx_b] + 1, prev[idx_b - 1] + cost))
        prev = curr
    return prev[-1]


def lcs_length(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    if not seq_a or not seq_b:
        return 0

    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    prev = [0] * (len(seq_b) + 1)
    for item_a in seq_a:
        curr = [0]
        for j, item_b in enumerate(seq_b, start=1):
            if item_a == item_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref_chars = list(normalize_text_for_cer(ref))
    hyp_chars = list(normalize_text_for_cer(hyp))

    if not ref_chars and not hyp_chars:
        return 0.0
    if not ref_chars:
        return 1.0

    return min(1.0, levenshtein(ref_chars, hyp_chars) / max(1, len(ref_chars)))


def wer(ref: str, hyp: str) -> float:
    ref_tokens = normalize_text_for_tokens(ref)
    hyp_tokens = normalize_text_for_tokens(hyp)

    if not ref_tokens and not hyp_tokens:
        return 0.0
    if not ref_tokens:
        return 1.0

    return min(1.0, levenshtein(ref_tokens, hyp_tokens) / max(1, len(ref_tokens)))


def reading_order_f1(ref: str, hyp: str) -> float:
    ref_lines = [line.strip().casefold() for line in ref.splitlines() if line.strip()]
    hyp_lines = [line.strip().casefold() for line in hyp.splitlines() if line.strip()]

    if not ref_lines and not hyp_lines:
        return 1.0
    if not ref_lines or not hyp_lines:
        return 0.0

    lcs = lcs_length(ref_lines, hyp_lines)
    return (2.0 * lcs) / (len(ref_lines) + len(hyp_lines))


def math_symbol_f1(ref: str, hyp: str) -> float:
    ref_counts: Dict[str, int] = defaultdict(int)
    hyp_counts: Dict[str, int] = defaultdict(int)

    for ch in normalize_text_for_cer(ref):
        if ch in MATH_SYMBOLS:
            ref_counts[ch] += 1

    for ch in normalize_text_for_cer(hyp):
        if ch in MATH_SYMBOLS:
            hyp_counts[ch] += 1

    ref_total = sum(ref_counts.values())
    hyp_total = sum(hyp_counts.values())

    if ref_total == 0 and hyp_total == 0:
        return 1.0
    if ref_total == 0 or hyp_total == 0:
        return 0.0

    overlap = 0
    for symbol, ref_count in ref_counts.items():
        overlap += min(ref_count, hyp_counts.get(symbol, 0))

    precision = overlap / hyp_total
    recall = overlap / ref_total

    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def evaluate(config_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    cfg = load_json(config_path)
    paths_cfg = cfg["paths"]

    manifests_dir = repo_root / paths_cfg["manifests_dir"]
    ocr_text_dir = repo_root / paths_cfg["ocr_text_dir"]
    metrics_dir = repo_root / paths_cfg["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pages_manifest_path = manifests_dir / "pages.jsonl"
    if not pages_manifest_path.exists():
        raise EvaluationError(f"missing pages manifest at {pages_manifest_path}")

    pages = read_jsonl(pages_manifest_path)
    by_sample = {row["sample_id"]: row for row in pages}

    gold_labels_path = repo_root / paths_cfg["gold_labels_path"]
    if not gold_labels_path.exists():
        raise EvaluationError(f"missing gold labels file at {gold_labels_path}")

    gold_rows_by_sample = {row["sample_id"]: row for row in read_jsonl(gold_labels_path)}
    missing_gold = sorted(row["sample_id"] for row in pages if row["sample_id"] not in gold_rows_by_sample)
    if missing_gold:
        raise EvaluationError(
            "gold labels missing sample_ids: " + ", ".join(missing_gold[:8])
            + (" ..." if len(missing_gold) > 8 else "")
        )

    for row in pages:
        row["reference_text"] = str(gold_rows_by_sample[row["sample_id"]].get("gold_text", ""))
        row["reference_origin"] = "gold"

    model_specs = cfg["models"]
    model_to_slug = {
        spec["id"]: re.sub(r"[^a-z0-9]+", "_", spec["id"].lower()).strip("_")
        for spec in model_specs
    }

    all_page_scores: List[Dict] = []
    model_summaries: List[Dict] = []
    tag_breakdown_rows: List[Dict] = []

    for spec in model_specs:
        model_id = spec["id"]
        model_slug = model_to_slug[model_id]
        result_path = ocr_text_dir / f"{model_slug}.jsonl"
        if not result_path.exists():
            raise EvaluationError(f"missing OCR result file for {model_id}: {result_path}")

        model_rows = read_jsonl(result_path)
        rows_by_sample = {row["sample_id"]: row for row in model_rows}

        per_page_metrics: List[Dict] = []

        for sample_id, page_meta in by_sample.items():
            result = rows_by_sample.get(sample_id)
            ref_text = page_meta.get("reference_text", "")
            has_ref = bool(normalize_text_for_cer(ref_text))

            if result is None:
                status = "missing"
                hyp_text = ""
                cost_usd = 0.0
                prompt_tokens = 0
                completion_tokens = 0
            else:
                status = result.get("status", "missing")
                hyp_text = result.get("ocr_text", "")
                cost_usd = float(result.get("cost_usd", 0.0) or 0.0)
                prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
                completion_tokens = int(result.get("completion_tokens", 0) or 0)

            if status == "ok" and has_ref:
                cer_score = cer(ref_text, hyp_text)
                wer_score = wer(ref_text, hyp_text)
                order_score = reading_order_f1(ref_text, hyp_text)
                math_score = math_symbol_f1(ref_text, hyp_text)
                token_recall, token_precision, token_f1 = token_overlap_scores(ref_text, hyp_text)
                lcs_recall = char_lcs_recall(ref_text, hyp_text)
            elif has_ref:
                cer_score = 1.0
                wer_score = 1.0
                order_score = 0.0
                math_score = 0.0
                token_recall = 0.0
                token_precision = 0.0
                token_f1 = 0.0
                lcs_recall = 0.0
            else:
                cer_score = 0.0
                wer_score = 0.0
                order_score = 1.0
                math_score = 1.0
                token_recall = 1.0
                token_precision = 1.0
                token_f1 = 1.0
                lcs_recall = 1.0

            row = {
                "model": model_id,
                "model_slug": model_slug,
                "sample_id": sample_id,
                "course": page_meta["course"],
                "page": page_meta["page"],
                "status": status,
                "has_reference": has_ref,
                "reference_origin": page_meta.get("reference_origin", "gold"),
                "cer": cer_score,
                "wer": wer_score,
                "reading_order_f1": order_score,
                "math_symbol_f1": math_score,
                "token_recall": token_recall,
                "token_precision": token_precision,
                "token_f1": token_f1,
                "char_lcs_recall": lcs_recall,
                "cost_usd": cost_usd,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "equation": bool(page_meta["tags"].get("equation", False)),
                "table": bool(page_meta["tags"].get("table", False)),
                "multi_column": bool(page_meta["tags"].get("multi_column", False)),
                "diagram": bool(page_meta["tags"].get("diagram", False)),
            }
            per_page_metrics.append(row)
            all_page_scores.append(row)

        eval_rows = [row for row in per_page_metrics if row["has_reference"]]
        ok_rows = [row for row in eval_rows if row["status"] == "ok"]

        cost_total = sum(row["cost_usd"] for row in per_page_metrics)
        prompt_total = sum(row["prompt_tokens"] for row in per_page_metrics)
        completion_total = sum(row["completion_tokens"] for row in per_page_metrics)

        mean_cer = mean(row["cer"] for row in eval_rows)
        mean_wer = mean(row["wer"] for row in eval_rows)
        mean_order = mean(row["reading_order_f1"] for row in eval_rows)
        mean_math = mean(row["math_symbol_f1"] for row in eval_rows)
        mean_token_recall = mean(row["token_recall"] for row in eval_rows)
        mean_token_precision = mean(row["token_precision"] for row in eval_rows)
        mean_token_f1 = mean(row["token_f1"] for row in eval_rows)
        mean_lcs_recall = mean(row["char_lcs_recall"] for row in eval_rows)

        coverage = len(ok_rows) / max(1, len(eval_rows))
        strict_accuracy_component = (
            (1.0 - mean_wer) + (1.0 - mean_cer) + mean_order + mean_math
        ) / 4.0
        robust_accuracy_component = (
            mean_token_recall + mean_lcs_recall + mean_order + mean_math
        ) / 4.0
        strict_accuracy_per_dollar = strict_accuracy_component / max(cost_total, 1e-9)
        robust_accuracy_per_dollar = robust_accuracy_component / max(cost_total, 1e-9)

        model_summary = {
            "model": model_id,
            "model_slug": model_slug,
            "n_pages": len(per_page_metrics),
            "n_eval_pages": len(eval_rows),
            "n_ok": len(ok_rows),
            "coverage": coverage,
            "mean_cer": mean_cer,
            "mean_wer": mean_wer,
            "mean_reading_order_f1": mean_order,
            "mean_math_symbol_f1": mean_math,
            "mean_token_recall": mean_token_recall,
            "mean_token_precision": mean_token_precision,
            "mean_token_f1": mean_token_f1,
            "mean_char_lcs_recall": mean_lcs_recall,
            "strict_accuracy_component": strict_accuracy_component,
            "robust_accuracy_component": robust_accuracy_component,
            "strict_accuracy_per_dollar": strict_accuracy_per_dollar,
            "robust_accuracy_per_dollar": robust_accuracy_per_dollar,
            "total_cost_usd": cost_total,
            "cost_per_eval_page_usd": cost_total / max(1, len(eval_rows)),
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
        }
        model_summaries.append(model_summary)

        for tag in ("equation", "table", "multi_column", "diagram"):
            tag_rows = [row for row in eval_rows if row[tag]]
            tag_breakdown_rows.append(
                {
                    "model": model_id,
                    "tag": tag,
                    "n": len(tag_rows),
                    "mean_cer": mean(row["cer"] for row in tag_rows),
                    "mean_wer": mean(row["wer"] for row in tag_rows),
                    "mean_reading_order_f1": mean(row["reading_order_f1"] for row in tag_rows),
                    "mean_math_symbol_f1": mean(row["math_symbol_f1"] for row in tag_rows),
                    "mean_token_recall": mean(row["token_recall"] for row in tag_rows),
                    "mean_token_f1": mean(row["token_f1"] for row in tag_rows),
                    "mean_char_lcs_recall": mean(row["char_lcs_recall"] for row in tag_rows),
                    "coverage": mean(1.0 if row["status"] == "ok" else 0.0 for row in tag_rows),
                }
            )

    if not model_summaries:
        raise EvaluationError("no model summaries computed")

    min_cost = min(row["total_cost_usd"] for row in model_summaries)

    for row in model_summaries:
        cost = row["total_cost_usd"]
        row["cost_efficiency"] = min_cost / max(cost, 1e-9)
        row["strict_balanced_score"] = (
            0.7 * row["strict_accuracy_component"] + 0.3 * row["cost_efficiency"]
        )
        row["robust_balanced_score"] = (
            0.7 * row["robust_accuracy_component"] + 0.3 * row["cost_efficiency"]
        )

    strict_accuracy_first = sorted(
        model_summaries,
        key=lambda r: (
            -r["strict_accuracy_component"],
            r["mean_wer"],
            r["mean_cer"],
            -r["coverage"],
        ),
    )[0]
    robust_accuracy_first = sorted(
        model_summaries,
        key=lambda r: (
            -r["robust_accuracy_component"],
            -r["mean_token_recall"],
            r["mean_wer"],
            -r["coverage"],
        ),
    )[0]

    cost_first = sorted(
        model_summaries,
        key=lambda r: (r["total_cost_usd"], -r["robust_accuracy_component"]),
    )[0]

    strict_balanced_best = sorted(
        model_summaries,
        key=lambda r: (
            -r["strict_balanced_score"],
            -r["strict_accuracy_component"],
            r["total_cost_usd"],
        ),
    )[0]

    robust_balanced_best = sorted(
        model_summaries,
        key=lambda r: (
            -r["robust_balanced_score"],
            -r["robust_accuracy_component"],
            r["total_cost_usd"],
        ),
    )[0]

    recommendations = {
        "accuracy_first_strict": {
            "model": strict_accuracy_first["model"],
            "reason": "Highest strict score over CER/WER/layout/math.",
        },
        "accuracy_first_robust": {
            "model": robust_accuracy_first["model"],
            "reason": "Highest robust score using token/character recall + layout/math.",
        },
        "cost_first": {
            "model": cost_first["model"],
            "reason": "Lowest measured total benchmark cost.",
        },
        "balanced_strict": {
            "model": strict_balanced_best["model"],
            "reason": "Best strict-accuracy/cost weighted score.",
        },
        "balanced_robust": {
            "model": robust_balanced_best["model"],
            "reason": "Best robust-accuracy/cost weighted score.",
        },
    }

    write_json(metrics_dir / "model_summary.json", {"models": model_summaries})
    write_json(metrics_dir / "recommendations.json", recommendations)

    with (metrics_dir / "model_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "n_eval_pages",
                "n_ok",
                "coverage",
                "mean_cer",
                "mean_wer",
                "mean_reading_order_f1",
                "mean_math_symbol_f1",
                "mean_token_recall",
                "mean_token_precision",
                "mean_token_f1",
                "mean_char_lcs_recall",
                "strict_accuracy_component",
                "robust_accuracy_component",
                "total_cost_usd",
                "cost_per_eval_page_usd",
                "strict_accuracy_per_dollar",
                "robust_accuracy_per_dollar",
                "cost_efficiency",
                "strict_balanced_score",
                "robust_balanced_score",
                "prompt_tokens",
                "completion_tokens",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in model_summaries:
            writer.writerow(row)

    with (metrics_dir / "per_page_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "sample_id",
                "course",
                "page",
                "status",
                "has_reference",
                "reference_origin",
                "cer",
                "wer",
                "reading_order_f1",
                "math_symbol_f1",
                "token_recall",
                "token_precision",
                "token_f1",
                "char_lcs_recall",
                "cost_usd",
                "equation",
                "table",
                "multi_column",
                "diagram",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in all_page_scores:
            writer.writerow(row)

    with (metrics_dir / "tag_breakdown.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "tag",
                "n",
                "coverage",
                "mean_cer",
                "mean_wer",
                "mean_reading_order_f1",
                "mean_math_symbol_f1",
                "mean_token_recall",
                "mean_token_f1",
                "mean_char_lcs_recall",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in tag_breakdown_rows:
            writer.writerow(row)

    table_headers = [
        "Model",
        "Coverage",
        "CER",
        "WER",
        "OrderF1",
        "MathF1",
        "TokRecall",
        "CharLCSRec",
        "CostUSD",
        "RobustAcc/$",
        "RobustBalanced",
    ]

    lines = []
    lines.append("| " + " | ".join(table_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(table_headers)) + "|")
    for row in sorted(model_summaries, key=lambda r: r["model"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    f"{row['coverage']:.3f}",
                    f"{row['mean_cer']:.4f}",
                    f"{row['mean_wer']:.4f}",
                    f"{row['mean_reading_order_f1']:.4f}",
                    f"{row['mean_math_symbol_f1']:.4f}",
                    f"{row['mean_token_recall']:.4f}",
                    f"{row['mean_char_lcs_recall']:.4f}",
                    f"{row['total_cost_usd']:.4f}",
                    f"{row['robust_accuracy_per_dollar']:.3f}",
                    f"{row['robust_balanced_score']:.4f}",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append(
        f"Accuracy-first (strict) winner: `{recommendations['accuracy_first_strict']['model']}`"
    )
    lines.append(
        f"Accuracy-first (robust) winner: `{recommendations['accuracy_first_robust']['model']}`"
    )
    lines.append(f"Cost-first winner: `{recommendations['cost_first']['model']}`")
    lines.append(f"Balanced (strict) winner: `{recommendations['balanced_strict']['model']}`")
    lines.append(f"Balanced (robust) winner: `{recommendations['balanced_robust']['model']}`")

    (metrics_dir / "model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    digest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models_evaluated": len(model_summaries),
        "reference_source": "gold",
        "gold_labels_path": str(gold_labels_path),
        "recommendations": recommendations,
        "output_files": {
            "model_summary_json": str((metrics_dir / "model_summary.json")),
            "model_summary_csv": str((metrics_dir / "model_summary.csv")),
            "model_summary_md": str((metrics_dir / "model_summary.md")),
            "per_page_scores_csv": str((metrics_dir / "per_page_scores.csv")),
            "tag_breakdown_csv": str((metrics_dir / "tag_breakdown.csv")),
            "recommendations_json": str((metrics_dir / "recommendations.json")),
        },
    }
    write_json(metrics_dir / "evaluation_digest.json", digest)

    print(json.dumps({"recommendations": recommendations, "models": model_summaries}, indent=2))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR benchmark outputs.")
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
        evaluate(config_path)
    except EvaluationError as exc:
        print(f"evaluate_benchmark error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
