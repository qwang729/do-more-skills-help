#!/usr/bin/env python3
"""Run the formal RQ1 retrieval-scaling experiment.

RQ1 asks whether skill retrieval accuracy decreases as the candidate skill
library grows. This script uses the Skill-Usage task queries and gold task-skill
mappings, samples random distractors at increasing pool sizes, ranks candidates
with a local BM25 retriever, and writes analysis-ready CSV/JSON/SVG outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
GT_OWNER = "benchflow-ai"
DEFAULT_POOL_SIZES = ["10", "50", "100", "500", "1000", "5000", "10000", "full"]
SUMMARY_METRICS = [
    "top1_accuracy",
    "hit@3",
    "hit@5",
    "hit@10",
    "recall@3",
    "recall@5",
    "recall@10",
    "mrr@10",
    "ndcg@10",
]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_json(path: Path):
    return json.loads(path.read_text())


def load_skill_docs(meta_path: Path) -> dict[str, dict]:
    docs = {}
    with meta_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("skill_id")
            if not sid:
                continue
            name = row.get("name") or row.get("skill_name") or row.get("skillId") or ""
            desc = row.get("description") or ""
            text = f"{name} {desc}"
            tokens = tokenize(text)
            docs[sid] = {
                "skill_id": sid,
                "name": name,
                "owner": row.get("owner", ""),
                "repo": row.get("repo", ""),
                "description": desc,
                "tokens": tokens,
                "token_set": set(tokens),
                "tf": Counter(tokens),
                "length": len(tokens),
            }
    return docs


def normalize_gt(raw_mapping: dict[str, list[str]]) -> dict[str, set[str]]:
    return {
        task: {f"{GT_OWNER}--{skill}" for skill in skills}
        for task, skills in raw_mapping.items()
    }


def bm25_rank(query: str, candidate_ids: list[str], docs: dict[str, dict], top_k: int) -> list[dict]:
    query_terms = tokenize(query)
    if not query_terms or not candidate_ids:
        return []

    n_docs = len(candidate_ids)
    doc_freq = Counter()
    lengths = []
    for sid in candidate_ids:
        doc = docs[sid]
        lengths.append(doc["length"])
        for term in doc["token_set"]:
            doc_freq[term] += 1

    avg_len = sum(lengths) / len(lengths) if lengths else 1.0
    k1 = 1.5
    b = 0.75
    scores = []
    positive_score_count = 0
    for sid in candidate_ids:
        doc = docs[sid]
        doc_len = doc["length"] or 1
        tf = doc["tf"]
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / avg_len)
            score += idf * (freq * (k1 + 1)) / denom
        if score > 0:
            positive_score_count += 1
        scores.append((score, sid))

    scores.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "rank": rank,
            "skill_id": sid,
            "score": round(score, 6),
            "positive_score": score > 0,
            "positive_score_count": positive_score_count,
            "name": docs[sid]["name"],
            "owner": docs[sid]["owner"],
            "repo": docs[sid]["repo"],
        }
        for rank, (score, sid) in enumerate(scores[:top_k], start=1)
    ]


def hit_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    return 1.0 if set(ranked_ids[:k]) & gold else 0.0


def recall_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranked_ids[:k]) & gold) / len(gold)


def reciprocal_rank_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    for index, sid in enumerate(ranked_ids[:k], start=1):
        if sid in gold:
            return 1.0 / index
    return 0.0


def first_gold_rank_at_k(ranked_ids: list[str], gold: set[str], k: int) -> int:
    for index, sid in enumerate(ranked_ids[:k], start=1):
        if sid in gold:
            return index
    return 0


def ndcg_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for rank, sid in enumerate(ranked_ids[:k], start=1):
        if sid in gold:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(gold), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def parse_pool_size(value: str, total: int) -> int:
    return total if value == "full" else int(value)


def sample_candidates(
    all_skill_ids: list[str],
    gold: set[str],
    pool_size: int,
    rng: random.Random,
) -> list[str]:
    gold_present = sorted(sid for sid in gold if sid in all_skill_ids)
    target_size = max(pool_size, len(gold_present))
    if target_size >= len(all_skill_ids):
        return list(all_skill_ids)

    gold_lookup = set(gold_present)
    distractor_count = target_size - len(gold_present)
    non_gold = [sid for sid in all_skill_ids if sid not in gold_lookup]
    distractors = rng.sample(non_gold, distractor_count)
    candidates = gold_present + distractors
    rng.shuffle(candidates)
    return candidates


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    return {key: mean(row[key] for row in rows) for key in SUMMARY_METRICS} | {"n": n}


def metric_stdev(rows: list[dict], metric: str) -> float:
    values = [row[metric] for row in rows]
    return pstdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def write_metric_svg(summary_rows: list[dict], total_skills: int, path: Path) -> None:
    width = 920
    height = 460
    left = 82
    right = 30
    top = 35
    bottom = 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    metrics = [
        ("top1_accuracy", "#1f77b4", "Top-1"),
        ("hit@10", "#2ca02c", "Hit@10"),
        ("recall@10", "#d62728", "Recall@10"),
        ("ndcg@10", "#9467bd", "NDCG@10"),
    ]

    numeric_sizes = [
        total_skills if row["pool_size"] == "full" else int(row["pool_size"])
        for row in summary_rows
    ]
    min_x = math.log10(min(numeric_sizes))
    max_x = math.log10(max(numeric_sizes))

    def sx(size: int) -> float:
        return left + (math.log10(size) - min_x) / (max_x - min_x) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - value) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="22" font-family="Arial" font-size="18" font-weight="700">RQ1 retrieval scaling: metric trends by skill pool size</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
    ]

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>')

    for row, size in zip(summary_rows, numeric_sizes):
        x = sx(size)
        label = row["pool_size"]
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#333"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')

    for metric, color, label in metrics:
        points = []
        for row, size in zip(summary_rows, numeric_sizes):
            points.append((sx(size), sy(row[metric])))
        point_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{point_str}"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')

    legend_x = left + plot_w - 118
    for i, (_, color, label) in enumerate(metrics):
        y = top + 18 + i * 22
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 32}" y="{y + 4}" font-family="Arial" font-size="13">{label}</text>')

    lines.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13">Candidate pool size (log scale)</text>')
    lines.append(f'<text x="18" y="{top + plot_h / 2:.2f}" transform="rotate(-90 18,{top + plot_h / 2:.2f})" text-anchor="middle" font-family="Arial" font-size="13">Metric value</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-usage-root", default="data/raw/Skill-Usage")
    parser.add_argument("--output-dir", default="data/experiments/rq1_retrieval_scaling")
    parser.add_argument("--pool-sizes", nargs="+", default=DEFAULT_POOL_SIZES)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=6002)
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all tasks")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.skill_usage_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = load_json(root / "data" / "task_queries.json")
    gt = normalize_gt(load_json(root / "data" / "task_skill_mapping.json"))
    docs = load_skill_docs(root / "skills-34k" / "skills_meta.jsonl")

    all_skill_ids = sorted(docs)
    tasks = sorted(set(queries) & set(gt))
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]

    per_query_rows = []
    repeat_summary_rows = []
    ranking_examples = {}
    error_cases = []
    summary_rows = []

    for pool_label in args.pool_sizes:
        pool_size = parse_pool_size(pool_label, len(all_skill_ids))
        repeat_count = 1 if pool_label == "full" else args.repeats
        pooled_rows = []

        for repeat in range(repeat_count):
            rng = random.Random(args.seed + repeat + pool_size * 1009)
            repeat_rows = []
            for task in tasks:
                gold = gt[task]
                query = " ".join(queries[task])
                candidates = sample_candidates(all_skill_ids, gold, pool_size, rng)
                ranked = bm25_rank(query, candidates, docs, args.top_k)
                ranked_ids = [row["skill_id"] for row in ranked]
                first_gold_rank = first_gold_rank_at_k(ranked_ids, gold, args.top_k)
                positive_score_count = ranked[0]["positive_score_count"] if ranked else 0

                row = {
                    "task": task,
                    "pool_size": pool_label,
                    "actual_pool_size": len(candidates),
                    "repeat": repeat,
                    "gold_count": len(gold),
                    "returned_count": len(ranked),
                    "positive_score_count": positive_score_count,
                    "top1_accuracy": 1.0 if ranked_ids and ranked_ids[0] in gold else 0.0,
                    "hit@3": hit_at_k(ranked_ids, gold, 3),
                    "hit@5": hit_at_k(ranked_ids, gold, 5),
                    "hit@10": hit_at_k(ranked_ids, gold, 10),
                    "recall@3": recall_at_k(ranked_ids, gold, 3),
                    "recall@5": recall_at_k(ranked_ids, gold, 5),
                    "recall@10": recall_at_k(ranked_ids, gold, 10),
                    "mrr@10": reciprocal_rank_at_k(ranked_ids, gold, args.top_k),
                    "ndcg@10": ndcg_at_k(ranked_ids, gold, 10),
                    "first_gold_rank@10": first_gold_rank,
                }
                per_query_rows.append(row)
                repeat_rows.append(row)
                pooled_rows.append(row)

                example_key = f"{pool_label}:{repeat}:{task}"
                if repeat == 0 and len(ranking_examples) < 30:
                    ranking_examples[example_key] = {
                        "query": query,
                        "gold": sorted(gold),
                        "top_results": ranked,
                    }

                if pool_label == "full" and row["top1_accuracy"] == 0.0:
                    error_cases.append({
                        "task": task,
                        "query": query,
                        "gold": sorted(gold),
                        "first_gold_rank@10": first_gold_rank,
                        "top_results": ranked,
                    })

            repeat_summary_rows.append({
                "pool_size": pool_label,
                "actual_pool_size": len(candidates),
                "repeat": repeat,
                "tasks": len(tasks),
                **summarize(repeat_rows),
            })

        metrics = summarize(pooled_rows)
        summary_row = {
            "pool_size": pool_label,
            "actual_pool_size": pool_size,
            "tasks": len(tasks),
            "repeats": repeat_count,
            **metrics,
        }
        for metric in SUMMARY_METRICS:
            summary_row[f"{metric}_std_across_queries"] = metric_stdev(pooled_rows, metric)
        summary_rows.append(summary_row)

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    repeat_summary_csv = output_dir / "repeat_summary.csv"
    per_query_csv = output_dir / "per_query_metrics.csv"
    examples_json = output_dir / "ranking_examples.json"
    error_cases_json = output_dir / "full_pool_error_cases.json"
    svg_path = output_dir / "metric_trends.svg"

    summary_json.write_text(json.dumps(summary_rows, indent=2) + "\n")
    examples_json.write_text(json.dumps(ranking_examples, indent=2) + "\n")
    error_cases_json.write_text(json.dumps(error_cases, indent=2) + "\n")
    write_csv(summary_csv, summary_rows)
    write_csv(repeat_summary_csv, repeat_summary_rows)
    write_csv(per_query_csv, per_query_rows)
    write_metric_svg(summary_rows, len(all_skill_ids), svg_path)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {repeat_summary_csv}")
    print(f"Wrote {per_query_csv}")
    print(f"Wrote {examples_json}")
    print(f"Wrote {error_cases_json}")
    print(f"Wrote {svg_path}")
    print()
    print(
        f"{'pool':>8} {'top1':>8} {'hit@10':>8} {'R@10':>8} "
        f"{'MRR@10':>8} {'NDCG@10':>10}"
    )
    for row in summary_rows:
        print(
            f"{row['pool_size']:>8} "
            f"{row['top1_accuracy']:>8.3f} "
            f"{row['hit@10']:>8.3f} "
            f"{row['recall@10']:>8.3f} "
            f"{row['mrr@10']:>8.3f} "
            f"{row['ndcg@10']:>10.3f}"
        )

    first = summary_rows[0]
    last = summary_rows[-1]
    top1_drop = first["top1_accuracy"] - last["top1_accuracy"]
    hit10_drop = first["hit@10"] - last["hit@10"]
    print()
    print(f"Top-1 drop from {first['pool_size']} to {last['pool_size']}: {top1_drop:.3f}")
    print(f"Hit@10 drop from {first['pool_size']} to {last['pool_size']}: {hit10_drop:.3f}")


if __name__ == "__main__":
    main()
