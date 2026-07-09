#!/usr/bin/env python3
"""Run the formal RQ2 distractor-type experiment.

RQ2 asks which distractor type is most likely to make the retriever select the
wrong skill. Skill-Usage gold skills do not include category/repo/tag metadata,
so this experiment uses reproducible proxy distractor families:

- random: uniformly sampled non-gold skills.
- query_overlap: non-gold skills with the highest token overlap with the query.
- bm25_hard: non-gold skills that BM25 itself ranks highly for the query.
- gold_skill_near: non-gold skills lexically similar to the gold skill text.
- embedding_semantic_near: non-gold skills closest to the gold skill embedding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from rq1_retrieval_scaling import (
    GT_OWNER,
    SUMMARY_METRICS,
    bm25_rank,
    first_gold_rank_at_k,
    hit_at_k,
    load_json,
    load_skill_docs,
    ndcg_at_k,
    normalize_gt,
    recall_at_k,
    reciprocal_rank_at_k,
    tokenize,
)


DEFAULT_POOL_SIZES = ["50", "100", "500", "1000"]
DEFAULT_DISTRACTOR_TYPES = [
    "random",
    "query_overlap",
    "bm25_hard",
    "gold_skill_near",
    "embedding_semantic_near",
]
DISTRACTOR_DESCRIPTIONS = {
    "random": "Uniform random non-gold skills.",
    "query_overlap": "Non-gold skills with the highest token Jaccard similarity to the task query.",
    "bm25_hard": "Non-gold skills that the same BM25 retriever ranks highest for the task query.",
    "gold_skill_near": "Non-gold skills with the highest token Jaccard similarity to the gold skill text.",
    "embedding_semantic_near": (
        "Non-gold skills nearest to the gold skill centroid in the official precomputed "
        "Skill-Usage embedding index."
    ),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_pool_size(value: str) -> int:
    return int(value)


def token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / len(left | right)


def rank_by_query_overlap(query: str, non_gold_ids: list[str], docs: dict[str, dict]) -> list[str]:
    query_tokens = set(tokenize(query))
    scored = []
    for sid in non_gold_ids:
        score = token_jaccard(query_tokens, docs[sid]["token_set"])
        if score > 0:
            scored.append((score, sid))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sid for _, sid in scored]


def rank_by_gold_skill_near(gold: set[str], non_gold_ids: list[str], docs: dict[str, dict]) -> list[str]:
    gold_tokens = set()
    for sid in gold:
        if sid in docs:
            gold_tokens.update(docs[sid]["token_set"])
    scored = []
    for sid in non_gold_ids:
        score = token_jaccard(gold_tokens, docs[sid]["token_set"])
        if score > 0:
            scored.append((score, sid))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sid for _, sid in scored]


def load_source_id_map(meta_path: Path) -> dict[str, str]:
    source_id_to_skill_id = {}
    with meta_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = row.get("id")
            skill_id = row.get("skill_id")
            if source_id and skill_id:
                source_id_to_skill_id[source_id] = skill_id
    return source_id_to_skill_id


def load_embedding_index(index_dir: Path, meta_path: Path) -> tuple[np.ndarray, dict[str, int], list[str]]:
    embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r")
    index_ids = json.loads((index_dir / "skill_ids.json").read_text())
    source_id_to_skill_id = load_source_id_map(meta_path)

    index_skill_ids = [source_id_to_skill_id[source_id] for source_id in index_ids]
    skill_id_to_index = {sid: index for index, sid in enumerate(index_skill_ids)}
    return embeddings, skill_id_to_index, index_skill_ids


def rank_by_embedding_semantic_near(
    gold: set[str],
    non_gold_ids: list[str],
    embeddings: np.ndarray,
    skill_id_to_index: dict[str, int],
    index_skill_ids: list[str],
) -> list[str]:
    gold_indices = [skill_id_to_index[sid] for sid in gold if sid in skill_id_to_index]
    if not gold_indices:
        return []

    gold_vectors = np.asarray(embeddings[gold_indices], dtype=np.float32)
    centroid = gold_vectors.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return []
    centroid = centroid / norm

    scores = np.asarray(embeddings @ centroid, dtype=np.float32)
    non_gold_lookup = set(non_gold_ids)
    ranked = []
    for index in np.argsort(scores)[::-1]:
        sid = index_skill_ids[int(index)]
        if sid in non_gold_lookup:
            ranked.append(sid)
    return ranked


def fill_from_ranked(
    ranked_ids: list[str],
    non_gold_ids: list[str],
    needed: int,
    rng: random.Random,
) -> tuple[list[str], int]:
    selected = ranked_ids[:needed]
    fallback_count = needed - len(selected)
    if fallback_count <= 0:
        return selected, 0
    selected_lookup = set(selected)
    fallback_pool = [sid for sid in non_gold_ids if sid not in selected_lookup]
    selected.extend(rng.sample(fallback_pool, fallback_count))
    return selected, fallback_count


def sample_candidates(
    distractor_type: str,
    gold: set[str],
    all_skill_ids: list[str],
    docs: dict[str, dict],
    pool_size: int,
    rng: random.Random,
    ranked_distractors: dict[str, list[str]],
) -> tuple[list[str], int]:
    gold_present = sorted(sid for sid in gold if sid in docs)
    target_size = max(pool_size, len(gold_present))
    distractor_count = target_size - len(gold_present)
    gold_lookup = set(gold_present)
    non_gold_ids = [sid for sid in all_skill_ids if sid not in gold_lookup]

    if distractor_type == "random":
        distractors = rng.sample(non_gold_ids, distractor_count)
        fallback_count = 0
    else:
        distractors, fallback_count = fill_from_ranked(
            ranked_distractors[distractor_type],
            non_gold_ids,
            distractor_count,
            rng,
        )

    candidates = gold_present + distractors
    rng.shuffle(candidates)
    return candidates, fallback_count


def summarize(rows: list[dict]) -> dict:
    summary = {key: mean(row[key] for row in rows) for key in SUMMARY_METRICS}
    summary["top1_error_rate"] = 1.0 - summary["top1_accuracy"]
    summary["hit10_miss_rate"] = 1.0 - summary["hit@10"]
    summary["total_fallback_distractors"] = sum(row["fallback_distractors"] for row in rows)
    summary["rows_with_fallback"] = sum(1 for row in rows if row["fallback_distractors"] > 0)
    summary["mean_fallback_distractors"] = mean(row["fallback_distractors"] for row in rows)
    summary["mean_hard_negative_purity"] = mean(row["hard_negative_purity"] for row in rows)
    summary["n"] = len(rows)
    return summary


def metric_stdev(rows: list[dict], metric: str) -> float:
    values = [row[metric] for row in rows]
    return pstdev(values) if len(values) > 1 else 0.0


def write_metric_svg(summary_rows: list[dict], path: Path) -> None:
    width = 960
    height = 500
    left = 82
    right = 170
    top = 38
    bottom = 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {
        "random": "#1f77b4",
        "query_overlap": "#d62728",
        "bm25_hard": "#9467bd",
        "gold_skill_near": "#2ca02c",
        "embedding_semantic_near": "#ff7f0e",
    }
    labels = {
        "random": "Random",
        "query_overlap": "Query overlap",
        "bm25_hard": "BM25 hard",
        "gold_skill_near": "Gold-skill near",
        "embedding_semantic_near": "Embedding near",
    }

    pool_sizes = sorted({int(row["pool_size"]) for row in summary_rows})
    min_x = math.log10(min(pool_sizes))
    max_x = math.log10(max(pool_sizes))

    def sx(size: int) -> float:
        return left + (math.log10(size) - min_x) / (max_x - min_x) * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - value) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-family="Arial" font-size="18" font-weight="700">RQ2 distractor type comparison: Top-1 accuracy</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>')

    for size in pool_sizes:
        x = sx(size)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#333"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{size}</text>')

    for distractor_type in DEFAULT_DISTRACTOR_TYPES:
        rows = [row for row in summary_rows if row["distractor_type"] == distractor_type]
        rows.sort(key=lambda row: int(row["pool_size"]))
        points = [(sx(int(row["pool_size"])), sy(row["top1_accuracy"])) for row in rows]
        point_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        color = colors[distractor_type]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{point_str}"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')

    legend_x = left + plot_w + 28
    for i, distractor_type in enumerate(DEFAULT_DISTRACTOR_TYPES):
        y = top + 18 + i * 24
        color = colors[distractor_type]
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 34}" y="{y + 4}" font-family="Arial" font-size="13">{labels[distractor_type]}</text>')

    lines.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13">Candidate pool size (log scale)</text>')
    lines.append(f'<text x="18" y="{top + plot_h / 2:.2f}" transform="rotate(-90 18,{top + plot_h / 2:.2f})" text-anchor="middle" font-family="Arial" font-size="13">Top-1 accuracy</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-usage-root", default="data/raw/Skill-Usage")
    parser.add_argument("--output-dir", default="data/experiments/rq2_distractor_types")
    parser.add_argument("--pool-sizes", nargs="+", default=DEFAULT_POOL_SIZES)
    parser.add_argument("--distractor-types", nargs="+", default=DEFAULT_DISTRACTOR_TYPES)
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=6002)
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all tasks")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--embedding-index-dir",
        default="data/raw/Skill-Usage/search_server/index",
        help="Directory containing official Skill-Usage embeddings.npy and skill_ids.json",
    )
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

    embeddings = None
    skill_id_to_index = {}
    index_skill_ids = []
    if "embedding_semantic_near" in args.distractor_types:
        embeddings, skill_id_to_index, index_skill_ids = load_embedding_index(
            Path(args.embedding_index_dir),
            root / "skills-34k" / "skills_meta.jsonl",
        )
        print(f"Loaded embedding index with {len(index_skill_ids)} skills")

    max_pool_size = max(parse_pool_size(value) for value in args.pool_sizes)
    ranked_by_task = {}
    for index, task in enumerate(tasks, start=1):
        gold = gt[task]
        query = " ".join(queries[task])
        gold_present = sorted(sid for sid in gold if sid in docs)
        non_gold_ids = [sid for sid in all_skill_ids if sid not in set(gold_present)]
        max_needed = max_pool_size - len(gold_present)
        ranked_by_task[task] = {
            "query_overlap": rank_by_query_overlap(query, non_gold_ids, docs)[:max_needed],
            "bm25_hard": [
                row["skill_id"]
                for row in bm25_rank(query, non_gold_ids, docs, max_needed)
            ],
            "gold_skill_near": rank_by_gold_skill_near(gold, non_gold_ids, docs)[:max_needed],
        }
        if "embedding_semantic_near" in args.distractor_types:
            ranked_by_task[task]["embedding_semantic_near"] = rank_by_embedding_semantic_near(
                gold,
                non_gold_ids,
                embeddings,
                skill_id_to_index,
                index_skill_ids,
            )[:max_needed]
        if index % 25 == 0:
            print(f"Prepared hard distractors for {index}/{len(tasks)} tasks")

    per_query_rows = []
    summary_rows = []
    error_examples = {}

    for distractor_type in args.distractor_types:
        repeat_count = args.random_repeats if distractor_type == "random" else 1
        for pool_label in args.pool_sizes:
            pool_size = parse_pool_size(pool_label)
            pooled_rows = []
            for repeat in range(repeat_count):
                rng = random.Random(args.seed + repeat + pool_size * 1009 + len(distractor_type) * 917)
                for task in tasks:
                    gold = gt[task]
                    query = " ".join(queries[task])
                    candidates, fallback_count = sample_candidates(
                        distractor_type,
                        gold,
                        all_skill_ids,
                        docs,
                        pool_size,
                        rng,
                        ranked_by_task[task],
                    )
                    ranked = bm25_rank(query, candidates, docs, args.top_k)
                    ranked_ids = [row["skill_id"] for row in ranked]
                    distractor_count = len(candidates) - len([sid for sid in gold if sid in docs])
                    hard_negative_count = (
                        max(distractor_count - fallback_count, 0)
                        if distractor_type != "random"
                        else 0
                    )
                    hard_negative_purity = (
                        hard_negative_count / distractor_count
                        if distractor_count and distractor_type != "random"
                        else 0.0
                    )
                    positive_score_count = ranked[0]["positive_score_count"] if ranked else 0
                    row = {
                        "task": task,
                        "distractor_type": distractor_type,
                        "pool_size": pool_label,
                        "actual_pool_size": len(candidates),
                        "repeat": repeat,
                        "gold_count": len(gold),
                        "distractor_count": distractor_count,
                        "hard_negative_count": hard_negative_count,
                        "hard_negative_purity": hard_negative_purity,
                        "fallback_distractors": fallback_count,
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
                        "first_gold_rank@10": first_gold_rank_at_k(ranked_ids, gold, args.top_k),
                    }
                    per_query_rows.append(row)
                    pooled_rows.append(row)

                    if row["top1_accuracy"] == 0.0:
                        key = f"{distractor_type}:{pool_label}:{repeat}:{task}"
                        if len(error_examples) < 80:
                            error_examples[key] = {
                                "query": query,
                                "gold": sorted(gold),
                                "first_gold_rank@10": row["first_gold_rank@10"],
                                "top_results": ranked,
                            }

            summary = summarize(pooled_rows)
            summary_row = {
                "distractor_type": distractor_type,
                "pool_size": pool_label,
                "tasks": len(tasks),
                "repeats": repeat_count,
                **summary,
            }
            for metric in SUMMARY_METRICS:
                summary_row[f"{metric}_std_across_queries"] = metric_stdev(pooled_rows, metric)
            summary_rows.append(summary_row)

    summary_rows.sort(key=lambda row: (row["distractor_type"], int(row["pool_size"])))
    output_payload = {
        "metadata_note": (
            "Skill-Usage gold skills have no category/repo/tag metadata in the local "
            "skills_meta.jsonl, so same-category and same-subcategory distractors are "
            "not directly evaluated. RQ2 uses lexical hard-negative proxies plus an "
            "embedding_semantic_near condition based on the official Skill-Usage "
            "precomputed skill embedding index."
        ),
        "gold_owner_prefix": GT_OWNER,
        "embedding_index_dir": args.embedding_index_dir,
        "distractor_descriptions": DISTRACTOR_DESCRIPTIONS,
        "summary": summary_rows,
    }

    (output_dir / "summary.json").write_text(json.dumps(output_payload, indent=2) + "\n")
    (output_dir / "error_examples.json").write_text(json.dumps(error_examples, indent=2) + "\n")
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "per_query_metrics.csv", per_query_rows)
    write_metric_svg(summary_rows, output_dir / "top1_by_distractor_type.svg")

    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'per_query_metrics.csv'}")
    print(f"Wrote {output_dir / 'error_examples.json'}")
    print(f"Wrote {output_dir / 'top1_by_distractor_type.svg'}")
    print()
    print(f"{'type':>16} {'pool':>6} {'top1':>8} {'err':>8} {'hit@10':>8} {'mrr':>8}")
    for row in summary_rows:
        print(
            f"{row['distractor_type']:>16} "
            f"{row['pool_size']:>6} "
            f"{row['top1_accuracy']:>8.3f} "
            f"{row['top1_error_rate']:>8.3f} "
            f"{row['hit@10']:>8.3f} "
            f"{row['mrr@10']:>8.3f}"
        )

    by_pool = {}
    for row in summary_rows:
        by_pool.setdefault(row["pool_size"], []).append(row)
    print()
    for pool_size, rows in sorted(by_pool.items(), key=lambda item: int(item[0])):
        worst = min(rows, key=lambda row: row["top1_accuracy"])
        print(
            f"Most harmful at pool {pool_size}: {worst['distractor_type']} "
            f"(top1={worst['top1_accuracy']:.3f}, error={worst['top1_error_rate']:.3f})"
        )


if __name__ == "__main__":
    main()
