#!/usr/bin/env python3
"""Run the RQ4 downstream skill exposure experiment.

RQ4 asks whether retrieving the right skill is enough for downstream task
performance. Running full SkillsBench agent evaluations requires external
model credentials and a sandbox runner, so this script measures a reproducible
necessary-condition proxy: which skills each retrieval condition would expose
to the downstream agent.

The experiment uses SkillsBench task-local curated skills as gold skills and
compares no-skill, oracle-gold, sparse retrieval, neural retrieval, hybrid
retrieval, and noisy-gold exposure conditions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from rq1_retrieval_scaling import tokenize
from rq3_retriever_comparison import bm25_full_scores, rank_candidates, reciprocal_rank_fusion


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEFAULT_TOP_KS = [1, 3, 5, 10]
SUMMARY_METRICS = [
    "any_gold_coverage",
    "complete_gold_coverage",
    "strict_skill_set_match",
    "gold_recall",
    "skill_precision",
    "top1_is_gold",
    "underload",
    "overload",
    "missing_gold_count",
    "extra_skill_count",
    "exposed_skill_count",
    "context_tokens",
    "gold_token_share",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_front_matter(markdown: str) -> tuple[dict, str]:
    if not markdown.startswith("---"):
        return {}, markdown
    parts = markdown.split("---", 2)
    if len(parts) < 3:
        return {}, markdown
    try:
        import yaml

        metadata = yaml.safe_load(parts[1]) or {}
    except Exception:
        metadata = {}
    return metadata, parts[2].strip()


def first_heading(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip() or fallback
    return fallback


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def stable_skill_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_skillsbench_tasks(tasks_root: Path) -> tuple[list[dict], list[dict]]:
    tasks = []
    skill_by_hash: dict[str, dict] = {}
    for task_dir in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
        task_md_path = task_dir / "task.md"
        skills_dir = task_dir / "environment" / "skills"
        if not task_md_path.exists() or not skills_dir.exists():
            continue

        task_raw = task_md_path.read_text(errors="replace")
        task_meta, task_body = parse_front_matter(task_raw)
        task_metadata = task_meta.get("metadata", {}) if isinstance(task_meta, dict) else {}
        gold_skill_ids = []
        gold_skill_paths = []
        for skill_path in sorted(skills_dir.glob("*/SKILL.md")):
            skill_raw = skill_path.read_text(errors="replace")
            skill_meta, skill_body = parse_front_matter(skill_raw)
            skill_id = stable_skill_id(skill_raw)
            name = skill_meta.get("name") if isinstance(skill_meta, dict) else None
            description = skill_meta.get("description") if isinstance(skill_meta, dict) else None
            name = name or skill_path.parent.name
            description = description or first_heading(skill_body, name)
            if skill_id not in skill_by_hash:
                skill_by_hash[skill_id] = {
                    "skill_id": skill_id,
                    "canonical_name": name,
                    "description": description,
                    "text": f"{name}\n{description}\n{skill_body}",
                    "token_count": count_tokens(skill_raw),
                    "example_path": str(skill_path),
                    "aliases": [],
                }
            skill_by_hash[skill_id]["aliases"].append(f"{task_dir.name}/{skill_path.parent.name}")
            gold_skill_ids.append(skill_id)
            gold_skill_paths.append(str(skill_path))

        tasks.append(
            {
                "task_id": task_dir.name,
                "query": task_body,
                "difficulty": task_metadata.get("difficulty", ""),
                "category": task_metadata.get("category", ""),
                "subcategory": task_metadata.get("subcategory", ""),
                "gold_skill_ids": sorted(set(gold_skill_ids)),
                "gold_skill_paths": gold_skill_paths,
                "gold_skill_count": len(set(gold_skill_ids)),
                "query_tokens": count_tokens(task_body),
            }
        )

    return tasks, sorted(skill_by_hash.values(), key=lambda row: row["skill_id"])


def dense_encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def rank_all(scores: np.ndarray, skill_ids: list[str], skill_id_to_index: dict[str, int]) -> list[str]:
    return rank_candidates(scores, skill_ids, skill_id_to_index, len(skill_ids))


def exposure_row(
    task: dict,
    condition: str,
    exposed_skill_ids: list[str],
    skill_by_id: dict[str, dict],
) -> dict:
    gold = set(task["gold_skill_ids"])
    exposed = list(dict.fromkeys(exposed_skill_ids))
    exposed_set = set(exposed)
    covered = gold & exposed_set
    extras = exposed_set - gold
    missing = gold - exposed_set
    context_tokens = sum(skill_by_id[sid]["token_count"] for sid in exposed)
    gold_context_tokens = sum(skill_by_id[sid]["token_count"] for sid in exposed if sid in gold)
    return {
        "task_id": task["task_id"],
        "condition": condition,
        "difficulty": task["difficulty"],
        "category": task["category"],
        "gold_skill_count": task["gold_skill_count"],
        "exposed_skill_count": len(exposed),
        "covered_gold_count": len(covered),
        "missing_gold_count": len(missing),
        "extra_skill_count": len(extras),
        "any_gold_coverage": 1.0 if covered else 0.0,
        "complete_gold_coverage": 1.0 if gold and covered == gold else 0.0,
        "strict_skill_set_match": 1.0 if gold and covered == gold and not extras else 0.0,
        "gold_recall": len(covered) / len(gold) if gold else 0.0,
        "skill_precision": len(covered) / len(exposed) if exposed else 0.0,
        "top1_is_gold": 1.0 if exposed and exposed[0] in gold else 0.0,
        "underload": 1.0 if missing else 0.0,
        "overload": 1.0 if extras else 0.0,
        "context_tokens": context_tokens,
        "gold_token_share": gold_context_tokens / context_tokens if context_tokens else 0.0,
        "exposed_skill_ids": ";".join(exposed),
        "missing_skill_ids": ";".join(sorted(missing)),
        "extra_skill_ids": ";".join(sorted(extras)),
    }


def summarize(rows: list[dict]) -> dict:
    summary = {metric: mean(row[metric] for row in rows) for metric in SUMMARY_METRICS}
    summary["tasks"] = len(rows)
    summary["median_context_tokens"] = median(row["context_tokens"] for row in rows)
    for metric in SUMMARY_METRICS:
        values = [row[metric] for row in rows]
        summary[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return summary


def pick_case_studies(rows: list[dict], tasks: list[dict], skill_by_id: dict[str, dict]) -> dict:
    task_by_id = {task["task_id"]: task for task in tasks}

    def enrich(row: dict) -> dict:
        task = task_by_id[row["task_id"]]
        exposed = [sid for sid in row["exposed_skill_ids"].split(";") if sid]
        missing = [sid for sid in row["missing_skill_ids"].split(";") if sid]
        extras = [sid for sid in row["extra_skill_ids"].split(";") if sid]
        return {
            "task_id": row["task_id"],
            "condition": row["condition"],
            "difficulty": row["difficulty"],
            "category": row["category"],
            "gold_skill_count": task["gold_skill_count"],
            "covered_gold_count": row["covered_gold_count"],
            "missing_gold_count": row["missing_gold_count"],
            "extra_skill_count": row["extra_skill_count"],
            "context_tokens": row["context_tokens"],
            "gold_skills": [skill_by_id[sid]["canonical_name"] for sid in task["gold_skill_ids"]],
            "exposed_skills": [skill_by_id[sid]["canonical_name"] for sid in exposed],
            "missing_skills": [skill_by_id[sid]["canonical_name"] for sid in missing],
            "extra_skills": [skill_by_id[sid]["canonical_name"] for sid in extras[:10]],
        }

    top1_correct_underloaded = [
        row
        for row in rows
        if row["condition"].endswith("_top1")
        and row["top1_is_gold"] == 1.0
        and row["complete_gold_coverage"] == 0.0
    ]
    full_coverage_overloaded = [
        row
        for row in rows
        if row["complete_gold_coverage"] == 1.0 and row["extra_skill_count"] > 0
    ]
    wrong_top1 = [
        row
        for row in rows
        if row["condition"].endswith("_top1") and row["top1_is_gold"] == 0.0
    ]

    top1_correct_underloaded.sort(key=lambda row: (-row["gold_skill_count"], row["extra_skill_count"], row["task_id"]))
    full_coverage_overloaded.sort(key=lambda row: (-row["extra_skill_count"], row["context_tokens"], row["task_id"]))
    wrong_top1.sort(key=lambda row: (-row["gold_skill_count"], row["task_id"]))
    return {
        "top1_correct_but_incomplete": [enrich(row) for row in top1_correct_underloaded[:5]],
        "complete_gold_but_noisy": [enrich(row) for row in full_coverage_overloaded[:5]],
        "wrong_top1": [enrich(row) for row in wrong_top1[:5]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skillsbench-root", default="data/raw/skillsbench")
    parser.add_argument("--output-dir", default="data/experiments/rq4_downstream_skill_exposure")
    parser.add_argument("--top-ks", nargs="+", type=int, default=DEFAULT_TOP_KS)
    parser.add_argument("--seed", type=int, default=6002)
    parser.add_argument("--neural-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--neural-batch-size", type=int, default=64)
    parser.add_argument("--limit-tasks", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_root = Path(args.skillsbench_root) / "tasks"

    tasks, skills = load_skillsbench_tasks(tasks_root)
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]
    skill_by_id = {skill["skill_id"]: skill for skill in skills}
    skill_ids = [skill["skill_id"] for skill in skills]
    skill_id_to_index = {sid: index for index, sid in enumerate(skill_ids)}
    skill_texts = [skill["text"] for skill in skills]

    build_times = {}
    start = time.perf_counter()
    count_vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-zA-Z0-9]+\b", lowercase=True)
    counts = count_vectorizer.fit_transform(skill_texts).astype(np.float32)
    counts_csc = counts.tocsc()
    doc_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    doc_lengths[doc_lengths == 0] = 1.0
    doc_freq = np.diff(counts_csc.indptr).astype(np.float32)
    avg_doc_len = float(doc_lengths.mean())
    build_times["bm25"] = time.perf_counter() - start

    start = time.perf_counter()
    tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z0-9]+\b", lowercase=True, norm="l2")
    tfidf_matrix = tfidf_vectorizer.fit_transform(skill_texts).astype(np.float32)
    build_times["tfidf"] = time.perf_counter() - start

    start = time.perf_counter()
    from sentence_transformers import SentenceTransformer

    neural_model = SentenceTransformer(args.neural_model_name, local_files_only=True)
    neural_doc_embeddings = dense_encode(neural_model, skill_texts, args.neural_batch_size)
    build_times["neural_minilm"] = time.perf_counter() - start
    build_times["hybrid_bm25_neural"] = build_times["bm25"] + build_times["neural_minilm"]

    per_task_rows = []
    score_times = Counter()
    rng = random.Random(args.seed)
    for task in tasks:
        query = task["query"]
        gold = set(task["gold_skill_ids"])

        per_task_rows.append(exposure_row(task, "no_skill", [], skill_by_id))
        per_task_rows.append(exposure_row(task, "oracle_gold_all", task["gold_skill_ids"], skill_by_id))
        non_gold = [sid for sid in skill_ids if sid not in gold]
        noisy_gold = task["gold_skill_ids"] + rng.sample(non_gold, min(5, len(non_gold)))
        per_task_rows.append(exposure_row(task, "oracle_gold_plus_5_noise", noisy_gold, skill_by_id))
        per_task_rows.append(exposure_row(task, "all_skills_visible", skill_ids, skill_by_id))

        start = time.perf_counter()
        bm25_scores = bm25_full_scores(query, count_vectorizer, counts_csc, doc_lengths, doc_freq, avg_doc_len)
        bm25_ranked = rank_all(bm25_scores, skill_ids, skill_id_to_index)
        score_times["bm25"] += time.perf_counter() - start

        start = time.perf_counter()
        q_tfidf = tfidf_vectorizer.transform([query]).astype(np.float32)
        tfidf_scores = np.asarray((tfidf_matrix @ q_tfidf.T).todense()).ravel().astype(np.float32)
        tfidf_ranked = rank_all(tfidf_scores, skill_ids, skill_id_to_index)
        score_times["tfidf"] += time.perf_counter() - start

        start = time.perf_counter()
        q_neural = dense_encode(neural_model, [query], args.neural_batch_size)[0]
        neural_scores = neural_doc_embeddings @ q_neural
        neural_ranked = rank_all(neural_scores, skill_ids, skill_id_to_index)
        score_times["neural_minilm"] += time.perf_counter() - start

        start = time.perf_counter()
        hybrid_ranked = reciprocal_rank_fusion(
            bm25_scores,
            neural_scores,
            skill_ids,
            skill_id_to_index,
            len(skill_ids),
        )
        score_times["hybrid_bm25_neural"] += time.perf_counter() - start

        for k in args.top_ks:
            per_task_rows.append(exposure_row(task, f"bm25_top{k}", bm25_ranked[:k], skill_by_id))
            per_task_rows.append(exposure_row(task, f"tfidf_top{k}", tfidf_ranked[:k], skill_by_id))
            per_task_rows.append(exposure_row(task, f"neural_minilm_top{k}", neural_ranked[:k], skill_by_id))
            per_task_rows.append(exposure_row(task, f"hybrid_bm25_neural_top{k}", hybrid_ranked[:k], skill_by_id))

    summary_rows = []
    for condition in sorted({row["condition"] for row in per_task_rows}):
        rows = [row for row in per_task_rows if row["condition"] == condition]
        retriever = re.sub(r"_top\d+$", "", condition)
        top_k_match = re.search(r"_top(\d+)$", condition)
        summary_rows.append(
            {
                "condition": condition,
                "retriever": retriever,
                "top_k": int(top_k_match.group(1)) if top_k_match else "",
                "build_seconds": build_times.get(retriever, 0.0),
                "score_seconds_total": score_times.get(retriever, 0.0),
                "score_seconds_per_task": score_times.get(retriever, 0.0) / len(tasks),
                **summarize(rows),
            }
        )

    def condition_sort(row: dict) -> tuple:
        condition = row["condition"]
        order = {
            "no_skill": 0,
            "oracle_gold_all": 1,
            "oracle_gold_plus_5_noise": 2,
            "all_skills_visible": 99,
        }
        if condition in order:
            return (order[condition], condition)
        return (10, row["retriever"], row["top_k"] or 0)

    summary_rows.sort(key=condition_sort)
    metadata = {
        "dataset": "SkillsBench default tasks",
        "tasks": len(tasks),
        "unique_skill_documents": len(skills),
        "task_gold_skill_references": sum(task["gold_skill_count"] for task in tasks),
        "neural_model_name": args.neural_model_name,
        "important_boundary": (
            "This experiment measures downstream skill exposure/readiness proxies, "
            "not actual agent pass rate. Full pass-rate evaluation requires an "
            "external model-backed SkillsBench agent run."
        ),
    }
    case_studies = pick_case_studies(per_task_rows, tasks, skill_by_id)

    write_csv(output_dir / "per_task_exposure.csv", per_task_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary_rows}, indent=2) + "\n"
    )
    (output_dir / "case_studies.json").write_text(json.dumps(case_studies, indent=2) + "\n")

    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'per_task_exposure.csv'}")
    print(f"Wrote {output_dir / 'case_studies.json'}")
    print()
    print(f"{'condition':>28} {'complete':>9} {'strict':>8} {'recall':>8} {'precision':>9} {'extra':>8} {'tokens':>8}")
    for row in summary_rows:
        print(
            f"{row['condition']:>28} "
            f"{row['complete_gold_coverage']:>9.3f} "
            f"{row['strict_skill_set_match']:>8.3f} "
            f"{row['gold_recall']:>8.3f} "
            f"{row['skill_precision']:>9.3f} "
            f"{row['extra_skill_count']:>8.2f} "
            f"{row['median_context_tokens']:>8.0f}"
        )


if __name__ == "__main__":
    main()
