#!/usr/bin/env python3
"""Run the RQ5 decision-budget stress test for LLM skill routing.

RQ5 asks: given that all required skills are visible, how do distractor count
and distractor hardness affect an LLM router's ability to select the complete
and correct skill set?

Design (docs/rq5_decision_budget/proposal.md):

- every candidate menu contains all gold skills (retrieval misses bypassed);
- the primary manipulation is the absolute distractor count n in {0,2,5,10,20};
- distractor types: uniform-random vs retrieval-hard (RQ3 hybrid BM25+MiniLM);
- one Qwen model acts as a multi-label router over name+description menus;
- outputs: selection precision/recall/F1, exact set match, diagnostics,
  token/latency cost, paired-bootstrap contrasts, and SVG plots.

The script supports --dry-run, --limit-tasks, --resume, --max-api-calls,
exponential-backoff retries, append-only JSONL logging, and deduplication by
stable condition IDs. The n=0 baseline is called once per task and shared by
both distractor types.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import requests

from rq1_retrieval_scaling import (
    load_json,
    load_skill_docs,
    normalize_gt,
)


DASHSCOPE_COMPATIBLE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_NOISE_COUNTS = [0, 2, 5, 10, 20]
DEFAULT_DISTRACTOR_TYPES = ["random", "hard"]
SHARED_BASELINE_TYPE = "shared"

ROUTER_SYSTEM = (
    "You are a skill router for an LLM agent. Select all and only the skills that are "
    "directly useful for completing the task. Do not solve the task. Do not select a "
    "skill merely because it shares broad keywords with the task. Return JSON only."
)


# ---------------------------------------------------------------------------
# Generic IO helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def condition_id(task: str, distractor_type: str, noise_count: int) -> str:
    return f"{task}|{distractor_type}|n{noise_count}"


def redact_base_url(base_url: str) -> str:
    """Replace the account-specific workspace subdomain with a placeholder.

    Workspace-dedicated endpoints look like
    https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/... and the WorkspaceId
    identifies the user's account, so it must not be written to tracked
    experiment outputs.
    """
    return re.sub(r"//[^./]+(\.[a-z0-9-]+\.maas\.aliyuncs\.com)", r"//<workspace>\1", base_url)


# ---------------------------------------------------------------------------
# Distractor construction
# ---------------------------------------------------------------------------

def build_random_sequence(task: str, gold: set[str], all_skill_ids: list[str], seed: int, max_needed: int) -> list[str]:
    """One deterministic permutation prefix of U - G_t per task (nested menus)."""
    non_gold = [sid for sid in all_skill_ids if sid not in gold]
    rng = random.Random(seed + stable_hash_int(task) % (2**31))
    rng.shuffle(non_gold)
    return non_gold[:max_needed]


def build_hard_rankings(
    tasks: list[str],
    queries: dict,
    gt: dict[str, set[str]],
    docs: dict[str, dict],
    all_skill_ids: list[str],
    max_needed: int,
    neural_model_name: str,
    neural_batch_size: int,
    doc_embedding_cache: Path,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Rank non-gold skills per task with the RQ3 hybrid BM25 + MiniLM retriever.

    Returns (ranked_non_gold_by_task, max_gold_similarity_by_task_skill) where
    the similarity map is keyed by f"{task}|{skill_id}" and holds the maximum
    MiniLM cosine similarity between that skill and any gold skill of the task.
    """
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sentence_transformers import SentenceTransformer

    from rq3_retriever_comparison import bm25_full_scores, reciprocal_rank_fusion

    skill_id_to_index = {sid: index for index, sid in enumerate(all_skill_ids)}
    desc_texts = [f"{docs[sid]['name']} {docs[sid]['description']}" for sid in all_skill_ids]

    count_vectorizer = CountVectorizer(token_pattern=r"(?u)\b[a-zA-Z0-9]+\b", lowercase=True)
    counts = count_vectorizer.fit_transform(desc_texts).astype(np.float32)
    counts_csc = counts.tocsc()
    doc_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    doc_lengths[doc_lengths == 0] = 1.0
    doc_freq = np.diff(counts_csc.indptr).astype(np.float32)
    avg_doc_len = float(doc_lengths.mean())

    model = SentenceTransformer(neural_model_name, local_files_only=True)
    if doc_embedding_cache.exists():
        doc_embeddings = np.load(doc_embedding_cache).astype(np.float32)
        if doc_embeddings.shape[0] != len(all_skill_ids):
            raise SystemExit(
                f"Cached embeddings at {doc_embedding_cache} have {doc_embeddings.shape[0]} rows "
                f"but the library has {len(all_skill_ids)} skills. Delete the cache and rerun."
            )
    else:
        doc_embeddings = np.asarray(
            model.encode(desc_texts, batch_size=neural_batch_size, normalize_embeddings=True, show_progress_bar=True),
            dtype=np.float32,
        )
        doc_embedding_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(doc_embedding_cache, doc_embeddings)

    query_texts = [" ".join(queries[task]) for task in tasks]
    query_embeddings = np.asarray(
        model.encode(query_texts, batch_size=neural_batch_size, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )

    ranked_by_task: dict[str, list[str]] = {}
    sim_by_task_skill: dict[str, float] = {}
    for task, q_emb in zip(tasks, query_embeddings):
        gold = gt[task]
        query = " ".join(queries[task])
        bm25_scores = bm25_full_scores(query, count_vectorizer, counts_csc, doc_lengths, doc_freq, avg_doc_len)
        neural_scores = doc_embeddings @ q_emb
        non_gold_ids = [sid for sid in all_skill_ids if sid not in gold]
        fused = reciprocal_rank_fusion(
            bm25_scores,
            neural_scores,
            non_gold_ids,
            skill_id_to_index,
            max_needed,
        )
        ranked_by_task[task] = fused

        gold_indices = [skill_id_to_index[sid] for sid in sorted(gold) if sid in skill_id_to_index]
        if gold_indices:
            gold_matrix = doc_embeddings[gold_indices]
            for sid in fused:
                sims = gold_matrix @ doc_embeddings[skill_id_to_index[sid]]
                sim_by_task_skill[f"{task}|{sid}"] = float(sims.max())
    return ranked_by_task, sim_by_task_skill


# ---------------------------------------------------------------------------
# Candidate menus
# ---------------------------------------------------------------------------

def display_order_key(task: str, skill_id: str, seed: int) -> str:
    return hashlib.sha256(f"{task}|{skill_id}|{seed}".encode("utf-8")).hexdigest()


def build_menu(
    task: str,
    gold: set[str],
    distractors: list[str],
    seed: int,
) -> list[str]:
    candidates = sorted(gold) + list(distractors)
    candidates.sort(key=lambda sid: display_order_key(task, sid, seed))
    return candidates


def validate_menu(
    task: str,
    distractor_type: str,
    noise_count: int,
    menu: list[str],
    gold: set[str],
    distractors: list[str],
) -> None:
    label = condition_id(task, distractor_type, noise_count)
    menu_set = set(menu)
    assert len(menu) == len(menu_set), f"{label}: duplicate skill IDs in menu"
    assert gold <= menu_set, f"{label}: gold skill missing from menu"
    assert not (set(distractors) & gold), f"{label}: distractor belongs to gold set"
    assert len(menu) == len(gold) + noise_count, (
        f"{label}: menu size {len(menu)} != gold {len(gold)} + noise {noise_count}"
    )


def build_experiment_plan(
    tasks: list[str],
    gt: dict[str, set[str]],
    noise_counts: list[int],
    distractor_types: list[str],
    random_sequences: dict[str, list[str]],
    hard_rankings: dict[str, list[str]],
    seed: int,
) -> list[dict]:
    """Build the full (task x distractor_type x noise_count) condition list.

    n=0 is emitted once per task with distractor_type='shared'.
    """
    plan: list[dict] = []
    for task in tasks:
        gold = gt[task]
        for noise_count in sorted(set(noise_counts)):
            if noise_count == 0:
                sources = [(SHARED_BASELINE_TYPE, [])]
            else:
                sources = []
                if "random" in distractor_types:
                    sources.append(("random", random_sequences[task][:noise_count]))
                if "hard" in distractor_types:
                    sources.append(("hard", hard_rankings[task][:noise_count]))
            for distractor_type, distractors in sources:
                if noise_count and len(distractors) < noise_count:
                    raise SystemExit(
                        f"Not enough {distractor_type} distractors for {task} at n={noise_count}"
                    )
                menu = build_menu(task, gold, distractors, seed)
                validate_menu(task, distractor_type, noise_count, menu, gold, distractors)
                gold_positions = sorted(index + 1 for index, sid in enumerate(menu) if sid in gold)
                plan.append(
                    {
                        "condition_id": condition_id(task, distractor_type, noise_count),
                        "task": task,
                        "distractor_type": distractor_type,
                        "noise_count": noise_count,
                        "gold_count": len(gold),
                        "menu_size": len(menu),
                        "gold_fraction": len(gold) / len(menu),
                        "menu": menu,
                        "gold": sorted(gold),
                        "distractors": list(distractors),
                        "gold_positions": gold_positions,
                    }
                )
    return plan


def check_nesting(plan: list[dict]) -> None:
    by_task_type: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in plan:
        if row["noise_count"] > 0:
            by_task_type[(row["task"], row["distractor_type"])].append(row)
    for (task, distractor_type), rows in by_task_type.items():
        rows.sort(key=lambda row: row["noise_count"])
        for smaller, larger in zip(rows, rows[1:]):
            assert set(smaller["distractors"]) <= set(larger["distractors"]), (
                f"{task}/{distractor_type}: n={smaller['noise_count']} menu not nested in n={larger['noise_count']}"
            )


# ---------------------------------------------------------------------------
# Router prompt and response parsing
# ---------------------------------------------------------------------------

def build_router_messages(query: str, menu: list[str], docs: dict[str, dict]) -> list[dict]:
    lines = [f"Task:\n{query}", "", "Available skills:"]
    for index, sid in enumerate(menu, start=1):
        doc = docs[sid]
        lines.append(f"[{index}] {doc['name']}")
        lines.append(f"Description: {doc['description']}")
        lines.append("")
    lines.append('Return exactly:')
    lines.append('{"selected": [integer indices]}')
    lines.append("")
    lines.append("Use an empty list only if none of the available skills is relevant.")
    user = "\n".join(lines)
    return [{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": user}]


def parse_selection(raw_text: str, menu_size: int) -> dict:
    """Parse {"selected": [...]} from a router response.

    Returns {"parse_status": ..., "selected_indices": [...], "parse_error": ...}.
    Invalid responses are NOT converted into an empty selection.
    """
    def interpret(obj) -> dict:
        if not isinstance(obj, dict) or "selected" not in obj:
            return {"parse_status": "invalid", "selected_indices": [], "parse_error": "missing 'selected' key"}
        values = obj["selected"]
        if not isinstance(values, list):
            return {"parse_status": "invalid", "selected_indices": [], "parse_error": "'selected' is not a list"}
        indices = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                return {"parse_status": "invalid", "selected_indices": [], "parse_error": f"non-integer value {value!r}"}
            if value < 1 or value > menu_size:
                return {"parse_status": "invalid", "selected_indices": [], "parse_error": f"index {value} out of range 1..{menu_size}"}
            indices.append(value)
        return {"parse_status": "ok", "selected_indices": sorted(set(indices)), "parse_error": ""}

    text = raw_text.strip()
    try:
        return {**interpret(json.loads(text)), "parse_mode": "strict"}
    except Exception:
        pass
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return {
            "parse_status": "invalid",
            "selected_indices": [],
            "parse_error": "no JSON object found",
            "parse_mode": "none",
        }
    try:
        return {**interpret(json.loads(match.group(0))), "parse_mode": "extracted"}
    except Exception as exc:
        return {
            "parse_status": "invalid",
            "selected_indices": [],
            "parse_error": f"extraction failed: {exc}",
            "parse_mode": "extracted",
        }


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_router(
    messages: list[dict],
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_completion_tokens: int,
    timeout: int,
    max_retries: int,
) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_completion_tokens,
        "stream": False,
        "enable_thinking": False,
    }
    last_error = None
    for attempt in range(max_retries):
        start = time.perf_counter()
        try:
            response = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(2**attempt, 30))
            continue
        latency = time.perf_counter() - start
        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage", {}) or {}
            return {
                "raw_text": data["choices"][0]["message"]["content"],
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "latency_seconds": latency,
                "retry_count": attempt,
                "api_status": "ok",
            }
        if response.status_code in {408, 429, 500, 502, 503, 504}:
            last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            time.sleep(min(2**attempt, 30))
            continue
        raise RuntimeError(f"Router API error {response.status_code}: {response.text[:500]}")
    raise RuntimeError(f"Router API call failed after {max_retries} retries: {last_error}")


def is_budget_or_access_error(exc: Exception) -> bool:
    text = str(exc)
    return any(
        marker in text
        for marker in [
            "AccessDenied.Unpurchased",
            "free quota has been exhausted",
            "insufficient",
            "quota",
            "403",
            "401",
        ]
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score_selection(selected: set[str], gold: set[str]) -> dict:
    correct = selected & gold
    precision_inclusive = len(correct) / len(selected) if selected else 0.0
    precision_conditional = len(correct) / len(selected) if selected else None
    recall = len(correct) / len(gold) if gold else 0.0
    f1 = (
        2 * precision_inclusive * recall / (precision_inclusive + recall)
        if (precision_inclusive + recall) > 0
        else 0.0
    )
    union = selected | gold
    return {
        "precision_inclusive": precision_inclusive,
        "precision_conditional": precision_conditional,
        "gold_recall": recall,
        "selection_f1": f1,
        "exact_set_match": 1.0 if selected == gold else 0.0,
        "complete_gold_coverage": 1.0 if gold <= selected else 0.0,
        "missing_gold_count": len(gold - selected),
        "extra_skill_count": len(selected - gold),
        "any_wrong_selection": 1.0 if selected - gold else 0.0,
        "empty_selection": 1.0 if not selected else 0.0,
        "jaccard": len(correct) / len(union) if union else 1.0,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def paired_bootstrap_ci(
    per_task_diffs: list[float],
    resamples: int,
    seed: int,
) -> dict:
    if not per_task_diffs:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_tasks": 0}
    rng = random.Random(seed)
    n = len(per_task_diffs)
    means = []
    for _ in range(resamples):
        sample = [per_task_diffs[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    return {
        "mean_diff": mean(per_task_diffs),
        "ci_low": percentile(means, 0.025),
        "ci_high": percentile(means, 0.975),
        "n_tasks": n,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def build_result_rows(plan_by_id: dict[str, dict], responses: list[dict]) -> list[dict]:
    rows = []
    for record in responses:
        plan = plan_by_id.get(record["condition_id"])
        if plan is None:
            continue
        menu = plan["menu"]
        gold = set(plan["gold"])
        parse_status = record.get("parse_status", "invalid")
        selected_ids = sorted(menu[i - 1] for i in record.get("selected_indices", []))
        metrics = score_selection(set(selected_ids), gold) if parse_status == "ok" else None
        row = {
            "condition_id": record["condition_id"],
            "task": plan["task"],
            "distractor_type": plan["distractor_type"],
            "noise_count": plan["noise_count"],
            "gold_count": plan["gold_count"],
            "menu_size": plan["menu_size"],
            "gold_fraction": round(plan["gold_fraction"], 6),
            "gold_positions": ";".join(str(p) for p in plan["gold_positions"]),
            "mean_distractor_gold_similarity": record.get("mean_distractor_gold_similarity", ""),
            "max_distractor_gold_similarity": record.get("max_distractor_gold_similarity", ""),
            "parse_status": parse_status,
            "parse_mode": record.get("parse_mode", ""),
            "selected_count": len(selected_ids),
            "selected_skill_ids": ";".join(selected_ids),
            "prompt_tokens": record.get("prompt_tokens", 0),
            "completion_tokens": record.get("completion_tokens", 0),
            "total_tokens": record.get("total_tokens", 0),
            "latency_seconds": round(record.get("latency_seconds", 0.0), 4),
            "retry_count": record.get("retry_count", 0),
        }
        if metrics:
            row.update(
                {
                    key: (round(value, 6) if isinstance(value, float) else value)
                    for key, value in metrics.items()
                    if value is not None
                }
            )
            row["precision_conditional"] = (
                round(metrics["precision_conditional"], 6)
                if metrics["precision_conditional"] is not None
                else ""
            )
        else:
            for key in [
                "precision_inclusive",
                "precision_conditional",
                "gold_recall",
                "selection_f1",
                "exact_set_match",
                "complete_gold_coverage",
                "missing_gold_count",
                "extra_skill_count",
                "any_wrong_selection",
                "empty_selection",
                "jaccard",
            ]:
                row[key] = ""
        rows.append(row)
    return rows


def expand_shared_baseline(rows: list[dict], distractor_types: list[str]) -> list[dict]:
    """Duplicate the shared n=0 baseline into each distractor-type curve for summaries."""
    expanded = []
    for row in rows:
        if row["distractor_type"] == SHARED_BASELINE_TYPE:
            for distractor_type in distractor_types:
                expanded.append({**row, "distractor_type": distractor_type})
        else:
            expanded.append(row)
    return expanded


def summarize_conditions(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["distractor_type"], row["noise_count"])].append(row)

    summary_rows = []
    for (distractor_type, noise_count), group in sorted(grouped.items()):
        valid = [row for row in group if row["parse_status"] == "ok"]
        non_empty = [row for row in valid if float(row["selected_count"]) > 0]
        summary = {
            "distractor_type": distractor_type,
            "noise_count": noise_count,
            "n_conditions": len(group),
            "n_valid": len(valid),
            "invalid_response_rate": round(1.0 - len(valid) / len(group), 6) if group else 0.0,
        }
        if valid:
            summary.update(
                {
                    "macro_f1": round(mean(float(r["selection_f1"]) for r in valid), 6),
                    "macro_precision_inclusive": round(mean(float(r["precision_inclusive"]) for r in valid), 6),
                    "macro_precision_conditional": (
                        round(mean(float(r["precision_conditional"]) for r in non_empty), 6)
                        if non_empty
                        else ""
                    ),
                    "macro_gold_recall": round(mean(float(r["gold_recall"]) for r in valid), 6),
                    "exact_set_match_rate": round(mean(float(r["exact_set_match"]) for r in valid), 6),
                    "complete_gold_coverage_rate": round(mean(float(r["complete_gold_coverage"]) for r in valid), 6),
                    "mean_missing_gold_count": round(mean(float(r["missing_gold_count"]) for r in valid), 6),
                    "mean_extra_skill_count": round(mean(float(r["extra_skill_count"]) for r in valid), 6),
                    "any_wrong_selection_rate": round(mean(float(r["any_wrong_selection"]) for r in valid), 6),
                    "empty_selection_rate": round(mean(float(r["empty_selection"]) for r in valid), 6),
                    "mean_jaccard": round(mean(float(r["jaccard"]) for r in valid), 6),
                    "mean_prompt_tokens": round(mean(float(r["prompt_tokens"]) for r in group), 1),
                    "mean_completion_tokens": round(mean(float(r["completion_tokens"]) for r in group), 1),
                    "mean_total_tokens": round(mean(float(r["total_tokens"]) for r in group), 1),
                    "median_latency_seconds": round(median(float(r["latency_seconds"]) for r in group), 4),
                    "p90_latency_seconds": round(percentile([float(r["latency_seconds"]) for r in group], 0.90), 4),
                }
            )
        summary_rows.append(summary)
    return summary_rows


def summarize_single_gold(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if int(row["gold_count"]) == 1:
            grouped[(row["distractor_type"], row["noise_count"])].append(row)
    summary_rows = []
    for (distractor_type, noise_count), group in sorted(grouped.items()):
        valid = [row for row in group if row["parse_status"] == "ok"]
        summary_rows.append(
            {
                "distractor_type": distractor_type,
                "noise_count": noise_count,
                "n_single_gold": len(group),
                "correct_selection_accuracy": (
                    round(mean(float(r["exact_set_match"]) for r in valid), 6) if valid else ""
                ),
                "wrong_selection_rate": (
                    round(mean(float(r["any_wrong_selection"]) for r in valid), 6) if valid else ""
                ),
                "refusal_rate": (
                    round(mean(float(r["empty_selection"]) for r in valid), 6) if valid else ""
                ),
                "invalid_rate": round(1.0 - len(valid) / len(group), 6) if group else 0.0,
            }
        )
    return summary_rows


def compute_contrasts(rows: list[dict], noise_counts: list[int], resamples: int, seed: int) -> list[dict]:
    """Pre-specified paired-bootstrap contrasts on per-task selection F1."""
    f1_by_task: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["parse_status"] != "ok":
            continue
        key = (row["distractor_type"], int(row["noise_count"]))
        f1_by_task[key][row["task"]] = float(row["selection_f1"])

    nonzero = sorted(n for n in set(noise_counts) if n > 0)
    if not nonzero:
        return []
    n_max = nonzero[-1]
    top3 = nonzero[-3:]

    contrasts = []

    def paired(cond_a: tuple[str, int], cond_b: tuple[str, int], label: str) -> None:
        a = f1_by_task.get(cond_a, {})
        b = f1_by_task.get(cond_b, {})
        shared_tasks = sorted(set(a) & set(b))
        diffs = [a[task] - b[task] for task in shared_tasks]
        contrasts.append({"contrast": label, "metric": "selection_f1", **paired_bootstrap_ci(diffs, resamples, seed)})

    for distractor_type in ["hard", "random"]:
        paired(
            (distractor_type, n_max),
            (distractor_type, 0),
            f"{distractor_type}_n{n_max}_minus_n0",
        )
    for n in top3:
        paired(("hard", n), ("random", n), f"hard_minus_random_n{n}")
    return contrasts


# ---------------------------------------------------------------------------
# SVG plots
# ---------------------------------------------------------------------------

def write_line_svg(
    summary_rows: list[dict],
    metric: str,
    title: str,
    y_label: str,
    path: Path,
    y_max: float | None = 1.0,
) -> None:
    width, height = 860, 440
    left, right, top, bottom = 82, 170, 40, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {"random": "#1f77b4", "hard": "#d62728"}

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in summary_rows:
        value = row.get(metric, "")
        if value == "" or value is None:
            continue
        series[row["distractor_type"]].append((int(row["noise_count"]), float(value)))
    if not series:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>\n")
        return

    all_noise = sorted({n for points in series.values() for n, _ in points})
    all_values = [v for points in series.values() for _, v in points]
    max_y = y_max if y_max is not None else max(all_values) * 1.15 or 1.0
    max_x = max(all_noise) or 1

    def sx(n: int) -> float:
        return left + n / max_x * plot_w

    def sy(value: float) -> float:
        return top + (1.0 - value / max_y) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-family="Arial" font-size="17" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for i in range(5):
        tick = max_y * i / 4
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>')
    for n in all_noise:
        x = sx(n)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#333"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{n}</text>')

    for distractor_type, points in sorted(series.items()):
        points.sort()
        color = colors.get(distractor_type, "#2ca02c")
        point_str = " ".join(f"{sx(n):.2f},{sy(v):.2f}" for n, v in points)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{point_str}"/>')
        for n, v in points:
            lines.append(f'<circle cx="{sx(n):.2f}" cy="{sy(v):.2f}" r="4" fill="{color}"/>')

    legend_x = left + plot_w + 24
    for i, distractor_type in enumerate(sorted(series)):
        y = top + 18 + i * 24
        color = colors.get(distractor_type, "#2ca02c")
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 34}" y="{y + 4}" font-family="Arial" font-size="13">{distractor_type}</text>')

    lines.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="13">Distractor count (noise_count)</text>')
    lines.append(f'<text x="18" y="{top + plot_h / 2:.2f}" transform="rotate(-90 18,{top + plot_h / 2:.2f})" text-anchor="middle" font-family="Arial" font-size="13">{y_label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def collect_case_studies(rows: list[dict], plan_by_id: dict[str, dict], queries: dict, docs: dict[str, dict], limit: int) -> list[dict]:
    failures = [
        row
        for row in rows
        if row["parse_status"] == "ok" and float(row["exact_set_match"]) == 0.0
    ]
    failures.sort(key=lambda row: (-float(row["extra_skill_count"] or 0) - float(row["missing_gold_count"] or 0), row["condition_id"]))
    cases = []
    for row in failures[:limit]:
        plan = plan_by_id[row["condition_id"]]
        selected = set(row["selected_skill_ids"].split(";")) if row["selected_skill_ids"] else set()
        gold = set(plan["gold"])
        if selected and not (selected - gold) and (gold - selected):
            error_type = "under_selection"
        elif selected - gold and not (gold - selected):
            error_type = "over_selection"
        elif not selected:
            error_type = "refusal"
        else:
            error_type = "mixed"
        cases.append(
            {
                "condition_id": row["condition_id"],
                "query": " ".join(queries[plan["task"]]),
                "distractor_type": plan["distractor_type"],
                "noise_count": plan["noise_count"],
                "gold_skills": [{"skill_id": sid, "name": docs[sid]["name"]} for sid in plan["gold"]],
                "selected_skills": [
                    {"skill_id": sid, "name": docs[sid]["name"]} for sid in sorted(selected)
                ],
                "distractor_skills": [
                    {"skill_id": sid, "name": docs[sid]["name"]} for sid in plan["distractors"]
                ],
                "error_type": error_type,
                "missing_gold_count": row["missing_gold_count"],
                "extra_skill_count": row["extra_skill_count"],
            }
        )
    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-usage-root", default="data/raw/Skill-Usage")
    parser.add_argument("--output-dir", default="data/experiments/rq5_llm_router")
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DASHSCOPE_BASE_URL", DASHSCOPE_COMPATIBLE_URL),
        help="Chat Completions endpoint. Defaults to $DASHSCOPE_BASE_URL if set; "
        "prefer the env var for workspace-dedicated URLs to keep the workspace ID "
        "out of shell history.",
    )
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key-prompt", action="store_true", help="Read API key with hidden terminal input.")
    parser.add_argument("--noise-counts", nargs="+", type=int, default=DEFAULT_NOISE_COUNTS)
    parser.add_argument("--distractor-types", nargs="+", default=DEFAULT_DISTRACTOR_TYPES, choices=["random", "hard"])
    parser.add_argument("--seed", type=int, default=6002)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--limit-tasks", type=int, default=0, help="0 means all tasks")
    parser.add_argument("--max-api-calls", type=int, default=1200, help="Safety cap on new API calls this run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-between-calls", type=float, default=0.25)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--case-study-limit", type=int, default=20)
    parser.add_argument("--neural-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--neural-batch-size", type=int, default=128)
    args = parser.parse_args()

    root = Path(args.skill_usage_root)
    if not (root / "data" / "task_queries.json").exists():
        raise SystemExit(
            f"Skill-Usage raw data not found under {root}. Restore data/raw/Skill-Usage before running RQ5 "
            "(this is the documented Day-1 prerequisite; do not reconstruct gold labels from RQ1-RQ3 outputs)."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = load_json(root / "data" / "task_queries.json")
    gt = normalize_gt(load_json(root / "data" / "task_skill_mapping.json"))
    docs = load_skill_docs(root / "skills-34k" / "skills_meta.jsonl")
    all_skill_ids = sorted(docs)

    tasks = sorted(set(queries) & set(gt))
    tasks = [task for task in tasks if all(sid in docs for sid in gt[task])]
    if args.limit_tasks:
        tasks = tasks[: args.limit_tasks]
    if not tasks:
        raise SystemExit("No tasks with fully resolvable gold skills were found.")

    max_noise = max(args.noise_counts)
    random_sequences = {
        task: build_random_sequence(task, gt[task], all_skill_ids, args.seed, max_noise)
        for task in tasks
    }
    hard_rankings: dict[str, list[str]] = {}
    sim_by_task_skill: dict[str, float] = {}
    if "hard" in args.distractor_types and max_noise > 0:
        hard_rankings, sim_by_task_skill = build_hard_rankings(
            tasks,
            queries,
            gt,
            docs,
            all_skill_ids,
            max_noise,
            args.neural_model_name,
            args.neural_batch_size,
            output_dir / "neural_doc_embeddings.npy",
        )

    plan = build_experiment_plan(
        tasks,
        gt,
        args.noise_counts,
        args.distractor_types,
        random_sequences,
        hard_rankings,
        args.seed,
    )
    check_nesting(plan)
    plan_by_id = {row["condition_id"]: row for row in plan}

    gold_count_distribution: dict[str, int] = defaultdict(int)
    for task in tasks:
        gold_count_distribution[str(len(gt[task]))] += 1

    metadata = {
        "model": args.model,
        "base_url": redact_base_url(args.base_url),
        "temperature": args.temperature,
        "max_completion_tokens": args.max_completion_tokens,
        "enable_thinking": False,
        "seed": args.seed,
        "noise_counts": sorted(set(args.noise_counts)),
        "noise_grid_status": "provisional_until_pilot_decision",
        "distractor_types": args.distractor_types,
        "hard_distractor_definition": "RQ3 hybrid BM25 + MiniLM (reciprocal-rank fusion) top non-gold skills",
        "task_count": len(tasks),
        "gold_count_distribution": dict(sorted(gold_count_distribution.items())),
        "planned_conditions": len(plan),
        "skill_library_size": len(all_skill_ids),
        "boundary": (
            "RQ5 measures multi-label routing selection quality with guaranteed gold visibility. "
            "Selected skills are not executed; results are not downstream task success."
        ),
    }
    decision_path = output_dir / "noise_grid_decision.json"
    if decision_path.exists():
        decision = load_json(decision_path)
        frozen_grid = sorted(set(int(n) for n in decision["final_grid"]))
        if sorted(set(args.noise_counts)) != frozen_grid:
            raise SystemExit(
                f"--noise-counts {sorted(set(args.noise_counts))} does not match the frozen grid "
                f"{frozen_grid} recorded in {decision_path}. Use the frozen grid or delete the "
                "decision file if it was written in error (never after full-run calls begin)."
            )
        metadata["noise_grid_status"] = "frozen_by_pilot_decision"
        metadata["noise_grid_decision"] = decision
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    plan_csv_rows = [
        {key: (";".join(str(v) for v in row[key]) if isinstance(row[key], list) else row[key]) for key in [
            "condition_id", "task", "distractor_type", "noise_count", "gold_count",
            "menu_size", "gold_fraction", "gold_positions",
        ]}
        for row in plan
    ]
    write_csv(output_dir / "experiment_plan.csv", plan_csv_rows)

    menus_path = output_dir / "candidate_menus.jsonl"
    menus_path.write_text("")
    for row in plan:
        append_jsonl(menus_path, row)

    print(f"Tasks: {len(tasks)}  planned conditions: {len(plan)}")
    print(f"Wrote {output_dir / 'experiment_metadata.json'}")
    print(f"Wrote {output_dir / 'experiment_plan.csv'}")
    print(f"Wrote {menus_path}")

    if args.dry_run:
        for row in plan[:5]:
            messages = build_router_messages(" ".join(queries[row["task"]]), row["menu"], docs)
            print("\n" + "=" * 70)
            print(f"DRY RUN condition {row['condition_id']} (menu size {row['menu_size']})")
            print(messages[1]["content"][:2000])
        print("\nDry run complete: no API calls were made.")
        return

    api_key = os.environ.get(args.api_key_env, "")
    if args.api_key_prompt and not api_key:
        api_key = getpass.getpass("DashScope API key: ").strip()
    if not api_key:
        raise SystemExit(f"No API key found. Set {args.api_key_env} or pass --api-key-prompt.")

    responses_path = output_dir / "raw_responses.jsonl"
    completed: dict[str, dict] = {}
    if args.resume:
        for record in load_jsonl(responses_path):
            if record.get("api_status") == "ok":
                completed[record["condition_id"]] = record
    elif responses_path.exists() and responses_path.stat().st_size > 0:
        raise SystemExit(
            f"{responses_path} already contains records. Pass --resume to continue, or move the file away."
        )

    calls_made = 0
    stopped_early = False
    total_conditions = len(plan)
    pending_at_start = total_conditions - len(completed)
    run_started = time.monotonic()
    if pending_at_start:
        print(f"Progress: {len(completed)}/{total_conditions} done, {pending_at_start} pending this run.")
    for row in plan:
        if row["condition_id"] in completed:
            continue
        if calls_made >= args.max_api_calls:
            print(f"Reached --max-api-calls={args.max_api_calls}; stopping. Re-run with --resume to continue.")
            stopped_early = True
            break
        messages = build_router_messages(" ".join(queries[row["task"]]), row["menu"], docs)
        try:
            result = call_router(
                messages,
                args.model,
                api_key,
                args.base_url,
                args.temperature,
                args.max_completion_tokens,
                args.timeout,
                args.max_retries,
            )
        except RuntimeError as exc:
            if is_budget_or_access_error(exc):
                print(f"Stopping cleanly after quota/permission error at {row['condition_id']}: {exc}")
                stopped_early = True
                break
            raise
        calls_made += 1
        parsed = parse_selection(result["raw_text"], row["menu_size"])
        distractor_sims = [
            sim_by_task_skill.get(f"{row['task']}|{sid}")
            for sid in row["distractors"]
        ]
        distractor_sims = [s for s in distractor_sims if s is not None]
        record = {
            "condition_id": row["condition_id"],
            "task": row["task"],
            "distractor_type": row["distractor_type"],
            "noise_count": row["noise_count"],
            "model": args.model,
            **result,
            **parsed,
            "mean_distractor_gold_similarity": round(mean(distractor_sims), 6) if distractor_sims else "",
            "max_distractor_gold_similarity": round(max(distractor_sims), 6) if distractor_sims else "",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        append_jsonl(responses_path, record)
        completed[row["condition_id"]] = record
        done = len(completed)
        pct = done / total_conditions * 100
        rate = (time.monotonic() - run_started) / calls_made
        eta_seconds = rate * (total_conditions - done)
        eta = f"{int(eta_seconds // 3600)}h{int(eta_seconds % 3600 // 60):02d}m" if eta_seconds >= 3600 else f"{int(eta_seconds // 60)}m{int(eta_seconds % 60):02d}s"
        bar_width = 24
        filled = int(bar_width * done / total_conditions)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"[{bar}] {done}/{total_conditions} {pct:5.1f}% ETA {eta} | "
            f"{row['condition_id']}: parse={parsed['parse_status']} "
            f"selected={len(parsed['selected_indices'])}/{row['menu_size']} "
            f"tokens={result['total_tokens']} calls={calls_made}/{args.max_api_calls}"
        )
        time.sleep(args.sleep_between_calls)

    result_rows = build_result_rows(plan_by_id, list(completed.values()))
    if not result_rows:
        print("No completed conditions yet; nothing to summarize.")
        return
    write_csv(output_dir / "per_condition_results.csv", result_rows)

    expanded = expand_shared_baseline(result_rows, args.distractor_types)
    summary_rows = summarize_conditions(expanded)
    single_gold_rows = summarize_single_gold(expanded)
    contrasts = compute_contrasts(expanded, args.noise_counts, args.bootstrap_resamples, args.seed)

    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "metadata": {**metadata, "completed_conditions": len(result_rows), "run_stopped_early": stopped_early},
                "summary": summary_rows,
                "single_gold_sensitivity": single_gold_rows,
                "paired_bootstrap_contrasts": contrasts,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    write_line_svg(
        summary_rows,
        "macro_f1",
        "RQ5: macro selection F1 vs distractor count",
        "Macro selection F1",
        output_dir / "selection_f1_vs_noise.svg",
    )
    write_line_svg(
        summary_rows,
        "exact_set_match_rate",
        "RQ5: exact set match vs distractor count",
        "Exact set match rate",
        output_dir / "exact_match_vs_noise.svg",
    )
    write_line_svg(
        summary_rows,
        "mean_prompt_tokens",
        "RQ5: prompt tokens vs distractor count",
        "Mean prompt tokens",
        output_dir / "tokens_vs_noise.svg",
        y_max=None,
    )
    case_studies = collect_case_studies(result_rows, plan_by_id, queries, docs, args.case_study_limit)
    (output_dir / "case_studies.json").write_text(json.dumps(case_studies, indent=2, ensure_ascii=False) + "\n")

    print(f"\nWrote {output_dir / 'per_condition_results.csv'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'selection_f1_vs_noise.svg'}")
    print(f"Wrote {output_dir / 'exact_match_vs_noise.svg'}")
    print(f"Wrote {output_dir / 'tokens_vs_noise.svg'}")
    print(f"Wrote {output_dir / 'case_studies.json'}")
    print()
    print(f"{'type':>8} {'n':>4} {'macroF1':>9} {'exact':>7} {'recall':>7} {'extra':>7} {'tokens':>8}")
    for row in summary_rows:
        if "macro_f1" not in row:
            continue
        print(
            f"{row['distractor_type']:>8} {row['noise_count']:>4} "
            f"{row['macro_f1']:>9.3f} {row['exact_set_match_rate']:>7.3f} "
            f"{row['macro_gold_recall']:>7.3f} {row['mean_extra_skill_count']:>7.2f} "
            f"{row['mean_prompt_tokens']:>8.0f}"
        )


if __name__ == "__main__":
    main()
