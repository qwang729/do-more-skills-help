#!/usr/bin/env python3
"""Read-only pipeline diagnostics over RQ5 outputs (safe to run mid-flight).

Three analyses, all computed directly from the append-only
`raw_responses.jsonl` plus the frozen `candidate_menus.jsonl` (never from
`per_condition_results.csv`, which is only rewritten when the main script
finishes a run):

1. Pipeline error decomposition (RQ3 x RQ5): combines the RQ3 hybrid
   retriever's complete-gold-coverage rate on the full 34k library with the
   RQ5 router's exact-set-match rate given guaranteed gold visibility, and
   reports the product as an estimated end-to-end correct-routing rate.
2. Failure-mode composition: correct / under_selection / over_selection /
   mixed / refusal / invalid shares per (distractor_type, noise_count).
3. Position bias: gold-skill selection rate by relative menu position.

This script only READS from data/experiments/rq5_llm_router/ and
data/experiments/rq3_retriever_enhanced/, and only WRITES to
data/experiments/rq5_pipeline_diagnostics/. It makes no API calls and never
touches files the RQ5 main script writes, so it cannot corrupt an in-progress
Step 3 run. Partial raw_responses.jsonl files are handled by skipping a
truncated final line; results are labeled with the completed-condition count.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

SHARED_BASELINE_TYPE = "shared"


# ---------------------------------------------------------------------------
# IO helpers (tolerant of a concurrently appended JSONL file)
# ---------------------------------------------------------------------------

def load_jsonl_tolerant(path: Path) -> list[dict]:
    """Load JSONL, skipping a truncated trailing line from a concurrent writer."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                print(f"Note: skipped truncated final line in {path} (writer mid-append).")
                continue
            raise
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Scoring (mirrors the RQ5 main script definitions)
# ---------------------------------------------------------------------------

def classify_error(selected: set[str], gold: set[str]) -> str:
    if selected == gold:
        return "correct"
    if not selected:
        return "refusal"
    extra = selected - gold
    missing = gold - selected
    if extra and missing:
        return "mixed"
    if extra:
        return "over_selection"
    return "under_selection"


def selection_f1(selected: set[str], gold: set[str]) -> float:
    if not selected:
        return 0.0
    tp = len(selected & gold)
    precision = tp / len(selected)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_records(menus: list[dict], responses: list[dict]) -> list[dict]:
    """Join raw responses with menus; deduplicate by condition_id (last ok wins)."""
    plan_by_id = {row["condition_id"]: row for row in menus}
    latest: dict[str, dict] = {}
    for record in responses:
        if record.get("api_status") == "ok":
            latest[record["condition_id"]] = record
    records = []
    for condition_id, record in latest.items():
        plan = plan_by_id.get(condition_id)
        if plan is None:
            continue
        menu = plan["menu"]
        gold = set(plan["gold"])
        parse_ok = record.get("parse_status") == "ok"
        selected = (
            {menu[i - 1] for i in record.get("selected_indices", [])} if parse_ok else set()
        )
        records.append(
            {
                "condition_id": condition_id,
                "task": plan["task"],
                "distractor_type": plan["distractor_type"],
                "noise_count": plan["noise_count"],
                "gold_count": plan["gold_count"],
                "menu": menu,
                "gold": gold,
                "parse_ok": parse_ok,
                "selected": selected,
                "error_type": classify_error(selected, gold) if parse_ok else "invalid",
                "f1": selection_f1(selected, gold) if parse_ok else None,
                "exact": (selected == gold) if parse_ok else None,
            }
        )
    return records


def expand_shared(records: list[dict], distractor_types: list[str]) -> list[dict]:
    expanded = []
    for record in records:
        if record["distractor_type"] == SHARED_BASELINE_TYPE:
            for distractor_type in distractor_types:
                expanded.append({**record, "distractor_type": distractor_type})
        else:
            expanded.append(record)
    return expanded


# ---------------------------------------------------------------------------
# Analysis 1: pipeline error decomposition (RQ3 retrieval x RQ5 routing)
# ---------------------------------------------------------------------------

def load_rq3_coverage(path: Path, retriever: str) -> dict[str, dict]:
    """Per-task retrieval stats for the given retriever on the full library."""
    per_task: dict[str, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["retriever"] != retriever or row["pool_size"] != "full":
                continue
            task = row["task"]
            if task in per_task:
                continue  # distractor_type is irrelevant at full pool; dedupe
            per_task[task] = {
                "recall_at_10": float(row["recall@10"]),
                "hit_at_10": float(row["hit@10"]),
                "complete_coverage_at_10": 1.0 if float(row["recall@10"]) >= 1.0 else 0.0,
            }
    return per_task


def pipeline_decomposition(
    records: list[dict],
    coverage: dict[str, dict],
    retriever: str,
) -> dict:
    shared_tasks = sorted({r["task"] for r in records} & set(coverage))
    stage1 = {
        "retriever": retriever,
        "library": "full (34k)",
        "n_tasks": len(shared_tasks),
        "complete_gold_coverage_at_10": round(
            mean(coverage[t]["complete_coverage_at_10"] for t in shared_tasks), 4
        ),
        "hit_at_10": round(mean(coverage[t]["hit_at_10"] for t in shared_tasks), 4),
        "mean_recall_at_10": round(mean(coverage[t]["recall_at_10"] for t in shared_tasks), 4),
    }

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["distractor_type"], record["noise_count"])].append(record)

    stage2_rows = []
    for (distractor_type, noise_count), group in sorted(grouped.items()):
        valid = [r for r in group if r["parse_ok"]]
        if not valid:
            continue
        exact_rate = mean(1.0 if r["exact"] else 0.0 for r in valid)
        macro_f1 = mean(r["f1"] for r in valid)
        task_products = [
            coverage[r["task"]]["complete_coverage_at_10"] * (1.0 if r["exact"] else 0.0)
            for r in valid
            if r["task"] in coverage
        ]
        stage2_rows.append(
            {
                "distractor_type": distractor_type,
                "noise_count": noise_count,
                "n_valid": len(valid),
                "routing_exact_set_match": round(exact_rate, 4),
                "routing_macro_f1": round(macro_f1, 4),
                "est_end_to_end_exact_routing": round(
                    stage1["complete_gold_coverage_at_10"] * exact_rate, 4
                ),
                "task_paired_end_to_end_exact": (
                    round(mean(task_products), 4) if task_products else ""
                ),
            }
        )
    return {
        "note": (
            "Stage 1 = P(all gold skills retrieved into top-10, full 34k library, RQ3). "
            "Stage 2 = P(exact correct skill set selected | all gold visible, RQ5). "
            "est_end_to_end multiplies the two macro rates (independence approximation); "
            "task_paired_end_to_end multiplies per task, then averages."
        ),
        "stage1_retrieval": stage1,
        "stage2_routing_and_composition": stage2_rows,
    }


# ---------------------------------------------------------------------------
# Analysis 2: failure-mode composition
# ---------------------------------------------------------------------------

ERROR_TYPES = ["correct", "under_selection", "over_selection", "mixed", "refusal", "invalid"]


def failure_modes(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["distractor_type"], record["noise_count"])].append(record)
    rows = []
    for (distractor_type, noise_count), group in sorted(grouped.items()):
        row = {
            "distractor_type": distractor_type,
            "noise_count": noise_count,
            "n_conditions": len(group),
        }
        for error_type in ERROR_TYPES:
            count = sum(1 for r in group if r["error_type"] == error_type)
            row[f"{error_type}_count"] = count
            row[f"{error_type}_rate"] = round(count / len(group), 4)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Analysis 3: gold-position bias
# ---------------------------------------------------------------------------

def position_bias(records: list[dict], min_menu_size: int, n_bins: int) -> list[dict]:
    """Gold selection rate by relative menu position (larger menus only)."""
    bins: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in records:
        if not record["parse_ok"] or len(record["menu"]) < min_menu_size:
            continue
        menu_size = len(record["menu"])
        for index, skill_id in enumerate(record["menu"]):
            if skill_id not in record["gold"]:
                continue
            relative = index / (menu_size - 1) if menu_size > 1 else 0.0
            bin_index = min(int(relative * n_bins), n_bins - 1)
            bins[(record["distractor_type"], bin_index)].append(
                1.0 if skill_id in record["selected"] else 0.0
            )
    rows = []
    for (distractor_type, bin_index), values in sorted(bins.items()):
        rows.append(
            {
                "distractor_type": distractor_type,
                "position_bin": bin_index,
                "bin_range": f"[{bin_index / n_bins:.2f}, {(bin_index + 1) / n_bins:.2f})",
                "n_gold_instances": len(values),
                "gold_selection_rate": round(mean(values), 4),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq5-dir", default="data/experiments/rq5_llm_router")
    parser.add_argument("--rq3-metrics", default="data/experiments/rq3_retriever_enhanced/per_query_metrics.csv")
    parser.add_argument("--output-dir", default="data/experiments/rq5_pipeline_diagnostics")
    parser.add_argument("--retriever", default="hybrid_bm25_neural")
    parser.add_argument("--position-min-menu-size", type=int, default=10)
    parser.add_argument("--position-bins", type=int, default=5)
    args = parser.parse_args()

    rq5_dir = Path(args.rq5_dir)
    output_dir = Path(args.output_dir)
    if output_dir.resolve() == rq5_dir.resolve():
        raise SystemExit("Refusing to write into the RQ5 experiment directory.")
    output_dir.mkdir(parents=True, exist_ok=True)

    menus = load_jsonl_tolerant(rq5_dir / "candidate_menus.jsonl")
    responses = load_jsonl_tolerant(rq5_dir / "raw_responses.jsonl")
    if not menus or not responses:
        raise SystemExit("Missing candidate_menus.jsonl or raw_responses.jsonl under the RQ5 directory.")

    records = build_records(menus, responses)
    distractor_types = sorted(
        {r["distractor_type"] for r in records if r["distractor_type"] != SHARED_BASELINE_TYPE}
    )
    expanded = expand_shared(records, distractor_types)
    completed = len(records)
    planned = len(menus)
    status = {
        "completed_conditions": completed,
        "planned_conditions": planned,
        "is_partial": completed < planned,
    }
    print(f"Completed conditions: {completed}/{planned}" + (" (PARTIAL - rerun after Step 3 finishes)" if completed < planned else ""))

    coverage = load_rq3_coverage(Path(args.rq3_metrics), args.retriever)
    decomposition = {"run_status": status, **pipeline_decomposition(expanded, coverage, args.retriever)}
    (output_dir / "pipeline_decomposition.json").write_text(
        json.dumps(decomposition, indent=2, ensure_ascii=False) + "\n"
    )
    write_csv(output_dir / "pipeline_decomposition.csv", decomposition["stage2_routing_and_composition"])

    failure_rows = failure_modes(expanded)
    write_csv(output_dir / "failure_modes.csv", failure_rows)

    position_rows = position_bias(
        expanded, args.position_min_menu_size, args.position_bins
    )
    write_csv(output_dir / "position_bias.csv", position_rows)

    print(f"Wrote {output_dir / 'pipeline_decomposition.json'}")
    print(f"Wrote {output_dir / 'pipeline_decomposition.csv'}")
    print(f"Wrote {output_dir / 'failure_modes.csv'}")
    print(f"Wrote {output_dir / 'position_bias.csv'}")

    stage1 = decomposition["stage1_retrieval"]
    print(
        f"\nStage 1 ({args.retriever}, full library): "
        f"complete gold coverage@10 = {stage1['complete_gold_coverage_at_10']}"
    )
    print(f"{'type':>8} {'n':>4} {'exact':>7} {'F1':>7} {'end2end':>8}")
    for row in decomposition["stage2_routing_and_composition"]:
        print(
            f"{row['distractor_type']:>8} {row['noise_count']:>4} "
            f"{row['routing_exact_set_match']:>7.3f} {row['routing_macro_f1']:>7.3f} "
            f"{row['est_end_to_end_exact_routing']:>8.3f}"
        )

    print(f"\n{'type':>8} {'n':>4} {'correct':>8} {'under':>7} {'over':>7} {'mixed':>7} {'refusal':>8}")
    for row in failure_rows:
        print(
            f"{row['distractor_type']:>8} {row['noise_count']:>4} "
            f"{row['correct_rate']:>8.3f} {row['under_selection_rate']:>7.3f} "
            f"{row['over_selection_rate']:>7.3f} {row['mixed_rate']:>7.3f} "
            f"{row['refusal_rate']:>8.3f}"
        )

    print(f"\nGold selection rate by relative menu position (menus >= {args.position_min_menu_size}):")
    print(f"{'type':>8} {'bin':>12} {'n_gold':>7} {'rate':>7}")
    for row in position_rows:
        print(
            f"{row['distractor_type']:>8} {row['bin_range']:>12} "
            f"{row['n_gold_instances']:>7} {row['gold_selection_rate']:>7.3f}"
        )


if __name__ == "__main__":
    main()
