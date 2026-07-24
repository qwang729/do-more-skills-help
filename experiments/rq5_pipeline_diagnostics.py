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

It also renders report figures (pure SVG, no plotting dependencies):
failure-mode stacked bars, a three-panel task-paired pipeline-decomposition
figure, precision/recall decomposition lines, cost-accuracy tradeoff,
single-gold subgroup accuracy, and position-bias bars.

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
        tp = len(selected & gold)
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
                "precision": ((tp / len(selected)) if selected else 0.0) if parse_ok else None,
                "recall": (tp / len(gold) if gold else 0.0) if parse_ok else None,
                "prompt_tokens": int(record.get("prompt_tokens", 0) or 0),
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
# Aggregates for report figures
# ---------------------------------------------------------------------------

def aggregate_condition_metrics(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["distractor_type"], record["noise_count"])].append(record)
    rows = []
    for (distractor_type, noise_count), group in sorted(grouped.items()):
        valid = [r for r in group if r["parse_ok"]]
        if not valid:
            continue
        rows.append(
            {
                "distractor_type": distractor_type,
                "noise_count": noise_count,
                "macro_precision": mean(r["precision"] for r in valid),
                "macro_recall": mean(r["recall"] for r in valid),
                "macro_f1": mean(r["f1"] for r in valid),
                "exact_rate": mean(1.0 if r["exact"] else 0.0 for r in valid),
                "mean_prompt_tokens": mean(r["prompt_tokens"] for r in group),
            }
        )
    return rows


def single_gold_accuracy(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        if record["gold_count"] == 1 and record["parse_ok"]:
            grouped[(record["distractor_type"], record["noise_count"])].append(record)
    rows = []
    for (distractor_type, noise_count), group in sorted(grouped.items()):
        rows.append(
            {
                "distractor_type": distractor_type,
                "noise_count": noise_count,
                "n_single_gold": len(group),
                "correct_rate": mean(1.0 if r["exact"] else 0.0 for r in group),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Report figures (pure SVG, no plotting dependencies)
# ---------------------------------------------------------------------------

TYPE_COLORS = {"random": "#1f77b4", "hard": "#d62728"}
STACK_COLORS = {
    "correct": "#2ca02c",
    "under_selection": "#1f77b4",
    "over_selection": "#ff7f0e",
    "mixed": "#d62728",
    "refusal": "#7f7f7f",
    "invalid": "#17becf",
}
FONT = 'font-family="Arial"'


def _svg_frame(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="26" {FONT} font-size="17" font-weight="700">{title}</text>',
    ]


def _axes(lines: list[str], left: int, top: int, plot_w: float, plot_h: float, y_max: float) -> None:
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>')
    for i in range(5):
        tick = y_max * i / 4
        y = top + (1.0 - tick / y_max) * plot_h
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" {FONT} font-size="12">{tick:.2f}</text>')


def write_failure_modes_svg(failure_rows: list[dict], path: Path) -> None:
    """Stacked bars: outcome composition per (distractor_type, noise_count)."""
    width, height = 960, 500
    left, top, right, bottom = 70, 46, 210, 96
    plot_w, plot_h = width - left - right, height - top - bottom
    stack_order = [e for e in ERROR_TYPES]
    types = sorted({row["distractor_type"] for row in failure_rows})
    by_type = {
        t: sorted((r for r in failure_rows if r["distractor_type"] == t), key=lambda r: r["noise_count"])
        for t in types
    }
    n_bars = sum(len(v) for v in by_type.values())
    slots = n_bars + max(len(types) - 1, 0)
    slot_w = plot_w / slots
    bar_w = slot_w * 0.68

    lines = _svg_frame(width, height, "RQ5 failure-mode composition by condition")
    _axes(lines, left, top, plot_w, plot_h, 1.0)

    slot = 0
    for t_index, t in enumerate(types):
        if t_index:
            slot += 1
        block_start = left + slot * slot_w
        for row in by_type[t]:
            x = left + slot * slot_w + (slot_w - bar_w) / 2
            y_cursor = top + plot_h
            for error_type in stack_order:
                rate = row.get(f"{error_type}_rate", 0.0) or 0.0
                if not rate:
                    continue
                h = rate * plot_h
                y_cursor -= h
                lines.append(
                    f'<rect x="{x:.2f}" y="{y_cursor:.2f}" width="{bar_w:.2f}" height="{h:.2f}" '
                    f'fill="{STACK_COLORS[error_type]}"/>'
                )
            lines.append(
                f'<text x="{x + bar_w / 2:.2f}" y="{top + plot_h + 18}" text-anchor="middle" {FONT} font-size="12">{row["noise_count"]}</text>'
            )
            slot += 1
        block_end = left + slot * slot_w
        lines.append(
            f'<text x="{(block_start + block_end) / 2:.2f}" y="{top + plot_h + 42}" text-anchor="middle" {FONT} font-size="13" font-weight="700">{t}</text>'
        )
    lines.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 14}" text-anchor="middle" {FONT} font-size="13">Distractor count (noise_count)</text>'
    )
    legend_x = left + plot_w + 26
    for i, error_type in enumerate(stack_order):
        y = top + 14 + i * 24
        lines.append(f'<rect x="{legend_x}" y="{y - 10}" width="16" height="16" fill="{STACK_COLORS[error_type]}"/>')
        lines.append(f'<text x="{legend_x + 24}" y="{y + 3}" {FONT} font-size="13">{error_type}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def write_pipeline_svg(stage1: dict, stage2_rows: list[dict], path: Path) -> None:
    """Three-panel task-paired attribution of retrieve-then-route outcomes.

    Every task is assigned to exactly one terminal outcome in Panel C:
    retrieval failure, routing failure after successful retrieval, or exact
    end-to-end routing success. This makes the stage contribution visible
    instead of presenting only a composed endpoint rate.
    """
    width, height = 1560, 590
    panel_top, panel_h = 78, 430
    panel_specs = [
        (30, 380, "A. Stage 1 — RQ3 retrieval gate"),
        (425, 535, "B. Stage 2 — conditional routing survival"),
        (975, 555, "C. Task-paired failure attribution"),
    ]
    lines = _svg_frame(width, height, "Task-paired retrieve-then-route failure decomposition")
    lines.append(
        f'<text x="20" y="50" {FONT} font-size="13" fill="#555">'
        "Each task is attributed to its first failed stage; success means both stages pass."
        "</text>"
    )

    def panel_frame(x: float, w: float, title: str) -> tuple[float, float, float, float]:
        lines.append(
            f'<rect x="{x}" y="{panel_top}" width="{w}" height="{panel_h}" rx="8" '
            'fill="#fafafa" stroke="#d8d8d8"/>'
        )
        lines.append(
            f'<text x="{x + 14}" y="{panel_top + 25}" {FONT} font-size="15" '
            f'font-weight="700">{title}</text>'
        )
        plot_left = x + 54
        plot_top = panel_top + 76
        plot_w = w - 74
        plot_h = 270
        return plot_left, plot_top, plot_w, plot_h

    def add_axes(
        left: float,
        top: float,
        plot_w: float,
        plot_h: float,
        y_max: float,
        ticks: int = 5,
    ) -> None:
        lines.append(
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>'
        )
        lines.append(
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
            f'y2="{top + plot_h}" stroke="#333"/>'
        )
        for i in range(ticks):
            value = y_max * i / (ticks - 1)
            y = top + (1.0 - value / y_max) * plot_h
            lines.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
                'stroke="#e5e5e5"/>'
            )
            lines.append(
                f'<text x="{left - 7}" y="{y + 4:.2f}" text-anchor="end" {FONT} '
                f'font-size="11">{value:.2f}</text>'
            )

    # Panel A: the three retrieval metrics make clear why complete coverage,
    # rather than hit@10 or mean recall, is the gate for exact routing.
    a_left, a_top, a_w, a_h = panel_frame(*panel_specs[0])
    add_axes(a_left, a_top, a_w, a_h, 1.0)
    lines.append(
        f'<text x="{panel_specs[0][0] + 15}" y="{a_top + a_h / 2:.2f}" '
        f'transform="rotate(-90 {panel_specs[0][0] + 15},{a_top + a_h / 2:.2f})" '
        f'text-anchor="middle" {FONT} font-size="11">Rate</text>'
    )
    stage1_metrics = [
        ("Hit@10", float(stage1["hit_at_10"]), "#72a0c1"),
        ("Mean recall@10", float(stage1["mean_recall_at_10"]), "#4c78a8"),
        ("Complete-gold", float(stage1["complete_gold_coverage_at_10"]), "#2f5f8f"),
    ]
    slot_w = a_w / len(stage1_metrics)
    bar_w = slot_w * 0.52
    for index, (label, value, color) in enumerate(stage1_metrics):
        x = a_left + index * slot_w + (slot_w - bar_w) / 2
        bar_h = value * a_h
        y = a_top + a_h - bar_h
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" '
            f'fill="{color}"/>'
        )
        lines.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" '
            f'{FONT} font-size="12" font-weight="700">{value:.3f}</text>'
        )
        lines.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{a_top + a_h + 19}" text-anchor="middle" '
            f'{FONT} font-size="11">{label}</text>'
        )
        if label == "Complete-gold":
            lines.append(
                f'<text x="{x + bar_w / 2:.2f}" y="{a_top + a_h + 34}" '
                f'text-anchor="middle" {FONT} font-size="11">coverage@10</text>'
            )
    lines.append(
        f'<text x="{panel_specs[0][0] + panel_specs[0][1] / 2:.2f}" '
        f'y="{panel_top + panel_h - 17}" text-anchor="middle" {FONT} font-size="12" '
        'fill="#555">Full 34k library; hybrid retriever; fixed Top-10</text>'
    )

    types = sorted({row["distractor_type"] for row in stage2_rows})
    by_type = {
        t: sorted(
            (row for row in stage2_rows if row["distractor_type"] == t),
            key=lambda row: row["noise_count"],
        )
        for t in types
    }
    noise_ticks = sorted({int(row["noise_count"]) for row in stage2_rows})

    def x_positions(left: float, plot_w: float) -> dict[int, float]:
        if len(noise_ticks) == 1:
            return {noise_ticks[0]: left + plot_w / 2}
        return {
            noise: left + index * plot_w / (len(noise_ticks) - 1)
            for index, noise in enumerate(noise_ticks)
        }

    def add_noise_axis(left: float, top: float, plot_w: float, plot_h: float) -> dict[int, float]:
        positions = x_positions(left, plot_w)
        for noise, x in positions.items():
            lines.append(
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" '
                f'y2="{top + plot_h + 5}" stroke="#333"/>'
            )
            lines.append(
                f'<text x="{x:.2f}" y="{top + plot_h + 20}" text-anchor="middle" '
                f'{FONT} font-size="11">{noise}</text>'
            )
        return positions

    stage1_coverage = float(stage1["complete_gold_coverage_at_10"])

    # Panel B: among tasks whose complete gold set was retrieved, show the
    # fraction that also achieved exact routing in the matching RQ5 condition.
    b_left, b_top, b_w, b_h = panel_frame(*panel_specs[1])
    add_axes(b_left, b_top, b_w, b_h, 1.0)
    lines.append(
        f'<text x="{panel_specs[1][0] + 15}" y="{b_top + b_h / 2:.2f}" '
        f'transform="rotate(-90 {panel_specs[1][0] + 15},{b_top + b_h / 2:.2f})" '
        f'text-anchor="middle" {FONT} font-size="11">P(routing exact | retrieval complete)</text>'
    )
    b_x = add_noise_axis(b_left, b_top, b_w, b_h)
    for distractor_type in types:
        color = TYPE_COLORS.get(distractor_type, "#2ca02c")
        rows = by_type[distractor_type]
        points = [
            (
                b_x[int(row["noise_count"])],
                b_top
                + (
                    1.0
                    - float(row["task_paired_end_to_end_exact"]) / stage1_coverage
                )
                * b_h,
            )
            for row in rows
        ]
        lines.append(
            f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}" '
            f'fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for x, y in points:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
    legend_y = panel_top + 49
    for index, distractor_type in enumerate(types):
        x = b_left + index * 92
        color = TYPE_COLORS.get(distractor_type, "#2ca02c")
        lines.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 24}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{x + 30}" y="{legend_y + 4}" {FONT} font-size="12">'
            f"{distractor_type}</text>"
        )
    lines.append(
        f'<text x="{b_left + b_w / 2:.2f}" y="{b_top + b_h + 42}" '
        f'text-anchor="middle" {FONT} font-size="12">RQ5 constructed distractor count</text>'
    )
    lines.append(
        f'<text x="{panel_specs[1][0] + panel_specs[1][1] / 2:.2f}" '
        f'y="{panel_top + panel_h - 17}" text-anchor="middle" {FONT} font-size="12" '
        f'fill="#555">Survival among the {round(stage1_coverage * int(stage1["n_tasks"]))} '
        "tasks with complete RQ3 retrieval</text>"
    )

    # Panel C: 100% stacked attribution. The shared n=0 baseline appears once;
    # all other conditions are grouped by distractor type.
    c_left, c_top, c_w, c_h = panel_frame(*panel_specs[2])
    add_axes(c_left, c_top, c_w, c_h, 1.0)
    lines.append(
        f'<text x="{panel_specs[2][0] + 15}" y="{c_top + c_h / 2:.2f}" '
        f'transform="rotate(-90 {panel_specs[2][0] + 15},{c_top + c_h / 2:.2f})" '
        f'text-anchor="middle" {FONT} font-size="11">Share of all tasks</text>'
    )

    baseline = by_type[types[0]][0]
    attribution_rows = [("shared", baseline)]
    for distractor_type in types:
        attribution_rows.extend(
            (distractor_type, row)
            for row in by_type[distractor_type]
            if int(row["noise_count"]) > 0
        )

    outcome_colors = {
        "retrieval failure": "#9e9e9e",
        "routing failure": "#f28e2b",
        "exact success": "#59a14f",
    }
    gap_after = {0, 4}
    total_slots = len(attribution_rows) + len(gap_after)
    slot_w = c_w / total_slots
    bar_w = slot_w * 0.68
    slot = 0
    group_bounds: dict[str, list[float]] = defaultdict(list)
    for row_index, (group, row) in enumerate(attribution_rows):
        x = c_left + slot * slot_w + (slot_w - bar_w) / 2
        success = float(row["task_paired_end_to_end_exact"])
        retrieval_failure = 1.0 - stage1_coverage
        routing_failure = stage1_coverage - success
        components = [
            ("exact success", success),
            ("routing failure", routing_failure),
            ("retrieval failure", retrieval_failure),
        ]
        y_cursor = c_top + c_h
        for outcome, value in components:
            component_h = value * c_h
            y_cursor -= component_h
            lines.append(
                f'<rect x="{x:.2f}" y="{y_cursor:.2f}" width="{bar_w:.2f}" '
                f'height="{component_h:.2f}" fill="{outcome_colors[outcome]}"/>'
            )
        label = "0" if group == "shared" else str(row["noise_count"])
        lines.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{c_top + c_h + 18}" '
            f'text-anchor="middle" {FONT} font-size="10">{label}</text>'
        )
        group_bounds[group].append(x + bar_w / 2)
        slot += 1
        if row_index in gap_after:
            slot += 1

    group_y = c_top + c_h + 35
    for group in ["shared", "hard", "random"]:
        centers = group_bounds[group]
        lines.append(
            f'<text x="{sum(centers) / len(centers):.2f}" y="{group_y}" '
            f'text-anchor="middle" {FONT} font-size="10" font-weight="700">{group}</text>'
        )

    legend_y = panel_top + 49
    legend_x = c_left
    legend_specs = [
        ("retrieval failure", "retrieval failure"),
        ("routing failure", "routing failure after retrieval"),
        ("exact success", "exact success"),
    ]
    for index, (outcome, label) in enumerate(legend_specs):
        x = legend_x + index * 145
        lines.append(
            f'<rect x="{x}" y="{legend_y - 9}" width="12" height="12" '
            f'fill="{outcome_colors[outcome]}"/>'
        )
        lines.append(
            f'<text x="{x + 17}" y="{legend_y + 2}" {FONT} font-size="10">{label}</text>'
        )
    lines.append(
        f'<text x="{panel_specs[2][0] + panel_specs[2][1] / 2:.2f}" '
        f'y="{panel_top + panel_h - 17}" text-anchor="middle" {FONT} font-size="12" '
        'fill="#555">Task-paired first-failure attribution; each bar sums to 100%</text>'
    )

    lines.append(
        f'<text x="{width / 2:.2f}" y="{height - 31}" text-anchor="middle" {FONT} '
        'font-size="12" fill="#444">Attribution order: retrieval failure first; among '
        "retrieval-complete tasks, routing failure or exact success.</text>"
    )
    lines.append(
        f'<text x="{width / 2:.2f}" y="{height - 13}" text-anchor="middle" {FONT} '
        'font-size="11" fill="#666">RQ3 is fixed at Top-10; RQ5 n=0 is one shared baseline; '
        "routing remains a controlled diagnostic rather than executed skill use.</text>"
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def write_xy_lines_svg(
    series: list[dict],
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    y_max: float = 1.0,
    x_ticks: list[float] | None = None,
) -> None:
    """Generic multi-series line chart.

    Each series dict: {label, color, dash (bool), points [(x, y)],
    point_labels (optional list[str])}.
    """
    width, height = 900, 460
    left, top, right, bottom = 82, 46, 250, 76
    plot_w, plot_h = width - left - right, height - top - bottom

    all_x = [x for s in series for x, _ in s["points"]]
    if not all_x:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>\n")
        return
    max_x = max(all_x) * 1.05 or 1.0

    def sx(x: float) -> float:
        return left + x / max_x * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - y / y_max) * plot_h

    lines = _svg_frame(width, height, title)
    _axes(lines, left, top, plot_w, plot_h, y_max)

    ticks = x_ticks if x_ticks is not None else [max_x * i / 5 for i in range(6)]
    for tick in ticks:
        x = sx(tick)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#333"/>')
        label = f"{tick:.0f}"
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle" {FONT} font-size="12">{label}</text>')

    for s in series:
        points = sorted(s["points"])
        dash = ' stroke-dasharray="7,5"' if s.get("dash") else ""
        point_str = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        lines.append(f'<polyline fill="none" stroke="{s["color"]}" stroke-width="3"{dash} points="{point_str}"/>')
        for index, (x, y) in enumerate(points):
            lines.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="{s["color"]}"/>')
            labels = s.get("point_labels")
            if labels:
                lines.append(
                    f'<text x="{sx(x):.2f}" y="{sy(y) - 10:.2f}" text-anchor="middle" {FONT} font-size="11" fill="#444">{labels[index]}</text>'
                )

    legend_x = left + plot_w + 22
    for i, s in enumerate(series):
        y = top + 14 + i * 24
        dash = ' stroke-dasharray="7,5"' if s.get("dash") else ""
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 26}" y2="{y}" stroke="{s["color"]}" stroke-width="3"{dash}/>')
        lines.append(f'<text x="{legend_x + 34}" y="{y + 4}" {FONT} font-size="13">{s["label"]}</text>')

    lines.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 14}" text-anchor="middle" {FONT} font-size="13">{x_label}</text>')
    lines.append(
        f'<text x="18" y="{top + plot_h / 2:.2f}" transform="rotate(-90 18,{top + plot_h / 2:.2f})" text-anchor="middle" {FONT} font-size="13">{y_label}</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def write_grouped_bars_svg(
    group_labels: list[str],
    series: list[dict],
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    y_max: float = 1.0,
) -> None:
    """Generic grouped bar chart. Each series dict: {label, color, values}."""
    width, height = 900, 460
    left, top, right, bottom = 82, 46, 210, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    slot_w = plot_w / len(group_labels)
    bar_w = slot_w * 0.7 / max(len(series), 1)

    lines = _svg_frame(width, height, title)
    _axes(lines, left, top, plot_w, plot_h, y_max)

    for g_index, label in enumerate(group_labels):
        x0 = left + g_index * slot_w + slot_w * 0.15
        for s_index, s in enumerate(series):
            value = s["values"][g_index]
            if value is None:
                continue
            h = value / y_max * plot_h
            lines.append(
                f'<rect x="{x0 + s_index * bar_w:.2f}" y="{top + plot_h - h:.2f}" width="{bar_w:.2f}" '
                f'height="{h:.2f}" fill="{s["color"]}"/>'
            )
        lines.append(
            f'<text x="{left + g_index * slot_w + slot_w / 2:.2f}" y="{top + plot_h + 18}" text-anchor="middle" {FONT} font-size="12">{label}</text>'
        )

    legend_x = left + plot_w + 26
    for i, s in enumerate(series):
        y = top + 14 + i * 24
        lines.append(f'<rect x="{legend_x}" y="{y - 10}" width="16" height="16" fill="{s["color"]}"/>')
        lines.append(f'<text x="{legend_x + 24}" y="{y + 3}" {FONT} font-size="13">{s["label"]}</text>')

    lines.append(f'<text x="{left + plot_w / 2:.2f}" y="{height - 14}" text-anchor="middle" {FONT} font-size="13">{x_label}</text>')
    lines.append(
        f'<text x="18" y="{top + plot_h / 2:.2f}" transform="rotate(-90 18,{top + plot_h / 2:.2f})" text-anchor="middle" {FONT} font-size="13">{y_label}</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def write_report_figures(
    output_dir: Path,
    agg_rows: list[dict],
    failure_rows: list[dict],
    position_rows: list[dict],
    single_gold_rows: list[dict],
    decomposition: dict,
) -> list[str]:
    written = []

    write_failure_modes_svg(failure_rows, output_dir / "failure_modes_stacked.svg")
    written.append("failure_modes_stacked.svg")

    write_pipeline_svg(
        decomposition["stage1_retrieval"],
        decomposition["stage2_routing_and_composition"],
        output_dir / "pipeline_decomposition.svg",
    )
    written.append("pipeline_decomposition.svg")

    types = sorted({r["distractor_type"] for r in agg_rows})
    noise_ticks = sorted({r["noise_count"] for r in agg_rows})

    pr_series = []
    for t in types:
        rows_t = sorted((r for r in agg_rows if r["distractor_type"] == t), key=lambda r: r["noise_count"])
        color = TYPE_COLORS.get(t, "#2ca02c")
        pr_series.append({"label": f"{t} precision", "color": color, "dash": False, "points": [(r["noise_count"], r["macro_precision"]) for r in rows_t]})
        pr_series.append({"label": f"{t} recall", "color": color, "dash": True, "points": [(r["noise_count"], r["macro_recall"]) for r in rows_t]})
    write_xy_lines_svg(
        pr_series,
        output_dir / "precision_recall_vs_noise.svg",
        "Precision vs recall decomposition under distractor noise",
        "Distractor count (noise_count)",
        "Macro rate",
        x_ticks=noise_ticks,
    )
    written.append("precision_recall_vs_noise.svg")

    cost_series = []
    for t in types:
        rows_t = sorted((r for r in agg_rows if r["distractor_type"] == t), key=lambda r: r["noise_count"])
        cost_series.append(
            {
                "label": t,
                "color": TYPE_COLORS.get(t, "#2ca02c"),
                "dash": False,
                "points": [(r["mean_prompt_tokens"], r["macro_f1"]) for r in rows_t],
                "point_labels": [f"n={r['noise_count']}" for r in rows_t],
            }
        )
    write_xy_lines_svg(
        cost_series,
        output_dir / "cost_accuracy_tradeoff.svg",
        "Cost-accuracy tradeoff: prompt tokens vs macro F1",
        "Mean prompt tokens",
        "Macro F1",
    )
    written.append("cost_accuracy_tradeoff.svg")

    sg_by_key = {(r["distractor_type"], r["noise_count"]): r["correct_rate"] for r in single_gold_rows}
    sg_series = [
        {
            "label": t,
            "color": TYPE_COLORS.get(t, "#2ca02c"),
            "values": [sg_by_key.get((t, n)) for n in noise_ticks],
        }
        for t in types
    ]
    write_grouped_bars_svg(
        [str(n) for n in noise_ticks],
        sg_series,
        output_dir / "single_gold_accuracy.svg",
        "Single-gold subgroup: correct selection accuracy",
        "Distractor count (noise_count)",
        "Correct selection rate",
    )
    written.append("single_gold_accuracy.svg")

    bin_labels = sorted({r["bin_range"] for r in position_rows})
    pos_by_key = {(r["distractor_type"], r["bin_range"]): r["gold_selection_rate"] for r in position_rows}
    pos_types = sorted({r["distractor_type"] for r in position_rows})
    pos_series = [
        {
            "label": t,
            "color": TYPE_COLORS.get(t, "#2ca02c"),
            "values": [pos_by_key.get((t, b)) for b in bin_labels],
        }
        for t in pos_types
    ]
    write_grouped_bars_svg(
        bin_labels,
        pos_series,
        output_dir / "position_bias.svg",
        "Gold selection rate by relative menu position (exploratory)",
        "Relative menu position bin",
        "Gold selection rate",
    )
    written.append("position_bias.svg")

    return written


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

    agg_rows = aggregate_condition_metrics(expanded)
    single_gold_rows = single_gold_accuracy(expanded)
    figure_names = write_report_figures(
        output_dir, agg_rows, failure_rows, position_rows, single_gold_rows, decomposition
    )

    print(f"Wrote {output_dir / 'pipeline_decomposition.json'}")
    print(f"Wrote {output_dir / 'pipeline_decomposition.csv'}")
    print(f"Wrote {output_dir / 'failure_modes.csv'}")
    print(f"Wrote {output_dir / 'position_bias.csv'}")
    for name in figure_names:
        print(f"Wrote {output_dir / name}")

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
