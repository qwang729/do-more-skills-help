#!/usr/bin/env python3
"""Prepare and audit the RQ4 real agent pass-rate validation protocol.

The previous RQ4 script measures skill exposure proxies. This script performs
the next practical step: select representative SkillsBench tasks, materialize
condition-specific task packages, and write the exact BenchFlow commands needed
to run real agents and verifiers.

It intentionally does not fabricate pass rates. If the local machine does not
have the SkillsBench runner stack (bench/uv/docker and model credentials), the
output records that pass-rate execution is blocked.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from rq4_downstream_skill_exposure import load_skillsbench_tasks


DEFAULT_SELECTED_TASKS = [
    "video-silence-remover",
    "drone-planning-control",
    "fix-erlang-ssh-cve",
    "financial-modeling-qa",
    "energy-market-pricing",
    "offer-letter-generator",
    "react-performance-debugging",
    "setup-fuzzing-py",
]
CONDITIONS = [
    "no_skill",
    "oracle_gold_all",
    "bm25_top10",
    "hybrid_bm25_neural_top10",
    "oracle_gold_plus_5_noise",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def command_output(command: list[str]) -> str:
    if not command_available(command[0]):
        return "not found"
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return f"error: {exc}"
    return (result.stdout or result.stderr).strip().splitlines()[0] if (result.stdout or result.stderr).strip() else ""


def detect_environment() -> dict:
    api_keys = {
        key: bool(os.environ.get(key))
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    }
    return {
        "bench_available": command_available("bench"),
        "bench_version": command_output(["bench", "--version"]),
        "uv_available": command_available("uv"),
        "uv_version": command_output(["uv", "--version"]),
        "docker_available": command_available("docker"),
        "docker_version": command_output(["docker", "--version"]),
        "codex_available": command_available("codex"),
        "codex_path": shutil.which("codex") or "",
        "api_keys_present": api_keys,
        "can_run_benchflow_agent": (
            command_available("bench")
            and command_available("docker")
            and any(api_keys.values())
        ),
    }


def load_exposure_rows(path: Path) -> dict[str, dict[str, dict]]:
    rows = list(csv.DictReader(path.open()))
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_task[row["task_id"]][row["condition"]] = row
    return by_task


def select_tasks(exposure: dict[str, dict[str, dict]], requested: list[str], limit: int) -> list[str]:
    selected = [task_id for task_id in requested if task_id in exposure]
    if len(selected) >= limit:
        return selected[:limit]

    # Fill any missing slots with diverse high-signal tasks from the proxy run.
    candidates = []
    for task_id, rows in exposure.items():
        if task_id in selected or "oracle_gold_all" not in rows:
            continue
        oracle = rows["oracle_gold_all"]
        bm25_top1 = rows.get("bm25_top1", {})
        hybrid_top10 = rows.get("hybrid_bm25_neural_top10", {})
        gold_count = int(float(oracle["gold_skill_count"]))
        top1_is_gold = float(bm25_top1.get("top1_is_gold", 0.0))
        complete_hybrid = float(hybrid_top10.get("complete_gold_coverage", 0.0))
        score = gold_count * 2 + (1.0 - complete_hybrid) + top1_is_gold
        candidates.append((score, task_id))
    for _, task_id in sorted(candidates, reverse=True):
        selected.append(task_id)
        if len(selected) == limit:
            break
    return selected


def copy_task_base(source_task: Path, destination_task: Path) -> None:
    if destination_task.exists():
        shutil.rmtree(destination_task)
    shutil.copytree(source_task, destination_task, ignore=shutil.ignore_patterns(".DS_Store"))
    skills_dir = destination_task / "environment" / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
    skills_dir.mkdir(parents=True, exist_ok=True)


def materialize_skills(destination_skills: Path, exposed_skill_ids: list[str], skill_by_id: dict[str, dict]) -> list[str]:
    copied = []
    used_names = defaultdict(int)
    for skill_id in exposed_skill_ids:
        skill = skill_by_id.get(skill_id)
        if not skill:
            continue
        src = Path(skill["example_path"]).parent
        base_name = src.name
        used_names[base_name] += 1
        name = base_name if used_names[base_name] == 1 else f"{base_name}-{used_names[base_name]}"
        dst = destination_skills / name
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"))
        copied.append(name)
    return copied


def build_agent_command(task_package: Path, condition: str, agent: str, model: str, sandbox: str, jobs_dir: Path) -> str:
    task_arg = str(task_package)
    job_arg = str(jobs_dir / condition)
    if condition == "no_skill":
        return (
            f"bench eval run --tasks-dir {task_arg} --agent {agent} --model {model} "
            f"--skill-mode no-skill --sandbox {sandbox} --jobs-dir {job_arg}"
        )
    return (
        f"bench eval run --tasks-dir {task_arg} --agent {agent} --model {model} "
        f"--skill-mode with-skill --skills-dir {task_arg}/environment/skills "
        f"--sandbox {sandbox} --jobs-dir {job_arg}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skillsbench-root", default="data/raw/skillsbench")
    parser.add_argument("--exposure-csv", default="data/experiments/rq4_downstream_skill_exposure/per_task_exposure.csv")
    parser.add_argument("--output-dir", default="data/experiments/rq4_agent_passrate_protocol")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_SELECTED_TASKS)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--agent", default="codex-acp")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--sandbox", default="docker")
    args = parser.parse_args()

    skillsbench_root = Path(args.skillsbench_root)
    output_dir = Path(args.output_dir)
    packages_dir = output_dir / "task_packages"
    jobs_dir = output_dir / "jobs"
    output_dir.mkdir(parents=True, exist_ok=True)
    packages_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    tasks, skills = load_skillsbench_tasks(skillsbench_root / "tasks")
    task_by_id = {task["task_id"]: task for task in tasks}
    skill_by_id = {skill["skill_id"]: skill for skill in skills}
    exposure = load_exposure_rows(Path(args.exposure_csv))
    selected_task_ids = select_tasks(exposure, args.tasks, args.limit)
    environment = detect_environment()

    selected_rows = []
    condition_rows = []
    command_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated RQ4 real-agent validation commands.",
        "# Install BenchFlow/SkillsBench dependencies and configure model API keys before running.",
        "",
    ]

    for task_id in selected_task_ids:
        task = task_by_id[task_id]
        selected_rows.append(
            {
                "task_id": task_id,
                "difficulty": task["difficulty"],
                "category": task["category"],
                "gold_skill_count": task["gold_skill_count"],
                "bm25_top1_is_gold": exposure[task_id]["bm25_top1"]["top1_is_gold"],
                "bm25_top10_complete": exposure[task_id]["bm25_top10"]["complete_gold_coverage"],
                "hybrid_top10_complete": exposure[task_id]["hybrid_bm25_neural_top10"]["complete_gold_coverage"],
                "reason": "representative SkillsBench validation task",
            }
        )

        source_task = skillsbench_root / "tasks" / task_id
        command_lines.append(f"# Task: {task_id}")
        for condition in CONDITIONS:
            row = exposure[task_id][condition]
            exposed_skill_ids = [sid for sid in row["exposed_skill_ids"].split(";") if sid]
            destination_task = packages_dir / task_id / condition
            copy_task_base(source_task, destination_task)
            copied_skill_dirs = materialize_skills(
                destination_task / "environment" / "skills",
                exposed_skill_ids,
                skill_by_id,
            )
            command = build_agent_command(
                destination_task,
                condition,
                args.agent,
                args.model,
                args.sandbox,
                jobs_dir / task_id,
            )
            command_lines.append(command)
            condition_rows.append(
                {
                    "task_id": task_id,
                    "condition": condition,
                    "task_package": str(destination_task),
                    "skills_dir": str(destination_task / "environment" / "skills"),
                    "exposed_skill_count": row["exposed_skill_count"],
                    "covered_gold_count": row["covered_gold_count"],
                    "gold_skill_count": row["gold_skill_count"],
                    "complete_gold_coverage": row["complete_gold_coverage"],
                    "extra_skill_count": row["extra_skill_count"],
                    "context_tokens": row["context_tokens"],
                    "copied_skill_dirs": ";".join(copied_skill_dirs),
                    "bench_command": command,
                    "pass_rate_status": "blocked" if not environment["can_run_benchflow_agent"] else "ready_to_run",
                }
            )
        command_lines.append("")

    metadata = {
        "selected_tasks": selected_task_ids,
        "conditions": CONDITIONS,
        "environment": environment,
        "pass_rate_status": (
            "blocked: bench/docker/model credentials are not all available"
            if not environment["can_run_benchflow_agent"]
            else "ready_to_run"
        ),
        "important_boundary": (
            "This protocol prepares real agent pass-rate runs and records local readiness. "
            "It does not report pass rates unless BenchFlow agent jobs are actually executed."
        ),
    }
    write_csv(output_dir / "selected_tasks.csv", selected_rows)
    write_csv(output_dir / "condition_matrix.csv", condition_rows)
    (output_dir / "run_commands.sh").write_text("\n".join(command_lines) + "\n")
    (output_dir / "protocol_summary.json").write_text(
        json.dumps({"metadata": metadata, "selected_tasks": selected_rows, "conditions": condition_rows}, indent=2) + "\n"
    )

    print(f"Wrote {output_dir / 'protocol_summary.json'}")
    print(f"Wrote {output_dir / 'selected_tasks.csv'}")
    print(f"Wrote {output_dir / 'condition_matrix.csv'}")
    print(f"Wrote {output_dir / 'run_commands.sh'}")
    print()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
