# RQ5 README: Decision-Budget Stress Test for LLM Skill Routing

**Script**: `experiments/rq5_llm_router_decision_budget.py`
**Proposal**: `docs/rq5_decision_budget/proposal.md`
**Output directory**: `data/experiments/rq5_llm_router/`
**Status**: Finished

---

## 1. Research Question

> Given that all required skills are visible, how do distractor count and distractor hardness affect an LLM router's ability to select the complete and correct skill set?

RQ5 isolates the **decision-budget** stage of the agent pipeline. RQ1-RQ3 study whether a retriever can find gold skills; RQ4 studies downstream plan readiness. RQ5 bypasses retrieval misses entirely: every candidate menu is guaranteed to contain all gold skills, and the only manipulation is how many and how confusing the non-gold candidates are.

```text
Skill library
    -> retrieval and ranking                 RQ1-RQ3: retrieval budget
    -> selected skills exposed to context    RQ4-A: context-exposure diagnostics
    -> LLM selects relevant visible skills   RQ5: decision budget   <-- this experiment
    -> skills are incorporated and executed  RQ4-B / future verifier run
```

---

## 2. Experimental Design

### Unit of analysis

```text
task x distractor_type x noise_count
```

- **Tasks**: all Skill-Usage tasks with fully resolvable gold skills (87 confirmed). Single-gold tasks (26) are a pre-specified sensitivity subgroup, not the main dataset.
- **Noise count** `n in {0, 2, 5, 10, 20}` (frozen on 2026-07-20 after the 10-task pilot per the adjustment rule in proposal Section 6; see `noise_grid_decision.json`).
- **Distractor types**:
  - `random`: one deterministic permutation of non-gold skills per task (seed 6002 + stable task hash); noise levels take nested prefixes;
  - `hard`: top-ranked non-gold skills from the RQ3 hybrid BM25 + MiniLM retriever (reciprocal-rank fusion), nested prefixes.
- The `n=0` baseline is identical for both types and is called **once per task** (recorded as `distractor_type="shared"`, expanded into both curves during analysis).

### Controls

- All gold skills visible in every menu; menu order blinded by a deterministic `sha256(task|skill_id|seed)` sort key; no gold/condition labels in prompts.
- One fixed Qwen model, `temperature=0`, `enable_thinking=false`, `max_completion_tokens<=256`, `stream=false`, seed 6002.
- Skill representation is name + description only (no full SKILL.md).

### Pre-call invariants (asserted before any API call)

- Every gold skill appears exactly once; no distractor is gold; no duplicate IDs.
- `menu_size == gold_count + noise_count`.
- Random and hard menus are nested within their own type.
- Menu indices map bijectively to skill IDs.

---

## 3. How to Run

### Prerequisites

1. **Restore raw data** to `data/raw/Skill-Usage/` (gitignored, download separately):
   - `data/task_queries.json`
   - `data/task_skill_mapping.json`
   - `skills-34k/skills_meta.jsonl`

  ```bash
  mkdir -p data/raw

  git clone \
    --depth 1 \
    --filter=blob:none \
    --sparse \
    https://github.com/UCSB-NLP-Chang/Skill-Usage.git \
    data/raw/Skill-Usage

  git -C data/raw/Skill-Usage sparse-checkout set data

  mkdir -p data/raw/Skill-Usage/skills-34k

  curl -L --fail --retry 3 \
    "https://huggingface.co/datasets/Shiyu-Lab/Skill-Usage/resolve/main/skills-34k/skills_meta.jsonl?download=true" \
    -o data/raw/Skill-Usage/skills-34k/skills_meta.jsonl
  ```

  Validation:
  ```bash
  test -s data/raw/Skill-Usage/data/task_queries.json &&
  test -s data/raw/Skill-Usage/data/task_skill_mapping.json &&
  test -s data/raw/Skill-Usage/skills-34k/skills_meta.jsonl &&
  echo "RQ5 raw data restored successfully"
  ```
  
2. **API key**: set `DASHSCOPE_API_KEY` or use `--api-key-prompt` (hidden terminal input). The key is never written to experiment files.

3. **Base URL** (optional): set `DASHSCOPE_BASE_URL` or use `--base-url`. If neither is provided, the shared endpoint `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` is used. For better performance and stability, Aliyun recommends the workspace-dedicated domain:

   ```bash
   export DASHSCOPE_BASE_URL="https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
   ```

   Replace `{WorkspaceId}` with your workspace ID from the Bailian console. Prefer the env var over `--base-url` to keep the workspace ID out of shell history. Requests use the full URL, but `experiment_metadata.json` / `summary.json` only store a redacted form (`<workspace>` placeholder), so the workspace ID is never written to tracked experiment files.

### Step 1: dry run (no API calls)

```bash
python3 experiments/rq5_llm_router_decision_budget.py --dry-run
```

Writes `experiment_metadata.json`, `experiment_plan.csv`, and `candidate_menus.jsonl`, then prints the first five router prompts for manual inspection.

**MiniLM cache** (only needed for `hard` distractors): `sentence-transformers/all-MiniLM-L6-v2` must be cached locally (same as RQ3 enhanced). Document embeddings are cached at `data/experiments/rq5_llm_router/neural_doc_embeddings.npy`.

### Step 2: 10-task pilot

```bash
python3 experiments/rq5_llm_router_decision_budget.py --limit-tasks 10 --api-key-prompt
```

After the pilot, apply the noise-grid adjustment rule (proposal Section 6) to the `hard, n=20` macro F1:

| Pilot macro F1 (hard, n=20) | Final grid |
|---|---|
| > 0.95 | `{0, 10, 30, 50, 100}` |
| 0.85 - 0.95 | `{0, 2, 5, 10, 20, 50}` |
| otherwise | keep `{0, 2, 5, 10, 20}` |

Record the decision once in `data/experiments/rq5_llm_router/noise_grid_decision.json` **by hand** (fields: `decided_on`, `pilot_macro_f1_hard_n20`, `rule_branch`, `final_grid`) and never revise it after full-run calls begin. 

Example content:

```json
{
  "decided_on": "2026-07-20",
  "pilot_command": "python3 experiments/rq5_llm_router_decision_budget.py --limit-tasks 10 --api-key-prompt",
  "pilot_macro_f1_hard_n20": 0.421,
  "rule_branch": "otherwise (< 0.85)",
  "final_grid": [0, 2, 5, 10, 20]
}
```

On every subsequent run the script merges this file into `experiment_metadata.json` (`noise_grid_status: frozen_by_pilot_decision`) and aborts if `--noise-counts` deviates from the frozen grid. Pilot calls under a superseded grid are reported separately as pilot data.

### Step 3: full run

```bash
# original grid
python3 experiments/rq5_llm_router_decision_budget.py --resume --api-key-prompt

# extended grid example
python3 experiments/rq5_llm_router_decision_budget.py --resume --api-key-prompt \
  --noise-counts 0 10 30 50 100
```

`--resume` deduplicates by stable `condition_id` (`task|distractor_type|nN`), so interrupted runs continue without repeating completed conditions. `--max-api-calls` (default 1200) is a per-run safety cap. Quota/permission errors trigger a clean shutdown; retryable HTTP errors use exponential backoff.

### Key CLI arguments

| Argument | Default | Purpose |
|---|---|---|
| `--skill-usage-root` | `data/raw/Skill-Usage` | Raw dataset root |
| `--output-dir` | `data/experiments/rq5_llm_router` | All outputs |
| `--model` | `qwen3.7-plus` | Fixed for the whole run; do not mix model IDs |
| `--base-url` | `$DASHSCOPE_BASE_URL` or DashScope compatible-mode endpoint | OpenAI-compatible Chat Completions. For workspace-dedicated URLs (`https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/...`), set the `DASHSCOPE_BASE_URL` env var instead of passing the flag to keep the workspace ID out of shell history; metadata files store a redacted URL (`<workspace>` placeholder) |
| `--noise-counts` | `0 2 5 10 20` | Overridden by the frozen post-pilot grid |
| `--distractor-types` | `random hard` | Distractor sources |
| `--seed` | `6002` | All deterministic sampling and ordering |
| `--limit-tasks` / `--max-api-calls` / `--dry-run` / `--resume` | - | Cost and recovery controls |

---

## 4. Outputs

All files under `data/experiments/rq5_llm_router/`:

| File | Content |
|---|---|
| `experiment_metadata.json` | Model, grid, task/gold distribution, claim boundary |
| `experiment_plan.csv` | One row per condition: sizes, gold fraction, gold positions |
| `candidate_menus.jsonl` | Full validated menus (reproducible from seed) |
| `noise_grid_decision.json` | Frozen post-pilot noise-grid decision record: decision date, pilot command, pilot macro F1 (hard, n=20) = 0.421, matched rule branch, final grid `{0, 2, 5, 10, 20}` |
| `raw_responses.jsonl` | Append-only raw API records: text, parse status, tokens, latency, retries |
| `per_condition_results.csv` | Per-condition metrics joined with covariates |
| `summary.csv` / `summary.json` | Macro metrics per (type, n), single-gold table, bootstrap contrasts |
| `selection_f1_vs_noise.svg` | Macro F1 curves, random vs hard |
| `exact_match_vs_noise.svg` | Exact set match curves |
| `tokens_vs_noise.svg` | Prompt-token cost curves |
| `case_studies.json` | Worst routing failures with error types |

---

## 5. Metrics

### Primary

- **`precision_inclusive`**: empty selection counted as 0 (used inside F1 so F1 is defined for every task).
- **`precision_conditional`**: averaged only over non-empty selections; isolates selection quality given the router selected something. Both versions are reported side by side with `empty_selection_rate`.
- **`gold_recall`**, **`selection_f1`** (macro F1 is the primary summary metric), **`exact_set_match`**.

### Diagnostics

`complete_gold_coverage`, `missing_gold_count` (under-selection), `extra_skill_count` (over-selection), `any_wrong_selection`, `empty_selection_rate`, `invalid_response_rate`, Jaccard, prompt/completion/total tokens, median and P90 latency, distractor-gold MiniLM similarity (mean/max per menu).

### Invalid-response policy

Strict JSON parse first; then one deterministic extraction of the first JSON object; missing `selected`, non-integer values, or out-of-range indices are marked `invalid` and are **not** converted into empty selections. Raw text, parse mode, and retry counts are preserved in `raw_responses.jsonl`.

### Statistical analysis

Task-level paired bootstrap (10,000 resamples, percentile 95% CIs) on per-task F1 differences:

1. `hard n=n_max` minus `n=0`;
2. `random n=n_max` minus `n=0`;
3. `hard` minus `random` at the three largest nonzero noise counts.

---

## 6. Boundary with RQ4

| Dimension | RQ4 | RQ5 |
|---|---|---|
| LLM role | Solver | Multi-label skill router |
| Skill representation | Truncated full guides | Name + description only |
| Gold visibility | Varies | Guaranteed |
| Output | Execution plan | Selected skill indices |
| Main metrics | Readiness score | Precision / recall / F1 / exact match |

RQ5 does not generate solutions, use an LLM judge, or claim task pass rate.

## 7. Claim Boundaries

RQ5 **may** claim: larger visible menus reduce multi-label routing quality; retrieval-hard distractors are more damaging than random ones; routing cost grows while quality plateaus or declines; multi-gold tasks show under-/over-selection under noise.

RQ5 **must not** claim: actual wrong skill invocation, downstream task-success degradation, OpenClaw performance, effects of stale/duplicate/malicious skills, or universal behavior across model families. Use the terms *wrong routing* / *false selection*, not *wrong invocation*.

> Intended conclusion: even after retrieval succeeds, a finite decision budget can limit how reliably an LLM agent routes among visible procedural memories; controlled candidate menus matter for reliable skill selection, not just retrieval efficiency.

## 8. Call Volume (original grid, 87 tasks)

```text
n=0 shared baseline:                 87 calls
4 nonzero n x 2 distractor types:   696 calls
Primary total:                      783 calls
```

Extended grid `{0,2,5,10,20,50}` raises the total to 957 calls.
