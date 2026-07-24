# Do More Skills Help?

**A scaling study of skill libraries for LLM agents**

This repository studies a practical question for skill-augmented LLM agents:

> When an agent is given access to more skills, does performance improve, or do retrieval noise, incomplete skill exposure, duplicate skills, and routing overload begin to dominate?

The project decomposes the problem into five connected stages: retrieval from a large skill library (RQ1), distractor robustness (RQ2), retriever design (RQ3), downstream skill exposure and task performance (RQ4), and final LLM routing among visible candidates (RQ5), plus a relational skill-graph extension (Question 8).

## Research Map

| Question | Focus | Main artifact | Status |
|---|---|---|---|
| RQ1 | Retrieval scaling | `experiments/rq1_retrieval_scaling.py` | Complete |
| RQ2 | Distractor type robustness | `experiments/rq2_distractor_types.py` | Complete |
| RQ3 | Sparse, dense, hybrid retrievers | `experiments/rq3_retriever_enhanced.py` | Complete |
| RQ4 | Downstream skill exposure and task performance | `experiments/rq4_free_lexical_proxy`, `experiments/rq4_downstream_skill_exposure`, `experiments/rq4_downstream_task_performance`, `docs/rq4/` | Complete analysis + judged pilot |
| RQ5 | LLM decision-budget stress test | `experiments/rq5_llm_router_decision_budget.py`, `experiments/rq5_pipeline_diagnostics.py`, `docs/rq5_decision_budget/` | Complete full run + diagnostics |
| Question 8 | Relational skill graph retrieval | `question_8/` | Complete |

## How the Questions Connect

The six questions form one causal chain: as a skill library scales, every budget along the agent pipeline is broken in turn.

```text
Skill library (34k skills)
   │
   │ ① Retrieval budget ────── RQ1: scale breaks it   RQ2: hard near-neighbors are why
   ▼                           RQ3: hybrid mitigates   Q8: graph completion attempt
Retrieved candidates / Top-K
   │
   │ ② Context budget ──────── RQ4: Top-1 underloads, Top-K overloads
   ▼
Skills actually exposed to the LLM
   │
   │ ③ Decision budget ─────── RQ5: even with gold visible, hard noise breaks routing
   ▼
Skills selected, interpreted and applied
   │
   ▼
Task execution and final result ── RQ4 judged pilot: LLM-judge pass-rate proxy
```

1. **RQ1**: library scale alone creates retrieval competition — Top-1 falls from 0.964 to 0.414.
2. **RQ2**: the damage comes from near-neighbor *hard* distractors, not noise volume; scale hurts because it breeds near-duplicates.
3. **RQ3**: stronger retrievers (hybrid) mitigate but cannot eliminate the degradation — retrieval-side fixes alone are insufficient.
4. **RQ4**: even correct retrieval is not enough; real tasks need 2.67 skills on average, so Top-1 underloads the agent and Top-K overloads it — the right target is the completeness and cleanliness of the whole exposed skill set, not Top-1 accuracy.
5. **RQ5**: idealize retrieval by guaranteeing gold visibility, and the LLM router is still broken by hard distractors — the decision budget is an independent bottleneck, and composing the RQ3 and RQ5 stages yields only ~7-31% end-to-end exact routing (task-paired; ~2-21% under an independence assumption).
6. **Question 8**: a relational skill graph tries to complete multi-skill sets within a fixed Top-10 budget — addressing the RQ4/RQ5 dilemma structurally — but leakage-free relations are currently too sparse to show a stable gain.

Note on ordering: in pipeline order, routing (RQ5) happens before downstream execution (RQ4); the questions are instead ordered by argument — RQ4 first shows retrieval accuracy is the wrong target metric, then RQ5 isolates the decision budget as the strictest controlled ablation of the overload mechanism RQ4 uncovered.

## Main Findings

### RQ1: Larger Skill Libraries Reduce Retrieval Accuracy

Using the Skill-Usage 34k skill library, BM25 retrieval degrades sharply as candidate-pool size grows:

| Pool size | Top-1 accuracy | Hit@10 | Recall@10 |
|---:|---:|---:|---:|
| 10 | 0.964 | 1.000 | 1.000 |
| Full library | 0.414 | 0.667 | 0.449 |

The core takeaway is that skill-library scale creates real retrieval competition, even before downstream execution begins.

### RQ2: Hard Distractors Are Much More Damaging Than Random Ones

At pool size 100, distractor type strongly changes retrieval difficulty:

| Distractor type | Top-1 accuracy |
|---|---:|
| Random | 0.889 |
| Query-overlap | 0.379 |
| BM25-hard | 0.425 |
| Embedding-semantic-near | 0.540 |
| Gold-skill-near | 0.563 |

Random distractors overestimate robustness. Realistic near-neighbor distractors are the harder and more informative stress test.

### RQ3: Hybrid Retrieval Helps, But Does Not Solve Scaling

The enhanced retriever experiment compares sparse, dense, hybrid, and full-document retrieval:

| Retriever | Top-1 | Hit@10 |
|---|---:|---:|
| Hybrid BM25 + MiniLM | 0.460 | 0.724 |
| BM25 full `SKILL.md` | 0.437 | 0.759 |
| BM25 description-only | 0.425 | 0.667 |
| MiniLM dense | 0.414 | 0.701 |

Hybrid retrieval improves the ranking frontier, but larger skill documents and near-duplicate skills still require careful retrieval and filtering design.

### RQ4: Correct Retrieval Is Necessary, But Not Sufficient

RQ4 combines three evidence chains: a free lexical proxy (**what** degrades), an exposure analysis (**why** it degrades), and a judged downstream pilot.

**What**: using each task's oracle solution + verifier text to rank all 34,396 skills, correct Top-1 retrieval yields 3.3x more relevant injected content than wrong retrieval (reciprocal relevance rank 0.200 vs 0.060). But real BM25 errors are far milder than constructed noise: wrong-but-retrieved skills (RRR 0.060) beat hard distractors (0.018) by 3x and random skills (0.0003) by ~200x, i.e. most real retrieval errors are near-misses, not garbage.

**Why**: SkillsBench tasks need 2.67 curated skills on average, so exposure faces an underload/overload dilemma:

| Condition | Complete gold coverage | Precision | Extra skills |
|---|---:|---:|---:|
| `bm25_top1` | 0.195 | - | - |
| `bm25_top10` | 0.667 | 0.217 | 7.83 |
| `hybrid_bm25_neural_top10` | 0.747 | 0.232 | 7.68 |
| `all_skills_visible` | 1.000 | 0.013 | 199.33 |

Top-1 underloads the agent (19.5% complete coverage even with 74.7% Top-1 hit rate); Top-10 fixes coverage but pollutes context with ~8 irrelevant skills per task.

**Judged pilot** (`qwen3.6-plus` solver, `qwen-plus-2025-07-28` judge, 21 tasks x 5 conditions x 1 repeat): `hard_distractor` is the worst condition on both metrics (pass rate 0/21, mean score 0.86), while `gold_skill` leads on mean score (1.52 vs 1.14 for `no_skill`). At this sample size only the `hard_distractor` collapse is a clear signal; the other pairwise differences are not statistically significant.

### RQ5: Even With Gold Skills Visible, LLM Routing Breaks Under Hard Noise

RQ5 fixes retrieval success by guaranteeing that every menu contains all gold skills, then asks a Qwen router to select all and only the relevant skills (87 tasks, 783 conditions, all responses parsed).

| Distractor type | Noise count | Macro F1 | Exact set match | Mean extra skills |
|---|---:|---:|---:|---:|
| Random | 0 | 0.818 | 0.540 | 0.00 |
| Random | 20 | 0.882 | 0.621 | 0.07 |
| Hard | 0 | 0.818 | 0.540 | 0.00 |
| Hard | 2 | 0.572 | 0.115 | 1.23 |
| Hard | 20 | 0.383 | 0.069 | 5.31 |

Key results, all pre-specified contrasts significant with 95% paired-bootstrap CIs excluding zero:

- **Distractor hardness, not count, breaks routing**: up to 20 random distractors are harmless (even mildly helpful for recall), while just 2 hard distractors cut exact set match from 54% to 11%.
- **The failure mode is over-selection of near-duplicates**: the router keeps gold skills (recall 0.66-0.72 under hard noise) but cannot reject semantically equivalent alternatives; in the worst cases it selects the entire menu.
- **Single-gold subgroup isolates the mechanism**: 26 single-gold tasks are solved 100% at n=0 and under all random noise, yet 2 hard distractors drop correct selection to 23%.
- **Cost scales linearly, value does not**: each distractor adds ~60 prompt tokens; under hard noise 5.6x more prompt tokens buy strictly worse accuracy.
- **Pipeline decomposition (diagnostics)**: composing RQ3 full-library gold coverage@10 (0.345) with RQ5 routing exact match gives at best ~31% end-to-end exact skill routing (task-paired; ~21% under the independence product), and ~7-10% under hard confusion — retrieval and decision budgets are multiplicative bottlenecks, and the two stages are positively correlated (retrieval-complete tasks are also easier to route).

### Question 8: Relation Graphs Show Potential, But No Stable Held-Out Gain Yet

Question 8 tests whether a relational skill graph can complete multi-skill sets from retrieval seeds under a fixed Top-10 budget (61 multi-skill tasks, 5,000-skill reduced hard pool, 5-fold held-out protocol). The leakage-free graph methods slightly improve complete coverage in some settings, but paired recall confidence intervals all include zero (e.g. CV all edges vs Hybrid: +0.008, 95% CI [-0.028, +0.046]).

| Method | Complete coverage | Gold recall |
|---|---:|---:|
| Hybrid Top-10 | 0.213 | 0.464 |
| Semantic graph | 0.246 | 0.475 |
| CV co-required | 0.230 | 0.467 |
| CV all edges | 0.262 | 0.472 |
| Transductive upper bound | 0.508 | 0.622 |

The transductive upper bound (which leaks test labels and is potential-only) shows that skill relations can matter, but current leakage-free relation sources are too sparse — 186/198 gold skills appear in only one task — to prove a stable average recall improvement.

## Repository Structure

```text
docs/
  do_more_skills_help_formal_proposal.md
  rq1_retrieval_scaling_analysis_2026-07-09.md
  rq2_distractor_type_analysis_2026-07-09.md
  rq3_retriever_enhanced_analysis_2026-07-09.md
  rq4/
    rq4_answer.md
    qwen3.6-plus.md
  rq5_decision_budget/
    proposal.md
    readme.md
    results.md

experiments/
  rq1_retrieval_scaling.py
  rq2_distractor_types.py
  rq3_retriever_comparison.py
  rq3_retriever_enhanced.py
  rq4_free_lexical_proxy
  rq4_downstream_skill_exposure
  rq4_downstream_task_performance
  rq5_llm_router_decision_budget.py
  rq5_pipeline_diagnostics.py

data/experiments/
  rq1_retrieval_scaling/
  rq2_distractor_types/
  rq3_retriever_enhanced/
  rq4_qwen_judged_passrate/
  rq5_llm_router/
  rq5_pipeline_diagnostics/

question_8/
  data/
  references/
  report/
  results/
  scripts/
```

## Data

Raw datasets are intentionally not committed because they are large. The committed repository contains experiment scripts, processed outputs, summaries, figures, and reports.

| Dataset | Source | Expected local path | Used by |
|---|---|---|---|
| Skill-Usage | `https://github.com/UCSB-NLP-Chang/Skill-Usage` | `data/raw/Skill-Usage` | RQ1, RQ2, RQ3, RQ4, RQ5, Question 8 |
| SkillsBench | `https://github.com/benchflow-ai/skillsbench` | `data/raw/skillsbench` | RQ4 exposure analysis |
| SWE-Skills-Bench | `https://github.com/GeniusHTX/SWE-Skills-Bench` | `data/raw/SWE-Skills-Bench` | Supporting downstream analysis |

## Environment

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Some experiments require locally cached model files or API credentials:

- RQ3, RQ5, and Question 8 use `sentence-transformers/all-MiniLM-L6-v2`.
- Qwen-based experiments (RQ4 downstream task performance, RQ5 router) require `DASHSCOPE_API_KEY` or the script's `--api-key-prompt` option.
- API keys should never be stored in repository files.

## Reproducing Experiments

RQ1 retrieval scaling:

```bash
python3 experiments/rq1_retrieval_scaling.py
```

RQ2 distractor-type analysis:

```bash
python3 experiments/rq2_distractor_types.py
```

RQ3 enhanced retriever comparison:

```bash
python3 experiments/rq3_retriever_enhanced.py
```

RQ4 lexical relevance proxy (What):

```bash
python3 experiments/rq4_free_lexical_proxy --skill-usage-root data/raw/Skill-Usage --repeats 3
```

RQ4 exposure table (Why):

```bash
python3 experiments/rq4_downstream_skill_exposure --skillsbench-root data/raw/skillsbench
```

RQ4 judged downstream pilot:

```bash
python3 experiments/rq4_downstream_task_performance \
  --skill-usage-root data/raw/Skill-Usage \
  --solver-model qwen3.6-plus \
  --judge-model qwen-plus-2025-07-28 \
  --limit-tasks 21 \
  --repeats 1
```

RQ5 decision-budget stress test:

```bash
python3 experiments/rq5_llm_router_decision_budget.py --dry-run
python3 experiments/rq5_llm_router_decision_budget.py --resume --api-key-prompt
```

RQ5 post-hoc pipeline diagnostics (read-only recomputation):

```bash
python3 experiments/rq5_pipeline_diagnostics.py
```

Question 8 relational skill graph experiment:

```bash
python3 question_8/scripts/run_relational_skill_graph_experiment.py
python3 question_8/scripts/analyze_results.py
```

## Reading Guide

For a fast overview, start with:

1. `docs/rq1_retrieval_scaling_analysis_2026-07-09.md`
2. `docs/rq2_distractor_type_analysis_2026-07-09.md`
3. `docs/rq3_retriever_enhanced_analysis_2026-07-09.md`
4. `docs/rq4/rq4_answer.md`
5. `docs/rq4/qwen3.6-plus.md`
6. `docs/rq5_decision_budget/results.md`
7. `question_8/report/Question8_关系化技能图实验报告.md`

## Claim Boundaries

This project supports claims about retrieval accuracy, skill exposure quality, LLM routing behavior, and judged downstream-readiness proxies. It does **not** claim universal agent pass-rate degradation across model families or production environments. RQ4 judged-pass results are LLM-judge signals on a small sample (21 tasks, 1 repeat), not sandbox execution ground truth; only the `hard_distractor` collapse is a clear signal there. RQ5 measures skill selection, not actual invocation or task completion. Question 8 results are held-out under a 5,000-skill reduced pool; the transductive upper bound leaks test labels and must not be cited as a generalization result.
