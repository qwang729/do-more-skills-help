# Do More Skills Help?

**A scaling study of skill libraries for LLM agents**

This repository studies a practical question for skill-augmented LLM agents:

> When an agent is given access to more skills, does performance improve, or do retrieval noise, incomplete skill exposure, duplicate skills, and routing overload begin to dominate?

The project decomposes the problem into four connected stages: retrieval from a large skill library, distractor robustness, retriever design, downstream skill exposure, and final LLM routing among visible candidates.

## Research Map

| Question | Focus | Main artifact | Status |
|---|---|---|---|
| RQ1 | Retrieval scaling | `experiments/rq1_retrieval_scaling.py` | Complete |
| RQ2 | Distractor type robustness | `experiments/rq2_distractor_types.py` | Complete |
| RQ3 | Sparse, dense, hybrid retrievers | `experiments/rq3_retriever_enhanced.py` | Complete |
| RQ4 | Downstream skill exposure and task readiness | `experiments/rq4_downstream_skill_exposure.py`, `experiments/rq4_adaptive_qwen_experiment.py`, `rq4/` | Complete pilot + judged analysis |
| RQ5 | LLM decision-budget stress test | `experiments/rq5_llm_router_decision_budget.py` | Complete |
| Question 8 | Relational skill graph retrieval | `question 8/` | Complete |

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

### RQ4: Correct Retrieval Is Helpful, But Not Sufficient

RQ4 separates two mechanisms:

1. **Content relevance**: when retrieval is correct, injected skill content is substantially more relevant than when retrieval is wrong.
2. **Exposure quality**: many real tasks require multiple skills, so Top-1 retrieval can underload the agent, while larger Top-K menus can overload it with irrelevant context.

In the exposure analysis, SkillsBench tasks need 2.67 curated skills on average:

| Condition | Complete gold coverage | Precision | Extra skills |
|---|---:|---:|---:|
| `bm25_top1` | 0.195 | - | - |
| `bm25_top10` | 0.667 | 0.217 | 7.83 |
| `hybrid_bm25_neural_top10` | 0.747 | 0.232 | 7.68 |
| `all_skills_visible` | 1.000 | 0.013 | 199.33 |

The Qwen judged downstream pilot further suggests that hard distractors can be worse than no skill at all, but the current judged sample is small and should be read as directional evidence rather than a final pass-rate claim.

### RQ5: Even With Gold Skills Visible, LLM Routing Breaks Under Hard Noise

RQ5 fixes retrieval success by guaranteeing that every menu contains all gold skills, then asks a Qwen router to select all and only the relevant skills.

| Distractor type | Noise count | Macro F1 | Exact set match | Mean extra skills |
|---|---:|---:|---:|---:|
| Random | 0 | 0.818 | 0.540 | 0.00 |
| Random | 20 | 0.882 | 0.621 | 0.07 |
| Hard | 0 | 0.818 | 0.540 | 0.00 |
| Hard | 20 | 0.383 | 0.069 | 5.31 |

Hard distractors mainly cause **over-selection of near-duplicate skills**. The router often keeps the gold skills, but cannot reject semantically similar alternatives.

### Question 8: Relation Graphs Show Potential, But No Stable Held-Out Gain Yet

Question 8 tests whether a relational skill graph can complete multi-skill sets from retrieval seeds under a fixed Top-10 budget. The leakage-free graph methods slightly improve complete coverage in some settings, but paired confidence intervals include zero.

| Method | Complete coverage | Gold recall |
|---|---:|---:|
| Hybrid Top-10 | 0.213 | 0.464 |
| Semantic graph | 0.246 | 0.475 |
| CV co-required | 0.230 | 0.467 |
| CV all edges | 0.262 | 0.472 |
| Transductive upper bound | 0.508 | 0.622 |

The transductive upper bound shows that skill relations can matter, but current leakage-free relation sources are too sparse to prove a stable average recall improvement.

## Repository Structure

```text
docs/
  data_usage_guide.md
  project_data_inventory.md
  do_more_skills_help_formal_proposal.md
  rq1_retrieval_scaling_analysis_2026-07-09.md
  rq2_distractor_type_analysis_2026-07-09.md
  rq3_retriever_enhanced_analysis_2026-07-09.md
  rq4_adaptive_qwen_analysis_2026-07-15.md
  rq5_decision_budget/

experiments/
  rq1_retrieval_scaling.py
  rq2_distractor_types.py
  rq3_retriever_comparison.py
  rq3_retriever_enhanced.py
  rq4_downstream_skill_exposure.py
  rq4_adaptive_qwen_experiment.py
  rq5_llm_router_decision_budget.py

data/experiments/
  rq1_retrieval_scaling/
  rq2_distractor_types/
  rq3_retriever_enhanced/
  rq4_downstream_skill_exposure/
  rq4_adaptive_qwen/
  rq5_llm_router/

rq4/
  rq4 answer
  qwen3.6-plus
  per_task_results

question 8/
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
| Skill-Usage | `https://github.com/UCSB-NLP-Chang/Skill-Usage` | `data/raw/Skill-Usage` | RQ1, RQ2, RQ3, RQ5, Question 8 |
| SkillsBench | `https://github.com/benchflow-ai/skillsbench` | `data/raw/skillsbench` | RQ4 |
| SWE-Skills-Bench | `https://github.com/GeniusHTX/SWE-Skills-Bench` | `data/raw/SWE-Skills-Bench` | Supporting downstream analysis |

See `docs/data_usage_guide.md` and `docs/project_data_inventory.md` for download notes, expected paths, and local data checks.

## Environment

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Some experiments require locally cached model files or API credentials:

- RQ3 and RQ5 use `sentence-transformers/all-MiniLM-L6-v2`.
- Qwen-based experiments require `DASHSCOPE_API_KEY` or the script's `--api-key-prompt` option.
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

RQ4 exposure table:

```bash
python3 experiments/rq4_downstream_skill_exposure.py
```

RQ4 adaptive Qwen pilot:

```bash
python3 experiments/rq4_adaptive_qwen_experiment.py \
  --max-tasks 6 \
  --token-budget 250000 \
  --model qwen3.6-flash \
  --resume \
  --api-key-prompt
```

RQ5 decision-budget stress test:

```bash
python3 experiments/rq5_llm_router_decision_budget.py --dry-run
python3 experiments/rq5_llm_router_decision_budget.py --resume --api-key-prompt
```

Question 8 relational skill graph experiment:

```bash
python3 "question 8/scripts/run_relational_skill_graph_experiment.py"
python3 "question 8/scripts/analyze_results.py"
```

## Reading Guide

For a fast overview, start with:

1. `docs/rq1_retrieval_scaling_analysis_2026-07-09.md`
2. `docs/rq2_distractor_type_analysis_2026-07-09.md`
3. `docs/rq3_retriever_enhanced_analysis_2026-07-09.md`
4. `rq4/rq4 answer`
5. `rq4/qwen3.6-plus`
6. `docs/rq5_decision_budget/results.md`
7. `question 8/report/Question8_关系化技能图实验报告.md`

## Claim Boundaries

This project supports claims about retrieval accuracy, skill exposure quality, LLM routing behavior, and judged downstream-readiness proxies. It does **not** claim universal agent pass-rate degradation across model families or production environments. RQ4 judged-pass results are LLM-judge signals, not sandbox execution ground truth. RQ5 measures skill selection, not actual invocation or task completion.
