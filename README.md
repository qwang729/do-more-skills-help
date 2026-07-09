# Do More Skills Help?

This repository contains our course project materials for:

**Do More Skills Help? A Scaling Study of Skill Libraries for LLM Agents**

The project studies whether larger skill libraries actually help LLM agents, or whether larger libraries introduce retrieval noise, skill competition, and context pollution.

## Research Questions

1. **RQ1:** Does skill retrieval accuracy decrease as skill library size grows?
2. **RQ2:** Which distractor type is most likely to make an agent select the wrong skill?
3. **RQ3:** How do different retrievers behave under large-scale skill libraries?
4. **RQ4:** Does retrieving the correct skill always improve downstream task performance?
5. **RQ5:** Does exposing more skill descriptions cause context pollution?

## Current RQ1 Experiment

We completed the formal RQ1 retrieval-scaling experiment using the Skill-Usage dataset.

- Tasks: 87
- Skill library size: 34,396
- Candidate pool sizes: 10, 50, 100, 500, 1000, 5000, 10000, full
- Retriever: lightweight local BM25 over skill name + description
- Ranking: Top-K results include zero-score candidates; `positive_score_count` is logged separately
- Repeats: 20 random-distractor repeats for non-full pools; 1 full-library run
- Metrics: Top-1 Accuracy, Hit@K, Recall@K, MRR@10, NDCG@10

Main result:

- Top-1 Accuracy drops from **0.964** at pool size 10 to **0.414** at full library.
- Hit@10 drops from **1.000** to **0.667**.
- Recall@10 drops from **1.000** to **0.449**.

See the full analysis in [`docs/rq1_retrieval_scaling_analysis_2026-07-09.md`](docs/rq1_retrieval_scaling_analysis_2026-07-09.md).

## Current RQ2 Experiment

We completed the formal RQ2 distractor-type experiment using the same Skill-Usage setup.

Because the local Skill-Usage gold skills do not include category/repo/tag metadata, we compare five reproducible distractor families:

- `random`: uniformly sampled non-gold skills
- `query_overlap`: non-gold skills with high token overlap with the task query
- `bm25_hard`: non-gold skills that BM25 itself ranks highly for the task query
- `gold_skill_near`: non-gold skills lexically similar to the gold skill text
- `embedding_semantic_near`: non-gold skills closest to the gold skill centroid in the official embedding index

Main result:

- At pool size 100, random distractors achieve **0.889** Top-1 Accuracy.
- Query-overlap distractors reduce Top-1 Accuracy to **0.379**.
- BM25-hard distractors reduce Top-1 Accuracy to **0.425**.
- Embedding-semantic-near distractors reduce Top-1 Accuracy to **0.540**.
- Gold-skill-near distractors reduce Top-1 Accuracy to **0.563**.

See the full analysis in [`docs/rq2_distractor_type_analysis_2026-07-09.md`](docs/rq2_distractor_type_analysis_2026-07-09.md).

## Current RQ3 Experiment

We completed an enhanced RQ3 retriever-comparison experiment that extends the original local baseline.

Retrievers:

- `bm25_desc`: corpus-level BM25 over skill name + description
- `tfidf_desc`: sparse TF-IDF cosine similarity over skill name + description
- `neural_minilm_desc`: cached `sentence-transformers/all-MiniLM-L6-v2` dense retriever
- `hybrid_bm25_neural`: reciprocal-rank fusion of BM25 and MiniLM dense retrieval
- `bm25_full_skill`: BM25 over name + description + full `SKILL.md`
- `tfidf_full_skill`: TF-IDF over name + description + full `SKILL.md`

Main full-library result:

- Hybrid BM25+MiniLM: **0.460** Top-1, **0.724** Hit@10.
- BM25 full `SKILL.md`: **0.437** Top-1, **0.759** Hit@10.
- BM25 description-only: **0.425** Top-1, **0.667** Hit@10.
- MiniLM dense: **0.414** Top-1, **0.701** Hit@10.
- TF-IDF description-only: **0.356** Top-1, **0.655** Hit@10.
- TF-IDF full `SKILL.md`: **0.241** Top-1, **0.517** Hit@10.

Main takeaways:

- A true cached neural retriever helps once it is fused with BM25.
- Full `SKILL.md` content helps BM25 under hard distractors and slightly improves full-library retrieval.
- Naive full-document TF-IDF hurts, so longer skill documents need more careful retrieval design.

See the enhanced analysis in [`docs/rq3_retriever_enhanced_analysis_2026-07-09.md`](docs/rq3_retriever_enhanced_analysis_2026-07-09.md). The original baseline remains documented in [`docs/rq3_retriever_comparison_analysis_2026-07-09.md`](docs/rq3_retriever_comparison_analysis_2026-07-09.md).

## Current RQ4 Experiment

We completed the first RQ4 downstream skill exposure experiment using SkillsBench.

This is a reproducible readiness proxy, not an actual agent pass-rate run. It measures whether each retrieval condition exposes the complete curated skill set needed by a task, and how many noisy non-gold skills are exposed at the same time.

- Tasks: 87 SkillsBench default tasks
- Unique skill documents: 202
- Task gold skill references: 232
- Average gold skills per task: 2.67
- Conditions: no skill, oracle gold, oracle gold + noise, BM25 top-K, TF-IDF top-K, MiniLM top-K, BM25+MiniLM top-K, all skills visible

Main result:

- BM25 Top-1 is gold for **0.747** of tasks, but complete gold coverage is only **0.195**.
- Hybrid BM25+MiniLM Top-10 has the best complete gold coverage: **0.747**.
- Hybrid Top-10 still exposes **7.68** extra non-gold skills per task on average.
- Exposing all skills gives complete coverage **1.000**, but adds **199.33** extra skills and about **165k** median context tokens.

Takeaway: retrieval correctness helps, but downstream readiness requires a complete and reasonably clean skill set, not just a correct Top-1 skill.

See the full analysis in [`docs/rq4_downstream_skill_exposure_analysis_2026-07-09.md`](docs/rq4_downstream_skill_exposure_analysis_2026-07-09.md).

We also prepared the real-agent pass-rate validation protocol for 8 representative SkillsBench tasks across five conditions: `no_skill`, `oracle_gold_all`, `bm25_top10`, `hybrid_bm25_neural_top10`, and `oracle_gold_plus_5_noise`. The command matrix is ready, but the current local environment does not have BenchFlow CLI, Docker, or model API credentials, so true pass rate is not reported yet. See [`docs/rq4_agent_passrate_protocol_2026-07-09.md`](docs/rq4_agent_passrate_protocol_2026-07-09.md).

## Previous Pilot Experiment

We completed a first retrieval-scaling pilot using the Skill-Usage dataset.

- Tasks: 87
- Candidate pool sizes: 10, 50, 100, 500, 1000, 5000, 10000, full
- Retriever: lightweight local BM25 over skill name + description
- Metrics: Top-1 Accuracy, Recall@3, Recall@5, Recall@10, MRR, NDCG@10

Note: in this pilot, `Recall@K` means the fraction of the full gold skill set recovered in the top K, not `Hit@K`. Therefore `Recall@3` can be lower than `Top-1 Accuracy` when a task has multiple gold skills.

Main result:

- Top-1 Accuracy drops from **0.963** at pool size 10 to **0.414** at full library.
- Recall@10 drops from **0.880** to **0.449**.

This supports our initial hypothesis that larger skill libraries can make correct skill retrieval substantially harder.

## Repository Structure

```text
docs/
  do_more_skills_help_formal_proposal.md
  rq1_retrieval_scaling_analysis_2026-07-09.md
  rq2_distractor_type_analysis_2026-07-09.md
  rq3_retriever_comparison_analysis_2026-07-09.md
  rq3_retriever_enhanced_analysis_2026-07-09.md
  rq4_downstream_skill_exposure_analysis_2026-07-09.md
  rq4_agent_passrate_protocol_2026-07-09.md
  first_experiment_retrieval_scaling_pilot.md
  data_usage_guide.md
  project_data_inventory.md

experiments/
  rq1_retrieval_scaling.py
  rq2_distractor_types.py
  rq3_retriever_comparison.py
  rq3_retriever_enhanced.py
  rq4_downstream_skill_exposure.py
  rq4_agent_passrate_protocol.py
  retrieval_scaling_pilot.py

data/experiments/
  rq1_retrieval_scaling/
    summary.csv
    summary.json
    repeat_summary.csv
    per_query_metrics.csv
    ranking_examples.json
    full_pool_error_cases.json
    metric_trends.svg
  rq2_distractor_types/
    summary.csv
    summary.json
    per_query_metrics.csv
    error_examples.json
    top1_by_distractor_type.svg
  rq3_retriever_comparison/
    summary.csv
    summary.json
    per_query_metrics.csv
    ranking_examples.json
    top1_by_retriever.svg
  rq3_retriever_enhanced/
    summary.csv
    summary.json
    per_query_metrics.csv
    neural_doc_embeddings.npy
  rq4_downstream_skill_exposure/
    summary.csv
    summary.json
    per_task_exposure.csv
    case_studies.json
  rq4_agent_passrate_protocol/
    selected_tasks.csv
    condition_matrix.csv
    protocol_summary.json
    run_commands.sh
  retrieval_scaling_pilot/
    summary.csv
    summary.json
    per_query_metrics.csv
    ranking_examples.json
```

## Data

Raw datasets are intentionally not committed because they are large. Download them from the official sources below.

| Dataset | Source | Expected local path | Project use |
|---|---|---|---|
| Skill-Usage | [GitHub](https://github.com/UCSB-NLP-Chang/Skill-Usage), [Hugging Face](https://huggingface.co/datasets/Shiyu-Lab/Skill-Usage) | `data/raw/Skill-Usage` | Main retrieval-scaling experiment |
| SkillsBench | [GitHub](https://github.com/benchflow-ai/skillsbench), [Hugging Face](https://huggingface.co/datasets/benchflow/skillsbench) | `data/raw/skillsbench` | Downstream validation |
| SWE-Skills-Bench | [GitHub](https://github.com/GeniusHTX/SWE-Skills-Bench), [Hugging Face](https://huggingface.co/datasets/GeniusHTX/SWE-Skills-Bench) | `data/raw/SWE-Skills-Bench` | SWE skill/no-skill comparison |

Recommended setup:

```bash
mkdir -p data/raw
cd data/raw

git clone https://github.com/UCSB-NLP-Chang/Skill-Usage.git
cd Skill-Usage
hf download Shiyu-Lab/Skill-Usage skills-34k/skills.zip skills-34k/skills_meta.jsonl --repo-type dataset --local-dir .
unzip skills-34k/skills.zip -d skills/
cp skills-34k/skills_meta.jsonl skills/
hf download Shiyu-Lab/Skill-Usage search_index/search_index.zip --repo-type dataset --local-dir .
unzip search_index/search_index.zip -d search_server/index/

cd ..
git clone https://github.com/benchflow-ai/skillsbench.git
git clone https://github.com/GeniusHTX/SWE-Skills-Bench.git
```

See `docs/data_usage_guide.md` and `docs/project_data_inventory.md` for more details about data size, expected paths, and usage notes.

## Reproducing RQ1

After downloading the raw Skill-Usage data to `data/raw/Skill-Usage`, run:

```bash
python3 experiments/rq1_retrieval_scaling.py
```

The script writes results to:

```text
data/experiments/rq1_retrieval_scaling/
```

## Reproducing RQ2

After downloading the raw Skill-Usage data to `data/raw/Skill-Usage`, run:

```bash
python3 experiments/rq2_distractor_types.py
```

The script writes results to:

```text
data/experiments/rq2_distractor_types/
```

## Reproducing RQ3

After downloading the raw Skill-Usage data to `data/raw/Skill-Usage`, run:

```bash
python3 experiments/rq3_retriever_comparison.py
```

This baseline script writes results to:

```text
data/experiments/rq3_retriever_comparison/
```

For the enhanced RQ3 addendum with a cached MiniLM dense retriever, hybrid retrieval, full `SKILL.md` conditions, and hard distractors, run:

```bash
python3 experiments/rq3_retriever_enhanced.py
```

It writes results to:

```text
data/experiments/rq3_retriever_enhanced/
```

## Reproducing RQ4

After downloading SkillsBench to `data/raw/skillsbench`, run:

```bash
python3 experiments/rq4_downstream_skill_exposure.py
```

The script writes results to:

```text
data/experiments/rq4_downstream_skill_exposure/
```

To prepare the real-agent pass-rate validation protocol, run:

```bash
python3 experiments/rq4_agent_passrate_protocol.py
```

This writes the selected task set, condition matrix, and BenchFlow command script to:

```text
data/experiments/rq4_agent_passrate_protocol/
```

## Reproducing the Pilot

After downloading the raw Skill-Usage data to `data/raw/Skill-Usage`, run:

```bash
python3 experiments/retrieval_scaling_pilot.py
```

The script writes results to:

```text
data/experiments/retrieval_scaling_pilot/
```
