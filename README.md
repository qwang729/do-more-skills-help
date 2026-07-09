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
  first_experiment_retrieval_scaling_pilot.md
  data_usage_guide.md
  project_data_inventory.md

experiments/
  rq1_retrieval_scaling.py
  rq2_distractor_types.py
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

## Reproducing the Pilot

After downloading the raw Skill-Usage data to `data/raw/Skill-Usage`, run:

```bash
python3 experiments/retrieval_scaling_pilot.py
```

The script writes results to:

```text
data/experiments/retrieval_scaling_pilot/
```
