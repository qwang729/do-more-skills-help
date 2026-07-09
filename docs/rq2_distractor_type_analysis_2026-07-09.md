# RQ2 正式实验分析：哪类 Distractor 最容易导致错误 Skill Retrieval

**日期**：2026-07-09  
**研究问题**：RQ2 - Which distractor type is most likely to make an agent select the wrong skill?  
**实验脚本**：`experiments/rq2_distractor_types.py`  
**输出目录**：`data/experiments/rq2_distractor_types/`

---

## 1. 结论摘要

RQ2 实验显示：**query-overlap distractors 和 BM25-hard distractors 最容易导致错误检索**。它们都比 random distractors 难得多。

主要结果：

- 在 pool size 100 时，random distractors 下 Top-1 Accuracy 为 **0.889**。
- 同样 pool size 下，query-overlap distractors 把 Top-1 Accuracy 降到 **0.379**。
- BM25-hard distractors 把 Top-1 Accuracy 降到 **0.425**。
- embedding-semantic-near distractors 把 Top-1 Accuracy 降到 **0.540**。
- gold-skill-near distractors 的破坏性中等，Top-1 Accuracy 为 **0.563**。

按四个 pool size 平均：

- query-overlap Top-1 Accuracy：**0.397**
- BM25-hard Top-1 Accuracy：**0.414**
- embedding-semantic-near Top-1 Accuracy：**0.503**
- gold-skill-near Top-1 Accuracy：**0.526**
- random Top-1 Accuracy：**0.819**

因此，RQ2 的当前答案是：**对 BM25 retrieval 来说，与 query 共享关键词、或者会被 BM25 自己排到前面的 hard negatives 最容易让 agent 选错 skill。真正 embedding semantic-near distractors 也明显比 random 更难，但破坏性低于 query-overlap 和 BM25-hard。**

---

## 2. 数据限制与实验调整

原始计划希望比较：

- random distractor
- same-category distractor
- same-subcategory distractor
- semantic-near distractor

但本地 Skill-Usage gold skills 在 `skills_meta.jsonl` 中存在一个重要限制：

- 198 个 unique gold skills 都使用 `benchflow-ai` owner；
- gold skills 的 `repo` 字段为空；
- gold skills 的 `category` 字段为空；
- gold skills 的 `tags` 字段为空。

因此，用 gold skill metadata 构造 same-category 或 same-repo distractors 会不公平，也不可复现。正式 RQ2 改用五类可由当前数据稳定构造的 distractor：

| Distractor type | Definition | Intended meaning |
|---|---|---|
| `random` | 从所有 non-gold skills 中均匀随机采样 | 普通背景噪声 |
| `query_overlap` | 与 task query token Jaccard similarity 最高的 non-gold skills | 与任务表面词高度重叠的混淆项 |
| `bm25_hard` | 当前 BM25 retriever 在全库中为该 query 排名最高的 non-gold skills | 对当前 retriever 最困难的 hard negatives |
| `gold_skill_near` | 与 gold skill text token Jaccard similarity 最高的 non-gold skills | 与正确 skill 描述相似的竞争 skill |
| `embedding_semantic_near` | 与 gold skill embedding centroid 最近的 non-gold skills | 真正基于官方 embedding index 的 semantic-near 竞争 skill |

这个调整仍然直接服务 RQ2：它比较的是不同来源的 distractor 对 retrieval accuracy 的破坏力。

---

## 3. 实验设置

### 数据

- 数据集：Skill-Usage
- 任务数：87
- Skill library size：34,396
- Query 来源：`data/raw/Skill-Usage/data/task_queries.json`
- Gold skill 来源：`data/raw/Skill-Usage/data/task_skill_mapping.json`
- Skill pool 来源：`data/raw/Skill-Usage/skills-34k/skills_meta.jsonl`
- Semantic embedding 来源：`data/raw/Skill-Usage/search_server/index/embeddings.npy`

### Candidate Pool

候选池大小：

| Pool size |
|---:|
| 50 |
| 100 |
| 500 |
| 1000 |

每个 candidate pool 都强制包含该任务的 gold skills，其余位置由指定 distractor type 填充。

### Repeats

- `random`：每个 pool size 重复 20 次随机采样。
- `query_overlap`、`bm25_hard`、`gold_skill_near`、`embedding_semantic_near`：确定性 hard-negative 构造，每个 pool size 运行 1 次。

### Retriever

与 RQ1 保持一致：

```text
BM25 over skill name + skill description
```

### Metrics

- Top-1 Accuracy
- Top-1 Error Rate
- Hit@10
- Hit@10 Miss Rate
- Recall@10
- MRR@10
- NDCG@10
- Mean hard-negative purity：非 random distractor 中，真正 hard negatives 占 distractors 的平均比例。
- Mean fallback distractors：当 hard negatives 不足以填满 candidate pool 时，平均用 random distractors 补齐的数量。

---

## 4. 主要结果

![RQ2 Top-1 by distractor type](../data/experiments/rq2_distractor_types/top1_by_distractor_type.svg)

| Distractor type | Pool size | Top-1 | Error | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| random | 50 | 0.908 | 0.092 | 0.992 | 0.941 |
| random | 100 | 0.889 | 0.111 | 0.989 | 0.924 |
| random | 500 | 0.763 | 0.237 | 0.953 | 0.836 |
| random | 1000 | 0.717 | 0.283 | 0.937 | 0.796 |
| query_overlap | 50 | 0.391 | 0.609 | 0.736 | 0.502 |
| query_overlap | 100 | 0.379 | 0.621 | 0.713 | 0.492 |
| query_overlap | 500 | 0.414 | 0.586 | 0.690 | 0.509 |
| query_overlap | 1000 | 0.402 | 0.598 | 0.690 | 0.505 |
| bm25_hard | 50 | 0.391 | 0.609 | 0.690 | 0.495 |
| bm25_hard | 100 | 0.425 | 0.575 | 0.678 | 0.512 |
| bm25_hard | 500 | 0.425 | 0.575 | 0.701 | 0.514 |
| bm25_hard | 1000 | 0.414 | 0.586 | 0.713 | 0.509 |
| embedding_semantic_near | 50 | 0.552 | 0.448 | 0.828 | 0.649 |
| embedding_semantic_near | 100 | 0.540 | 0.460 | 0.805 | 0.630 |
| embedding_semantic_near | 500 | 0.460 | 0.540 | 0.770 | 0.568 |
| embedding_semantic_near | 1000 | 0.460 | 0.540 | 0.759 | 0.558 |
| gold_skill_near | 50 | 0.563 | 0.437 | 0.908 | 0.681 |
| gold_skill_near | 100 | 0.563 | 0.437 | 0.862 | 0.654 |
| gold_skill_near | 500 | 0.494 | 0.506 | 0.782 | 0.591 |
| gold_skill_near | 1000 | 0.483 | 0.517 | 0.770 | 0.575 |

---

## 5. Average Across Pool Sizes

| Distractor type | Avg Top-1 | Avg Error | Avg Hit@10 | Avg Hit@10 Miss | Avg MRR@10 | Avg NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.819 | 0.181 | 0.968 | 0.032 | 0.874 | 0.791 |
| gold_skill_near | 0.526 | 0.474 | 0.830 | 0.170 | 0.625 | 0.526 |
| embedding_semantic_near | 0.503 | 0.497 | 0.790 | 0.210 | 0.601 | 0.502 |
| bm25_hard | 0.414 | 0.586 | 0.695 | 0.305 | 0.508 | 0.413 |
| query_overlap | 0.397 | 0.603 | 0.707 | 0.293 | 0.502 | 0.419 |

The most harmful distractor type by average Top-1 Accuracy is `query_overlap`.

`bm25_hard` is very close, and in pool size 50 it ties `query_overlap`. This is expected because both strategies select skills that are lexically attractive to the same BM25 retriever.

---

## 6. Pool Size 100 Detailed Comparison

Pool size 100 is a useful midpoint because RQ1 showed random distractors were still relatively easy at this size.

| Distractor type | Top-1 | Drop vs random | Hit@10 |
|---|---:|---:|---:|
| random | 0.889 | 0.000 | 0.989 |
| gold_skill_near | 0.563 | 0.325 | 0.862 |
| embedding_semantic_near | 0.540 | 0.348 | 0.805 |
| bm25_hard | 0.425 | 0.463 | 0.678 |
| query_overlap | 0.379 | 0.509 | 0.713 |

At pool size 100, query-overlap distractors reduce Top-1 Accuracy by **0.509** compared with random distractors. This shows that distractor *quality* can matter as much as, or more than, distractor *quantity*.

### Hard-negative purity

| Distractor type | Avg hard-negative purity | Avg fallback distractors |
|---|---:|---:|
| query_overlap | 0.953 | 31.986 |
| bm25_hard | 1.000 | 0.000 |
| gold_skill_near | 1.000 | 0.000 |
| embedding_semantic_near | 1.000 | 0.000 |

`query_overlap` occasionally needs random fallback distractors at larger pool sizes because some queries do not have enough non-gold skills with positive token overlap to fill the full pool. The fallback rate is reported explicitly, and the main conclusion remains stable because query-overlap is still the most harmful distractor type by average Top-1 Accuracy.

---

## 7. Interpretation

### 7.1 Query-overlap is highly damaging

The query-overlap setting selects skills that share task keywords with the query. BM25 relies heavily on lexical matching, so these distractors directly attack the retriever's decision boundary. This is why Top-1 Accuracy drops sharply even at pool size 50.

### 7.2 BM25-hard confirms retriever-specific vulnerability

`bm25_hard` distractors are the non-gold skills that BM25 itself finds most appealing before the final candidate-pool ranking. Their strong negative effect confirms that the retriever has systematic blind spots, not just random noise sensitivity.

### 7.3 Gold-skill-near is less harmful than query-near

Gold-skill-near distractors are still much harder than random distractors, but they are less harmful than query-overlap and BM25-hard. This suggests that, for this dataset and retriever, surface alignment with the task query is more dangerous than lexical similarity to the gold skill description.

### 7.4 Embedding semantic-near is genuinely harder than random

`embedding_semantic_near` uses the official precomputed Skill-Usage embedding index instead of token Jaccard. It reduces average Top-1 Accuracy from 0.819 under random distractors to 0.503. This validates the concern that semantic-near skills create real retrieval competition. However, because the final retriever is BM25, lexical query-near distractors remain more damaging than embedding-near distractors.

### 7.5 Top-10 remains vulnerable

Random distractors preserve Hit@10 above 0.93 for all tested pool sizes. In contrast, query-overlap and BM25-hard reduce Hit@10 to roughly 0.69-0.74. That means hard distractors do not merely reorder the top result; they can push all gold skills out of the top-10.

---

## 8. Failure Patterns

At pool size 100:

- query-overlap Top-1 failures：54 / 87
- BM25-hard Top-1 failures：50 / 87
- embedding-semantic-near Top-1 failures：40 / 87
- gold-skill-near Top-1 failures：38 / 87
- query-overlap and BM25-hard shared failures：49 tasks

Frequently affected tasks include:

- `azure-bgp-oscillation-route-leak`
- `citation-check`
- `court-form-filling`
- `crystallographic-wyckoff-position-analysis`
- `data-to-d3`
- `econ-detrending-correlation`
- `enterprise-information-search`
- `financial-modeling-qa`
- `find-topk-similiar-chemicals`
- `fix-build-agentops`
- `fix-druid-loophole-cve`

完整错误样例见：

- `data/experiments/rq2_distractor_types/error_examples.json`

---

## 9. 当前限制

- 由于 gold metadata 缺失，本实验没有直接评估 same-category 或 same-subcategory distractors。
- `query_overlap` 和 `gold_skill_near` 使用 token Jaccard，是 lexical proxy；`embedding_semantic_near` 已补充真正的 embedding-based semantic-near distractors。
- `bm25_hard` 是 retriever-specific hard negatives；换 dense retriever 后 hard-negative 集合可能不同。
- 当前实验仍只评估 retrieval，不评估 downstream task completion。

---

## 10. 下一步

1. **RQ3**：比较 BM25、dense embedding、hybrid retrieval 和 reranker，看 hard distractor 下是否仍有同样退化。
2. **Retriever-specific semantic comparison**：在 dense retriever 下重新构造 dense-hard negatives，检查 embedding semantic-near 是否会变成最有害 distractor。
3. **Manual error taxonomy**：抽样标注 query-overlap、BM25-hard 和 embedding-semantic-near 的错误案例，区分 broad keyword collision、tool-domain collision、task-format collision 等错误来源。
4. **Dataset enrichment**：如果后续能获得 gold skill category/subcategory，补跑 same-category 和 same-subcategory distractor。

---

## 11. Reproducibility

复现实验：

```bash
python3 experiments/rq2_distractor_types.py
```

主要输出：

- `data/experiments/rq2_distractor_types/summary.csv`
- `data/experiments/rq2_distractor_types/summary.json`
- `data/experiments/rq2_distractor_types/per_query_metrics.csv`
- `data/experiments/rq2_distractor_types/error_examples.json`
- `data/experiments/rq2_distractor_types/top1_by_distractor_type.svg`
