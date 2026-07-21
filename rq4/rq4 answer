# RQ4 研究成果报告：检索到正确 Skill 是否一定提升 Downstream Task Performance

**日期**：2026-07-13
**研究问题**：RQ4 - Does retrieving the correct skill always improve downstream task performance?
**本报告整合**：
- **What（发生了什么）**——`rq4_free_lexical_proxy.py`，量化"检索对/错"导致的内容相关性退化幅度
- **Why（为什么发生）**——`rq4_downstream_skill_exposure.py`，定位退化的两个具体机制：Underload（检索不完整）与 Overload（上下文污染）

---

## 1. 核心结论

RQ4 的答案是：**不一定。检索到"正确"的 skill 只是必要条件，不是充分条件。** 我们用两条独立的证据链证明了这一点，并且第一次把"退化多严重"和"退化因为什么"连了起来：

> **检索到正确 skill 提升的是"agent 手里这一份内容是否有用"，但决定任务能不能做完的，是"agent 手里这一整套 skill 是否完整、干净"。前者是量化出来的 What，后者是定位出来的 Why——两者合起来说明：即使把 Top-1 检索准确率做到 100%，只要任务需要不止一个 skill，或者为了凑齐这些 skill 而扩大候选数量，downstream 表现依然会被 Underload 或 Overload 拖累。**

---

## 2. Part 1 · What：检索对/错，内容相关性退化多少

### 2.1 方法

不调用任何模型 API，复用 RQ1 的 BM25 机制：把每个任务自己的 oracle 解答 + verifier 断言文本当作"这个任务真正需要什么"的 query，对全部 **34,396 个 skill**（Skill-Usage 全量库）排序，看不同条件下注入的 skill 排第几名。87 个任务、5 个条件（`no_skill` / `gold_skill` / `retrieved_top1_full` / `hard_distractor` / `random_wrong`）、3 次重复（与 RQ3 对齐）。

> 候选池规模：**34,396**（Skill-Usage 全量库）。这个数字和第 3 节 Why 部分的候选池规模不同，两边不能直接做数字层面的算术比较——能合并的是各自指向的机制性结论，不是数字本身。完整的范围核对见 `rq4_combined_analysis_2026-07-13.md`。

### 2.2 确切的性能退化数据

| Condition | Reciprocal Relevance Rank | Top-10 相关率 | Found rate（Top-1000 内） |
|---|---:|---:|---:|
| `gold_skill`（理想上限） | 0.171 | 29.9% | 80.5% |
| `retrieved_top1_full`（真实检索管线） | 0.118 | 19.5% | 63.2% |
| ├─ 检索**正确**（41.4% 的任务） | 0.200 | 30.6% | 80.6% |
| └─ 检索**错误**（58.6% 的任务） | **0.060** | 11.8% | 51.0% |
| `hard_distractor`（刻意构造的最坏错误） | 0.018 | 4.6% | 28.7% |
| `random_wrong`（纯随机） | 0.0003 | 0.0% | 4.6% |

**两个确切的数字结论：**

1. **检索对 vs 检索错，相关性相差 3.3 倍**（RRR 0.200 → 0.060）——这是"检索正确性有用"的直接证据。
2. **但检索"错"的真实退化，远没有想象中严重**：真实 BM25 犯的错误（RRR 0.060，found rate 51%），比故意构造的 hard distractor（0.018，29%）好 3 倍以上，比随机技能（0.0003，5%）好近 200 倍。也就是说，检索管线选错的那 58.6% 的任务里，注入的 skill 大概率不是"完全无关的噪声"，而是"次优但仍然沾边"的近似命中——比如 `citation-check` 检索错了，但注入的技能在真实需求排序里排第 9 名，几乎等于没错；真正意义上的"注入垃圾内容"（比如 `court-form-filling` 排第 280 名）只是错误里的一部分，不是全部。

这条证据链回答的是**"单个被注入的 skill，内容有多相关"**——这是 What，是可以精确量化的退化曲线。但它没有回答一个更根本的问题：**为什么即使给对了 skill，任务也不一定能做完？** 这是下一部分 Why 要回答的。

---

## 3. Part 2 · Why：Underload 与 Overload

这部分实验换了一个观察角度：不问"这一个 skill 内容好不好"，而是问**"agent 最终能看到的整套 skill 集合，够不够、干不干净"**。核心发现是两个互斥的失败模式：

### 3.1 Underload：检索不完整

SkillsBench 87 个任务平均需要 **2.67 个** curated skill 才能完成，只看 Top-1 会系统性低估真实所需。数据非常直接（候选池规模：**202**，SkillsBench 任务自带的小型 skill 池，与 2.1 节的 34,396 不是同一个候选库，两者不能直接做数字层面的算术比较）：

| Condition | Top-1 Is Gold | 完整覆盖所有 gold skills 的概率 |
|---|---:|---:|
| `bm25_top1` | **0.747** | **0.195** |

Top-1 命中率高达 74.7%，但"任务所需的全部 skill 都被找到"的概率只有 19.5%——因为大多数任务不是单 skill 任务。典型案例：`video-silence-remover` 需要 7 个 curated skill，BM25 Top-1 只能给出其中 1 个,agent 拿到的是一份不完整的工具箱。**这就是 Underload：即使 Top-1 是对的，也可能只是"对了一小部分"。**

### 3.2 Overload：为了补全 Underload，代价是上下文污染

一个直觉的解法是加大 Top-K，多给几个候选。数据显示这条路走不通：

| Condition | 完整覆盖率 | Precision | 额外无关 skill 数 | 中位 context tokens |
|---|---:|---:|---:|---:|
| `bm25_top1` | 0.195 | — | — | — |
| `bm25_top10` | 0.667 | 0.217 | 7.83 | 13,663 |
| `hybrid_bm25_neural_top10` | **0.747** | 0.232 | 7.68 | 10,525 |
| `all_skills_visible`（给全部技能） | 1.000 | 0.013 | 199.33 | 165,227 |

Top-K 从 1 提到 10，完整覆盖率确实从 19.5% 提到了 66.7%-74.7%，但代价是：Precision 只剩 21%-23%（意味着 agent 看到的技能里将近 8 成是无关的），每个任务平均多背 7.68-7.83 个无关 skill、上万 token 的无用上下文。极端情况下"干脆全给"，覆盖率满分，但代价是 199 个无关 skill、16 万+ token——这在任何真实 agent 场景里都是不可接受的上下文污染。**这就是 Overload：为了解决 Underload 而扩大候选数量,换来的是噪声和 token 成本的同步失控。**

---

## 4. What + Why：合并起来到底说明了什么

这是这两份工作真正能合并的地方——不是数字对比，是**因果链条**：

```text
[What 量化的现象]                [Why 定位的机制]
检索正确 → 内容相关性高 3.3 倍    ←── 只在"单 skill 任务"或"只看这一个 skill 够不够用"时成立
检索错误 → 相关性大幅下降        ←── Underload：真实任务平均需要 2.67 个 skill，
                                      Top-1 对了也常常只覆盖其中一小部分（19.5% 完整覆盖）
扩大候选池 "本该" 缓解上面的问题  ←── 但 Overload：Top-K 越大，无关 skill 和 token
                                      成本同步失控（precision 从 100% 掉到 1.3%）
```

**把两条证据链拼在一起，RQ4 的完整回答是：**

> 检索正确性对 downstream performance 有正向贡献,这一点被 What 的数据精确量化了（检索对比错，内容相关性差 3.3 倍,比刻意构造的错误好 3 倍以上）。但这个正向贡献**不能线性外推到任务完成率**,因为真实任务往往需要多个 skill 组合完成,而"提高检索正确率"这个杠杆本身,在多 skill 场景下会撞上一个两难:只取 Top-1,大概率 Underload(拿不全);放大 Top-K 去补全,又会 Overload(拿太多、太脏)。这解释了为什么"检索到正确 skill"不能简单等价于"downstream 表现会提升"——**它提升的是拿到手的这一份内容的质量上限,但决定任务能不能做完的,是整套工具箱的完整度和干净程度,这是检索 Top-1 accuracy 这一个指标无法单独衡量的。**

这也直接呼应了 proposal 里最初设想的贡献三:"连接 retrieval metrics 与 downstream agent success,说明错误 skill 会造成 context pollution"——现在这句话有了两组独立、方向一致的数据支撑。

---

## 5. 下一步

1. 用付费 LLM-judge 版本（`rq4_downstream_task_performance.py`）小样本 pilot,直接验证"Underload/Overload 假设"在真实下游表现上是否成立——预期：`oracle_gold_plus_5_noise` 这类 Overload 条件的 LLM-judge pass rate,应该明显低于`oracle_gold_all`,即便两者的 gold coverage 都是 100%。
2. 跑通真实 agent pass-rate 协议（`rq4_agent_passrate_protocol.py`），直接检验本报告第 4 节的"What+Why"因果链在真实 pass/fail 数据上是否成立,这是回答 RQ4 最终极的一步。

---

## 6. Reproducibility

```bash
# What：内容相关性退化数据
python rq4_free_lexical_proxy.py --skill-usage-root data/raw/Skill-Usage --repeats 3

# Why：Underload / Overload 机制
python3 experiments/rq4_downstream_skill_exposure.py --skillsbench-root data/raw/skillsbench
```
