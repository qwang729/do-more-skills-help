# RQ4 正式实验分析：LLM-Judge Downstream Task Performance（qwen3.6-plus 实测）

**日期**：2026-07-21
**研究问题**：RQ4 - Does retrieving the correct skill actually improve downstream task performance?
**实验脚本**：`rq4_downstream_task_performance.py`
**Solver 模型**：`qwen3.6-plus`
**Judge 模型**：`qwen-plus-2025-07-28`

---

## 1. 结论摘要

21 个任务、5 个条件、1 次重复，跑出的真实 pass rate 数据：

| Condition | Pass Rate | Mean Score（0-5 分） |
|---|---:|---:|
| `no_skill` | 0.045 | 1.14 |
| `gold_skill` | **0.143** | **1.52** |
| `retrieved_top1_full` | 0.095 | 1.24 |
| `hard_distractor` | **0.000** | **0.86** |
| `random_wrong` | 0.143 | 1.33 |

两个方向性结果：

1. **`hard_distractor` 在两个指标上都是全场最差**（pass rate 0、mean score 0.86），符合"注入一个刻意构造的强误导性 skill 应该比不给 skill 更差"这个直觉方向。
2. **`gold_skill` 在 mean score 上明显领先**（1.52，比 `no_skill` 高 33%），但 pass rate 上和 `random_wrong` 打平（都是 0.143）。

但在下结论之前，必须先看第 3 节的置信区间——**当前样本量下，这些差异大多数不具备统计意义**，只有 `hard_distractor` 明显偏离其他条件这一点相对更值得关注。

---

## 2. 实验设置

| 项目 | 取值 |
|---|---|
| 任务数 | 21 |
| 条件 | `no_skill` / `gold_skill` / `retrieved_top1_full` / `hard_distractor` / `random_wrong` |
| 重复次数 | 1 |
| Solver 模型 | `qwen3.6-plus` |
| Judge 模型 | `qwen-plus-2025-07-28` |
| 打分刻度 | 0-5 整数（metadata 里已明确标注：`"mean_score_scale": "0-5 integer"） |

**重要边界**：`judged_pass` 是 LLM-judge 对照 oracle 解答 + verifier 文本打的分，不是真实沙箱执行结果，应当视为一个必要条件式的信号，不是 ground-truth pass rate。

---

## 3. 统计显著性：现在能不能下结论

用二项比例的 95% 置信区间（Wald 近似，下界截断在 0）：

| Condition | Pass Rate | 通过数/总数 | 95% 置信区间 |
|---|---:|---:|---:|
| `no_skill` | 0.045 | 1/22 | [0.000, 0.132] |
| `gold_skill` | 0.143 | 3/21 | [0.000, 0.293] |
| `retrieved_top1_full` | 0.095 | 2/21 | [0.000, 0.221] |
| `hard_distractor` | 0.000 | 0/21 | [0.000, 0.000] |
| `random_wrong` | 0.143 | 3/21 | [0.000, 0.293] |

除 `hard_distractor`（21 个任务全部判定失败，区间退化为一个点）之外，其余四个条件的置信区间都相互重叠、且下界都贴着 0。**在 n=21、仅 1 次重复的情况下，`no_skill`、`gold_skill`、`retrieved_top1_full`、`random_wrong` 这四者两两之间的差异都不能被认为是统计显著的**——`gold_skill`（3/21）和 `random_wrong`（3/21）通过数完全相同，`retrieved_top1_full`（2/21）介于两者和 `no_skill`（1/22）之间，差距都只是一两个任务的量级，样本量太小，任何一个任务的判断翻转都足以改变排序。

`hard_distractor` 是唯一一个和其他条件有清晰区隔的条件——21 个任务全部未通过，是本次实验里最明确的信号。

---

## 4. 值得记录的观察（不代表已证实的结论）

- **`hard_distractor` 全军覆没**：pass rate 0.000，mean score 0.86，两个指标都是最低。这是本次结果里唯一一个"偏离其他条件足够多、值得单独拿出来说"的现象。
- **`gold_skill` 的 mean score 优势比 pass rate 优势更明显**：mean score 从 `no_skill` 的 1.14 提升到 1.52（+33%），但 pass rate 只从"1/22"提升到"3/21"，样本太小导致这个提升暂时体现不出统计意义。
- **`no_skill` 消耗的 solver token 明显更多**：`mean_solver_tokens` 在 `no_skill` 下是 5529，其余四个有 skill 注入的条件是 4544-5437，`no_skill` 明显更高；`mean_judge_tokens` 也是同样模式（`no_skill` 4592 vs 其余 3025-3631）。一个可能的读法是：没有 skill 指导时，模型要花更多篇幅自己从头摸索解法，生成更长的推理过程；给了 skill 之后（不论对错），模型的解答更收敛、更短。这只是一个观察，本次数据不足以确认因果关系。
- **整体 pass rate 偏低**：五个条件的 pass rate 都在 0-14.3% 区间，即使是表现最好的 `gold_skill`/`random_wrong` 也只有 14.3%。说明这批任务对当前的 solver 模型整体偏难，或者 judge 的通过标准比较严格；不论哪种原因，这都会让"条件之间差多少"这个问题在小样本下更难看清楚，因为大多数任务本来就落在"失败"这一侧。

---

## 5. 当前限制

- **样本量和重复次数不足以做显著性检验**：n=21，仅 1 次重复，第 3 节已经量化说明，除 `hard_distractor` 外的条件间差异都不能视为可靠结论。
- **`judged_pass`/`mean_score` 是 LLM-judge 代理指标**，不是沙箱执行的 ground truth。
- **单次重复意味着任何一次判断的随机性都会直接体现在最终排序上**，尤其是在整体 pass rate 普遍偏低（多数任务判定失败）的情况下，少数几个任务的通过/失败翻转就能改变条件之间的相对顺序。

---

## 6. 下一步

1. 在现有 21 个任务的基础上增加重复次数（哪怕只是 2-3 次），先把 `gold_skill` 和 `random_wrong` 目前打平的 3/21 这个结果的方差缩小到可以判断的程度。
2. 针对 `hard_distractor` 这个目前唯一清晰的信号，抽样读几条 `judge_reason` 和 solution 文本，确认"全部判定失败"是 skill 内容确实起到了误导作用，还是恰好这批任务本身对该模型偏难。
3. 如果要提升整体 pass rate 偏低带来的判断难度，可以考虑换一批任务难度分布更均衡的子集，让"通过/失败"不至于大多数条件都堆在同一侧。

---

## 7. Reproducibility

```bash
python rq4_downstream_task_performance.py \
  --skill-usage-root data/raw/Skill-Usage \
  --solver-model qwen3.6-plus \
  --judge-model qwen-plus-2025-07-28 \
  --limit-tasks 21 \
  --repeats 1
```
