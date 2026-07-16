# RQ5 实验设计：暴露更多 skill description 是否导致 context pollution？

**日期**：2026-07-15  

---

## 1. 研究问题与假设

当我们把 top-K 个 skill description 暴露给 LLM chooser 或 Agent 时，K 越大表现是否持续提升，还是会出现 context pollution（token cost 上升、wrong skill invocation 上升、judged pass rate 下降）？

**概念区分（贯穿最终报告的叙事框架，避免与 RQ1 混淆）**：

- **library scale**：库里一共有多少 skills——已由 RQ1 回答（库规模扫描）；
- **retrieval depth**：检索器取 Top-K——RQ1/RQ3 的 Hit@K / Recall@K 维度；
- **context exposure**：实际向模型展示多少个候选 skills——**RQ5 操纵的是这一层（exposed candidate count K），不是 library size**。

### 假设

- **H5.1（coverage–pollution trade-off，仅限 `natural_topk`）**：随 K 增大，`gold_visibility_rate` 预期单调不减，而 `conditional_selection_accuracy` 可能因 context interference 下降；二者共同决定的 `end_to_end_accuracy` 可能呈倒 U 或平台。**此假设只绑定 `natural_topk` 条件**。
- **H5.2（可见即污染，`gold_injected_topk`）**：即使 gold 保证在候选列表里（K=1 的 gold_injected 即 oracle 上界），K 增大仍导致准确率不升反降。理论预期是**单调不增、平台或下降，不是倒 U**。把 retrieval miss 和 pollution 解耦，这是 RQ5 与 RQ1 的核心区别。
- **H5.3（噪声质量效应）**：retrieval-ranked non-gold distractors 比 uniform-random distractors 导致更低的 conditional selection accuracy（呼应 RQ2 的 random vs query-overlap 结论）。注意检索排名靠前首先代表 query relevance，不必然等于与 gold skill 的相似度；query/distractor/gold 三方 similarity 仅作为机制分析变量，不写进假设本体。
- **H5.4（成本收益递减）**：prompt token cost 随 K 近似线性增长，但准确率的边际收益递减甚至为负；latency 随 K 的变化作为经验性工程指标测量，不预设严格线性关系（受服务负载、缓存、网络影响）。
- **H5.5（位置偏置，次要探索性假设）**：gold 在列表中的**相对位置**影响被选中概率（serial-position / lost-in-the-middle 效应），仅在受控位置子实验（Exp B，K ∈ {10, 20}）中检验。


### 与前面 RQ 的分工边界

- RQ1/RQ3 测"retriever 能否把 gold 排进 top-K"（retrieval 侧）。
- RQ4 分两部分：已完成的 exposure 实验测"不同检索条件下的 gold coverage 与 noise exposure"（SkillsBench，静态代理）；`rq4_adaptive_qwen_experiment.py` 的 **paired downstream-readiness 实验**（同一 task 多条件的 solver 输出由 judge 一次性校准对比，输出 `readiness_score`/`likely_pass`）已有 1 个任务的 pilot 结果（qwen3.6-flash，配额中断待续跑）；均非沙箱真实 agent pass rate。
- **RQ5 测"top-K 列表本身作为 context 的影响"**：固定 gold 可见性，只操纵暴露数量 K 与列表构成，观察 selection 和 downstream 两级表现。这是 "context budget" 论点的直接证据。

---

## 2. 数据集与模型

### 2.1 数据集

| 数据集 | 规模 | 用于 | 说明 |
|---|---|---|---|
| **Skill-Usage**（UCSB-NLP-Chang，`data/raw/Skill-Usage`） | 87 tasks，34,396 skills | Exp A / B（chooser 主数据集） | 与 RQ1–RQ3 完全同源：任务 query 取 `data/task_queries.json`，gold mapping 取 `data/task_skill_mapping.json`（两者交集 87 任务）；skill 的 name+description 取 `skills-34k/skills_meta.jsonl`（Exp A/B 注入用） |
| **SkillsBench**（benchflow-ai，`data/raw/skillsbench`） | 87 tasks，202 curated skills（平均 2.67 gold/task） | Exp C（下游 noise sweep，沿用 RQ4 选定任务）与选做 Exp D；另用于 Exp A 与 RQ4 exposure 结果的交叉验证（K ≤ 10） | 任务自带 task.md + oracle + verifier 文本（Exp C 的 judge 参照）与沙箱环境，是真实 agent pass-rate 的唯一载体 |

### 2.2 LLM 与 embedding 模型

| 角色 | 模型 | 用于 | 选型理由 |
|---|---|---|---|
| Chooser（主） | Qwen 低成本档 `qwen3.6-flash` | Exp B 全量 | 调用量最大（≈4,000 次）但任务最简单（列表选编号）；低成本模型适合高频 skill routing/chooser 场景，具有实际部署代表性，并使约 4,000 次受控调用在预算内可执行 |
| Chooser（档位对照，默认不做） | `qwen3.7-plus`、`qwen3.7-max` | Exp B 的 K ∈ {5, 20, 50} 子集 | **仅在主结果全部完成且日程有余时才做**；同干扰条件下扫模型档位，回答"强模型是否更抗污染" |
| Solver | `qwen3.7-plus` | Exp C | 注入多个完整 skill 时 prompt 较长，需要中高档能力真正"消化"多个 skill 内容；调用量小（~30 次），成本可控；若配额受限可改用 qwen3.6-flash，但**必须在开跑前选定并锁定，整套 Exp C 所有任务与条件用同一固定 solver model；中途换模型则整套重跑，不得合并两个模型的结果**（否则模型差异与 noise level/task/运行时间混杂） |
| Judge | `qwen3.7-max` | Exp C | 与 solver **不同模型且更强**，LLM-as-judge 最佳实践，**降低（不能保证消除）** self-preference 偏置（RQ4 实跑 solver/judge 同为 qwen3.6-flash，RQ5 在此处改进）；judge 模型与快照版本同样在开跑前锁定，不得中途更换；沿用 RQ4 的 **task-level paired judge**：同一 task 全部条件一次对比，调用数 = 任务数 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`（本地缓存） | Exp A 的 `distractor_similarity`；所有实验的 hybrid BM25+MiniLM 检索 | 对齐 RQ1–RQ3 的检索器，免费、离线可复现，不走外部 API |

**与 RQ4 基线的关系**：若 Exp C solver 用 qwen3.7-plus（与 RQ4 实跑的 qwen3.6-flash 不同），与 RQ4 的 readiness 数值只做**定性**对照；因果对比不受影响，因为 Exp C 自带 no_skill 与 gold+0-noise 两个内部基线，所有条件在同一模型内部比较；且 paired judge 对同 task 全条件一次校准比较，进一步降低模型差异带来的尺度漂移。

**chooser 主实验统一 `temperature=0`、关闭 thinking mode**——本地 seed=6002 控制不了云端采样随机性，默认 temperature 亦可能随 SDK/模型版本漂移。模型快照 ID、temperature、thinking 设置、SDK 版本与完整请求参数全部写入脚本 CLI 默认值与 `summary.json`，保证可复现。若需研究生成随机性，另做单独的 temperature sensitivity 子实验，不与主实验混合。

所有 API 调用沿用 `rq4_adaptive_qwen_experiment.py` 的实现惯例：**OpenAI 兼容接口，requests 直连**（不依赖 openai SDK）；`--base-url` 可配——RQ4 实跑用的是 Bailian workspace 专属 endpoint（形如 `https://{WORKSPACE_ID}.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1/...`），不一定是标准 `dashscope.aliyuncs.com`；API key 通过 `DASHSCOPE_API_KEY` 环境变量或 `--api-key-prompt` 隐藏输入，不硬编码、不入仓。同时继承其 **token-budget 硬闸门**（`--token-budget`，超预算自动停）、**配额/权限错误识别**（`AccessDenied.Unpurchased`、free quota exhausted、403 → 干净停机保留已完成输出）与 **`--resume` 续跑**机制——RQ4 实跑中途配额耗尽，这套机制已被验证为必需。每次调用的 `prompt_tokens`/`completion_tokens` 落盘 `raw_log.jsonl`，既是 H5.4 的数据也是成本审计。

---

## 3. 总体设计：三层实验（+ 选做 Exp D）

### Experiment A（免费本地）：dry-run exposure / token 统计

Exp A 的内容在 Exp B 脚本的 `--dry-run` 阶段顺带产出（同一套候选列表构造代码，免 API），作为 Exp B 报告的附表：

1. **K vs `context_tokens`**：description-only 与 full-SKILL.md 两种注入模式分别统计，K ∈ {1, 3, 5, 10, 20, 50}；
2. **K vs gold visibility**：`natural_topk` 下 gold 落入候选列表的比例（即 Hit@K，与 RQ1 衔接）；
3. **`distractor_similarity`**：非 gold 候选与 gold 的平均语义相似度（MiniLM cosine），刻画"污染物毒性"，为 Exp B 的错选归因提供解释变量。

检索器用 RQ3 最优的 hybrid BM25+MiniLM。RQ4 exposure 实验（SkillsBench，K ≤ 10）的已有数字在报告中直接引用做跨数据集交叉验证，**不重跑**。

### Experiment B（LLM chooser，核心主实验）：K 扫描下的 skill 选择

**数据**：Skill-Usage 87 tasks（与 RQ1–RQ3 相同的 query 和 gold mapping）。

**候选列表构造（关键控制）**，每个 task × 每个 K 构造三种列表：

1. `natural_topk`：hybrid 检索的真实 top-K（gold 可能不在其中）→ 反映端到端真实系统（验证 H5.1）；
2. `gold_injected_topk`：强制把 gold skill 放入列表，其余 K−1 个取检索 top 非 gold 结果 → 排除 retrieval miss，重点估计 gold 可见条件下的 context interference（验证 H5.2）；
3. `gold_plus_random`：gold + 全库随机抽取的 K−1 个非 gold skill → **解耦数量与噪声质量的对照臂**（验证 H5.3）。

**nested 候选集（严格 paired K-sweep 的前提）**：不同 K 之间的比较必须只反映数量变化，不能混入"被抽中 skill 身份变化"。因此：

- `gold_injected_topk` 的噪声位取检索排名前缀，天然 nested；
- `gold_plus_random` **为每个 task 预先生成一个长度 49 的固定随机 distractor 序列**（seed=6002），各 K 取前缀：K=3 → gold + distractors[:2]，K=5 → gold + distractors[:4]，K=20 → gold + distractors[:19]，以此类推。**禁止每个 K 独立重新抽样**。

**混淆控制说明**：`gold_injected_topk` 中 K 增大时，新增干扰项的边际相关度天然下降，因此 K 的效应混杂了"数量增加"与"噪声质量漂移"。沿 `gold_plus_random` 这条线看 K 的效应，估计的是 **uniform-random distractor distribution 下的数量效应**：nested prefixes 保证新增候选身份可追踪，但固定序列的前缀组成仍可能碰巧漂移，故通过 nested prefixes、task-level pairing 与 distractor similarity diagnostics 降低候选构成混淆，**不声称绝对纯粹的数量因果效应**。同一 K 下 `gold_injected_topk` 与 `gold_plus_random` 之差称为 **distractor construction/type effect**（数量恒定下的构造方式差异，不声称纯语义质量因果）。这形成 2×K 因子设计，可回答"污染主要来自 token 体积还是语义竞争"，直接呼应 RQ2 的 random vs query-overlap 结论（Top-1 0.889 vs 0.379）。gold 本身在所有条件下是同一个 skill、同样内容，有用信号的质量恒定。**similarity 诊断**：每个 task × K × 条件记录 mean/max distractor–gold similarity（MiniLM cosine，复用 Exp A 附表代码），检查 similarity 是否随 K 系统性变化；必要时在回归分析中作为协变量。

**gold 多重性处理**：Skill-Usage 的 gold 是集合（87 任务中仅约 26 个 single-gold，其余 2–7 个 gold）。约定：(a) **注入侧只注入单个 canonical gold**（沿用 RQ4 的 `sorted(gold)[0]`）——注意这是任意选择、不一定是任务最核心的 skill，在 limitations 中说明；(b) 注入列表（`gold_injected_topk` / `gold_plus_random`）的噪声位从**排除整个 gold 集合**的池子里取（沿用 RQ4 的 `non_gold_ids` 惯例），保证列表内 gold 恰好 1 个；(c) **判分侧按整个 gold 集合**——全部 87 tasks 作为主分析，选中任意 gold 都算对；记录 `selected_is_injected_gold` 与 `selected_is_any_gold` 两个字段，主报告用后者；(d) 26 个 single-gold tasks 单独报告，作为最干净的 sensitivity analysis。**措辞约束**：injected 条件在报告中称 "one relevant skill is exposed"，不称完整 downstream solution；chooser accuracy 不解释为完整 task readiness。

**K 取值**：{1, 3, 5, 10, 20, 50}。K=1 的 gold_injected 即 oracle 上界。

**Chooser prompt**：向 LLM 展示 task query + K 个 skill 的 name+description（带编号），要求返回 JSON：`{"selected": <编号或 none>, "confidence": float}`。允许回答 `none`（测 skill underloading：gold 明明在列表里却拒选）。

**invalid-response 处理（预定义，不得事后临时决定）**：≈4,000 次调用中必然出现非 JSON 输出、JSON 截断、`selected` 编号越界、字段缺失、API timeout、transport error。处理规则：

1. 网络/API 错误自动重试（记 `retry_count`），**不算模型行为**；
2. JSON 无法解析时进行一次格式修复（记 `parse_repair_success_rate`）；
3. 仍无法解析（含编号越界、字段缺失）则记 `invalid_response=1`；
4. **不将 invalid response 自动转换为 `none`**——invalid 与 refusal 是不同行为，混合会污染 `refusal_rate`。

**模型**：主跑 Qwen 低成本档。plus/max 档位对照**默认不做**，仅在主结果完成后考虑。

**重复术语与 order blocks（不笼统称 repeats）**：`temperature=0`、关闭 thinking mode 下相同输入近似确定性，因此不存在统计意义上的独立 repeats。术语约定：**candidate-set replicate**（不同随机候选集，本设计不用）/ **order block**（相同候选集的不同顺序，本设计采用）/ **model replicate**（相同输入重复调用，temperature=0 下无意义，不用）。具体：

- `natural_topk`：**保持检索排名顺序，只跑一次**——打乱后就不再是自然 top-K menu；
- `gold_injected_topk` / `gold_plus_random`：候选身份固定、distractor 内部顺序固定，gold 分别放在 **first / middle / last 三个 order blocks**；主 K 效应对三个位置取平均；
- **K=1 退化处理**：K=1 时 first/middle/last 三个 order blocks 的输入逐字节相同，**只调用一次**（temperature=0 下重复调用无意义且浪费配额），统计聚合时按唯一输入计权，**不得把重复输入当成三份独立证据**。

**位置子实验（H5.5，与主实验共享调用）**：仅在 K ∈ {10, 20}，将 `gold_injected_topk` 的 gold 位置从三个扩展到 first / early-middle / middle / late-middle / last 五个**相对位置**（候选集完全相同，只新增 early-middle / late-middle 两个位置的调用），比较选中率。不直接比较 position 1–50 的绝对位置。

**指标（分解式报告，分母明确）**：

`natural_topk` 的准确率必须拆解为 P(select gold) = P(gold visible) × P(select gold | gold visible)，否则无法区分曲线变化来自 retrieval coverage 还是 context pollution——这一分解是 RQ1 与 RQ5 的连接点，也是 H5.1 两个反向机制的分离手段：

- `gold_visibility_rate`：候选列表里存在任一 gold 的比例（分母：全部任务；即 Hit@K，与 RQ1 直接衔接）；
- `conditional_selection_accuracy`：仅在 gold-visible 样本中，chooser 选中 gold 的比例（pollution 的干净体现）；
- `end_to_end_accuracy`：全部任务中最终选中 gold 的比例（上两者之积）；
- `wrong_invocation_rate`：选了非 gold 的比例，**分母为 gold-visible cases**（H5.2 核心指标）；
- `refusal_rate`：答 none 的比例，**分母为 gold-visible cases**（underloading）；
- `retrieval_miss`：gold 不可见的样本单独归类，**不计入 wrong invocation**；
- `invalid_response_rate` / `api_error_count` / `retry_count` / `parse_repair_success_rate`：数据质量指标（见 invalid-response 处理规则）；
- `position_bias`：位置子实验中按 gold 相对位置分桶的准确率（H5.5）；
- `prompt_tokens` / `latency`：每次调用记录（H5.4；latency 为经验性指标）；
- `distraction_source`：选错时，错选对象与 gold 的相似度排名（衔接 RQ2 的 distractor 分析）。

`gold_injected_topk` / `gold_plus_random` 中 gold 恒可见，`conditional_selection_accuracy` 即主指标。

**数据质量审计（分析前必做）**：gold-visible 样本中验证恒等式 `conditional_selection_accuracy + wrong_invocation_rate + refusal_rate + invalid_response_rate = 1`，不满足即分类逻辑有 bug。

**成本估算**：注入两臂 87 tasks × 6 K × 2 列表 × 3 order blocks ≈ 3,132 次 + `natural_topk` 87 × 6 × 1 = 522 次 + 位置子实验新增 87 × 2 K × 2 位置 = 348 次，合计 ≈ 4,000 次（K=1 时部分退化重合，实际略少）；description-only 注入下 K=50 时 prompt 约 4–6k tokens；主跑低成本档总成本估计 ¥15–30，可控。**主要风险不是预算而是 API rate limit、失败重试、JSON 解析与分析时间**，预留重试与清洗时间。若预算/时间紧张，`gold_plus_random` 只跑 K ∈ {5, 20, 50}。先用 `--dry-run` 验证 prompt（同时产出 Exp A 附表），再用 `--limit-tasks 10` 试跑。

### Experiment C（LLM solver 下游，小规模）：downstream context sensitivity pilot

**定位声明**：Exp C 把注入的完整 skill 全部交给 solver，中间没有 chooser 选择环节，因此它测的是**多个完整 skill 内容带来的长上下文与内容冲突效应**。注意噪声数量增加同时改变 token volume 与 semantic conflict，Exp C 估计的是二者综合的 **context pollution effect**，不能单独解释为纯语义冲突；机制拆分依靠 Exp B 的 `gold_injected` vs `gold_plus_random` 对照辅助解释，报告中照此措辞。

基于 `rq4_adaptive_qwen_experiment.py` 改造，直接继承其 solver → **task-level paired judge** 流程（同一 task 的全部条件输出放进一次 judge 调用校准对比，对 K-sweep 尤其合适：相邻条件的细微差异在同一次对比内分辨，且 judge 调用数从每条件一次降到每任务一次）、任务选择逻辑、token-budget/resume/dry-run 机制，以及 LLM-judge 非沙箱执行的 boundary 声明。改动点：

- **条件从 RQ4 的"skill 质量维度"换成 hard-noise sweep**：`no_skill`、`gold_all_plus_noise_{0,5,10,20}`——gold 集合恒在（沿用 RQ4 `oracle_gold_all` 构造），只变噪声 skill 数。**噪声类型明确定义为 hard noise（hybrid 检索排名最高的 non-gold skills），与 RQ4 的 random +5 不是同一种 noise**：RQ4 的 `oracle_gold_plus_5_noise` 是全库随机采样（`rng.sample(non_gold, 5)`），只作为研究动机与历史注释引用，**不合并进 RQ5 的 noise–readiness 曲线**；RQ5 的 `{0,5,10,20}` 全部条件重新跑。选 hard noise 是因为它更接近真实 retrieval menu、更易暴露语义冲突，且与 RQ2/Exp B 的 hard distractor 逻辑一致；
- **噪声池构造（不读现有 CSV）**：现有 `per_task_exposure.csv` 最深只存 `hybrid_bm25_neural_top10`，且 Top-10 内可能含多个 gold（实测 non-gold 数在 4–9 之间波动），连 noise=10 都未必够。因此 **Exp C 脚本内重新计算完整 hybrid ranking**（复用 RQ4 exposure 脚本的检索代码，确定性、本地免费），过滤全部 gold 后取 non-gold 前缀：

  ```python
  hard_noise_pool = [sid for sid in full_hybrid_ranking if sid not in gold_skill_ids]
  noise_5, noise_10, noise_20 = hard_noise_pool[:5], hard_noise_pool[:10], hard_noise_pool[:20]
  ```

  保证严格 nested；
- **任务用 SkillsBench**（非 Skill-Usage）：沿用 RQ4 自动选出的 6 个高信号多-gold 任务（video-silence-remover 等），可扩到 10–12 个；SkillsBench 任务自带 oracle+verifier 文本供 judge 参照，judge 质量远高于无参照的 Skill-Usage 任务；已完成的 RQ4 pilot（video-silence-remover）仅作研究动机与定性参照（其 +5 为 random noise，与本实验 hard noise 不可比），不作为 sweep 数据点；
- **每条件 1 run**（solver `temperature=0` 下相同输入重复调用不构成独立 replicate，与 Exp B 术语原则一致；task diversity 比重复调用更有价值）；
- **显式关闭 thinking（运行前必改配置）**：Exp C 的 **solver 与 judge 所有调用都必须在 HTTP payload 中显式传 `enable_thinking=false`，并从返回的 response metadata（usage 中是否出现 reasoning tokens）验证确实生效**——仅设 `temperature=0` 不等于关闭 thinking。这直接影响 token cost、latency、不同条件间的预算公平性与可复现性。若模型或 region 不支持关闭：将 reasoning tokens 计入预算，先完整跑 1 个 task 用实际 usage 更新剩余预算，不直接全量启动；
- **gold block 恒定与截断控制（因果前提，不沿用 RQ4 截断逻辑）**：RQ4 的 `exposure_skill_context` 在 `--skills-total-cap`（默认 6500）触顶时直接 break，默认参数下约 5–6 个 skill 即触顶——照搬会导致 `gold+10` 与 `gold+20` 的 prompt 可能完全相同，sweep 高端失效，readiness 变化也无法归因（pollution？gold 被截？noise 没注入？）。Exp C 必须改为：(a) gold block 始终放最前，全部条件中**逐字节一致**，total cap 不得截断 gold；(b) 每个 noise skill 用固定表示与固定长度（`name + description + 正文前 N 字符`），gold 用固定完整/较长 procedural content；(c) 每条件落盘 `prompt_visible_gold_ids` / `prompt_visible_noise_ids` / `prompt_visible_gold_count` / `prompt_visible_noise_count` / `truncated_skill_count` / `actual_skill_prompt_tokens`；(d) 分析前校验：gold block hash 在全部条件一致、noise 条件严格 nested、noise=20 实际确有 20 个 noise 进入 prompt；
- **solver/judge 条件匿名化（修复 RQ4 的 label bias）**：RQ4 judge prompt 按 condition 名称输出且 metadata 含 `complete_gold_coverage`，judge 可预知哪个条件理论上应最好。Exp C 改为：条件匿名为 A/B/C/D/E，**排列用 5×5 balanced Latin square / cyclic assignment（task *i* 取第 *i* mod 5 行，seed=6002 固定行顺序），而非普通随机排列**——仅 6–12 个任务时随机排列不保证五个真实条件在 judge prompt 的 first/middle/last 位置上均衡，可能出现 `noise_20` 碰巧总在末位，judge 位置偏差与 noise level 混杂；judge 不见 `gold_all_plus_noise_20` 等名称与任何泄漏条件质量的 metadata，`best_condition` 返回匿名 ID，评分完成后离线映射回真实条件，`condition_order` 落盘供审计；solver prompt 同样只含 task instructions + visible skill guides，不含 `oracle` / `noise_20` 等标签；
- **judge-side truncation audit（与 gold block hash audit 同等重要）**：paired judge 一次要看 5 个 solver 输出 + oracle/verifier 参照，judge prompt 可能是全实验最长单条 prompt；若尾部条件被截断，固定排列也无法补救。每次 judge 调用落盘：`judge_prompt_tokens`、每个匿名条件的实际可见字符/token 数、是否发生 judge-side truncation；五个 solver outputs 必须使用**相同的截断规则与长度上限**；分析前校验无任何条件被截；
- **指标命名与 RQ4 统一**：`readiness_score`（0–10）、`likely_pass`（保守 judged 代理）、`delta_vs_no_skill`、`major_gaps`；**禁止称无修饰的 pass_rate / task success**——只有 SkillsBench sandbox verifier 的输出才可称 task pass rate；另记录 `prompt_tokens` 与 judge rationale 是否引用错误 skill（人工抽查 case study）。

**成本估算**：6–12 tasks × 5 条件 × 1 run ≈ 30–60 次 solver + 6–12 次 paired judge；参照 RQ4 dry-run（6 任务 5 条件 ≈ 174k tokens），noise sweep 会更长，预留 ≈ 300–500k token 预算，并按 `Budget_required = DryRunEstimate × 1.8`（safety factor 1.5–2.0）申请——RQ4 实测消耗为 dry-run 估计的 1.6 倍；先完整跑 1 个 task，用实际 usage 校准剩余预算后再全量。**首要约束是 Bailian 配额（RQ4 实跑已因配额中断）**，必须带 `--token-budget` 与 `--resume`。

**Primary contrasts（预先指定，不得事后挑选）**：`gold+5` vs `gold+0`、`gold+10` vs `gold+0`、`gold+20` vs `gold+0`，及 noise count 的单调趋势。**`no_skill` 只作背景基线，不作为 noise-effect 的主要 baseline**：`no_skill → gold+0` 衡量 skill 的帮助，`gold+0 → gold+noise` 才是真正的 context-pollution effect。

**统计定位**：任务数少、功效很低，Exp C 坚持作为 **descriptive pilot**：主要报告每个 task 的 noise–readiness trajectory、`score(noise_n) − score(noise_0)`、单调下降与稳健任务的计数、case-level failure mechanism；不以平均 likely-pass rate 的显著性为核心结论，任务数 ≤ 6 时不过度强调 bootstrap CI。

**外部有效性限定**：Exp C 的任务是有意 enriched 的 high-signal 多-gold 任务，利于检测效应但不代表全部任务——结论主要适用于 complex multi-skill tasks，不外推到完整 SkillsBench 分布；报告 limitations 中明确说明。若预算允许，可加少量 single-gold / 低复杂度任务作 controls（可选，不阻塞）。

### Experiment D（选做，不在关键路径）：真实 agent pass-rate 验证 protocol

**定位**：诚实性 artifact。Exp C 是 LLM-judge 文本代理而非沙箱真跑，本实验把"真跑该怎么跑"完整物化，**不伪造任何 pass rate**：环境不具备（bench/Docker/API key 缺失）时如实记录 `pass_rate_status: blocked`。


---

## 4. 统计与严谨性

- 所有随机性统一 seed=6002（与 RQ1–RQ4 一致）。
- 主统计方法：**task-clustered paired bootstrap 95% CI**（以 task 为 cluster 重采样，10k 次）——同一 task 的三个 order blocks 高度相关，**不得当作独立样本**。
- **Primary contrasts**：injected K=1 vs K=20；injected K=1 vs K=50；injected vs random（distractor construction effect）at K ∈ {20, 50}。
- **辅助模型（时间允许时）**：logistic regression / GEE：`correct ~ log2(K) * distractor_type + relative_position + similarity`，standard errors clustered by task。不用 McNemar：order blocks 按 task 聚合需引入任意的二值化决策且丢失信息，clustered bootstrap + effect size + GEE 已足够。
- 图必带误差条；报告 per-task 方差，不只报均值。
- 结果按 task 难度（gold skill 数、query 长度）分桶做细粒度分析。
- 所有 raw response 落盘 `raw_log.jsonl`（复用 `rq4_adaptive_qwen_experiment.py` 的 jsonl append + `--resume` 机制，配额中断可续跑）。

---

## 5. 交付物

```text
experiments/
  rq5_llm_chooser.py                   # Exp B（API；--dry-run 顺带产出 Exp A 附表）
  rq5_downstream_noise_sweep.py        # Exp C（API，基于 rq4_adaptive_qwen_experiment.py 改造）
  rq5_agent_passrate_protocol.py       # Exp D（选做）

data/experiments/
  rq5_llm_chooser/{raw_log.jsonl,summary.csv,summary.json,per_query.csv,position_bias.csv,exposure_stats.csv,case_studies.json,accuracy_vs_k.svg}
  rq5_downstream_noise_sweep/{solutions.jsonl,judged_tasks.jsonl,per_condition_results.csv,summary.csv,summary.json,readiness_vs_noise.svg}
  rq5_agent_passrate_protocol/{selected_tasks.csv,condition_matrix.csv,protocol_summary.json,run_commands.sh}  # 选做

docs/
  rq5_context_pollution_analysis_2026-07-XX.md   # 正式分析报告
```

**核心图表（海报/报告用）**：

1. K vs accuracy 主图：`gold_injected` / `gold_plus_random` 的 conditional accuracy 两条线 + `natural_topk` 的三指标分解（`gold_visibility_rate` × `conditional_selection_accuracy` = `end_to_end_accuracy`）+ CI；natural end-to-end 预期倒 U 或平台（H5.1），gold_injected 预期平台或下降（H5.2）；
2. K vs token cost（左轴）+ accuracy（右轴）双轴图——"成本涨、收益不涨"（H5.4）；
3. gold 相对位置 vs 选中率（位置子实验柱状图，K ∈ {10, 20}）（H5.5）；
4. 噪声数 vs `readiness_score` / `likely_pass`（Exp C pilot；RQ4 的 random +5 结果仅作历史注释标注，不并入 hard-noise 曲线——二者噪声类型不同）；
5. 案例表：gold 可见但被高相似 distractor 抢走的 3–5 个典型 case。

---

## 6. 日程（7-15 → 7-26）

| 日期 | 任务 | 产出 |
|---|---|---|
| 7-15 | 定稿本设计；控制台确认 chooser model ID | 设计定稿、模型锁定 |
| 7-16 | 写 Exp B 脚本；dry-run（顺带产出 Exp A 附表）+ 10 任务试跑 | exposure/token 附表、prompt 验证、成本实测 |
| 7-17–7-18 | Exp B 全量（temperature=0，3 order blocks）+ 位置子实验扩展位置 | chooser 全部数据 |
| 7-19 | Exp B 分析：accuracy/位置/错选来源 | 中期图表 |
| 7-20–7-21 | 改造并跑 Exp C pilot（SkillsBench 6–12 tasks，noise ∈ {0,5,10,20}，1 run，含续跑 RQ4 未完成部分可共用配额） | downstream 数据 |
| 7-22 | 统计检验（task-clustered bootstrap CI、GEE/clustered regression）+ 出全部 SVG 图 | 最终图表 |
| 7-23–7-24 | 写 `rq5_context_pollution_analysis.md` + case studies + error taxonomy + limitations | 分析报告 |
| 7-25 | 更新 README RQ5 章节；与 RQ1–RQ4 结论串联成整体叙事 | 集成 |
| 7-26 | Buffer：补跑、校对、备份数据；（若有余力）Exp D protocol；（若有余力）plus/max 档位对照 | 提交 |

**固定优先级（时间不足时从后往前砍，不牺牲排序靠前项的数据清洗、统计与报告质量）**：

1. `gold_injected_topk`（H5.2 主线）
2. `natural_topk`（H5.1 + 指标分解）
3. `gold_plus_random`（H5.3）
4. error analysis 与统计
5. Exp C pilot
6. 位置子实验扩展位置
7. Exp D、plus/max 档位对照

**风险与降级路径**：

- **Bailian 配额耗尽（RQ4 实跑已发生，首要风险）** → 所有脚本带 `--token-budget` 硬闸门 + 配额错误干净停机 + `--resume` 续跑；先跑优先级高的条件（gold_injected 主线优先于 gold_plus_random 与位置扩展），保证中断时已有可分析数据。
- 无 API key / 预算不足 → Exp B 用本地开源小模型（如 Qwen2.5-7B via ollama）替代，仍可回答 H5.1–H5.3；Exp C 降级为 RQ4 exposure 式静态代理。
- 时间不足 → 按上述优先级从后往前砍（先砍 Exp D/档位对照，再砍位置扩展与 Exp C）；Exp B（gold_injected 主线）是不可砍的核心；Exp A 附表随 Exp B dry-run 免费产出，不占独立时间。
- 结果不显著（K 大也不掉点）→ 本身即是有价值的负结果："pollution 主要来自 retrieval miss 而非 in-context 干扰"，与 RQ1 呼应。

---

## 7. 依赖清单

- **`data/raw/`（Skill-Usage 34k 与 SkillsBench）在本仓库磁盘上不存在**（gitignore 且未下载）：Exp A/B 开跑前必须先恢复 Skill-Usage 数据（query/gold mapping/skills_meta），Exp C/D 需恢复 SkillsBench tasks；`skills_full.db` 也需重建或从备份恢复。列为 **7-16 的前置任务**。
- RQ4 的 `per_task_exposure.csv`（`data/experiments/rq4_downstream_skill_exposure/`）已在本仓库，可用于任务选择与交叉验证；**但不能用于构造 Exp C 的 noise 池**（最深只存 Top-10 且含 gold，non-gold 实测仅 4–9 个），Exp C 噪声必须由脚本内重算的完整 hybrid non-gold ranking 提供（见 Exp C 噪声池构造）。
- `DASHSCOPE_API_KEY` 环境变量或 `--api-key-prompt`（Exp B/C：chooser=qwen3.6-flash，solver=qwen3.7-plus（配额受限时降为 qwen3.6-flash），judge=qwen3.7-max）；HTTP 层用 `requests` 直连 OpenAI 兼容 endpoint（沿用 RQ4 实现，不依赖 openai SDK），`--base-url` 按实际 workspace 配置。
- 本地缓存的 `sentence-transformers/all-MiniLM-L6-v2`（distractor_similarity 计算，RQ3 已缓存）。
- 选做 Exp D：还需 bench CLI + Docker + 模型 API key（当前缺失，protocol 会如实记录 blocked）。
