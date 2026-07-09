# RQ4 真实 Agent Pass-Rate 验证协议

**日期**：2026-07-09  
**目标**：推进 RQ4 下一步，即从 skill exposure proxy 走向真实 SkillsBench agent pass rate。  
**协议脚本**：`experiments/rq4_agent_passrate_protocol.py`  
**输出目录**：`data/experiments/rq4_agent_passrate_protocol/`

---

## 1. 本次完成了什么

根据上一轮 RQ4 exposure proxy 的结果，本次完成了三件事：

1. 选择 8 个代表性 SkillsBench tasks；
2. 为每个 task 生成 5 个 agent 条件；
3. 生成可直接运行的 BenchFlow command matrix。

五个条件对应截图中的下一步：

| Condition | Meaning |
|---|---|
| `no_skill` | 不暴露任何 skill |
| `oracle_gold_all` | 暴露任务本地 curated gold skills |
| `bm25_top10` | 暴露 RQ4 proxy 中 BM25 检索到的 Top-10 skills |
| `hybrid_bm25_neural_top10` | 暴露 BM25 + MiniLM hybrid 检索到的 Top-10 skills |
| `oracle_gold_plus_5_noise` | 暴露全部 gold skills，再加入 5 个 noisy skills |

---

## 2. 代表性任务

选择任务时覆盖了多 skill composition、retrieval failure、retrieval success、不同难度和不同领域。

| Task | Difficulty | Category | Gold Skills | BM25 Top-10 Complete | Hybrid Top-10 Complete |
|---|---|---|---:|---:|---:|
| `video-silence-remover` | hard | media-content-production | 7 | 0 | 0 |
| `drone-planning-control` | medium | industrial-physical-systems | 6 | 1 | 1 |
| `fix-erlang-ssh-cve` | hard | cybersecurity | 6 | 0 | 0 |
| `financial-modeling-qa` | hard | finance-economics | 2 | 0 | 0 |
| `energy-market-pricing` | hard | industrial-physical-systems | 4 | 1 | 1 |
| `offer-letter-generator` | easy | office-white-collar | 1 | 1 | 1 |
| `react-performance-debugging` | hard | software-engineering | 2 | 1 | 1 |
| `setup-fuzzing-py` | medium | cybersecurity | 3 | 1 | 1 |

这些任务形成了一个小规模但信息量较高的 pass-rate validation set：

- 有单 skill task，也有 6-7 skill composition task；
- 有检索完整覆盖的任务，也有 Top-10 仍无法完整覆盖的任务；
- 有 easy / medium / hard；
- 覆盖 office、software engineering、cybersecurity、finance、industrial systems、media production。

---

## 3. 生成的文件

| File | Purpose |
|---|---|
| `selected_tasks.csv` | 8 个代表性任务及其 proxy 指标 |
| `condition_matrix.csv` | 8 tasks × 5 conditions 的 skill exposure 和 command matrix |
| `run_commands.sh` | 可直接执行的 `bench eval run` 命令 |
| `protocol_summary.json` | 协议 metadata、环境审计、完整条件表 |

本地还生成了 `task_packages/`，包含 8 × 5 个条件化 SkillsBench task package。但这个目录约 189MB，是可再生成产物，因此已加入 `.gitignore`，不提交到 GitHub。

---

## 4. 当前环境审计

本机当前无法直接运行真实 SkillsBench agent pass-rate：

| Requirement | Status |
|---|---|
| `bench` CLI | not found |
| `uv` | not found |
| Docker | not found |
| `OPENAI_API_KEY` | not set |
| `ANTHROPIC_API_KEY` | not set |
| `GEMINI_API_KEY` | not set |
| Codex app binary | found |

因此，本次没有报告真实 pass rate。这样处理是必要的：没有 BenchFlow runner、Docker sandbox 和模型 credentials 时，任何 pass-rate 数字都不可靠。

---

## 5. 如何运行真实 Pass Rate

在具备 BenchFlow/SkillsBench 运行环境后，执行：

```bash
bash data/experiments/rq4_agent_passrate_protocol/run_commands.sh
```

默认命令使用：

```text
--agent codex-acp
--model gpt-5
--sandbox docker
```

如果要换成 Claude 或 Gemini agent，可以重新生成协议：

```bash
python3 experiments/rq4_agent_passrate_protocol.py \
  --agent claude-agent-acp \
  --model <model-name>
```

---

## 6. 预期分析方式

真实 agent jobs 运行完成后，应从每个 job 的 result / verifier 输出中提取：

- `passed`
- `reward`
- task id
- condition
- agent/model
- token usage
- runtime

然后与 `condition_matrix.csv` 中的 proxy 指标对齐，比较：

- proxy complete coverage 是否预测 pass；
- noisy skills 是否降低 pass rate；
- `bm25_top10` 与 `hybrid_bm25_neural_top10` 的真实 pass-rate 差异；
- `oracle_gold_all` 与 `oracle_gold_plus_5_noise` 的差异。

这一步将真正回答：

**exposure proxy 是否能预测 downstream success。**

---

## 7. 当前结论

本次已经完成真实 pass-rate 实验的准备工作，但真实 pass-rate 仍被本地环境阻塞。

可以在报告中这样表述：

> We selected 8 representative SkillsBench tasks and materialized five agent conditions for each task. The generated command matrix is ready for real BenchFlow evaluation. However, the current local environment does not include the BenchFlow CLI, Docker sandbox, or model API credentials, so we do not report task success rate yet.

这比把 exposure proxy 误写成 pass rate 更稳，也为后续真正运行 verifier 留好了完整路径。

---

## 8. Reproducibility

重新生成协议：

```bash
python3 experiments/rq4_agent_passrate_protocol.py
```

主要输出：

- `data/experiments/rq4_agent_passrate_protocol/selected_tasks.csv`
- `data/experiments/rq4_agent_passrate_protocol/condition_matrix.csv`
- `data/experiments/rq4_agent_passrate_protocol/run_commands.sh`
- `data/experiments/rq4_agent_passrate_protocol/protocol_summary.json`
