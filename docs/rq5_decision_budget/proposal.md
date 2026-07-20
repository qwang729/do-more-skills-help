# RQ5 Experiment Proposal: Decision-Budget Stress Test for LLM Skill Routing

**Date**: 2026-07-20

---

## 1. Executive Summary

```
Skill library
   │
   │ ① Retrieval budget
   ▼
Retrieved candidates / Top-K
   │
   │ ② Context budget
   ▼
Skills actually exposed to the LLM
   │
   │ ③ Decision budget
   ▼
Skills selected, interpreted and applied
   │
   ▼
Task execution and final result
```

RQ1-RQ3 study whether a retriever can find relevant skills as the library grows. RQ4 studies the static skill-exposure frontier and uses a small Qwen-judged pilot to examine downstream plan readiness. A remaining systems question is located at the **decision-budget** stage:

> When all required skills are already visible, can an LLM router still identify the complete and correct skill set as the candidate menu becomes larger and more confusing?

RQ5 will isolate this stage by bypassing retrieval misses. For every task, the candidate menu will always contain all gold skills. The experiment will add a controlled number of either random or retrieval-hard non-gold skills, ask an Alibaba Cloud Qwen model to select all and only the relevant skills, and score the returned selection against the gold skill set.

The main experiment uses all available Skill-Usage tasks, supports multi-gold routing, and treats single-gold tasks as a sensitivity subgroup rather than the main dataset. The primary manipulation is the absolute number of distractors:

```text
noise_count = 0 / 2 / 5 / 10 / 20   (provisional; may be raised after the pilot, see Section 6)
```

For each observation, the analysis will also record menu size, gold fraction, prompt tokens, and latency. This design measures routing quality and cost, not actual skill execution or downstream task success.

---

## 2. Research Question

### RQ5

> **Given that all required skills are visible, how do distractor count and distractor hardness affect an LLM router's ability to select the complete and correct skill set?**

中文表述：

> **在所有必要 skills 均可见的条件下，干扰项数量和干扰项难度如何影响 LLM Router 选择完整且正确 skill set 的能力？**

### Position in the Agent Pipeline

```text
Skill library
    -> retrieval and ranking                 RQ1-RQ3: retrieval budget
    -> selected skills exposed to context    RQ4-A: context-exposure diagnostics
    -> LLM selects relevant visible skills   RQ5: decision budget
    -> skills are incorporated and executed  RQ4-B / future verifier run
```

RQ5 fixes retrieval success by injecting all gold skills. It therefore estimates routing errors conditional on gold visibility rather than repeating Hit@K or Recall@K.

### Scope Evolution from the Original Proposal RQ5

The original project proposal stated RQ5 as a context-pollution question: whether exposing more skill descriptions degrades downstream task pass rate and increases token cost and wrong skill invocation. The present design deliberately narrows that scope:

- downstream execution and task pass rate are already covered by the RQ4 paired-readiness design and are not repeated here;
- "wrong skill invocation rate" is replaced by the measurable proxy "false selection / wrong routing", because selected skills are not executed.

---

## 3. Boundary with RQ4

RQ4 and RQ5 must use different dependent variables.

| Dimension | RQ4 | RQ5 |
|---|---|---|
| Main question | Does skill exposure improve downstream plan readiness? | Can the router select the correct visible skill set? |
| LLM role | Solver | Multi-label skill router |
| Skill representation | Truncated full skill guides | Name + description only |
| Gold visibility | Varies across retrieved conditions | Guaranteed in every experimental condition |
| Output | Execution plan | Selected skill indices |
| Main metrics | Readiness score, likely-pass proxy | Precision, recall, F1, exact set match |
| Downstream execution | Not executed | Not executed |

RQ5 will not generate task solutions, use an LLM judge, or claim task pass rate. This prevents duplication with RQ4 and keeps the experiment feasible within six days.

---

## 4. Hypotheses

### H5.1: Menu-size interference

With all gold skills visible, increasing the number of non-gold candidates will reduce macro selection F1 and exact set match.

### H5.2: Distractor-hardness effect

At the same noise count, retrieval-hard distractors will produce lower selection F1 and more false selections than uniform-random distractors.

### H5.3: Under-selection and over-selection

As noise count increases, the router may respond in two different ways:

- select too few skills, increasing missing-gold count;
- select too many skills, increasing extra-skill count.

Both behaviors will be reported instead of collapsing all failures into one accuracy value.

### H5.4: Routing-cost growth

Prompt tokens and end-to-end API latency will increase with candidate-menu size. Any accuracy plateau or decline alongside higher token usage will indicate a less efficient use of the context and decision budgets.

### Exploratory H5.5: Task-complexity interaction

The effect of distractors may differ by gold-skill count. This is exploratory because task complexity is observed rather than randomized.

---

## 5. Dataset and Experimental Unit

### Primary Dataset

Use the same **Skill-Usage** task queries, gold mappings, and 34k skill metadata used by RQ1-RQ3:

```text
data/raw/Skill-Usage/data/task_queries.json
data/raw/Skill-Usage/data/task_skill_mapping.json
data/raw/Skill-Usage/skills-34k/skills_meta.jsonl
```

Expected task count: 87 tasks after intersecting query and gold-mapping keys. Exact gold-count distribution must be recomputed and written to the RQ5 experiment metadata before API calls begin.

### Current Data Blocker

At proposal time, `data/raw/` is absent from the current working tree. Restoring or relinking the previously used Skill-Usage raw data is a Day-1 prerequisite. The experiment must not silently reconstruct gold labels from RQ1-RQ3 aggregate outputs.

### Experimental Unit

The unit of analysis is a task under one routing condition:

```text
task x distractor_type x noise_count
```

The primary analysis uses all tasks, including multi-gold tasks. Single-gold tasks are reported as a pre-specified sensitivity subgroup because they provide an unambiguous single-choice case.

---

## 6. Variables

### Independent Variables

#### Noise Count

```text
n in {0, 2, 5, 10, 20}
```

##### Pilot-Based Noise-Level Adjustment Rule

The noise grid above is provisional until the 10-task pilot completes. Modern instruction-tuned models may route near-perfectly at menus of 20-30 name+description candidates, which would yield an uninformative null result across the entire matrix. Therefore:

- if the pilot shows macro F1 > 0.95 under `hard, n=20`, replace the noise grid with `{0, 10, 30, 50, 100}` before the full run;
- if the pilot shows macro F1 between 0.85 and 0.95 under `hard, n=20`, extend the grid to `{0, 2, 5, 10, 20, 50}`;
- otherwise keep the original grid.

The adjustment decision must be made once, immediately after the pilot, recorded in `experiment_metadata.json`, and never revised after full-run API calls begin. Only one final grid may appear in the primary analysis; pilot calls under a superseded grid are reported separately as pilot data.

For task `t` with `g_t` gold skills:

```text
menu_size_t = g_t + n
gold_fraction_t = g_t / (g_t + n)
noise_to_gold_ratio_t = n / g_t
```

Absolute noise count is the primary manipulation because it supports clear paired statements such as "adding 10 distractors to the same task." Gold fraction and actual prompt tokens are secondary explanatory variables that make tasks with different gold counts easier to compare.

#### Distractor Type

1. `random`: uniform random non-gold skills;
2. `hard`: highest-ranked non-gold skills from the RQ3 hybrid BM25 + MiniLM retriever.

The `n=0` condition is identical for both types and must be called only once per task.

### Controlled Variables

- all gold skills are visible in every menu;
- the same task query and gold set are used across conditions;
- one fixed Qwen model and endpoint are used for the complete run;
- temperature is 0;
- thinking mode is disabled;
- skill representation is limited to name + description;
- maximum completion length is fixed;
- random seed is 6002;
- no condition label reveals `gold`, `random`, or `hard` to the model.

### Recorded Covariates

- gold-skill count;
- total candidate count;
- gold fraction;
- actual prompt/completion/total tokens returned by the API;
- total description characters;
- request latency;
- mean and maximum distractor-gold MiniLM similarity;
- gold relative positions in the displayed menu.

---

## 7. Candidate-Menu Construction

Let `G_t` be the complete gold set for task `t` and `U` be the global skill library.

### Random Distractors

For every task, generate one deterministic random permutation of `U - G_t` using seed 6002 plus a stable task hash. Each noise level takes a prefix:

```text
random_noise_2  = random_sequence[:2]
random_noise_5  = random_sequence[:5]
random_noise_10 = random_sequence[:10]
random_noise_20 = random_sequence[:20]
```

This guarantees nested menus and prevents each noise level from receiving an unrelated random sample.

### Hard Distractors

Run the RQ3 hybrid BM25 + MiniLM retriever for the task query, remove the entire gold set, and take ranked prefixes:

```text
hard_noise_2  = ranked_non_gold[:2]
hard_noise_5  = ranked_non_gold[:5]
hard_noise_10 = ranked_non_gold[:10]
hard_noise_20 = ranked_non_gold[:20]
```

Hard distractors represent plausible retrieval competition, not stale, duplicated, or adversarial skills. Conclusions must use the term `retrieval-hard distractors` rather than claiming coverage of every real-world noise type.

### Display Order

Candidate identities and candidate order are separate variables. For the main experiment:

1. assign every candidate a deterministic pseudo-random order key based on `(task_id, skill_id, seed)`;
2. sort the current menu by that key;
3. do not display original retrieval scores or gold labels;
4. record the relative positions of all gold skills.

This yields a stable, blinded order and distributes gold positions across tasks. It does not fully eliminate order effects. If the main run finishes by July 23, add a small order-sensitivity analysis on at most 20 tasks under `hard, n=20`, using three deterministic menu permutations. This is optional and must not delay the primary run.

### Candidate Validation Invariants

Before any API call, assert that:

- every gold skill appears exactly once;
- no distractor belongs to the gold set;
- the menu contains no duplicate skill IDs;
- menu size equals `gold_count + noise_count`;
- random and hard menus are nested within their own distractor type;
- menu indices map bijectively to skill IDs.

---

## 8. Alibaba Cloud LLM Configuration

### Primary Model

Use one Alibaba Cloud Model Studio text model for the entire experiment:

```text
qwen3.7-plus
```

Qwen-Plus is selected as a balance between routing ability, throughput, and operational stability. Although monetary cost is not a constraint, using a single model avoids turning RQ5 into a model-comparison study and reduces implementation and analysis time.

If `qwen3.7-plus` is unavailable in the configured workspace, select one supported replacement before the pilot, record the exact model ID, and use that model for every condition. Do not combine results from different model IDs.

### API

Use the OpenAI-compatible Chat Completions endpoint already used by RQ4:

```text
https://{WORKSPACE_ID}.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions
```

Required settings:

```text
temperature = 0
enable_thinking = false
max_completion_tokens <= 256
stream = false
```

The API key must come from `DASHSCOPE_API_KEY` or hidden terminal input and must never be stored in experiment files.

Official references:

- [Alibaba Cloud Model Studio model overview](https://www.alibabacloud.com/help/en/model-studio/what-is-model-studio)
- [Qwen OpenAI-compatible API example](https://www.alibabacloud.com/help/en/model-studio/first-api-call-to-qwen)
- [Qwen Function Calling documentation](https://www.alibabacloud.com/help/en/model-studio/qwen-function-calling)

### Why Prompt-Based JSON Instead of Function Calling

The primary experiment will use a numbered menu and JSON response rather than registering every skill as an executable tool. Skills in this dataset are procedural documents, not guaranteed callable functions. Prompt-based multi-label selection also supports selecting several skills in one response and keeps the experimental representation consistent across menu sizes.

Function Calling can be mentioned as a future structured-router implementation, but it is outside the July 26 critical path.

---

## 9. Router Prompt and Response

### System Prompt

```text
You are a skill router for an LLM agent. Select all and only the skills that are
directly useful for completing the task. Do not solve the task. Do not select a
skill merely because it shares broad keywords with the task. Return JSON only.
```

### User Prompt Template

```text
Task:
{task_query}

Available skills:
[1] {skill_name_1}
Description: {skill_description_1}

[2] {skill_name_2}
Description: {skill_description_2}

...

Return exactly:
{"selected": [integer indices]}

Use an empty list only if none of the available skills is relevant.
```

### Response Interpretation

For ordinary Chat Completions, the selected indices are parsed from:

```text
choices[0].message.content
```

Token counts are read from the API response metadata:

```text
usage.prompt_tokens
usage.completion_tokens
usage.total_tokens
```

Latency is measured by the client around the complete HTTP request. It includes network and service queue time and must not be described as pure model inference time.

### Invalid-Response Policy

1. Retry network, timeout, 429, and retryable 5xx errors; these are API errors, not routing errors.
2. Parse strict JSON first.
3. If the response contains surrounding prose, attempt one deterministic extraction of the first JSON object.
4. Treat missing `selected`, non-integer values, or out-of-range indices as invalid.
5. Do not convert invalid responses into an empty selection.
6. Preserve the raw response, parse status, retry count, and error message.

---

## 10. Evaluation Metrics

For task `t`, let `G_t` be the gold skill set and `S_t` be the router-selected skill set.

### Primary Metrics

#### Selection Precision

```text
|S_t intersection G_t| / |S_t|
```

Defining precision as 0 when `S_t` is empty would conflate refusal (empty selection) with fully wrong selection in a single number. Precision is therefore reported in two parallel versions:

1. `precision_inclusive`: empty `S_t` counted as precision 0. This penalizes refusal and is the version used inside Selection F1 so that F1 remains defined for every task.
2. `precision_conditional`: computed only over tasks with non-empty `S_t`; empty-selection tasks are excluded from the average. This isolates selection quality given that the router selected something.

Both versions must appear side by side in the summary tables, together with `empty_selection_rate` (already a diagnostic metric), so that a drop in `precision_inclusive` can be attributed to either refusal or wrong selection. Since all gold skills are visible in every menu, an empty selection is always an error, but it is a different error mode from selecting distractors.

#### Gold Recall

```text
|S_t intersection G_t| / |G_t|
```

#### Selection F1

```text
2 * precision * recall / (precision + recall)
```

Macro F1 across tasks is the primary summary metric.

#### Exact Set Match

```text
1 if S_t == G_t else 0
```

This is the strictest complete-routing metric.

### Diagnostic Metrics

- `complete_gold_coverage`: whether `G_t` is a subset of `S_t`;
- `missing_gold_count`: `|G_t - S_t|`;
- `extra_skill_count`: `|S_t - G_t|`;
- `any_wrong_selection`: whether `S_t - G_t` is non-empty;
- `empty_selection_rate`: whether `S_t` is empty;
- `invalid_response_rate`;
- Jaccard similarity between selected and gold sets;
- mean prompt/completion/total tokens;
- median and P90 request latency.

The report will use `wrong routing` or `false selection`, not `actual wrong invocation`, because the selected skills are not executed.

### Single-Gold Sensitivity Metrics

For tasks with exactly one gold skill, additionally report:

- correct-selection accuracy;
- wrong-selection rate;
- refusal rate;
- accuracy by gold relative-position bucket.

---

## 11. Primary Contrasts and Statistical Analysis

### Pre-Specified Contrasts

Contrasts are defined relative to the frozen post-pilot noise grid. Let `n_max` be the largest noise count in the final grid (20 under the original grid, 50 or 100 under an extended grid):

1. `hard n=n_max` minus `n=0` for macro F1;
2. `random n=n_max` minus `n=0` for macro F1;
3. `hard` minus `random` at the three largest nonzero noise counts of the final grid;
4. monotonic trend across noise count for each distractor type.

### Uncertainty

Use task-level paired bootstrap with 10,000 resamples:

1. resample tasks with replacement;
2. preserve all conditions for each sampled task;
3. recompute paired metric differences;
4. report percentile 95% confidence intervals.

API calls and order permutations from the same task must not be treated as independent observations.

### Secondary Analyses

- plot F1 against absolute noise count;
- plot F1 against gold fraction;
- plot F1 and token usage together to show the decision-quality/cost frontier;
- stratify tasks by gold count: 1, 2-3, and 4+;
- inspect whether hard-distractor similarity predicts false selection;
- report per-task trajectories and representative failures.

Avoid fitting a regression containing noise count, menu size, gold fraction, and prompt tokens simultaneously without checking collinearity, because these variables are mathematically related.

---

## 12. API Call Volume and Time Scope

With 87 tasks:

```text
n=0 shared baseline:                 87 calls
4 nonzero n x 2 distractor types:   696 calls
Primary total:                      783 calls
```

If the pilot triggers the extended grid `{0, 2, 5, 10, 20, 50}`, the primary total grows to 957 calls. The optional order sensitivity adds at most 60 calls.

Monetary cost is not used to reduce the experimental matrix. Implementation time, API rate limits, failure recovery, and analysis time remain constraints. The script must therefore support:

- `--dry-run`;
- `--limit-tasks`;
- `--resume`;
- retry with exponential backoff;
- append-only JSONL logging;
- deduplication by a stable condition ID;
- a maximum-call safety argument;
- clean shutdown on quota or permission errors.

---

## 13. Implementation Plan

### Proposed Script

```text
experiments/rq5_llm_router_decision_budget.py
```

Reuse tested helpers where appropriate:

- Skill-Usage loaders and gold normalization from RQ1;
- query-overlap and embedding utilities from RQ2;
- BM25, MiniLM, and reciprocal-rank fusion from RQ3;
- Alibaba Cloud request/retry/resume patterns from RQ4.

Do not import the RQ4 solver/judge prompt construction because RQ5 has a different behavioral task.

### Required CLI

```text
--skill-usage-root
--output-dir
--model
--base-url
--api-key-env
--api-key-prompt
--noise-counts 0 2 5 10 20   # default; overridden by the frozen post-pilot grid
--distractor-types random hard
--seed 6002
--temperature 0
--max-completion-tokens 256
--limit-tasks
--max-api-calls
--dry-run
--resume
```

### Output Files

```text
data/experiments/rq5_llm_router/
  experiment_plan.csv
  experiment_metadata.json
  candidate_menus.jsonl
  raw_responses.jsonl
  per_condition_results.csv
  summary.csv
  summary.json
  selection_f1_vs_noise.svg
  exact_match_vs_noise.svg
  tokens_vs_noise.svg
  case_studies.json
```

### Analysis Document

```text
docs/rq5_decision_budget/analysis.md
```

---

## 14. Verification Checklist

### Before API Calls

- [ ] Raw Skill-Usage data restored and paths verified.
- [ ] Task count and gold-count distribution recorded.
- [ ] Candidate menus pass all invariants.
- [ ] Random and hard prefixes are nested.
- [ ] Gold labels and condition names are absent from prompts.
- [ ] Dry-run output manually inspected for at least five tasks.
- [ ] `--limit-tasks 10` pilot completed.

### After Pilot

- [ ] Pilot `hard, n=20` macro F1 checked against the noise-level adjustment rule; final noise grid recorded in `experiment_metadata.json`.
- [ ] JSON parse success rate is acceptable.
- [ ] No selected index maps outside the displayed menu.
- [ ] API `usage` fields are stored.
- [ ] Thinking mode is confirmed disabled.
- [ ] Resume does not duplicate completed conditions.
- [ ] Token and latency fields are non-negative.

### Before Analysis

- [ ] Every planned task-condition has exactly one terminal record.
- [ ] `selected_correct + selected_wrong` set calculations match raw IDs.
- [ ] Empty selection and invalid response remain separate.
- [ ] `n=0` is not duplicated across distractor types.
- [ ] Summary metrics can be regenerated from raw JSONL.

---

## 15. Threats to Validity

### Construct Validity

- Gold mappings may contain useful skills that are not all strictly necessary.
- Selecting a skill description is not equivalent to loading, invoking, or correctly applying the skill.
- Prompt-based routing differs from native tool/function calling.
- Exact set match may be overly strict for partially redundant gold skills; precision, recall, and case studies must accompany it.

### Internal Validity

- Candidate order may influence selection despite deterministic shuffling.
- Hard distractors are defined by the chosen hybrid retriever and are retriever-specific.
- Temperature 0 improves reproducibility but does not guarantee identical cloud responses across model revisions.
- Noise count, menu size, gold fraction, and token count are related and must not be interpreted as independent causal variables.

### External Validity

- The experiment uses one Alibaba Cloud Qwen model.
- Skill-Usage tasks and descriptions are not a direct OpenClaw deployment.
- The largest menu in this experiment remains much smaller than the full 34k library because the Router receives retrieved candidates, not the entire library.
- Results concern routing decisions, not downstream task success.

### Operational Risks

- Raw data is currently absent and must be restored first.
- API rate limits or endpoint permissions may interrupt the full run.
- Invalid JSON and partial responses may require deterministic repair rules.
- A model or endpoint change during the run would invalidate direct condition comparisons.

---

## 16. Expected Figures and Tables

1. **Macro F1 vs noise count**, with random and hard distractor curves and paired-bootstrap confidence intervals.
2. **Exact set match vs noise count**.
3. **Precision and recall vs noise count**, showing over-selection and under-selection separately; precision is plotted in both inclusive and conditional versions alongside the empty-selection rate.
4. **Prompt tokens vs noise count**, optionally paired with F1 as a cost-quality frontier.
5. **Single-gold sensitivity table** with correct, wrong, refusal, and invalid rates.
6. **Case-study table** showing query, gold skills, selected skills, hard distractors, and error type.

---

## 17. Claim Boundaries

RQ5 may support claims such as:

- larger visible skill menus reduce multi-label routing quality;
- retrieval-hard distractors are more damaging than random distractors;
- routing cost increases while decision quality plateaus or declines;
- multi-gold tasks exhibit under-selection or over-selection under noisy menus.

RQ5 must not claim:

- actual wrong skill invocation, unless selected skills are executed;
- downstream task-success degradation;
- performance of OpenClaw itself;
- effects of outdated, duplicate, or malicious skills;
- universal behavior across LLM providers or model families.

The intended conclusion is conditional and systems-oriented:

> Even after retrieval succeeds, a finite decision budget can limit how reliably an LLM agent routes among visible procedural memories. Controlled candidate menus are therefore necessary not only for retrieval efficiency, but also for reliable skill selection.

---

## 18. Definition of Done

RQ5 is complete only when:

1. the raw dataset and exact task/gold distribution are documented;
2. candidate menus are validated and reproducible;
3. all primary conditions have terminal API records or explicitly documented failures;
4. raw responses, token usage, latency, and selected IDs are preserved;
5. multi-gold metrics and single-gold sensitivity results are generated;
6. paired uncertainty estimates and required plots are produced;
7. the analysis distinguishes routing selection from invocation and downstream success;
8. the final noise grid and the pilot-adjustment decision are documented in the experiment metadata;
9. README, final report, and poster contain the scope-evolution paragraph explaining the narrowing from the original context-pollution RQ5 to the decision-budget routing RQ5;
10. README and final report state the final scope without overstating RQ5 evidence.
