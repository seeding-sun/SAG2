# D1 分级评分臂：20 题验收报告

日期：2026-11-12
实验臂：`D1`（分级 rubric + 种子阈值，不填充）
对照臂：`A1-H1`（BM25 + Qwen 0/1 二值判定，种子补满 10 个）

---

## 1. 本轮要解决的问题

前两轮 100 题实验（`SAG2_H0_100Q_Report_zh.md`、`SAG2_H1_100Q_Report_zh.md`）暴露了一个结构性缺陷：Qwen 的证据判定退化成了常数。

实测 `A1-H1` 的 20,789 次判定里，**98.3% 给 0.0**，平均每题只有 1.6~1.9 个候选拿到非零分。而 `search_sag2` 的种子选择是：

```python
frontier = [c for c in ranked() if c["id"] not in expanded][:seed_k]
```

这个切片**不做任何分数过滤**。于是：

- 每题 10 个种子席位里，8 个左右是 0 分填充（100 题上 839/1000，即 83.9%）；
- 这 8 个 0 分种子照常参与扩展词提取，`TF * IDF * (1 + context) * seed_weight` 里 `score/max_score = 0`，但词本身的 TF·IDF 还在，等于**用无关段落的高频词去污染扩展查询**；
- hop 2 的历史候选排序是 `(-jev_score, id)`，在全员 0 分时退化为**按 id 字典序**取 80 条。

所以问题不是"Qwen 判得不准"，而是**判定的分辨率不足 + 空位被噪声填满**。

本轮验证的假设：把二值判定换成 6 档评级，并规定"低于阈值不要、也不填充"，能否消除填充噪声并改善排序。

---

## 2. 方法改动

### 2.1 判定刻度：0/1 → 6 档 rubric

新增 `GRADE_RUBRIC`（`scripts/sag2.py`）：

| 档位 | 含义 |
|---:|---|
| 0 | 跑题，或谈的是问题所指之外的实体 |
| 2 | 主题相关但无可用的具体事实（只有关键词重叠） |
| **4** | **提供答案链上的一段事实，但需要另一段段落来识别实体或补全该步** |
| 6 | 明确确立链条的一个中间步骤，但单独不能回答问题 |
| 8 | 提供关键的桥接或回答事实，仅剩轻微歧义 |
| 10 | 直接且完整地提供链条上必要的一环，无歧义 |

关键设计：**阈值 4 = "上链"边界**。4 分的意思是"这段在链上，但自己不够用"，正是 SAG2 需要的桥接证据；0/2 分是"不在链上"，不能当种子。

### 2.2 两层分离：10 分制不外泄

```
第 1 层（模型输出）: grade ∈ {0,2,4,6,8,10}   字段名 grades
第 2 层（对外接口）: jev_score = grade/10 ∈ [0,1]
第 3 层（trace）  : 额外记录 grade 原值
```

下游（`ranked()`、DPP kernel、扩展词权重、报告脚本）全部只读 `jev_score`，**零改动**。`CRITERION` 一字未动，因为 `JevClient` 和本地 reranker 的 true/false 路径还在用它。

### 2.3 不填充

`search_sag2` 新增 `grade_threshold`（默认 `0.0`，即不过滤）：

```python
frontier = [
    c for c in ranked()
    if c["id"] not in expanded and c["jev_score"] >= grade_threshold
][:seed_k]
```

不足 `seed_k` 就少给，甚至给 0 个。0 个种子时循环在下一次扩展前 `break`，不会空转。hop 2 的 `previous` 同样过滤，并顺带修掉了缺失的 `-fusion_score` 次级排序键（原来 tie 全靠 `id`）。

### 2.4 为什么默认值都是旧行为

`scoring_scale="binary"` 与 `grade_threshold=0.0` 两个默认值下，所有既有路径逐字节等价。这让改动可以安全落地：A1/B1/C1 的历史结果一个都没动。

### 2.5 代码改动清单

| 文件 | 改动 |
|---|---|
| `scripts/sag2.py` | `GRADE_RUBRIC` / `GRADE_CRITERION` / `GRADE_THRESHOLD` 常量；`grade_threshold` 参数；frontier 与 hop-2 `previous` 过滤；hop-2 tie 补 `-fusion_score`；trace 记录 `grade`、`previous_eligible_count`、`eligible_count` |
| `scripts/qwen_evidence_scorer.py` | `parse_grades`（只接受 6 个整数锚点，拒浮点）；`scoring_scale` 参数；grade 模式下 `jev_score=grade/10` 并保留 `jev_grade` |
| `scripts/run_hybrid_compare.py` | 接出 `--scoring-scale`、`--grade-threshold`，并写入 `config.json` |
| `tests/test_grade_threshold.py` | 新增 6 项：默认不填充、阈值丢弃、阈值高于全部分数时跳过扩展、hop-2 过滤、hop-2 默认不过滤 |
| `tests/test_qwen_evidence_scorer.py` | 新增 10 项：6 档各值、拒浮点/字符串/越界/错信封、删位修复、binary 向后兼容 |
| `scripts/diagnostics/diagnose_grades.py` | 新诊断脚本，所有指标从 `details.jsonl` 重算并自校验 |

提交：

```text
7fcf236  Repair single deleted digit in Qwen candidate keys   （前置，让哈希可追溯）
d6bc3f9  Add graded scoring scale and seed threshold to SAG2
0f5cfe2  Add grade-scale acceptance diagnostics and 20-question results
0f08787  Expose hop-2 history eligibility in the trace and test it for real
```

测试：`126 passed`（110 原有 + 16 新增），`ruff` 全过。

---

## 3. 实验设置

### 3.1 样本

`dataset/musique_20_for_grades.json`，20 题，按父样本 `musique_100_for_ab.json` 的 40/30/30 比例分层取 **8/6/6**（2hop/3hop/4hop），且**从 A1 的同源 100 题中按 A1 运行顺序抽取**。

这一点很关键：样本与 A1 完全重叠，因此可以做**逐题配对**比较，而不是比较两个不同题集的平均值。

### 3.2 两个配置

| 臂 | 打分刻度 | 阈值 | batch | 目的 |
|---|---|---|---|---|
| `A1-20` | 0/1 | — | 10 | 基线，复现 A1 |
| `D1-b10` | 6 档 | 0.4 | 10 | **与 A1 同 batch，隔离"刻度改动"本身** |
| `D1-b100` | 6 档 | 0.4 | 100 | 检验批大小对校准的影响 |

其余参数全部固定：`--max-hops 1 --candidate-k 100 --seed-k 10 --top-k 10 --retrieval-mode bm25`，与 A1-H1 一致。

### 3.3 成本

| 臂 | 判定次数 | 请求数 | 输入 token | 输出 token | 墙钟 |
|---|---:|---:|---:|---:|---:|
| A1-20 | 4,103 | 420 | 1,363,210 | 47,859 | 803 s |
| D1-b100 | 4,134 | **60** | **729,812** | 50,476 | **707 s** |

单批打分的**请求数少 7 倍、输入 token 少 46%**。原因是 `shared_context`（原问题 + 上轮证据）在每个 batch 里重复一次，10 批就要重复 10 次。这是批大小的直接收益，与刻度无关。

---

## 4. 结果

### 4.1 结构性指标（决定性）

| 指标 | A1-20（0/1） | D1-b10 | D1-b100 | 判定 |
|---|---:|---:|---:|---|
| 实际使用档位数 | 2 | **6** | **6** | 通过 |
| hop-0 非零占比 | 1.85% | 15.5% | 4.75% | — |
| 每题种子数（均值） | 10.00（全填） | **6.95** | **3.50** | 见 §5.3 |
| 填充席位（score < 0.4） | **163 / 200（81.5%）** | **0 / 139** | **0 / 70** | 通过 |
| 无种子题 | 0% | 0% | 0% | 通过 |
| 种子精度（含填充） | 16.50% | **23.74%** | **50.00%** | 通过 |

hop-0 档位直方图（每臂 2,000 个候选）：

| 档位 | 0 | 2 | 4 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| `D1-b10` | 1691 | 122 | 110 | 49 | 14 | 14 |
| `D1-b100` | 1905 | 16 | 40 | 25 | 6 | 8 |

A1 的对照分布：`{0.0: 1963, 1.0: 37}`——2,000 个候选里只有 37 个非零。

### 4.2 检索指标（重算，逐题配对）

| 臂 | R@1 | R@2 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| `A1-20` | **0.2833** | 0.4917 | 0.6292 | 0.6542 |
| `D1-b10` | 0.2292 | 0.4583 | 0.6708 | 0.6875 |
| `D1-b100` | 0.2417 | **0.5125** | **0.7000** | **0.7583** |

逐题变化（vs A1）：

| 臂 | R@10 | 提升题 | 下降题 | 持平 |
|---|---:|---:|---:|---:|
| `D1-b10` | 0.6875 | 5 | 3 | 12 |
| `D1-b100` | 0.7583 | 5 | 2 | 13 |

### 4.3 批大小的排序一致性

`D1-b10` vs `D1-b100` 的 hop-0 逐题 Kendall τ：**均值 0.9835，最小 0.8404，低于 0.8 的题 0/20**。

### 4.4 自校验

`diagnose_grades.py` 从 trace 重放排序，再与 `retrieved_docs` 反查回 document id 的集合比对。`rank_dpp` 只重排 `candidate_k` 前缀，不改变集合，因此集合必须完全一致：

```text
D1-b10 : 20/20
D1-b100: 20/20
A1-20  : 20/20
```

三个 run 全部通过，§4.1 与 §4.2 的所有重算指标建立在正确的重放逻辑上。

---

## 5. 分析

### 5.1 "不填充"是机制上被验证的，不是统计上的巧合

A1 的 200 个种子席位里 163 个是 0 分填充（81.5%），与 100 题上的 83.9% 一致。D1 两个配置的填充席位都是精确的 **0**。这是确定性计数，不受采样波动影响。

### 5.2 一个重要反直觉发现：A1 的"敢说 1"其实并不差

只看 A1 中**真正过阈值的 37 个种子**，精度是 **67.6%**（25/37），比 D1-b100 的 50% 还高。

所以收益的来源要说准：

- **不是**"分级判断比二值判断更准"；
- **而是** A1 每题只敢说 1.85 个候选是证据，剩下的 8.15 个席位被 0 分噪声占满，这些噪声**直接进入扩展词提取**；
- D1 把"没有证据"显式化为"0 个/少量种子"，避免噪声污染扩展查询。

这解释了为什么 `H1_100Q_Report` 里"扩展轮没有放大召回优势"——扩展查询本身被污染了。

### 5.3 阈值 0.4 不是跨批次不变量

`batch=10` 与 `batch=100` 的 τ = 0.9835，相对排序几乎不变，但绝对分数整体下移：

- 过阈值候选：9.35/题 → 3.95/题（少 58%）
- 0 分占比：84.6% → 95.3%

单批打分更"严格"也更极化。因此**换批大小必须重新标定阈值**，0.4 只在 batch=10 下经过标定。

这也说明我原定的验收线"每题 ≥5 个种子"设计错了：`batch=100` 只有 3.50 个种子，却拿到最高 R@10。种子数量不是目标，**种子质量与下游 Recall** 才是。

### 5.4 检索指标的收益集中在 R@5/R@10，R@1/R@2 反而下降

| 臂 | R@1 变化 | R@2 变化 | R@5 变化 | R@10 变化 |
|---|---:|---:|---:|---:|
| `D1-b10` | −0.054 | −0.033 | +0.042 | +0.033 |
| `D1-b100` | −0.042 | +0.021 | +0.071 | +0.104 |

这不是噪声的形状——两个配置都是"浅层变差、深层变好"。合理机制：种子变少后扩展词变少，召回覆盖面变宽（R@10 升），但精确的头部排序没有得到同等收益；同时 `ranked()` 在分数拉开后，`fusion_score` 的权重相对下降，可能让 BM25 排序得更前的段落掉出 top-1。

R@1 下降是一个**真实代价**，需要在 100 题上确认是否持续。若持续，说明需要调 DPP 或引入分数与融合分的显式组合，而不是简单用 `jev_score` 主导排序。

### 5.5 唯一不能声称的事：Recall 提升不显著

`D1-b100` vs `A1-20` 的 R@10 差值 +0.1042，配对 Bootstrap（2000 次）95% CI = **[−0.0208, +0.2375]**，含 0；符号检验 p = 0.227（5 升 / 2 降 / 13 平）。

20 题只有 7 题发生变化，**样本量决定了不可能得到显著结论**。能够确定的只有结构性的部分：档位数、填充归零、种子精度、以及"扩展查询不再被 0 分噪声污染"这个机制。

---

## 6. 结论

### 6.1 可以支持的表述

> 把证据判定从 0/1 换成 6 档 rubric 并禁止种子填充后，种子席位中的 0 分噪声从 81.5% 降到 0%，种子精度从 16.5% 提升到 23.7%（batch=10）/ 50.0%（batch=100）；20 题上 R@10 从 0.6542 提升到 0.6875/0.7583，但该提升在 n=20 下不显著。

### 6.2 不能支持的表述

> 分级评分使 SAG2 显著优于二值判定。

> 阈值 0.4 是通用最优阈值。

> 批大小不影响结果。

### 6.3 与已有结论的关系

`H1_100Q_Report` 的结论是"扩展轮没有放大初始召回优势，反而抹平了它"，并把原因归于 seed 质量（当时 seed precision 18%）。本轮提供了机制层的解释：**seed 里有 81.5% 是 0 分填充，它们直接进入扩展词**。

该 20 题子集上 BM25 初始召回的 R@5/R@10 是 0.4417/0.4667，三个臂都大幅超过它，所以"扩展有用"这一点本身没有争议；争议在于扩展**放大得够不够**。修复填充后，`D1-b100` 把 R@10 从 A1 的 0.6542 推到 0.7583（+0.104），是第一个在 20 题上明显拉开差距的配置。这是对上一轮结论的补充，不是推翻。

### 6.4 遗留问题

1. **R@1/R@2 下降**：两个配置一致出现，需在 100 题上确认。
2. **阈值未在 batch=100 下重标定**：可考虑 0.3，但本轮坚持 0.4 是为了隔离刻度改动。
3. **0 分占比仍然很高**（84.6% / 95.3%）：说明 rubric 的 2 分档几乎没被使用，可能档位描述需要再调，或 Qwen 对"主题相关但无事实"的辨别力有限。

---

## 7. 产物位置

```text
dataset/musique_20_for_grades.json                     验收样本（20 题）
output/smoke20-grades/D1-b10/                          配置一结果
output/smoke20-grades/D1-b100/                         配置二结果
SAG-Benchmark/sag2/scripts/diagnostics/diagnose_grades.py   诊断脚本
doc/SAG2_Scoring_Scale_Redesign_zh.md                  设计与全部决策记录
```

---

## 8. 复现命令

```bash
cd SAG-Benchmark/sag2
set -a; . ./.env; set +a

# 配置一：与 A1 同 batch，隔离刻度改动
.venv/bin/python scripts/run_hybrid_compare.py \
  --retrieval-mode bm25 --dataset musique --dataset-dir ../../dataset \
  --sample-file ../../dataset/musique_20_for_grades.json \
  --output-dir ../../output/smoke20-grades/D1-b10 \
  --max-hops 1 --candidate-k 100 --seed-k 10 --top-k 10 \
  --scoring-scale grade --grade-threshold 0.4 \
  --evidence-batch-size 10 --evidence-concurrency 4 \
  --cache-dir .cache/sag2

# 配置二：单批打分
.venv/bin/python scripts/run_hybrid_compare.py \
  --retrieval-mode bm25 --dataset musique --dataset-dir ../../dataset \
  --sample-file ../../dataset/musique_20_for_grades.json \
  --output-dir ../../output/smoke20-grades/D1-b100 \
  --max-hops 1 --candidate-k 100 --seed-k 10 --top-k 10 \
  --scoring-scale grade --grade-threshold 0.4 \
  --evidence-batch-size 100 --evidence-concurrency 4 \
  --cache-dir .cache/sag2

# 诊断（含自校验与配对比）
.venv/bin/python scripts/diagnostics/diagnose_grades.py \
  --run ../../output/smoke20-grades/D1-b100 \
  --baseline /path/to/A1-20 \
  --compare ../../output/smoke20-grades/D1-b10 \
  --threshold 0.4
```
