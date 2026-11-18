# SAG2 桥接打分：阶段结论 + n100 计划 + 相关论文

日期：2026-11-17

---

## 1. 阶段结论（先记清楚）

### 1.1 收敛点

| 决策 | 内容 |
|---|---|
| 召回 | **dense-mq**（分解子问并行 Dense + RRF）+ hop1 followup，保留 |
| 打分主臂 | **dual 无闸门**：`max(原问, 自填子问)`，不再默认叠门 |
| 停做 | 再调抽取提示 / 冲突门 / 逐段 verifier / 槽值语义校验当主线 |

### 1.2 证据摘要

| 实验 | 结果 |
|---|---|
| never11 dual vs MQ | **0.477 vs 0.417（+6.1pp）**；有单题回撤 |
| 门/verifier/语义 | 稳但 ≤ gated 0.432；修不好 Sazerac 分数 |
| 冻结池三枪 | 原问 0.364 < 自填 0.455 < oracle 0.523（池覆盖 0.765） |
| n30 dual vs MQ | **0.778 vs 0.772（+0.6pp）**；4↑24平2↓ |
| 冻结池 bge 重排 | 原问 0.386；自填 0.364（**弱于 JEV 自填**）；oracle 0.530≈JEV；远低于 0.77 |

**一句话：** 子问打分方向对；门家族性价比低；easy 切片增益小；难切片仍有空间。当前「改门/改提示」线视为到顶。**裸 cross-encoder 重排不能吃掉池内选型缺口**（见实验日志 §11）。

细节见 `SAG2_Bridge_Scoring_Experiment_Log_zh.md`。

---

## 1.3 n100 + BridgeRAG 冒烟结果（2026-11-18）

详见 **`SAG2_n100_MQ_Dual_and_BridgeRAG_Smoke_Results_zh.md`**。  
摘要：C6-MQ-H2-dual H2 R@10 **0.836**（vs MQ 0.788）；冒烟上 dual R@5 0.858 > BridgeRAG-ours 0.733，但 dual 慢约 30×。

---

## 2. n100 大样本计划（已完成）

样本：`dataset/musique_100_for_ab.json`（与旧矩阵相同）。  
模型：`qwen3.8-27b`。

「这一套」= **dense-mq ± dual 无闸门** × **二值 / 六档**，各跑 **一次 H2**（`max_hops=2`）：

| 臂名 | 召回 | 评分 | dual |
|---|---|---|---|
| `C2-MQ-H2` | dense-mq | 二值 | 否 |
| `C2-MQ-H2-dual` | dense-mq | 二值 | 是（无闸门） |
| `C6-MQ-H2` | dense-mq | 六档 τ=0.4 | 否 |
| `C6-MQ-H2-dual` | dense-mq | 六档 τ=0.4 | 是（无闸门） |

共 **4 臂**，不是 12。H0/H1 是同一次扩展里 hop0/hop1 的中间排名，写入 `metrics.json` 的 `hop_metrics`（`H0_Recall@10` / `H1_…` / `H2_…`），**不另开 H0/H1 重跑**。  
产物：`output/mq-dual-100/`；链：`run_chain_h2_four.sh`。  
说明：旧矩阵曾按 H0/H1/H2 **分三次**跑是历史对照习惯；本轮按「跑完 H2 即覆盖」处理。

### 2.1 旧矩阵 H0 基线（已有，先记）

来源：`output/27b-100/H1/*-H0-qwen3.8-27b/metrics.json`（与全矩阵报一致）。

| | BM25 (A) | Hybrid (B) | Dense (C) |
|---|---:|---:|---:|
| **二值 H0** | 0.6017 | 0.6517 | **0.6633** |
| **六档 H0** | 0.6008 | 0.6533 | 0.6542 |

重打后 A2/A6-H0 不变（0.6017 / 0.6008）——首轮池未变。mq-dual 的 H0 跑完后填下表对照 C-H0。

---

## 3. 相关论文（初检）

与「多跳召回分解 / 桥文档不能只按原问打分」直接相关：

| 工作 | 要点 | 链接 |
|---|---|---|
| **IRCoT** (Trivedi et al., 2023) | 检索与 CoT 交错；用推理句当下一跳查询 | https://arxiv.org/abs/2212.10509 |
| **MDR** (Xiong et al., 2021) | 多跳 Dense：查询表示条件于原问+已检索段落 | https://arxiv.org/abs/2009.12756 |
| **BridgeRAG** (2026) | 二跳应按 \(s(q,b,c)\) 桥条件打分，而非只 \(s(q,c)\)；MuSiQue 上 training-free R@5 提升明显 | https://arxiv.org/abs/2604.03384 |
| **Verification + Decomposition** (2026) | 多跳上按原问验段会失败；应对照**子问**做 entailment（MuSiQue 金分解有大幅提升） | https://arxiv.org/html/2608.00585 |
| **Semantically Aligned RAG** (2025) | 子问 Dense 召回 + **原问 cross-encoder 重排**，过滤「子问有效但与主问无关」的段 | https://arxiv.org/abs/2508.09755 |
| **DaPT** (2026) | dual-path 分解图 + 子问填槽推进 + Judge 校验；强调分解图与路径融合 | https://arxiv.org/html/2603.19097 |
| **GRITHopper** | 无显式分解的多跳 Dense 另一路线 | https://arxiv.org/abs/2503.07519 |
| **RI²VER / Inter-Passage Verification** (ACL 2025 Findings) | 答案候选再拆成验证子问并二次取证过滤假阳性 | https://aclanthology.org/2025.findings-acl.354/ |

**和我们结论的对齐：**  
文献普遍支持「桥/中间跳不能只拿原问当相关度」——与冻结池 oracle/自填抬分一致；也支持「分解质量决定上限」——与自填 < oracle、错槽回撤一致。BridgeRAG / 子问条件验证与我们 dual 锚思路同族，但实现细节不同（他们更强调 bridge passage 条件，我们是子问字符串再打分）。Semantically Aligned RAG 的「子问召回 + 原问重排」与我们 dense-mq + dual 评分互补：一个偏覆盖过滤，一个偏评分抬桥。

**补充（§11 实测）：** 同冻结池上裸 `bge-reranker-v2-m3(q,doc)` 不能逼近池覆盖 0.77，且自填 max 重排弱于 JEV dual——印证「相关度 ≠ 链上证据效用」；若再试重排，应对齐 BridgeRAG 式条件化，而非直接上原问/子问 CE。
