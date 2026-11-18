# n100 MQ±dual 与 BridgeRAG 冒烟结果（归档）

日期：2026-11-17 ~ 2026-11-18  
说明：完整 `output/`（含 details/requests）体量大且在 `.gitignore`；本页归档**指标与结论**。本地产物路径见各节。

---

## 1. n100：dense-mq ± dual × 二值/六档 × H2

样本：`dataset/musique_100_for_ab.json`  
模型：`qwen3.8-27b` + `bge-m3`  
产物：`output/mq-dual-100/`（`run_chain_h2_four.sh`）  
H0/H1 来自同次 H2 的 `hop_metrics`。

| 臂 | H0 R@10 | H1 R@10 | H2 R@10 | H2 R@5 |
|---|---:|---:|---:|---:|
| C6-MQ-H2 | 0.6517 | 0.7750 | 0.7883 | 0.7183 |
| **C6-MQ-H2-dual** | 0.6650 | 0.8275 | **0.8358** | **0.7642** |
| C2-MQ-H2 | 0.6375 | 0.6900 | 0.7242 | 0.6692 |
| C2-MQ-H2-dual | 0.6400 | 0.6550 | 0.7008 | 0.6450 |

读数：六档 dual **+4.8pp**（H2）；二值 dual **−2.3pp**。增益主要在 H0→H1。

墙钟（metrics.wall_seconds）：C6-MQ ~2.5h；C6-dual ~9.6h；C2 ~2.3h；C2-dual ~5.7h。

---

## 2. BridgeRAG 对齐冒烟（30 题）

样本：`dataset/musique_30_bridgerag_smoke.json`  
固定语料/bge/27B；只换方法。  
产物：`output/bridgerag-compare-smoke30/`、`GATE_smoke30.md`

| 臂 | R@5 | R@10 | 约秒/题 |
|---|---:|---:|---:|
| BridgeRAG-ours（论文结构 + 我们模型） | 0.733 | 0.814 | ~9 |
| **C6-MQ-H2-dual** | **0.858** | **0.922** | ~270 |

题级 R@5：dual 8↑ / BridgeRAG 2↑ / 20 平。  
论文宣称 BridgeRAG ~4–5s/题（70B/A100）；不能直接对表 0.8146。

dual smoke hop（R@10）：H0 0.778 → H1 0.903 → H2 0.922。

---

## 3. 相关文档

- 桥接实验日志：`SAG2_Bridge_Scoring_Experiment_Log_zh.md`（含冻结池、rerank §11）  
- 阶段结论：`SAG2_Bridge_Phase_Conclusion_and_n100_Plan_zh.md`  
- 对齐计划：`SAG2_BridgeRAG_Matched_Compare_Plan_zh.md`  
- 代码：`scripts/hop_anchor.py`、`multi_query_retriever.py`、`diagnostics/run_c6_mq_h1.py`、`run_bridgerag_smoke.py`、冻结池/重排探针等
