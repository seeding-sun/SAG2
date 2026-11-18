# BridgeRAG 对齐对比实验计划

日期：2026-11-18  
目的：在**同一语料 / 同一题集 / 同一向量 / 同一大模型**下，只换「方法」，比较 BridgeRAG 式流水线 vs 我们的 dense-mq + dual。

---

## 1. 对比设计（成立）

固定：

| 项 | 取值 |
|---|---|
| 题集 | HippoRAG2 / BridgeRAG 同款 MuSiQue 评测集（约 999–1000 题；优先全量，可先 100 题冒烟） |
| 语料 | 同款 passage 库（本地 `musique_corpus.json` 已是 **11656** 篇，与 HippoRAG2 表一致；论文写的 ~21k 更像笔误/另一统计） |
| 向量 | **我们的** `bge-m3`（`192.168.2.108:8013`） |
| 大模型 | **我们的** `qwen3.8-27b`（证据/改写/judge） |
| 指标 | **R@5**（及附报 R@10）；定义 mean(\|gold∩top-k\|/\|gold\|） |

两臂（只换方法）：

| 臂 | 方法 |
|---|---|
| **A. BridgeRAG-ours-models** | 按论文：hop1 ANN → 桥 \(b\) → SVO+双实体扩池 → LLM \(s(q,b,c)\) 精排（用 27B 实现 judge/抽实体/SVO，不用 70B） |
| **B. SAG2-mq-dual** | 现有 dense-mq + dual 无闸门 + 六档（C6-MQ-H2-dual 同配置），接到**同一题集语料** |

这样比的是**方法**，不是「谁家模型更强」。  
不能宣称复现论文表里的 0.8146（那是 NV-Embed + 70B）；只能说「同资源下谁更好」。

---

## 2. 现状缺口

| 缺口 | 说明 |
|---|---|
| BridgeRAG **可运行代码** | 公开仓目前主要是 README / PDF / `eval/pilot_audit/*.json` **结果产物**，几乎没有可直接跑的 pipeline → **需按论文重实现** |
| 题集划分核对 | 确认本地 `musique.json`（1000）是否 = 他们 999；必要时按 HippoRAG2 / 产物里的 id 列表对齐 |
| \(s(q,b,c)\) 提示与批打分 | 按论文 tripartite judge 用 27B 重做；输出标量/档位需标定 |
| SVO 生成 + 实体抽取 | 2 次 LLM 调用/题，接入现有 chat API |
| 扩池 ANN | 用现有 bge-m3 向量缓存 / VectorRetriever，不必上 pgvector |
| 臂 B 全量 | 现有结果是 **100 题分层集**；要对齐需在全量（或同一 id 列表）上重跑 |

---

## 3. 建议推进顺序

1. **核对 id 列表**：用 BridgeRAG `eval/pilot_audit/*musique*` 或 HippoRAG2 公布划分，导出 `musique_bridgerag_ids.json`。  
2. **冒烟 20–30 题**：实现最小 BridgeRAG 臂（hop1 + 桥条件打分 + 简单扩池），与 C6-MQ-dual 同题对比 R@5。  
3. **扩到全量**（或 999）：两边都跑完再写报告。  
4. （可选）再加一臂「仅换 \(s(q,b,c)\)、召回仍用 mq」做消融。

### 3.1 冒烟已启动（2026-11-18）

- 样本：`dataset/musique_30_bridgerag_smoke.json`（20×2hop + 5×3hop1 + 5×4hop*）  
- 臂 A：`scripts/diagnostics/run_bridgerag_smoke.py` → `output/bridgerag-compare-smoke30/BridgeRAG-ours/`  
- 臂 B：`run_c6_mq_h1.py` dual H2 → `…/C6-MQ-H2-dual/`  
- 进度：`bash output/bridgerag-compare-smoke30/progress.sh`

---

## 4. 读数时注意

- 27B vs 论文 70B：桥条件判断可能变弱，但不影响「两臂相对高低」。  
- bge-m3 vs NV-Embed：池覆盖可能整体偏低，两臂同吃亏。  
- 若臂 A ≪ 臂 B：说明在我们模型上 dual/mq 更吃香；若臂 A ≥ 臂 B：桥段落条件打分值得迁入主线。
