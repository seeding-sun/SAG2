# 跨数据集实验预案：2Wiki / HotpotQA

> 日期：2026-11-15  
> 前提：MuSiQue 全历史重打矩阵已完成；主瓶颈为**池内 gold 判 0**；最强臂 C6-H2=0.7892。  
> 数据：HippoRAG 2 钉死修订 `5ec05b38…`，已下载至 `dataset/`。

---

## 1. 三套数据长什么样（已摸底）

| | MuSiQue（已做） | 2WikiMultihopQA | HotpotQA |
|---|---|---|---|
| 题数 / 语料 | 1000 / 11656 | 1000 / **6119** | 1000 / **9811** |
| 难度轴 | id 跳数 2/3/4 | `type`：compositional 413 / comparison 244 / bridge_comparison 235 / inference 108 | `type`：bridge 811 / comparison 189；**level 全 hard** |
| 每题 gold 文档数 | 2 / 3 / 4（=跳数） | **2（765）或 4（235，bridge_comparison）** | **恒为 2** |
| gold⊂corpus | 是 | **0 missing** | **0 missing** |
| 现成 100 题分层 | `musique_100_for_ab` 40/30/30 | 无 | 无 |
| BGE / sqlite 缓存 | 有 | **无** | **无** |

要点：

- Hotpot / 2Wiki **没有 MuSiQue 那种 3–4hop 深度轴**；Hotpot 全是 2 文档桥接/比较。  
- 2Wiki 的 `bridge_comparison`（4 gold）最接近「难多跳」。  
- 语料更小 → 索引/向量比 MuSiQue 便宜。  
- Loader 已支持（`benchmark_data.py`：Hotpot 句子直接拼接，2Wiki 空格拼接）。

---

## 2. 这一轮要回答什么

MuSiQue 上已成立的说法，换数据是否还成立：

| # | 问题 | 最小证据 |
|---|---|---|
| Q1 | 全历史重打 + 六档，是否仍明显优于二值？ | 同数据 A6-H1 vs A2-H1 |
| Q2 | Dense 是否仍略优于 BM25（最终 R@10）？ | C6-H1/H2 vs A6-H1/H2 |
| Q3 | **主瓶颈是否仍是「池内判 0」而非池外漏召？** | 段落命运漏斗（never / in_pool_miss / top10） |
| Q4 | 难子集（2Wiki bridge_comparison、Hotpot bridge）是否更像 MuSiQue 4hop？ | 分层 R@10 + 漏斗 |

**先不做什么：** 全 18 格矩阵、换 DeepSeek 全量臂、上 JEV、调 RRF 权重——等 Q3 在新数据上复现后再动判定器。

---

## 3. 建议实验顺序

### Phase 0 — 工程准备（两套并行，无判定费）

1. 建 sqlite 索引：`run_sag2_benchmark.py --index-only --datasets 2wikimultihopqa hotpotqa`  
2. 建 BGE-M3 向量：`build_passage_embeddings.py`（复用现网 embedding）  
3. 固定抽 **100 题**样本文件（seed 写死，可复现）：  
   - **2wiki_100_for_ab.json**：按 type 分层，建议  
     `compositional 40 / comparison 20 / bridge_comparison 25 / inference 15`（合计 100；抬高 4-gold 难例占比）  
   - **hotpot_100_for_ab.json**：`bridge 80 / comparison 20`（贴近全库 811:189）  
4. 输出目录隔离：`output/2wiki-100/H1-rescore/`、`output/hotpot-100/H1-rescore/`（**禁止**与 musique 的 request cache 混用）

### Phase 1 — 冒烟（每库 10 题，可选）

- 臂：`A6-H1`、`C6-H1`（BM25/Dense × 六档 × 1 跳）  
- 目的：协议、模型审计、漏斗脚本是否在 Hotpot/2Wiki id 上跑通  
- 墙钟：远小于 100 题；不过夜也行

### Phase 2 — 主对比（每库 100 题，判定器固定 `qwen3.8-27b`，重打代码）

**最小必要臂（优先）：**

| 顺序 | 臂 | 作用 |
|---|---|---|
| 1 | A6-H1 | 六档 BM25 锚 |
| 2 | A2-H1 | 二值对照 → 答 Q1 |
| 3 | C6-H1 | Dense 六档 → 答 Q2（H1） |
| 4 | A6-H2 / C6-H2 | 第二跳是否还涨（六档在 MuSiQue 上 H2 仍小幅正） |

**可砍：** 先只跑到 H1（4 臂 × 2 库 = 8 跑）；H2 视 H1 增益再开。  
**可缓：** B6（hybrid）——MuSiQue 上夹在 A/C 中间，不是第一刀。

粗算（按 musique H1 ~55–60 min/臂）：  
8 臂 × ~1h ≈ **8–10 小时**；加 H2 再翻倍量级（有 H1 cache 会短不少）。

### Phase 3 — 诊断（零/低判定费）

对每库最优臂（预期 C6 或 A6）跑与 MuSiQue 同款漏斗：

- 池外 never / 池内 miss / top10  
- 池内 miss 的 grade 分布（是否仍 ≈ 全 0）  
- 按 type 分层：bridge_comparison、bridge 是否「大便」更集中  

脚本：`analyze_never_recalled.py` 需扩到 Hotpot/2Wiki 的 gold 构造（supporting_facts + context），或复用 `benchmark_data.get_question_records()` 的 `gold_docs`。

### Phase 4 — 分支（仅当 Phase 3 成立/不成立时）

| 若… | 则… |
|---|---|
| 新数据仍是「池内判 0」主导 | 判定器升级（DeepSeek API 小样本 → 再谈腾卡/JEV）；**停止**加检索臂 |
| 新数据变成「池外漏」主导 | 再开 B6/C 配额、扩大 candidate_k；对照 MuSiQue 机制差异写报告 |
| 六档相对二值无增益 | 记录「D1/重打红利数据集依赖」，Hotpot 可能更浅、二值已够 |

---

## 4. 成功标准（写进报告用）

**可支持的表述（需数字）：**

- 同代码、同判定器下，2Wiki/Hotpot 的 A6 vs A2、C6 vs A6 的 ΔR@10 与 CI。  
- 段落命运：池内 miss 中 grade&lt;4 占比；与 MuSiQue C6-H2（~48/52）对照。  

**不要预先承诺：** 「换数据就能涨分」或「Dense 一定第一」——预案只保证**可比较、可证伪**。

---

## 5. 风险与依赖

| 风险 | 处置 |
|---|---|
| 网关再次静默换模型 | 每臂 `check_model_consistency`；`strict_model` |
| embedding 服务不稳 | 向量可断点续建；先 index-only |
| Hotpot corpus 带 `idx` 字段 | Loader 只用 `title`/`text`，已兼容 |
| 100 题分层与论文 1000 题不可比 | 报告写明 seed/strata；全量 1000 留作后续 |
| GPU/判定配额 | 与 musique 共用 `qwen3.8-27b`；串行链，concurrency 保持 6 |

---

## 6. 建议你拍板的两点

1. **先做哪个库？** 建议 **2Wiki 先**（有 4-gold 难型，更像压判定）；Hotpot 跟其后。  
2. **H2 是否首轮就开？** 建议 **先 H1 四臂看漏斗**；若池内判 0 已主导，H2 对结论帮助有限，可省。

确认后按 Phase 0 → 2 开工。

---

## 7. 执行日志

| 时间 | 事项 |
|---|---|
| 2026-11-15 | 用户确认：2Wiki 优先、首轮只到 H1 |
| 2026-11-15 | Phase 0：索引 6245 chunks；BGE 向量已写；样本 `dataset/2wiki_100_for_ab.json`（seed=20261115，40/20/25/15） |
| 2026-11-15 | Phase 2 启动：`scripts/run_2wiki_rescore_h1_chain.sh` → `output/2wiki-100/H1-rescore/`（A6-H1 → A2-H1 → C6-H1），日志 `chain.log` |
| 2026-11-15 | H1 完成：A6 **0.985** / C6 0.9825 / A2 0.975；审计干净。`compare_arms.py` 仍写死 MuSiQue gold，配对报错不影响 metrics |
| 2026-11-15 | **漏斗（与 MuSiQue 相反）**：A6 250 gold → top10=246，**池内漏=0**，全程池外=**4**（2 bridge_comparison 导演名 + 2 inference 亲属链）。首轮池外约 65，扩展捞回大半。主瓶颈在此库是**召回覆盖**不是判 0。产物 `output/2wiki-100/H1-rescore/diagnostics/` |
| 2026-11-15 | 专报 `SAG2_2Wiki_H1_100Q_Report_zh.md`；启动 Hotpot：样本 `hotpot_100_for_ab.json`（bridge80/comparison20），索引 9847 chunks + BGE；`run_hotpot_rescore_h1_chain.sh` → `output/hotpot-100/H1-rescore/` |
