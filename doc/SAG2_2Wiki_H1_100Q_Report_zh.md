# 2WikiMultihopQA H1 100 题报告（全历史重打代码）

> 日期：2026-11-15  
> 样本：`dataset/2wiki_100_for_ab.json`（seed=20261115；compositional 40 / comparison 20 / bridge_comparison 25 / inference 15）  
> 判定：`qwen3.8-27b`；代码：历史全量重打（与 musique `H1-rescore` 同版）  
> 产物：`output/2wiki-100/H1-rescore/`  
> 预案：`SAG2_CrossDataset_Experiment_Plan_zh.md`

---

## 1. 目的

在 MuSiQue 上结论是「主瓶颈 = 池内判 0」。换 2Wiki 检验：

1. 六档是否仍优于二值？  
2. Dense 是否仍略优于 BM25？  
3. 剩余误差是池外漏召还是池内判 0？

---

## 2. 配置

| 项 | 值 |
|---|---|
| 检索臂 | A6=BM25×六档；A2=BM25×二值；C6=Dense×六档 |
| 跳数 | H1 only |
| candidate_k / seed_k / top_k | 100 / 10 / 10 |
| τ（六档） | 0.4；`fill_missing_score=0` |
| 语料 | HippoRAG 2 `5ec05b38…`；6119 段 → 索引 6245 chunks |
| 脚本 | `scripts/run_2wiki_rescore_h1_chain.sh` |

模型审计：三臂 batches 均为 100% `qwen3.8-27b`。

---

## 3. 结果（R@k）

| 臂 | R@1 | R@2 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| **A6-H1** | 0.4375 | 0.8600 | **0.9850** | **0.9850** |
| C6-H1 | 0.4375 | 0.8450 | 0.9775 | 0.9825 |
| A2-H1 | 0.4375 | 0.8400 | 0.9700 | 0.9750 |

- 六档相对二值：A6−A2 ≈ **+1.0pp**（天花板效应）。  
- Dense **未**超过 BM25（与 MuSiQue C6>A6 相反）。  
- 绝对水平远高于 MuSiQue 同设置（~0.76–0.79）。

注：`compare_arms.py` 仍按 MuSiQue 装 gold，对本目录配对会 KeyError；上表直接来自各臂 `metrics.json`。

---

## 4. 段落命运漏斗（250 gold）

| 命运 | A6 | C6 | A2 |
|---|---:|---:|---:|
| 进最终 top10 | **246** | 245 | 244 |
| 进过池、未进 top10 | **0** | **0** | **0** |
| 全程未进任何跳的池 | **4** | 5 | 6 |

A6 未满分仅 **4 题**，均为某条 gold **从未进池**：

| 题型 | 漏掉的 gold（标题） | 问句要点 |
|---|---|---|
| bridge_comparison | Maurice Elvey | 哪部电影导演出生更晚… |
| bridge_comparison | Duilio Coletti | 同上结构 |
| inference | William Rose Benét | Marjorie Flack 的 sibling-in-law |
| inference | Tamatoa V | Teriimaevarua III 的 paternal grandmother |

首轮 100 池约含 185/250 gold（漏 ~65）；扩展捞回大半，最终只剩 4 条池外。  
**池内漏 = 0 → 没有「在池里打成 0」的判定瓶颈。**

明细：`output/2wiki-100/H1-rescore/diagnostics/never_A6_H1.json`。

---

## 5. 与 MuSiQue 对照

| | MuSiQue C6-H2 | 2Wiki A6-H1 |
|---|---|---|
| R@10 | ~0.79 | **0.985** |
| 池外 | 17/290 | **4/250** |
| 池内未进 top10 | 52（多为 grade 0） | **0** |
| 主瓶颈 | **判定欠打** | **召回覆盖（难实体）** |

同一套重打代码 + 同一判定器，瓶颈随数据集翻转。

---

## 6. 可支持 / 不可支持

**可以支持：**

- 2Wiki 100 题分层子集上，H1 已接近饱和；六档相对二值增益很小。  
- 剩余误差全部是池外漏召；扩展有效但补不完导演名/亲属链类桥接。  
- 「MuSiQue 的池内判 0 是该数据特有主矛盾」——至少在 2Wiki 此设置下不成立。

**不能支持：**

- Dense 在 2Wiki 上优于 BM25。  
- 在此库继续堆判定器升级会明显抬 R@10（可改进空间 <2pp，且全是 never-recall）。

---

## 7. 下一步

按预案开 **HotpotQA** 同构 H1（A6/A2/C6），看瓶颈更像 MuSiQue 还是 2Wiki。
