# HotpotQA H1 100 题报告（全历史重打代码）

> 日期：2026-11-15～16  
> 样本：`dataset/hotpot_100_for_ab.json`（seed=20261115；bridge 80 / comparison 20）  
> 判定：`qwen3.8-27b`；与 musique / 2Wiki 同版全历史重打  
> 产物：`output/hotpot-100/H1-rescore/`  
> 预案：`SAG2_CrossDataset_Experiment_Plan_zh.md`

---

## 1. 结果（R@k）

| 臂 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| **A6-H1** BM25×六档 | 0.470 | 0.960 | **0.985** |
| **C6-H1** Dense×六档 | 0.470 | 0.980 | **0.985** |
| A2-H1 BM25×二值 | 0.480 | 0.935 | 0.955 |

审计三臂均为 100% `qwen3.8-27b`。六档相对二值约 **+3.0pp**；A6 与 C6 的 R@10 打平。

---

## 2. 段落命运（200 gold，每题恒 2）

| 命运 | A6 | C6 | A2 |
|---|---:|---:|---:|
| 进最终 top10 | **197** | 197 | 191 |
| 进过池、未进 top10 | **2** | 1 | 6 |
| 全程未进池 | **1** | 2 | 3 |
| R@10&lt;1 的题 | 3 | 3 | 9 |

A6 首轮池内 gold ≈ 190/200；扩展后 AE@10 = **97/100**。

### A6 三道未满分

| 题 | 命运 | 漏掉的 gold |
|---|---|---|
| PokerFace host from where? | **NEVER** | `Ant &amp; Dec`（标题实体/编码） |
| Boltzmann machine 同名… | **POOL_MISS**，全程 grade 0 | `Ludwig Boltzmann` |
| actor son of Jeremiah Porter | **POOL_MISS**（无有效 grade 记录） | `Ben Cook (actor)` |

C6 未满分另有：`Verve (1937–60)`、`Spider-Man (pinball)` 等易混淆同名段；与 A6 **无交集**——两边漏的不是同一批。

---

## 3. 与 2Wiki / MuSiQue

| | MuSiQue C6-H2 | 2Wiki A6-H1 | Hotpot A6-H1 |
|---|---|---|---|
| R@10 | ~0.79 | 0.985 | **0.985** |
| 池外 / 池内漏 | 17 / **52** | **4** / 0 | **1** / **2** |
| 主瓶颈 | **池内判 0** | 池外难实体 | 近饱和；残余 = 偶发池外 + 极少判 0 |

Hotpot 更接近 **2Wiki**（天花板），不像 MuSiQue。

---

## 4. 结论

- Hotpot 100 题上 H1 已近饱和；继续加跳/换判定器预期收益 &lt;2pp。  
- 残余误差以 **bridge 题的难实体/同名混淆** 为主，夹杂个别池内 grade 0。  
- 明细：`output/hotpot-100/H1-rescore/diagnostics/`。
