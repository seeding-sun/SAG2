# MuSiQue 两大瓶颈复盘 + 跨数据集对照

> 日期：2026-11-16  
> 范围：全历史重打代码下的 musique / 2Wiki / Hotpot 100 题实验  
> 相关：`SAG2_Rescore_and_Judge_Probe_Report_zh.md`、`SAG2_2Wiki_H1_100Q_Report_zh.md`、`SAG2_Hotpot_H1_100Q_Report_zh.md`

---

## 0. 一句话

MuSiQue 从始至终卡在两件事上：

1. **召不回**：gold 从未进入任何跳的候选池  
2. **池里捞不到**：进了池，判定给 0（或长期 &lt;4），进不了 top10  

换 Dense/Hybrid、做全历史重打，主要是在这两条线上挪数字；**2Wiki / Hotpot 几乎碰不到第 2 条**，所以分数贴天花板——不能用来否定 MuSiQue 上的判定问题。

---

## 1. 跨数据集总表（同判定器 `qwen3.8-27b`，重打代码）

| 数据集 | 代表臂 | R@10 | 池外 never | 池内漏 | 主瓶颈 |
|---|---|---:|---:|---:|---|
| **MuSiQue** | C6-H2 | 0.789 | 17/290 | **52/290** | **池内判 0** |
| 2Wiki | A6-H1 | 0.985 | 4/250 | **0** | 偶发池外 |
| Hotpot | A6/C6-H1 | 0.985 | 1–2/200 | 1–2/200 | 近饱和 |

首轮就注定差距（BM25/Dense 初始）：

| | Init R@10 | 首轮全中 | gold 标题≈出现在问句 |
|---|---:|---:|---:|
| Hotpot | 0.855 | 0.90 | ~50% |
| 2Wiki | 0.675 | 0.48 | ~60% |
| MuSiQue | 0.507 | 0.37 | ~**29%** |

Hotpot/2Wiki：浅多跳、题面常点名实体、语料更小。  
MuSiQue：深桥接、题面缺中间实体 → 检索难 + 判定更难。

---

## 2. MuSiQue：两条线从 H0 走到 H2

固定样本 `musique_100_for_ab`（290 gold）。命运定义：

- **never**：全程任何跳的召回/打分池都没出现  
- **miss（池里捞不到）**：进过池，最终未进 top10  
- **miss_g0**：miss 且全程 max grade &lt;4（含全 0）

### 2.1 六档 BM25：扩展在补「召不回」，捞不出「判 0」

| 臂 | R@10 | top10 | never | miss | 其中 grade&lt;4 |
|---|---:|---:|---:|---:|---:|
| A6-H0 | 0.601 | 165 | **98** | 27 | 27 |
| A6-H1 | 0.762 | 210 | 37 | 43 | 41 |
| A6-H2 | 0.778 | 216 | **30** | **44** | 39 |

读法：

- H0→H1：never 98→37（**扩展把池外捞进来**），R@10 +16pp。  
- H1→H2：never 再降一点；**miss 不降反升**（37→44）——更多金进了池，仍判不上。  
- 池内漏几乎全是低分：H2 上 39/44 的 max&lt;4。

### 2.2 换检索：少漏池外，多堆池内漏

| 臂 | R@10 | never | miss | miss 中 &lt;4 |
|---|---:|---:|---:|---:|
| A6-H2 BM25 | 0.778 | 30 | 44 | 39 |
| B6-H2 Hybrid | 0.784 | 25 | 45 | 44 |
| **C6-H2 Dense** | **0.789** | **17** | **52** | **49** |
| A2-H2 二值 | 0.712 | 33 | 61 | 61 |

- Dense：never 最少（17），其中 **14/17 是 4hop**。  
- 但 miss 最多（52），且 **49 条 max&lt;4**；约 46 条在某跳 **已有 evidence 仍低分**。  
- 账：少漏池外 8（相对 B）≈ 多池内漏 7 → 净 top10 只 +1（此前已拆过）。  
- **换检索解决不了「池里捞不到」。**

### 2.3 两条线的跳数分布（C6-H2）

| | 2hop | 3hop | 4hop |
|---|---:|---:|---:|
| never | 1 | 2 | **14** |
| miss | 8 | 17 | **27** |

4hop 两条线都最重：既召不齐，召进来也判不上。

### 2.4 机制对应（实验史上已验证的）

| 问题 | 机制 | 有效干预 | 无效/有限 |
|---|---|---|---|
| 召不回 | 题面缺实体；桥接段词面/语义离查询远 | Dense 覆盖↑；扩展补召回（H0→H1 主增益） | 等权 RRF 不保证优于 Dense |
| 池里捞不到 | hop0 无 evidence 打 0；扩展后仍把桥接打 0；τ 曾挡重打 | **全历史重打**让六档重新压过二值（A6-H2 0.778） | 改 criterion prompt；单纯加跳；换 hybrid |

判定器探测（同专报）：27B 上改文案无效；DeepSeek 小样本能抬桥接分 → **第 2 条更像模型能力**。

---

## 3. 「从始至终」时间线（只保留与两瓶颈相关）

1. **初始召回**：BM25 池内 gold ~1.9/题；池外就是后天 never 的来源。  
2. **二值 vs 六档（旧实现）**：六档丢 0 分历史；二值 H2「赢」靠盲猜 leftover。  
3. **全历史重打**：给桥接第二次带上下文的机会 → 六档回来；但大量桥接 **有 evidence 仍 0**。  
4. **Dense/Hybrid 重打矩阵**：never↓、miss↑ 或持平 → 瓶颈从「外面」转到「里面」。  
5. **2Wiki/Hotpot**：never/miss 都接近 0 → 暴露 MuSiQue 特有的深桥接难度。

---

## 4. 现在还剩什么（MuSiQue，按优先级）

**P0 — 池里捞不到（量更大，C6 上 ~52）**  
- 对象：有 evidence、仍 grade 0 的桥接段  
- 方向：更强判定器；**本地量化 deepseek-v4-flash 已否决为整臂替换**（探针能抬桥接，整臂 R@10 下降；细档/抬 τ 无效）。详见判定专报 §5.5。  
- 27B 上 **六档略优于 0–10 十档**（n30 C6-H1：0.756 vs 0.733）。  

**P1 — 召不回（C6 上 ~17，三路交集 ~15）**  
- 对象：4hop 题面无实体的中间段  
- 方向：查询改写 / 多查询；再堆单路 Dense 边际已薄  
- 探针 v1：iterative 7/17（门禁未过）。  
- **探针 v2（27B 改写 + 种子追问）**：union **10/17** 进检索池（过门禁）。三臂含义见判定专报 **§5.6**（`decomp_only` / `iterative` / `seed_followup` + RRF）。  
- **已接入** `dense-mq` hop0 + hop1 dense followup；never11：池内 never 金 **0→9/17**，top10 **1/17**，R@10 持平——捞进池后仍判不上。产物 `output/probe-never-multiquery-v2-27b/`。  
- **子问锚点判定（§5.7）**：oracle 金分解 5/10≥4；裸 retrieve 锚否决。  
- **dual-anchor**：无闸门 **0.417→0.477**；§5.8 gated **0.432**（挡歧义，未追上无闸门；HQ→Frankfort 仍漏）。见 §5.8.6。  

**不必优先**  
- 2Wiki/Hotpot 上再开 H2 或换判定器（可改进空间很小）  
- 在 MuSiQue 上继续扩 hybrid 矩阵当主线  
- 本地 deepseek 整臂替换 / 27B 改十档  

---

## 5. 产物索引

| 内容 | 路径 |
|---|---|
| MuSiQue 重打 + 判定探测（含 §5.5 deepseek/十档） | `doc/SAG2_Rescore_and_Judge_Probe_Report_zh.md` |
| 2Wiki H1 | `doc/SAG2_2Wiki_H1_100Q_Report_zh.md` |
| Hotpot H1 | `doc/SAG2_Hotpot_H1_100Q_Report_zh.md` |
| 跨库预案 | `doc/SAG2_CrossDataset_Experiment_Plan_zh.md` |
| MuSiQue 漏斗 JSON | `output/27b-100/H1-rescore/diagnostics/` |
| 2Wiki / Hotpot 漏斗 | `output/2wiki-100/.../diagnostics/`、`output/hotpot-100/.../diagnostics/` |
| 本地 deepseek / 十档对照 | `output/smoke10-judge-ds/` |
| never 多查询探针 | `output/probe-never-multiquery/` |
