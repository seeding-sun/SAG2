# SAG2 历史全量重打 + 判定器探测报告

> 日期：2026-11-14～11-15  
> 代码分界：`845e354`（`search_sag2` 每跳对全历史重打）  
> `sag2_code_sha256`：`74610b95a7b21e422f149c50cfc3d458d6dbeae059496bc6fb8ad2cfab08c19e`  
> 相关：工程记录 §22；现状文档 §0.11；旧矩阵 `SAG2_38b_Full_Matrix_Report_zh.md`（重打前，不可与本轮混比）

---

## 1. 动机

两件独立但接续的事：

1. **上下文变了却不重打**：hop0 无 evidence 时桥接 gold 常被打成 0；hop≥1 旧实现只重打种子/protected，六档 hop2 又用 τ 把 0 分历史挡在门外。二值 H2「赢」六档，很大一块是 τ=0 把 leftover 留在旧池里盲猜。
2. **判定器是否欠打**：重打后仍有大量「池内 gold 最终未进 top‑10」。怀疑 rubric/prompt 与「用 evidence 定位桥接」冲突，以及 `qwen3.8-27b` 本身是否偏严；用 criterion A/B 与 DeepSeek 小样本探测。

论文侧 JEV 为云 API；本轮未部署本地 JEV。服务器盘够（`/data` ~18T）、GPU 几乎占满，起不了额外大判定实例。

---

## 2. 改动摘要（上下文重打）

| 项 | 旧行为 | 新行为 |
|---|---|---|
| hop≥1 打分池 | protected + 种子 + 新召回（~110） | **`latest` 全历史 ∪ 本轮新召回** |
| hop2 | 旧池几乎只要 `score≥τ` + 新 20 | **历史全量重打** + 新召回仍 ≤20 |
| τ | 同时卡谁能进重打池 / frontier | **只卡 frontier / `previous_eligible_count`**，不再决定谁进重打池 |

工程细节与测试：`SAG2_Engineering_Incidents_and_Code_Changes_zh.md` §22。  
**旧 `output/27b-100/H1/*` 与新代码不可混比；hop≥1 禁止复用旧 request cache。**

---

## 3. smoke10（机制验收）

目录：`output/smoke10-rescore/`，模型 `qwen3.8-27b`，同 10 题对旧矩阵 H2。

| 臂 | 旧 H2 R@10 | 新 R@10 | Δ | judgments |
|---|---:|---:|---:|---:|
| A6-H2 | 0.6500 | **0.7833** | **+13.3pp** | 4760 |
| A2-H2 | 0.7000 | 0.7250 | +2.5pp | 5200 |

hop0⊆hop1 = 1.0，hop1 池 = 200 → 全历史重打生效。小样本上六档收益远大于二值。

---

## 4. 100 题重打：A6 / A2 × H0–H2

- 目录：`output/27b-100/H1-rescore/`
- 脚本：`scripts/run_38b_rescore_ab_chain.sh`（日志 `chain.log` / `h0_chain.log`）
- 检索：**仅 BM25**（本轮未重跑 B/C）
- 判定：`qwen3.8-27b`，各臂模型审计 **100%**
- 样本：`musique_100_for_ab.json`（与旧矩阵相同）

### 4.1 总表（R@10）

| 臂 | H0 | H1 | H2 | H0→H1 | H1→H2 |
|---|---:|---:|---:|---:|---:|
| **A2** 二值 | 0.6017 | 0.7183 | 0.7117 | **+11.67pp** | −0.67pp |
| **A6** 六档 | 0.6008 | **0.7617** | **0.7783** | **+16.08pp** | **+1.67pp** |

同臂相对**旧矩阵**（重打前代码）：

| 臂 | 旧 R@10 | 新 R@10 | Δ |
|---|---:|---:|---:|
| A2-H1 | 0.6717 | 0.7183 | +4.67pp |
| A2-H2 | 0.7075 | 0.7117 | +0.42pp |
| A6-H1 | 0.7142 | 0.7617 | +4.75pp |
| A6-H2 | 0.7083 | **0.7783** | **+7.00pp** |

H0 与旧矩阵一致（0.6017 / 0.6008）——首轮池未变，差异来自扩展轮重打。

### 4.2 配对（同目录、同代码；baseline = A2-H1）

摘自 `chain.log`：

| 对比 | Δ R@10 | 95% CI | Holm p |
|---|---:|---|---:|
| A6-H1 − A2-H1 | +0.0433 | [+0.006, +0.082] | 0.091 |
| A6-H2 − A2-H1 | **+0.0600** | [+0.017, +0.104] | **0.024** |
| A2-H2 − A2-H1 | −0.0067 | [−0.031, +0.020] | 1.000 |

### 4.3 分跳（R@10）

| 臂 | 2hop (n=40) | 3hop (n=30) | 4hop (n=30) |
|---|---:|---:|---:|
| A2-H1 | 0.8250 | 0.7444 | 0.5500 |
| A2-H2 | 0.8375 | 0.7222 | 0.5333 |
| A6-H1 | 0.9000 | 0.7556 | 0.5833 |
| A6-H2 | 0.9000 | 0.7778 | 0.6167 |

### 4.4 判定量

| 臂 | judgments |
|---|---:|
| A2-H0 / A6-H0 | 10,000 |
| A2-H1 | 29,940 |
| A6-H1 | 29,503 |
| A2-H2 | 51,880 |
| A6-H2 | 43,926 |

### 4.5 本轮结论（重打）

**可以支持：**

- 全历史重打后，**六档重新压过二值**：A6-H2 0.7783 > A2-H1/H2；相对 A2-H1 的 +6.0pp Holm p=0.024。
- 增益主要在 **H0→H1**；A6 的 H1→H2 仍小幅正（+1.7pp），与旧矩阵六档 H2 掉点相反。
- 二值 H2 相对 H1 **无增益**（−0.67pp）——与「二值靠旧池盲猜」的旧叙事一致被削弱。

**不能支持 / 尚未做：**

- 重打代码下 Hybrid/Dense 是否仍翻转（B/C 未重跑）。
- 「换更强本地 JEV」的全量臂（仅有小样本 API 探测，见下节）。

---

## 5. 判定器 / Prompt 探测

脚本：`scripts/diagnostics/probe_grade_criterion_ab.py`  
设定：从 A6-H2 重打结果中抽样「池内 gold、有 evidence、跑分 &lt;4」的桥接段 + 非 gold 对照；同一上下文，只换 criterion 文本。

- **A**：现行 `GRADE_CRITERION`（强调勿把 evidence 事实算到候选上）  
- **B**：明确允许用 evidence **定位**链上一步，候选本身有可用事实即可 ≥4  

### 5.1 `qwen3.8-27b`（`output/probe-criterion-ab/`，n=60：bridge 30 + control 30）

| | bridge ≥4 | control ≥4 | B&gt;A（bridge） |
|---|---:|---:|---:|
| grade_run（原跑） | 0/30 | 0/30 | — |
| criterion A | 4/30 | 0/30 | — |
| criterion B | 3/30 | 0/30 | **0**（A&gt;B=1） |

**结论：改 prompt 几乎抬不动 27B 上的桥接 0 分；对照保持 0，说明不是乱抬分。**

### 5.2 DeepSeek（云 API，`output/probe-criterion-ab-deepseek/`，n=20：bridge 10 + control 10）

| | bridge ≥4 | control ≥4 |
|---|---:|---:|
| grade_run（27B 原跑） | 0/10 | 0/10 |
| DeepSeek A | **7/10** | 0/10 |
| DeepSeek B | **7/10** | 0/10（有 3 条 control=2，无 ≥4） |

A/B 几乎无差；**模型换掉后桥接 ≥4 从 0/10 → 7/10**。

### 5.3 更难子集（`output/probe-deepseek-mini/`，n=10：bridge 5 + control 5）

| | bridge ≥4 | control ≥4 |
|---|---:|---:|
| DeepSeek A | **1/5** | 0/5 |

难例上 DeepSeek 也紧，但对照仍干净。

### 5.4 探测结论（早期）

| 假设 | 证据 |
|---|---|
| 「改 criterion 就能救 27B 欠打」 | **不支持**（A/B 无提升） |
| 「更强判定器能抬桥接分」 | **方向支持**（官网 DeepSeek 7/10），样本小，非正式臂 |
| 「本地部署 JEV/大判定」 | 盘够、**卡不够**；论文 JEV 非本机权重 |

### 5.5 本地量化 `deepseek-v4-flash` 整臂对照（2026-11-16）

网关模型：`deepseek-v4-flash`（**本地量化**，非官网 API）。Scorer 侧对 deepseek 固定 `reasoning_effort=none` + `thinking=false`，并修复偶发裸数字 id（缺 `candidate_` 前缀）。

#### 5.5.1 探针（仍支持「能抬桥接欠分」）

| 设定 | bridge ≥4 | control ≥4 |
|---|---:|---:|
| C6-H2 池内低分桥接 30 + 对照 30（`probe-c6-deepseek-v4-flash/`） | A **8/30**，B **10/30** | **0/30** |
| 早先 A6 抽样 10+10（`probe-criterion-ab-deepseek-v4-flash/`） | A **4/10** | **0/10** |

对照干净；相对 27B 同协议（约 4/30）有提升，但弱于官网 flash 的 7/10。

#### 5.5.2 整臂 smoke / n30：R@10 变差

设定：dense × 六档或细档 × H1 × τ=0.4（除非另注）。基线为同题子集上的 `C6-H1-qwen3.8-27b`。

| 臂 | n | R@10 | vs 27B |
|---|---:|---:|---:|
| 27B 六档 | 10 | **0.775** | — |
| deepseek 六档 | 10 | 0.592 | −18.3pp |
| deepseek 0–10 细档 | 10 | 0.600 | −17.5pp |
| 27B 六档 | 30 | **0.756** | — |
| deepseek 0–10 细档 τ=0.4 | 30 | 0.675 | −8.1pp |
| deepseek 0–10 细档 τ=0.6 | 30 | 0.600 | −15.6pp |

典型掉分机制（非「金均分低于非金」）：金均分仍远高于非金，但 **≥4 / ≥6 高分段被非金淹没**。例：Olivera/Serbian 题上 deepseek hop0 eligible 11（27B 仅 2），top10 被 Phoenix 德台、Plymouth、一串「官方语言国家」占满，金被挤出。

n=30 分位纯度（段最大分）：

| | ≥4 金占比 | ≥6 金占比 | hop0 eligible 均值 |
|---|---:|---:|---:|
| 27B 六档 | 42% | 54% | 3.2 |
| deepseek 细档 τ=0.4 | **19%** | **38%** | 6.1 |
| deepseek 细档 τ=0.6 | — | — | 2.0（R@10 更差） |

**抬 τ「加强筛选」反而更差**：τ=0.6 砍掉大量只到 4–5 的真桥，却留住已到 6+ 的噪声。  
**加档 ≠ 自动变严**：τ 仍 0.4 时，进种子条件仍是 ≥4；打到 6 的干扰细档后往往仍是 6。

#### 5.5.3 27B：六档 vs 0–10 十档（n=30，C6-H1，τ=0.4）

| | R@10 | R@5 | hop0 eligible | 逐题 |
|---|---:|---:|---:|---|
| **六档** | **0.756** | 0.714 | 3.2 | — |
| 十档 | 0.733 | 0.683 | 2.7 | 2↑ / 26平 / 2↓ |

十档 ≥4 金占比略升（42%→51%），但 R@10 **−2.2pp**。  
**结论：27B 上维持六档；十档无端到端增益。**

#### 5.5.4 判定器线小结

| 结论 | 证据 |
|---|---|
| 官网 DeepSeek 探针能抬桥接欠分 | 7/10 |
| 本地量化 flash **探针能抬、整臂变差** | 高分段不纯 → 乱扩 |
| 细档 / 抬 τ 救不回本地 deepseek 整臂 | n10/n30 上仍显著低于 27B |
| 27B 六档优于十档（小样本） | n30 −2.2pp |

产物目录：`output/smoke10-judge-ds/`（含 `summary.json`、`summary_n30.json`、`summary_27b_g6_vs_g10_n30.json`）。

---

## 6. 产物与复现

```text
output/smoke10-rescore/
output/27b-100/H1-rescore/
  A{2,6}-H{0,1,2}-qwen3.8-27b/
  B{2,6}-H{1,2}-qwen3.8-27b/
  C{2,6}-H{1,2}-qwen3.8-27b/
  chain.log / h0_chain.log / bc_chain.log
  diagnostics/never_all_ABC_H2.json
  diagnostics/never_C6_H2_fixed.json
  diagnostics/miss_C6_H2_fixed.json
output/probe-criterion-ab/
output/probe-criterion-ab-deepseek/
output/probe-criterion-ab-deepseek-v4-flash/
output/probe-c6-deepseek-v4-flash/
output/probe-deepseek-mini/
output/smoke10-judge-ds/
  C6-H1-deepseek-v4-flash/
  C6-H1-deepseek-v4-flash-g10/
  C6-H1-deepseek-v4-flash-g10-n30/
  C6-H1-deepseek-v4-flash-g10-n30-tau06/
  C6-H1-qwen3.8-27b-g10-n30/
output/probe-never-multiquery/
output/probe-never-multiquery-v2/
output/probe-never-multiquery-v2-27b/
  C6-MQ-H1-never11/
output/probe-subq-judge/
output/probe-subq-judge-v2/
output/probe-subq-judge-oracle/
output/probe-never-multiquery-v3/
```

### 5.6 Never 改写 + 多路召回（2026-11-16）

#### 5.6.1 要解决什么

C6 上约 **17** 条 gold **全程从未进任何跳的候选池**（never）。题面是多跳，中间桥文档的标题/用词和原问对不上，单路 Dense/BM25 用原问打就捞不着。  
目标先验证：**把原问改成多条可检索子问，多路召回再合并，能否把 never 金捞进池**（先不要求进 top10）。

#### 5.6.2 总流程（抓住对象）

```
原问
  → LLM 拆成有序子问（可带 {1}/{2} 占位）
  → 三种用法各自「多路 dense top-k」
  → 每路内部用 RRF（1/(60+名次)）合成一份候选列表
  → 探针再把三臂结果 RRF 成 union
```

**RRF 不是模型加权**：同一文档被越多路命中、名次越靠前，合并分越高；切 top-N 即该臂召回池。

代码：`scripts/multi_query_retriever.py`；探针：`scripts/diagnostics/probe_never_multiquery.py`；整臂：`scripts/diagnostics/run_c6_mq_h1.py`（hop0=`dense-mq`，hop1=BM25 扩展 ∪ dense 种子追问）。

#### 5.6.3 三种臂分别干什么

| 臂名 | 一句话 | 具体做法 |
|---|---|---|
| **decomp_only** | 拆完就盲搜，占位不填 | 检索词 = 原问 + 全部子问**原文**（含未替换的 `{1}`）。各路 dense → RRF。后跳子问常因占位未填而几乎废查。 |
| **iterative** | 链式填空再搜 | 子问1 搜 → LLM 从 top 段抽短实体 → 填进子问2 的 `{1}` → 再搜 → …；各步结果 RRF。对「必须先认出桥实体」的题最有效。 |
| **seed_followup** | 用已召回种子写下一跳短 query | 拿 hop0 高分/frontier 段落当种子，LLM 写 3–5 条**具体名词短语**追问（不复述整题），各路 dense → RRF。正式臂挂在 **hop1**。 |

探针额外报告 **union** = 上述三臂池再 RRF 一次（谁捞到算谁）。

**真题对照**（Casa Loma）：

- 原问：`… castle … birthplace … performer of Falling Out`  
- 子问：谁唱 Falling Out → `{1}` 出生地 → `{2}` 有什么城堡  
- `iterative` 填成 Serena Ryder → Millbrook → 第 3 路 top 出现 **Casa Loma**（进池）  
- `decomp_only` 第 3 问仍带 `{2}`，Casa Loma **未进**  
- `seed_followup` 种子已是专辑页，追问偏「出生地」，本例未直接捞到城堡  

#### 5.6.4 检索探针数字（27B 改写，17 never 金）

| 臂 | 金进池 |
|---|---:|
| baseline（原 C6 池） | 0/17 |
| decomp_only | 3/17 |
| iterative | 6/17 |
| seed_followup | **8/17** |
| **union** | **10/17**（过 ≥8 门禁） |

本地 deepseek-v4-flash 改写 union 仅 7/17，改写侧仍用 27B。

#### 5.6.5 整臂接入（C6-MQ-H1，never11 题）

| | C6-H1 | C6-MQ-H1 |
|---|---:|---:|
| R@10 | 0.417 | 0.417 |
| never 金进池 | 0/17 | **9/17** |
| never 金进 top10 | 0/17 | **1/17** |

**结论：** 多路召回解决的是「捞进池」；进池后 27B 仍几乎抬不进 top10 → 回到 P0 **池内判定**。下一刀试：**判定锚点改为子问（原问只占辅分）**，见 §5.7。

产物：`output/probe-never-multiquery/`、`…-v2/`、`…-v2-27b/`（含 `GATE.md`、`C6-MQ-H1-never11/`）。

### 5.7 子问锚点判定 + 改写优化（2026-11-16）

**动机：** 多路已把 never 金捞进池，但用**整句原问**打分仍全是 0。假设：桥文档应对齐「填槽后的子问」，原问只占辅分。

#### 5.7.1 v1：对着我们的 iterative 子问打分

空 evidence；27B 六档；union 进池 **10** 金 + **22** distractor。`probe_subq_judge.py`。

| | 金 n=10 | distractor n=22 |
|---|---:|---:|
| ≥4 占比 orig | **0%** | 27% |
| ≥4 占比 `sub_max`（iter 填槽） | **20%**（2/10） | 23% |
| ≥4 占比 blend07 | **20%** | 18% |

抬起：`Sno*Drift`、`Oh Yeah` 专辑。其余仍 0。

#### 5.7.2 v2：对着「捞到它的那条 query」打分

失败尸检：很多金是 **seed_followup** 捞进池的，v1 却只拿 iterative 子问打 → 锚点打错。  
`probe_subq_judge_v2.py` 增加 `follow_max` / `retrieve_max` / evidence 补打。

| | 金 ≥4 | distractor ≥4 |
|---|---:|---:|
| orig | 0% | 27% |
| `all_max`（iter∪follow∪捞金 query） | **仍 20%**（还是那 2 条） | **50%** ↑ |

**否决「裸 retrieve 锚」当整臂规则：** 金几乎不再涨，脏文档被任意子问抬爆。Evidence cascade 也没再救金。

#### 5.7.3 Oracle：对着 MuSiQue **官方金分解** 打分

`probe_subq_judge_oracle.py`（用 gold `question_decomposition` 填槽后的 hop 问句）。

| | ≥4 |
|---|---:|
| orig | 0/10 |
| **oracle 本 hop 子问** | **5/10** |

抬起的典型：

| 金 | 错锚（我们的） | 对锚（官方） | 分 |
|---|---|---|---:|
| Casa Loma | castle in **Millbrook** | castle in **Toronto** | 0→10 |
| Sacramento 史 | cities bordering Sacramento | work location of **Samuel Brannan** | 0→6 |
| Saudi Arabia | region north of Middle East | when was Kingdom of SA **established** | 0→10 |

仍 0 的（即便 oracle）：数据集金页与 hop 不对齐的噪声（如 cocktail 页 `Sazerac` 对 HQ 问；`Ahmed Salah Hosny` 对 Mohamed Salah 队籍）。

**结论：27B 判定在「子问正确」时可用；当前瓶颈是改写/抽槽保真，不是再换判定模型。**

#### 5.7.4 改写加固试了一刀（v3）→ 召回回退

收紧 extract + hop 查询变体后重跑 never 探针：

| | v2 | v3 |
|---|---:|---:|
| iterative 进池 | **6**/17 | 3/17 |
| union 进池 | **10**/17 | 8/17 |

Casa Loma 链：v3 第一步抽空 / 出生地抽成 Houston，更差。  
**已回退有害的变体检索；保留「自然语言分解 + followup 显式关系问」提示。召回仍以 v2 为准。**

#### 5.7.5 总读数（优化方向）

1. 多路召回：继续用 v2（进池有效）。  
2. 判定换锚：方向对，但**必须先有正确子问**；错锚 + retrieve 锚会抬脏。  
3. 下一刀应打在 **抽槽保真**（出生地勿用专辑文案；人/机构抽出后强制 `place of birth` / `headquarters` 二次检索），再用「正确子问」重打判定；不要再裸上 retrieve-max。

#### 5.7.6 已接入整臂：dual-anchor + verify（never11）

实现：`hop_anchor.py`（`HopAnchorPlanner` + `DualAnchorScorer`）；`search_sag2(..., hop_planner=)`；`run_c6_mq_h1.py --dual-anchor`。

行为：

- 每跳打分：`max(对原问, 对本 hop 已填子问)`（同 evidence 包）  
- hop 结束后：从高分段抽槽；对人/机构关系强制二次 Dense（`place of birth` / `headquarters` 等）再抽  

| | C6-MQ-H1 | **+dual max + verify** |
|---|---:|---:|
| R@10（never11） | 0.417 | **0.477（+6.1pp）** |
| 逐题 | — | 3↑ / 7平 / 1↓ |

↑ 例：`Oh Yeah` 链抽出 Charles Mingus → Arizona → Phoenix（0.25→0.75）。  
↓ 例：McAfee/Sazerac 题（0.67→0.33）——子问锚仍可能抬错段。  

**读数：** 这是第一次在 never 硬切片上，换锚真正抬动端到端 R@10；可继续在 miss 切片 / 更大样本验证，并收紧「何时启用 hop 锚」（例如仅当抽槽非空且子问无未填占位）。

产物：`output/probe-never-multiquery-v2-27b/C6-MQ-H1-dual-never11/`、`GATE_dual.md`、`arm_compare_dual_never11.json`。  
另：`probe-subq-judge/`、`probe-subq-judge-v2/`、`probe-subq-judge-oracle/`、`probe-never-multiquery-v3/`。

### 5.8 方向对齐与修正规格（2026-11-16）

#### 5.8.1 问题链（已实验对齐）

| 阶段 | 结论 |
|---|---|
| 正式臂痛点 | 子问只参与**召回**，打分锚死盯 `original_question` → 桥文档进池仍 grade 0 |
| MQ 召回 | never 金进池 ↑，R@10 持平（判定仍卡） |
| 裸换子问打分 | 子问对才抬；错锚/捞金 query 锚抬脏（否决） |
| Oracle 金分解 | 5/10≥4 → **判定可用**，瓶颈在填槽 |
| 无闸门 dual-max | never11 **+6.1pp**，但 Sazerac↓：错实体加固 |

填槽统计（dual 跑）：约 **对 9 / 错 6 / 空 25**。  
对：池内有显式单值关系句；错：歧义/近义串台/同名；空：无关系句或前环已断。

#### 5.8.2 两种「子问」必须分开

| 种类 | 是什么 | 用途 |
|---|---|---|
| **静态分解子问** | 开题就拆好，无依赖中间答案 | **仅召回**（MQ / decomp） |
| **动态桥接子问** | `{实体 E} + {关系 R}`，由上一跳 E 与本跳 R 模板生成 | **仅打分锚**（可进 max） |

禁止把 LLM 自由长句当打分锚（emb/语义漂移、关系改歪）。

#### 5.8.3 打分融合（方案 A，已采用语义）

```
s1 = grade(候选 | 原问, 证据)
s2 = grade(候选 | 桥接子问, 证据)   # 仅闸门通过时
score = max(s1, s2)                 # 不是向量逐维 max
```

含义：能答总问 **或** 能答本跳桥关系 → 应留在 frontier。  
原问支路**永不关闭**；子问只是加分探索。

#### 5.8.4 三道准入闸门（防错链加固）

全部满足才启用 s2，否则 `score = s1`：

1. **实体置信** `E.jev_score ≥ τ_e`（默认 0.6）  
2. **实体消歧**：已知歧义名 / 二次检索与初抽冲突 → 禁用  
3. **关系单值**：`birth_place` / `headquarters` / `capital_of` / `manufacturer` / `performer` 等允许；`highest_scorer` / 历任 `member` / 多值邻接等**禁止**

#### 5.8.5 下一版实现要点

1. 桥问结构化：`{E} {relation_phrase}?` 固定短语表  
2. 闸门 + 日志：`s1/s2`、是否启用、E、R、拒因  
3. 召回仍 MQ；双 frontier 保底可第二步（先用「未过闸=只用原问」近似）

代码入口：`hop_anchor.py`（gated）、`run_c6_mq_h1.py --dual-anchor --bridge-gate`。

#### 5.8.6 Gated 整臂对照（never11，已落地）

| | C6-MQ-H1 | dual 无闸门 | **gated §5.8** |
|---|---:|---:|---:|
| R@10 | 0.417 | 0.477 | **0.432** |
| vs MQ | — | +6.1pp | **+1.5pp** |
| 逐题 vs MQ | — | 3↑7平1↓ | **2↑8平1↓** |

实现要点：模板桥问；三闸门；hop0 抽槽池 pad fusion+baseline；标题命中置信地板 0.7；专辑标题 fallback。  
闸门 accept/reject 见 `GATE_gated.md`。

**读数：**

- 规格方向正确：射手/member 等歧义关系被拒；Oh Yeah 能抽出 Mingus→Arizona 并 `gate_accept`。  
- 尚未追上无闸门 +6pp：H1 深度不够走完 capital 链；**headquarters 仍放行 Frankfort**（Sazerac↓仍在）。  
- 下一刀曾试冲突+逐段 verifier（实验 I）：never11 **仍 0.432**，Sazerac 仍 0.33 → **失败**；见 `SAG2_Bridge_Scoring_Experiment_Log_zh.md` §5。
- 当前刀（实验 J）：**槽值语义校验**（强总部/出生表述可单篇 accept；弱表述或打架 reject；不要求多源篇数）。

产物：`C6-MQ-H1-gated-never11/`、`GATE_gated.md`、`arm_compare_gated_never11.json`；I/J 详见桥接实验日志。

```bash
cd SAG-Benchmark/sag2
.venv/bin/python -m scripts.diagnostics.check_model_consistency \
  ../../output/27b-100/H1-rescore/C6-H2-qwen3.8-27b
.venv/bin/python scripts/diagnostics/compare_arms.py \
  --baseline ../../output/27b-100/H1-rescore/A6-H2-qwen3.8-27b \
  --arm ../../output/27b-100/H1-rescore/C6-H2-qwen3.8-27b
.venv/bin/python scripts/diagnostics/analyze_never_recalled.py   # 需用 1-based doc_id 修正版逻辑；见 §8 数字
```

---

## 7. 对下一轮的含义（AB 完成后、BC 前的表述；已被 §8 修订）

1. 全历史重打后六档压过二值（A6-H2）。  
2. 判定欠打仍在；prompt 微调无效。  
3. 已在同代码下重跑 B/C → 见 §8。

---

## 8. BC 重打完成 + gold 命运（2026-11-15）

脚本：`run_38b_rescore_bc_chain.sh` → `bc_chain.log`。模型审计各臂 100% `qwen3.8-27b`。

### 8.1 R@10 总表（重打代码）

| | H1 | H2 |
|---|---:|---:|
| A6 BM25×六档 | 0.7617 | 0.7783 |
| B6 Hybrid×六档 | 0.7708 | 0.7842 |
| **C6 Dense×六档** | 0.7783 | **0.7892** |
| A2 BM25×二值 | 0.7183 | 0.7117 |
| B2 Hybrid×二值 | 0.7633 | 0.7300 |
| C2 Dense×二值 | 0.7508 | 0.7117 |

- 六档：Dense ≥ Hybrid ≥ BM25（与「向量更善捞」同向；增益小）。  
- 二值 H2 再掉点；旧矩阵「B2-H2 全表第一」在重打代码下不成立。

### 8.2 段落级命运（290 gold，`document_id = corpus_index + 1`）

| 命运 | C6-H2 | B6-H2 | A6-H2 |
|---|---:|---:|---:|
| 进最终 top10 | 221 | 220 | 216 |
| **进过池、未进 top10** | **52** | 45 | 44 |
| **全程未进任何跳的池** | **17** | 25 | 30 |

首轮 100 池内 gold：C 232 / B 224 / A 192。C 的 hop0 漏召中扩展又捞回 41，最终只剩 17 全程池外。  
三路都捞不到：**15**（多为 4hop 桥接，问句词与 gold 标题不对齐）。

**为何 C 的「池内漏」52 > B 的 45（不是 C 更差）：**

| | |
|---|---|
| C 少漏池外 | 25→17（−8） |
| C 多池内漏 | 45→52（+7） |
| 净进 top10 | 220→221（+1） |

C 的 52 拆开：B 也池内漏 32；**B 已 top10、仅 C 漏 12**（同金不同判）；**B 池外、C 独捞进池又抬不起 8**（增量库存，几乎全 grade 0）。  
Dense 网更大，多捞进的难例堆在「进池未进 top10」栏，故 52 偏大，净收益只有 +1。

### 8.3 池内漏的本质

各臂池内未进 top10 的主体仍是 **末次 grade=0**（B：43/45；C：约 48/52）。  
有 evidence 仍给 0 的桥接段占绝大多数——与重打前「leftover gold 打成 0」同一瓶颈。  
**换 dense/hybrid 不能解决这批；检索只把池外从 ~30 压到 ~17。**

明细：`output/27b-100/H1-rescore/diagnostics/never_all_ABC_H2.json` 等。

---

## 9. 换数据集（待办备忘）

论文/校验脚本默认三套：`musique`、`2wikimultihopqa`、`hotpotqa`。  
**本仓库 `dataset/` 目前只有 MuSiQue**（见工程记录事故：缺 `2wikimultihopqa.json`）。  
换数据集前需补齐语料与 100 题分层样本、重建索引与 BGE 向量；不要与 MuSiQue 的 `H1-rescore` 混目录。

---

## 10. 当前结论（2026-11-15）

1. 重打代码下最强已跑臂：**C6-H2 = 0.7892**（略高于 A6-H2 0.7783）。  
2. 检索方向正确但边际小；**主瓶颈仍是池内 gold 被判 0**。  
3. 下一步用户意向：换 2Wiki / HotpotQA 看是否同构（数据未入库，先准备再跑）。
