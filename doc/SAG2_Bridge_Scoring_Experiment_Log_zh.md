# SAG2 桥接打分实验进程日志（MuSiQue never 切片）

日期：2026-11-16  
范围：never11 / never 金 17 条硬切片；判定模型默认 `qwen3.8-27b`。

---

## 0. 要解决的问题是什么

**现象：** Dense 多跳检索（C6）上，一部分金文档**从来进不了候选池**（never）；更多是**进了池但进不了 top10**（池内判 0）。

**机制诊断：**

1. 多跳桥文档往往只支撑「中间一跳」，用**整句原问**去打分会被打成 0。  
2. 查询改写/多路召回能把 never 金**捞进池**，但正式臂打分锚仍是 `original_question` → R@10 不涨。  
3. 若改用「子问」打分，子问本身来自上一跳抽取，**带噪声**；填错会加固错链。

**一句话问题：**  
召回已能多路找桥，判定却死盯原问；要让桥文档存活，又不能把错误子问当成真值。

---

## 0.1 问题共性（整理，I 轮之后）

三条主线：

| # | 共性 | 表现 | 已尝试 |
|---|---|---|---|
| 1 | **锚不对** | 桥文档对原问常判 0 | MQ 捞池有效，R@10 不涨 |
| 2 | **槽不准** | 子问对了才涨（oracle/Oh Yeah）；错了就伤（Sazerac） | dual +6pp；瓶颈在填槽 |
| 3 | **门太浅** | 只查「能不能开桥」，不查「槽是不是主事实」 | gated 防了歧义 R，拦不住 Frankfort |

失败形态共性：

| 形态 | 例子 | 共同点 |
|---|---|---|
| 近义错槽 | Sazerac→Frankfort（厂址当总部） | 形式合法、语义近义错 |
| 槽没填全 | Oh Yeah：gated 停在 state | 正确链开不完 |
| 槽空/拒 | Casa Loma 出生地；射手歧义 R | 不伤分也不涨分 |
| 事后门控无效 | 冲突 0 次；verifier 削噪不改 R@10 | 错槽常是**唯一**答案；关 s2 救不回已被带偏的排序 |

**收束：** 真问题是对已放行单值 R，校验槽值是否为目标关系的**主事实**（强表述可单篇通过；弱表述或打架才拒），而不是再给 s2 叠门。

---

## 1. 进程总表（按时间）

| 步 | 尝试 | 问题针对 | 做法 | 结果 | 结论 |
|---|---|---|---|---|---|
| A | 命运漏斗 / 固定证据 | 池外 vs 池内 | never/miss 切片 | never≈17；池内大量 grade0 | 双瓶颈：召不回 + 判不上 |
| B | 换判定模型 deepseek | 池内判 0 | 本地 flash 整臂 | 探针能抬桥，整臂 R@10↓ | **否决**整臂换模型 |
| C | 六档 vs 十档 | 粒度 | 27B n30 | 十档略差 | 维持六档 |
| D | 多路召回探针 | never 召不回 | decomp / iterative / seed_followup + RRF | union **10/17** 进池 | 召回侧有效 |
| E | C6-MQ-H1 | 接入召回 | hop0 dense-mq + hop1 followup；**打分仍原问** | 进池↑，R@10 **0.417 持平** | 瓶颈转回判定锚 |
| F | 子问打分探针 v1/v2/oracle | 换锚是否有效 | 原问 vs 子问 vs 捞金 query | 自写子问 2/10≥4；捞金 query 抬脏；**oracle 5/10≥4** | 判定可用，填槽是瓶颈 |
| G | dual 无闸门整臂 | max(原问,子问) | 抽出就用 | **0.417→0.477**；Sazerac↓ | 方向对，错链会伤 |
| H | gated 三闸门 | 防错链 | E置信+消歧+单值R；模板 `{E}+{R}` | **0.432**（+1.5pp） | 更稳，未追上无闸门；**HQ→Frankfort 仍漏** |
| I | 冲突+逐段 verifier | 近义主/附属事实 | 同E+R多答案冲突；高风险R逐段 valid/invalid/uncertain | **0.432（与 gated 持平）**；Sazerac 仍 0.33 | **尝试失败**；见 §5 |
| J | 槽值语义校验 | 唯一错槽仍放行 | 强/弱表述校验 `(E,R,V)`；弱或打架拒闸 | **0.432（与 gated 平）**；Sazerac 机制对、分仍 0.33 | 见 §6 |

三臂含义（H 阶段对照）：

| 臂 | 召回 | 填槽 | 打分锚 |
|---|---|---|---|
| **MQ** | 多路 | 不给打分用 | 只用原问 |
| **dual 无闸门** | 多路 | 抽出就填长句 | `max(原问, 任意已填子问)` |
| **gated** | 多路 | 只许 `{E}+{R}` 模板 | 过闸才 `max(原问,桥问)`，否则只用原问 |

---

## 2. 每一步：问题 → 解法 → 结果（展开）

### 2.1 多路召回（D→E）

**问题：** never 金题面无实体，单路 Dense 搜不着。  
**解法：** 静态分解多路 + 迭代填槽检索 + 种子追问；RRF 合并。  
**结果：** 探针 union 10/17；整臂进池 0→9/17；R@10 仍 0.417。  
**结论：** 召回问题部分解决；**打分仍用原问**，桥进池仍抬不起。

### 2.2 子问打分探针（F）

**问题：** 进池后是否因「锚是原问」而判 0？  
**解法：** 固定空证据，对原问 / 填槽子问 / 捞金 query 重打。  
**结果：**

- 原问：金全 0  
- 自写子问：2/10≥4  
- 捞金 query：金不涨、脏文档暴涨 → **否决**  
- 官方金分解 oracle：5/10≥4  

**结论：** 换锚有效的前提是**子问正确**；瓶颈在填槽保真，不是再换 27B。

### 2.3 dual 无闸门（G）

**问题：** 整臂接上 `max(s_原问, s_子问)` 能否涨 R@10？  
**解法：** 抽槽+verify 检索；无准入，自由/半自由子问都可进 max。  
**结果：** never11 **0.477（+6.1pp）**；Oh Yeah 大涨；Sazerac 因 Frankfort 错总部下跌。  
**结论：** 证实「桥问打分」端到端有用；**无闸门会加固错链**。

### 2.4 gated 三闸门（H）

**问题：** 如何保留桥问收益、减少错链？  
**解法：**

1. 静态分解只用于召回  
2. 打分只用模板桥问 `{E}+{R}`  
3. 三闸门：E 置信 / 消歧 / R 单值白名单  
4. 原问支路永不关  

**结果：** never11 **0.432（+1.5pp）**；射手/member 等被拒；Oh Yeah 能抽出 Mingus→Arizona 但 H1 吃不到 Phoenix 链完整收益；**Sazerac 仍 gate_accept headquarters 且抽出 Frankfort**。  

**漏点定性：** 不是「关系类型不该放行」，而是 **单值 R 下的近义事实**（corporate HQ vs distillery city）。

### 2.5 冲突 + 逐段 verifier（I）— **结果偏坏**

**问题：** 第一层闸门放行后，如何区分主事实 vs 附属地点？  
**解法：** 多答案冲突关桥；HQ/birth_place 逐段 verifier 门控 s2；后补 `bridge_killed`。  
**结果：** never11 **0.432 = gated**；Sazerac **仍 0.33**。详见 §5。  
**结论：** 再叠 s2 门控这条路走不通；下一刀改打槽值语义。

---

## 3. 关键实例（帮助理解三臂差）

| 题 | MQ | dual | gated | verify(I) | 原因摘要 |
|---|---:|---:|---:|---:|---|
| Oh Yeah | 0.25 | **0.75** | 0.25 | 0.25 | dual 填到 Phoenix；gated 只开到 state_of |
| Sazerac | **0.67** | 0.33 | 0.33 | 0.33 | HQ 抽成 Frankfort；I 轮关桥仍不回 MQ |
| EPL 射手 | 0.50 | 0.50 | 0.50 | 0.50 | gated 拒歧义 R；打平 |
| Casa Loma | 0.67 | 0.67 | 0.67 | 0.67 | 出生地未填出 Toronto |

---

## 4. 产物索引

| 内容 | 路径 |
|---|---|
| 多路召回 | `output/probe-never-multiquery-v2-27b/` |
| dual 无闸门 | `.../C6-MQ-H1-dual-never11/`、`GATE_dual.md` |
| gated | `.../C6-MQ-H1-gated-never11/`、`GATE_gated.md` |
| 冲突+verifier（I） | `.../C6-MQ-H1-verify-never11/`、`GATE_verify.md`、`arm_compare_verify_never11.json` |
| Sazerac bridge_killed 复测 | `.../C6-MQ-H1-verify-sazerac1/` |
| 槽值语义校验（J） | `.../C6-MQ-H1-semantic-never11/`、`GATE_semantic.md`（待写） |
| 判定专报交叉引用 | `SAG2_Rescore_and_Judge_Probe_Report_zh.md` §5.6–5.8 |

---

## 5. 实验 I 完整记录（冲突 + 逐段 verifier）

### 5.1 假设

在第一层闸门已放行后：

- 若同 E+R 抽出 ≥2 个不同槽值 → 关闭本跳桥问；  
- 否则对 HQ / birth_place 逐段 verifier：invalid 抹 s2，uncertain 衰减 s2。  

预期：Sazerac 不再因 Frankfort 段被抬进 top10。

### 5.2 实现

- `hop_anchor.py`：`slot_conflict`、`RelationVerifier`、`DualAnchorScorer` 逐段门控  
- CLI：`--slot-conflict --passage-verify`  
- 后补：`invalid ≫ valid` → `bridge_killed`（整跳 s2=0）

### 5.3 结果

| 臂 | R@10 | 备注 |
|---|---:|---|
| MQ | 0.417 | 基线 |
| dual 无闸门 | 0.477 | |
| gated | 0.432 | |
| **I: conflict+verify** | **0.432** | 与 gated **逐题全平** |

**分析：**

1. Layer-2 冲突 **0 次**：Sazerac 只抽出唯一错槽 Frankfort → 「多答案」逻辑用不上。  
2. Layer-3 有生效：相对 gated 去掉 Ten High，部分威士忌 grade 2→0；市长链金仍不进 top10。  
3. `bridge_killed` 单题复测：`valid=0, invalid=14, bridge_killed=true`，**R@10 仍 0.33** → 关 s2 后仍不回 MQ，残差含 hop1 原问重打 / 池与证据上下文差异。  

**总评：偏坏 / 尝试失败。** 有局部削噪，无指标收益；确认不应再叠 s2 门控。

---

## 6. 实验 J：槽值语义校验（进行中）

### 6.1 动机（相对「多源一致」的修正）

曾提「双源一致」：≥2 篇同城才信。  
担心：**只有 1 篇真总部也会错杀**。  

改为：

| 情况 | 动作 |
|---|---|
| ≥1 篇**强表述**支持 `(E,R,V)`（headquartered in / born in …） | **accept，可开门**（不要求多篇） |
| 仅有弱表述（厂区/品牌/家乡口吻） | **reject，拒闸，只用原问** |
| 多篇强表述互相打架 | **reject** |

校验对象是**槽值 V 本身**，不是再门控 s2 段落分。

### 6.2 规格

- 范围：`headquarters`、`birth_place`（与高风险 R 对齐）  
- 时机：verify 检索合并证据后，对抽出的 V 做一次语义判定  
- reject → 对该 E 打 `conflict`（及 `slot_semantic=reject`），一层闸门关闭桥问  
- accept → 保持现有 gated 行为  
- 原问 s1 永不关  

### 6.3 验收

1. Sazerac：Frankfort 应为 reject → 不开 HQ 桥；R@10 目标 ≥0.67（回 MQ）或至少明显好于 0.33  
2. never11：均值 ≥ gated 0.432；不把已涨题打崩  

### 6.4 结果（已跑完）

| 臂 | R@10 | 备注 |
|---|---:|---|
| MQ | 0.417 | |
| gated | 0.432 | |
| I verify | 0.432 | |
| **J semantic** | **0.432** | 与 gated **逐题全平** |

**机制侧（成功）：**

- Sazerac：`slot_semantic reject` Frankfort，理由明确为生产厂址非 corporate HQ → `gate_reject`  
- 未要求「必须两篇」；单篇弱表述即拒  

**指标侧（未好）：**

- never11 均值不变；Sazerac 仍 0.33  
- accept 从 gated 的 6 降到 2（多了 `slot_semantic_missing` / reject），但本切片逐题 R@10 未变  

**结论：** 语义校验**做对了该做的事**（拦住错槽开门），但 **治不好 Sazerac 的分数**——与 I 轮 `bridge_killed` 一致：关 HQ 桥后排名仍不回 MQ。残差要另查（hop1 原问重打 / followup 池差异等），不是继续收紧槽门。

产物：`C6-MQ-H1-semantic-never11/`、`C6-MQ-H1-semantic-sazerac1/`、`GATE_semantic.md`、`arm_compare_semantic_never11.json`。

---

## 7. 对错对照深挖（验证是否「无解」）

### 7.1 填对案例：Oh Yeah（`4hop1__726391_…`）

题：*Who won the Indy car race in the largest city and capitol of the state where the Oh Yeah performer is from?*

| 臂 | R@10 | 关键 top |
|---|---:|---|
| MQ | 0.25 | Indiana / Ohio song / Oklahoma…（桥文档起不来） |
| **dual 无闸门** | **0.75** | Arizona / Mingus / Oh Yeah album… |
| gated / semantic | 0.25 | Boston / Pennsylvania 占头；Arizona 链没抬起来 |

**逐步（dual 为何涨）：**

1. hop0 用自由子问打分（无闸门允许）→ `Oh Yeah (Charles Mingus album)` **grade 8**，进 frontier  
2. 抽出 `Charles Mingus` → 再抽 `Arizona` → 甚至 `Phoenix`  
3. hop1 用已填子问再打 → `Arizona`/`Mingus` **grade 10** → R@10 到 0.75  

**逐步（gated 为何吐回去）：**

1. hop0 **强制只用原问** → 专辑文档抬不起来（top 是 Pennsylvania/Boston grade 4）  
2. 虽仍抽出 Mingus，且 `gate_accept state_of`（桥问：*What state is Charles Mingus from?*）  
3. hop1 只开到 state，**吃不到** dual 那种「自由子问 + 更深填槽（Phoenix）」的收益  
4. 结果掉回 0.25  

→ **填对时的涨分，很大一块来自「无闸门、hop0 就能用子问」；gated 为防错，把这段收益砍掉了。**

### 7.2 填错案例：Sazerac（`3hop1__786067_…`）

题：*When did the city that headquarters the manufacturer of McAfee's benchmark elect its first black Mayor?*

| 臂 | R@10 | 关键现象 |
|---|---:|---|
| MQ | **0.67** | 池头就有市长文档；威士忌 grade 0 |
| dual / gated | 0.33 | 开 HQ 桥 → 威士忌 grade 2，挤掉市长链 |
| semantic | 0.33 | **正确 reject Frankfort 并关桥**；威士忌回 0，但仍不回 0.67 |

**逐步（dual/gated 如何变差）：**

1. 召回分解写成嵌套 HQ 长句 → 池头变 McAfee/Seattle/Maine（相对 MQ 的市长开头）  
2. 抽出 `Sazerac Company`，总部槽抽成 **Frankfort**（厂址近义）  
3. `gate_accept` 桥问 *Where is the headquarters of Sazerac Company?*  
4. hop1：Blanton's / Kentucky Tavern / Ten High 等 **0→2**；Black people 等掉出有效位 → 0.33  

**逐步（semantic 关桥后）：**

1. `slot_semantic reject`：明确判定 Frankfort 只是 Buffalo Trace 生产地  
2. hop1 不再开桥，威士忌分回到 0  
3. 但 hop0 池头仍是 McAfee 那一版，市长金排不进 top10 → **仍 0.33**  

→ **错槽开门会伤；关对了门也救不回，因为召回名单已经偏了。**

### 7.3 对照后的判断：是不是无解？

| 命题 | 是否成立 |
|---|---|
| 子问打分方向错了？ | **否**（Oh Yeah dual +0.50 已证伪） |
| 现在能稳定填对槽？ | **否**（Sazerac 近义错；gated 下 Oh Yeah 也吃不全链） |
| 防错门能保住涨分又修好错题？ | **否**（gated/semantic 均值停在 +1.5pp；Sazerac 仍 DOWN） |
| 因此「子问路线无解」？ | **不成立**——无解的是当前这条 **「自由子问涨分 ↔ 闸门防错」的实现折中**，不是理论不可能 |

**结构性死结（当前实现）：**

```
涨分需要：早用子问（含 hop0）+ 填得深（Arizona→Phoenix）
防错需要：晚用/少用子问 + 严校验
两者抢同一开关 → 打开则 Sazerac 类受伤，关上则 Oh Yeah 类涨分吐回
```

**尚未证成「无解」的出路（需另开实验，不是再叠同一类门）：**

1. **召回子问与打分子问彻底解耦并固化**（避免分解抖动改池）  
2. **按关系非对称**：performer/state 可更早开桥；HQ/birth 默认不开，除非强表述 accept  
3. **真把槽填对**（HQ 抽 New Orleans 而非 Frankfort）——这才是错题根治  

在 1/2/3 做成之前，never11 上会继续表现为：dual 净赚但单题回撤大；gated 家族稳但涨不动、错题救不回。

---

## 8. 冻结池三枪（原问 / 自填 / oracle）— 已补齐

**设定：** 冻结 MQ 臂候选 id；不重搜、不扩展；空 evidence 下分别用原问 / dual 自填锚 / 金分解 oracle 锚打分取 max。

| | R@10 |
|---|---:|
| 池内金覆盖 | 0.765 |
| 原问 | 0.364 |
| 自填 | **0.455** |
| oracle | **0.523** |

**结论：** 同池下 **原问 < 自填 < oracle**。子问锚有效；自填仍低于 oracle。Sazerac 同池三枪都是 0.67（整臂 dual 变差需另看池/扩展）。详见 `GATE_frozen_three_arms.md`。

---

## 9. 决策：主臂先认 dual，放大样本验证（已完成 n30）

不再叠门/verifier/改抽取提示。  
主臂候选：**dense-mq + dual 无闸门自填**。

| 切片 | MQ R@10 | dual 无闸门 | Δ | 逐题 |
|---|---:|---:|---:|---|
| never11（难） | 0.417 | **0.477** | **+6.1pp** | 3↑7平1↓ |
| **n30（更广）** | 0.772 | **0.778** | **+0.6pp** | 4↑24平2↓ |

**读数：** 更大样本上 dual 仍净正，但增益变小（基线已高）。未出现「错槽把整体打崩」。  
产物：`output/probe-mq-dual-n30/`、`GATE_n30.md`、`arm_compare_n30.json`。

---

## 10. 阶段收敛 + n100（2026-11-17）

正式结论与论文初检见 **`SAG2_Bridge_Phase_Conclusion_and_n100_Plan_zh.md`**。  
n100：**4 臂 H2**（`C2/C6-MQ-H2` ± dual）；H0/H1 从同次 `hop_metrics` 取，不重复跑 → `output/mq-dual-100/run_chain_h2_four.sh`。  
旧矩阵 H0 基线见阶段结论 §2.1。

---

## 11. 冻结池 bge 重排探针（2026-11-17）

**动机：** 冻结池显示池内金覆盖 ~0.77，但 JEV 原问/自填/oracle 只到 0.36/0.46/0.52——瓶颈是「进池选不到」。怀疑用更强 cross-encoder 重排能抬 top10。

**设定：** 与 §8 同一冻结池（MQ hop0 ∪ jev-scored ids，不重搜）；`bge-reranker-v2-m3` @ `http://192.168.2.108:8014/v1/rerank`；never11；无 LLM JEV。

| 臂 | 查询 | R@10 |
|---|---|---:|
| 池内金覆盖 | — | 0.765 |
| JEV 原问（昨 §8） | 原问 | 0.364 |
| JEV 自填（昨 §8） | max(原问, 自填子问) | **0.455** |
| JEV oracle（昨 §8） | max(原问, 金分解) | 0.523 |
| rerank 原问 | 原问 | 0.386 |
| rerank 自填 max | max(原问, 自填子问) | 0.364 |
| rerank oracle max | max(原问, 金分解) | 0.530 |

**结果：** 接口可用；rerank 原问仅 +2pp；**自填重排弱于 JEV 自填（−9pp）**；oracle 与 JEV oracle 持平，仍远低于 0.77。

**读数（为何没用）：**  
cross-encoder 量的是「切片对**当前这句话**有多相关 / 能否直接支撑它」。多跳金切片常只覆盖中间一跳，对整问信息不完整 → 分低。即使用子问，目标仍是语义相似，不是「是否推进推理链」；`max` 还可能把「像子问但不在链上」的段顶进 top10。  
→ **裸 `bge(q, doc)` 解决不了池内选型缺口**；要有效需桥条件化（如 \(s(q,b,c)\)）或继续 JEV 类证据档位。

产物：`output/probe-never-multiquery-v2-27b/frozen-pool-rerank/`（`GATE_rerank.md`、`summary.json`、`probe_frozen_pool_rerank.py`）。