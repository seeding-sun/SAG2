# SAG2 后续实验方案

## 1. 文档目的

本文在重新核对 SAG2 论文后，给出后续实验的完整路线：

- 要用哪些数据集；
- 每个实验解决什么问题；
- 控制哪些变量；
- 使用哪些指标；
- 跑多少道题、多少个扩展轮次和多少次独立重复；
- 在什么环境下运行；
- 什么条件下继续、停止或进入全量实验。

配套文档：

```text
D:\learn\知识库\RAG\SAG2\doc\SAG2_Hybrid_Experiment_Plan.md
D:\learn\知识库\RAG\SAG2\doc\SAG2_Hybrid_Implementation_Plan.md
```

本文重新规划实验顺序，并特别区分两个容易混淆的概念：

1. **H 轮次**：SAG2 内部的检索与证据评分轮次。H=0 表示初始评分；H=1 表示追加一次扩展；H=2 表示追加两次扩展。
2. **独立重复次数**：同一组实验重新执行多少次，用于衡量 Qwen 等模型带来的波动。

## 2. 论文回顾

### 2.1 论文要解决的问题

SAG2 关注的不是最终答案生成，而是多跳问题中的证据检索。

典型失败场景是：

```text
问题中的表面词只能找到第一跳证据
第二跳证据必须知道第一跳的新实体后才能找到
一次性 BM25 或向量检索不会根据已发现证据改变下一次检索
```

因此，仅扩大一次性的 top-k，可能仍然找不到完整的证据链。

### 2.2 论文方法

论文流程：

```text
原问题
  ↓
SQLite FTS5 / BM25 top-100
  ↓
JEV 对候选进行证据判断
  ↓
从最多 10 个未扩展 seed 中选择最多 20 个扩展词
  ↓
term-specific BM25
  ↓
交错、去重并补充最多 100 个新候选
  ↓
JEV 在共享证据上下文中重新评分
  ↓
最多 H=2 次追加扩展
  ↓
DPP 仅重排最终 top-10
```

论文默认参数：

```text
初始候选 B = 100
最终候选上限 = 100
seed s = 10
扩展词 L = 20
受保护初始候选 p = 5
追加扩展轮数 H = 2
问题和 HTTP 并发 = 20
BM25 字段权重 = 2,2,1
```

### 2.3 论文指标

主指标：

```text
Recall@k
AllEvidence@k
```

定义：

```text
Recall@k
    top-k 唯一原始 passage 中的 gold evidence 比例

AllEvidence@k
    top-k 是否已经包含该题全部 gold evidence
    是 1，否 0，再对题目求平均
```

这两个指标都只评价证据检索，不评价最终答案正确性。

### 2.4 论文核心结果

MuSiQue 1,000 题上的主要结果：

| 配置 | R@10 | AE@10 | Passage judgments |
|---|---:|---:|---:|
| BM25 | 54.41 | 17.40 | 0 |
| BM25+JEV, H=0 | 67.93 | 36.00 | 99,925 |
| SAG2, H=1 | 83.36 | 63.10 | 211,081 |
| SAG2, H=2 | 86.62 | 69.50 | 331,325 |
| BM25+JEV，预算匹配 | 69.82 | 39.60 | 329,194 |
| SAG2，无共享上下文 | 76.25 | 52.20 | 330,990 |

从论文中可以提炼出四项直接与后续实验有关的事实：

1. 第一次扩展把 R@10 从 67.93 提高到 83.36，增加 15.43 个百分点；
2. 第二次扩展只再增加 3.26 个百分点，H=2 的边际收益明显下降；
3. 把一次性的 BM25+JEV 判断预算提高到接近 H=2 后，R@10 仍只有 69.82，说明收益不是简单来自“多判断候选”；
4. 去掉共享上下文后，R@10 从 86.62 降到 76.25，说明后续评分依赖前面发现的事实。

### 2.5 按问题 hop 的结果

论文 Table A3 的 R@10：

| 问题类型 | 题数 | BM25 | H=0 | H=1 | H=2 |
|---|---:|---:|---:|---:|---:|
| 2-hop | 518 | 61.68 | 75.00 | 90.54 | 92.86 |
| 3-hop | 316 | 51.37 | 66.88 | 82.07 | 85.34 |
| 4-hop | 166 | 37.50 | 47.89 | 63.40 | 69.58 |

四跳问题第二轮扩展的增益最大，但最终 AE@10 只有 28.31%。这说明：

- 扩展轮次对更长证据链更重要；
- 即使 R@10 较高，也不代表完整证据链容易全部找齐；
- 后续必须同时报告 R@k 和 AE@k，不能只看 Recall。

### 2.6 论文没有回答的问题

论文没有直接比较：

```text
BM25+JEV
vs
BM25+BGE-M3+JEV
```

论文中的 BGE 主要来自其它系统基线，不是与 SAG2 初始召回融合的对照。

因此，我们准备做的混合召回实验属于新实验，不是论文中已有消融的复现。

论文还把以下问题列为局限：

- 主要依赖 hosted JEV，模型权重不公开；
- 不能独立在完全本地的环境复现原 JEV；
- 没有证明 JEV 一定优于其它 evidence scorer；
- 固定 development subset 且 MuSiQue 参与方法开发；
- 词法访问仍会漏掉 paraphrase、alias 和没有表面词连接的证据；
- 四跳完整证据覆盖仍然很低；
- 未评估最终答案生成质量。

### 2.7 对当前实验的直接启示

第一，不能用普通 Qwen Chat 结果冒充论文 JEV 结果。

当前方案是：

```text
Qwen Chat 替代 JEV
```

所以准确名称应是：

```text
SAG2-style retrieval with a Qwen evidence scorer
```

而不是“严格复现 SAG2”。

第二，仓库已有的本地 reranker 替代实验显示：

```text
SAG2+JEV       R@10 = 86.61
SAG2+Jina      R@10 = 72.13
SAG2+Qwen 0.6B R@10 = 74.18
```

这些结果不是同一种 Qwen Chat evidence scorer，但足以说明：替换 JEV 后性能可能明显下降。Qwen 评分器必须先单独校准，再进入主实验。

第三，论文的“预算匹配”和“去共享上下文”是非常重要的控制实验。混合召回即使提升，也可能只是增加了候选或判断预算，必须用对照排除这种解释。

## 3. 研究问题与假设

### RQ1：BGE 是否能补足 BM25 的初始入口？

H1：

> 在同样返回 top-100 的条件下，BM25+BGE-M3 RRF 比纯 BM25 包含更多 gold passage，并提高至少一个 gold passage 的入口覆盖。

对应指标：

```text
InitialEntryRecall
InitialFullRecall
Initial R@10
HybridAddedEvidence
HybridNoise
```

### RQ2：初始召回收益能否穿过 Qwen 评分与 SAG2 扩展？

H2：

> Hybrid 的初始入口优势在 Qwen 证据判断和 H=1/H=2 扩展后仍然存在。

如果初始召回提高了，但最终 R@10/AE@10 没提高，说明：

- Qwen 没有识别新增 gold evidence；
- seed 选择没有用上新增证据；
- 扩展阶段引入了更多噪声；
- 或者收益被后续排序抵消。

### RQ3：收益是否只是来自更多候选或更多判断？

H3：

> 在相同 passage judgment 预算下，Hybrid 仍然优于 one-shot BM25+Qwen。

若不能通过该控制，则“Hybrid 更强”不能成立，只能说明多判断了一些候选。

### RQ4：共享上下文在 Hybrid 下是否仍然有效？

H4：

> 去掉前序证据上下文会降低 Hybrid 的 R@10 和 AE@10。

该实验检验 Qwen 是否真的在使用多跳上下文，而不是只对每个 passage 做独立相关性判断。

### RQ5：收益是否值得成本？

H5：

> Hybrid 的精度收益能够抵消 query embedding、RRF 和额外候选判断的成本。

指标：

```text
query embedding latency
Qwen requests
passage judgments
tokens
p50/p95 latency
失败率
```

## 4. 实验总原则

### 4.1 先阶段门，不一次跑完

实验必须按门控推进：

```text
环境与接口可用
  ↓
初始召回有解释
  ↓
Qwen 评分稳定
  ↓
10 题 smoke
  ↓
100 题主对照
  ↓
机制控制
  ↓
稳定性重复
  ↓
可选全量/跨数据集
```

上一阶段不通过时，不运行下一阶段全量实验。

### 4.2 主实验只改一个变量

主对照固定：

```text
question set
corpus
BM25 index
Qwen model
Qwen prompt
temperature = 0
candidate_k = 100
seed_k = 10
expansion_term_limit = 20
max_hops
DPP
metric implementation
```

唯一变化：

```text
BM25 initial top-100
vs
BM25+BGE RRF initial top-100
```

### 4.3 所有候选使用同一单位

BM25、BGE 和 RRF 都返回 `chunks.id`。

评测再按原始 passage 去重，避免：

- BM25 多 chunk 计多次；
- 向量 passage 与 BM25 chunk 使用不同编号；
- RRF 把两个单位的“第 1 名”错误融合。

### 4.4 不把论文 JEV 数值当成本实验的目标线

论文数值只能作为外部参考。

当前系统的 Qwen 分数：

```text
jev_score
```

只是兼容字段名，不代表原 JEV 输出。

### 4.5 先冻结参数，再跑正式集

主实验开始前冻结：

- 100 题 sample ID；
- corpus hash；
- BM25 index signature；
- embedding manifest；
- Qwen 模型名；
- Qwen prompt hash；
- Qwen batch size；
- temperature；
- RRF 公式和 k；
- 所有 SAG2 参数；
- 代码 commit/hash。

### 4.6 调参集与确认集分开

100 题是方法开发子集，不是独立测试集。

如果后续调整：

- RRF 权重；
- candidate_k；
- seed_k；
- expansion terms；

应预先将 100 题固定划分为：

```text
50 题调参
50 题确认
```

或者直接在更大的 1,000 题上做“开发/确认”分层。

主实验使用论文默认 RRF，不做参数搜索，因此可以直接用完整 100 题。

## 5. 环境与资源

### 5.1 当前机器检查

已核对：

```text
操作系统：Windows 11 10.0.26200
Python：3.12.14
Python 环境：D:\learn\.venv
D 盘可用空间：约 181.9 GiB
MuSiQue 数据：约 17.91 MB
现有 BM25 索引：约 26.20 MB
```

当前 `D:\learn\.venv` 中尚未安装项目所需包：

```text
numpy
aiohttp
jieba
python-dotenv
regex
tiktoken
pytest
```

因此正式实验前先执行依赖同步和导入检查。依赖安装成功后再运行任何模型请求。

### 5.2 第一阶段推荐环境

```text
CPU：4 核以上
内存：8 GiB 可运行，建议 16 GiB
GPU：不需要
本地磁盘：至少 2 GiB 可用
网络：能够访问 BGE 与 Qwen 内网服务
系统：当前 Windows 环境即可
```

原因：

- SQLite FTS5 和 NumPy 检索规模很小；
- 11,656 × 1,024 的 float32 向量约 47.8 MB；
- BGE 与 Qwen 都在远程服务运行；
- 第一阶段不需要本地大模型。

### 5.3 并发

建议分三档：

```text
接口 smoke：1-2
10 题测试：2-4
正式 100 题：10
稳定后最高：20
```

论文使用 20 并发，但正式实验不应从 20 并发直接起跑。先观察服务错误率、429、超时和延迟。

### 5.4 必须记录的环境元数据

每次正式实验记录：

```text
OS
Python version
关键依赖版本
数据集 SHA256
BM25 index signature
embedding cache manifest
embedding model
Qwen model
RRF k
prompt hash
temperature
batch size
concurrency
运行日期
代码 hash
```

实际返回模型名与服务配置不一致时，实验无效或必须重新分组。

## 6. 数据与固定样本

### 6.1 第一阶段

数据集：

```text
MuSiQue
```

Corpus：

```text
11,656 passages
```

固定 100 题：

```text
2-hop  40
3-hop  30
4-hop  30
```

使用固定 seed：

```text
20261111
```

A/B 的 sample file 必须完全相同。

### 6.2 第二阶段

如果 100 题结果出现正向信号：

```text
MuSiQue 完整 1,000 题
```

### 6.3 泛化阶段

在 MuSiQue 上确认结论后，再考虑：

```text
2WikiMultiHopQA 前 100 题
HotpotQA 前 100 题
```

通过小规模泛化门后，才决定是否扩展到各 1,000 题。

论文中的完整语料规模：

```text
2WikiMultiHopQA   6,119 passages
HotpotQA          9,811 passages
```

EnterpriseRAG-Bench 第一阶段不做，因为它有 511,962 个源文档，属于独立的规模测试。

## 7. 实验路线

### E0：环境、数据和接口预检

目的：

- 确认代码能启动；
- 确认 Qwen 返回合法 JSON；
- 确认 BGE embedding 维度及 batch；
- 确认候选 ID 与索引一致；
- 确认不存在 gold 信息泄漏。

内容：

```text
pytest
ruff
BM25 index-only
1 条 embedding 请求
5 个候选的 Qwen 评分
3 道题的 BM25 检索
3 道题的 BGE 检索
RRF top-10 手工核对
```

环境：

```text
Windows + D:\learn\.venv
concurrency = 1
```

轮次：

```text
每个接口 1 次
Qwen 格式测试重复 3 次
```

通过条件：

- 测试通过；
- 索引返回 11,656 个 chunk；
- Qwen JSON 解析成功；
- BGE 向量维度和数量正确；
- 候选无重复；
- prompt 中不含 gold evidence、answer、is_supporting。

失败处理：

```text
不进入 E1，先修接口或实现
```

### E1：初始召回能力实验

目的：

回答 RQ1，确认 BGE 是否真的补足 BM25 的入口，而不是只增加语义相似噪声。

实验臂：

| 组别 | 初始召回 |
|---|---|
| I-BM25 | BM25 top-100 |
| I-Dense | BGE-M3 dense top-100 |
| I-Hybrid | BM25 top-100 + BGE top-100 → RRF top-100 |

此实验不调用 Qwen，不做扩展。

数据：

```text
固定 100 题
完整 11,656 passage corpus
```

轮次：

```text
每个实验臂 1 次确定性检索
共 300 个 question-arm 结果
```

环境：

```text
本地 BM25 + NumPy
BGE query embedding 远程服务
concurrency = 2-4
```

指标：

```text
Initial R@1
Initial R@2
Initial R@5
Initial R@10
InitialEntryRecall
InitialFullRecall
HybridAddedEvidence
HybridNoise
gold median rank
```

为什么不用 Qwen：

先把“候选生成”与“证据判断”分开。否则无法知道提升来自 BGE 还是 Qwen。

结果解释：

- Dense 单独强：BGE 值得融合；
- Hybrid 强于两路：RRF 有互补；
- Hybrid 只增加噪声：不应继续给 Qwen 增加负担；
- Hybrid 入口提升但全证据覆盖不升：BGE 找到的是零散证据，后续要看扩展。

是否进入 E2：

```text
Hybrid 的 InitialEntryRecall 或 Initial R@10 有正向结果
或
Dense 新增证据比例足够高，值得测试 Qwen 是否能过滤噪声
```

### E2：Qwen 证据评分器校准

目的：

- 验证 Qwen 是否按 SAG2 criterion 判断；
- 识别模型是否把“主题相关”误当成“必要证据”；
- 验证 JSON 协议；
- 发现 Qwen 与论文 JEV 的能力差距；
- 在实际运行前估算成本和报错率。

输入：

```text
固定 20 题，分层为 8 道 2-hop、6 道 3-hop、6 道 4-hop
```

实验臂：

| 组别 | 候选池 | 评分器 |
|---|---|---|
| Q-BM25 | BM25 top-100 | Qwen |
| Q-Hybrid | Hybrid top-100 | Qwen |

都是 H=0，不做扩展。

轮次：

```text
每个实验臂 3 次独立执行
20 题 × 2 组 × 3 次 = 120 个 question-run
```

环境：

```text
concurrency = 2 起
Qwen temperature = 0
每次独立重复使用不同响应缓存命名空间
```

注意：

如果三次重复复用了完全相同的 HTTP 缓存，就只能测代码稳定性，不能测模型波动。

指标：

```text
Recall@10
AE@10
JSON 解析成功率
候选分数覆盖率
gold / non-gold 的 score 分布
AUROC（仅诊断）
3 次重复的分数 Spearman 相关
3 次重复的 top-10 Jaccard
每题失败类型
judgments / requests / tokens / latency
```

进入主实验的最低条件：

- 所有格式化失败都能被重试机制捕获；
- 经过重试后，候选评分覆盖率为 100%；
- 没有静默补 0；
- Qwen prompt 未接触 gold；
- 三次重复虽然允许波动，但整体排序不能完全失稳；
- 报告 gold 与 non-gold 的分数分离情况。

如果 Qwen 评分器明显无法区分：

```text
暂停 Hybrid+Qwen 主实验
先改评分协议或更换本地 scorer
```

### E3：核心 A/B 主实验

目的：

回答 RQ2，判断 Hybrid 的初始召回优势是否能穿过 Qwen 和 SAG2 扩展。

实验臂：

| 组别 | 初始召回 | 证据评分 | H |
|---|---|---|---|
| A1-H0 | BM25 | Qwen | 0 |
| B1-H0 | Hybrid | Qwen | 0 |
| A1-H1 | BM25 | Qwen | 1 |
| B1-H1 | Hybrid | Qwen | 1 |

固定：

```text
100 题
candidate_k = 100
seed_k = 10
expansion_term_limit = 20
rrf_k = 60
temperature = 0
Qwen batch size 固定
```

轮次：

```text
每个实验臂正式运行 1 次
共 4 个 question-arm 配置
100 题 × 4 = 400 个 question-run
H=0 是 1 个评分轮次
H=1 是 2 个评分轮次
```

环境：

```text
正式主实验 concurrency = 10
若服务稳定且错误率低于 1%，可升到 20
每一步保持 checkpoint
```

主指标：

```text
R@10
AE@10
```

次要指标：

```text
R@1
R@2
R@5
AE@5
```

过程指标：

```text
SeedPrecision
BridgeRecovery
InitialEntryRecall
InitialFullRecall
HybridAddedEvidence
HybridNoise
```

成本指标：

```text
query embedding latency
judgments
requests
input/output tokens
p50/p95 latency
失败率
```

核心对照：

```text
B1-H0 - A1-H0：初始池和 Qwen 过滤的收益
B1-H1 - A1-H1：一轮扩展后的净收益
B1-H1 - B1-H0：扩展在 Hybrid 上的收益
A1-H1 - A1-H0：扩展在 BM25 上的收益
```

进入 H=2 的条件：

- B1-H1 在 R@10 或 AE@10 上优于 A1-H1；
- 改善不是只集中在一个 hop；
- seed precision 或 bridge recovery 有合理改善；
- 成本增加可接受；
- 没有出现大面积解析失败。

如果 H0 没有收益，但初始入口有收益：

```text
运行 H1，检查 Qwen 是否浪费了新增证据
```

如果 H0/H1 都没有收益且 HybridNoise 明显升高：

```text
停止 H=2 和全量实验，转入错误分析
```

### E4：两轮扩展确认实验

目的：

- 判断第二轮扩展是否仍值得；
- 检查四跳问题是否获益更多；
- 判断 Hybrid 是否在更长证据链上更有价值。

实验臂：

| 组别 | 初始召回 | 证据评分 | H |
|---|---|---|---|
| A1-H2 | BM25 | Qwen | 2 |
| B1-H2 | Hybrid | Qwen | 2 |

轮次：

```text
100 题 × 2 组 × 1 次 = 200 个 question-run
H=2 最多 3 个评分轮次
```

环境：

```text
与 E3 相同
```

指标：

```text
全部主指标
按 2/3/4-hop 分层的 R@10、AE@10
MarginalGain(H2-H1)
MarginalJudgments(H2-H1)
BridgeRecovery
```

预期解释方式：

- H1 提升大、H2 提升小：一篇论文式收益递减；
- H2 只在 4-hop 提升：证据链长度是主要因素；
- Hybrid 在 H2 反而下降：后期扩展被向量噪声污染；
- H2 成本翻倍但指标不升：应把 H1 作为默认产品配置。

### E5：机制控制实验

目的：

排除“Hybrid 只是用了更多判断预算”或“Qwen 只是做一次普通 rerank”的解释。

只在 E3/E4 出现正向结果后运行。

#### C1：判断预算匹配

对照：

```text
Hybrid+Qwen H=2
vs
one-shot BM25+Qwen，其 passage judgment 数逐题匹配 Hybrid
```

为什么：

论文中 matched-budget 把 69.82 与 86.62 分开，这是证明“扩展机制有效”的关键控制。

指标：

```text
R@10
AE@10
实际 judgments
input tokens
```

判定：

- Hybrid 仍胜：收益不是单纯候选预算；
- Hybrid 不胜：主要收益来自更多判断，不应宣传为结构优势。

轮次：

```text
100 题 × 1 次
```

#### C2：去除共享上下文

对照：

```text
Hybrid+Qwen H=1/H=2 with context
vs
Hybrid+Qwen H=1/H=2 without shared context
```

为什么：

论文去上下文产生 10.37 个百分点的 R@10 下降；要确认 Hybrid 环境中上下文仍有作用。

指标：

```text
R@10
AE@10
SeedPrecision
BridgeRecovery
```

轮次：

```text
100 题 × 2 组 × 1 次
```

#### C3：仅最终 Qwen 评分

对照：

```text
Hybrid 中间评分 + 最终评分
vs
Hybrid 只做词法/向量扩展，只在最终候选上调用 Qwen
```

为什么：

判断收益是否来自“中间轮次动态选择 seed”，还是仅来自最后 rerank。

指标：

```text
R@10
AE@10
judgments
seed precision
```

轮次：

```text
100 题 × 1 次
```

#### C4：RRF 权重敏感性

只在主结论成立后做。

固定：

```text
Hybrid+Qwen H=1
```

比较：

| 配置 | BM25 权重 | Dense 权重 |
|---|---:|---:|
| RRF-balanced | 1.0 | 1.0 |
| RRF-BM25 | 1.0 | 0.5 |
| RRF-dense | 0.5 | 1.0 |

因为要调参，必须使用预先划分的 50 题调参集，另外 50 题确认。

轮次：

```text
3 个配置 × 50 题调参 = 150 个 question-run
选中的 1 个配置 × 50 题确认 = 50 个 question-run
```

如果只想保持论文式简洁，可以跳过 C4，把等权 RRF 作为主配置。

### E6：Qwen 稳定性独立重复

目的：

估计同一输入下的模型波动，避免把一次 100 题结果当成精确值。

样本：

```text
从固定 100 题中预先抽 20 题
8 道 2-hop
6 道 3-hop
6 道 4-hop
```

配置：

```text
A1-H1
B1-H1
```

轮次：

```text
2 个实验臂 × 20 题 × 3 次独立执行
= 120 个 question-run
```

要求：

- 每次使用独立请求缓存目录；
- prompt、模型、temperature、batch 不变；
- 不使用同一缓存响应冒充重复；
- 记录每次实际模型名。

指标：

```text
每次运行的 R@10、AE@10
三次均值和标准差
top-10 Jaccard
候选分数 Spearman
跨重复的 Win/Tie/Loss
解析失败率
平均延迟
```

为什么只需要 20 题：

- 重复实验的目的不是重新估计总体精度；
- 它主要估计模型输出噪声和实现稳定性；
- 100 题主实验已经用于 paired bootstrap。

### E7：泛化实验

触发条件：

```text
MuSiQue 100 题主实验和至少一个机制控制出现正向结果
```

数据：

```text
2WikiMultiHopQA 前 100 题
HotpotQA 前 100 题
```

实验臂：

```text
A1-H1
B1-H1
```

轮次：

```text
每个数据集 2 个实验臂 × 100 题 × 1 次
2 个数据集共 400 个 question-run
```

指标：

```text
R@10
AE@10
R@5
AE@5
cost
```

不直接扩展到各 1,000 题，除非：

- 两个数据集方向一致；
- 没有严重的数据格式问题；
- 成本可接受；
- 人工检查确认没有 benchmark 单位错配。

### E8：BGE Reranker 可选实验

目的：

在初始召回已确定后，判断 cross-encoder 精排是否有独立价值。

不能这样比较：

```text
A：BM25 → Qwen
B：Hybrid → BGE Reranker → Qwen
```

因为这会把“混合召回”和“精排”混在一起。

正确对照：

```text
A2：BM25 top-100 → BGE Reranker → Qwen
B2：Hybrid top-100 → BGE Reranker → Qwen
```

可选扩展：

```text
A3：BM25 top-100 → Qwen（无 Reranker）
B3：Hybrid top-100 → Qwen（无 Reranker）
```

指标：

```text
R@10
AE@10
reranker latency
reranker score distribution
候选池最大可召回上限
```

轮次：

```text
100 题 × 2 个 Reranker 实验臂 × 1 次
```

注意：

Reranker 只能重排已有候选，不能作为全语料召回器。它的召回上限由候选池决定。

### E9：全量确认实验

触发条件：

```text
100 题 H=1/AE@10 有稳定正向收益
且 E5 至少解释清楚收益来源
```

运行：

```text
MuSiQue 1,000 题
A1-H1
B1-H1
```

轮次：

```text
2 个实验臂 × 1,000 题 × 1 次
= 2,000 个 question-run
```

如果成本允许，再加：

```text
B1-H2
```

全量实验使用 paired bootstrap 2,000 次，按题配对。

### E10：性能与成本实验

目的：

量化混合召回的实际部署代价。

测量：

```text
BM25 query latency
BGE query embedding latency
RRF latency
Qwen per-batch latency
end-to-end per-question latency
embedding cache size
Qwen requests
passage judgments
input/output tokens
cache hit rate
failure/retry count
```

环境：

```text
同一台机器
同一服务地址
同一并发
冷启动和热启动分开记录
```

重复：

```text
20 道题 × 5 次热运行
```

报告：

```text
mean
p50
p95
max
```

不能把论文中的索引时间与本机索引时间混在一起比较，除非硬件、分词、磁盘缓存和输入完全相同。

## 8. 实验矩阵汇总

| ID | 目的 | 数据 | 配置 | H | 独立运行 | 优先级 |
|---|---|---|---:|---:|---:|---|
| E0 | 环境与接口 | 3 题 | BM25/Dense/Qwen | 0 | 1 | 必须 |
| E1 | 初始召回 | 100 题 | BM25/Dense/Hybrid | 0 | 1 | 必须 |
| E2 | 评分器校准 | 20 题 | BM25/Hybrid+Qwen | 0 | 3 | 必须 |
| E3 | 核心主对照 | 100 题 | A/B+Qwen | 0,1 | 1 | 必须 |
| E4 | 两轮扩展 | 100 题 | A/B+Qwen | 2 | 1 | 条件 |
| E5 | 机制控制 | 100 题 | 预算/上下文/最终评分 | 1,2 | 1 | 条件 |
| E6 | Qwen 波动 | 20 题 | A/B H=1 | 1 | 3 | 必须 |
| E7 | 跨数据集 | 2×100 题 | A/B H=1 | 1 | 1 | 条件 |
| E8 | Reranker | 100 题 | A2/B2 | 1 | 1 | 可选 |
| E9 | 全量确认 | MuSiQue 1,000 | A/B H=1 | 1 | 1 | 条件 |
| E10 | 性能成本 | 20 题 | 最优配置 | 1 | 5 次热跑 | 条件 |

## 9. 指标总表

### 9.1 检索效果

```text
R@1
R@2
R@5
R@10
AE@5
AE@10
```

主指标：

```text
R@10
AE@10
```

次要指标：

```text
R@1
R@2
R@5
AE@5
```

### 9.2 初始召回诊断

```text
InitialEntryRecall
InitialFullRecall
GoldMedianRank
HybridAddedEvidence
HybridAddedCandidateCount
HybridNoiseRate
```

定义：

```text
HybridAddedEvidence
    B 初始集合相对 A 初始集合新增的 gold passage 数

HybridNoiseRate
    B 相对 A 新增候选中非 gold passage 的比例
```

### 9.3 SAG2 过程诊断

```text
SeedPrecision
SeedRecall
BridgeRecovery
NewEvidenceAtH1
NewEvidenceAtH2
```

### 9.4 错误类型

```text
no-entry
wrong-seed
no-bridge
wrong-bridge
incomplete
judge-parse-failed
model-request-failed
```

每题只能有一个主错误类型，其它错误作为附加标签。

### 9.5 Qwen 评分器诊断

```text
JSON parse success rate
candidate score coverage
gold vs non-gold score distribution
AUROC
repeat Spearman
repeat top-10 Jaccard
```

AUROC 和相关性只用于诊断，不作为主结论。

### 9.6 成本

```text
embedding requests
embedding tokens
embedding build time
embedding cache bytes
query embedding requests
Qwen requests
Qwen passage judgments
Qwen input tokens
Qwen output tokens
p50/p95 end-to-end latency
retry rate
```

## 10. 统计方法

### 10.1 配对比较

A/B 必须按 `dataset_sample_id` 配对：

```text
对每道题先算：
ΔR@10 = R@10(B) - R@10(A)
ΔAE@10 = AE@10(B) - AE@10(A)
再对题目求均值
```

不能只比较两个不配对的总体均值。

### 10.2 Bootstrap

正式 100 题和 1,000 题使用：

```text
paired percentile bootstrap
2,000 resamples
95% interval
seed = 20261015
```

每个 resample 保留同一道题在 A/B 中的配对关系。

### 10.3 分层

按问题 hop 报告：

```text
2-hop
3-hop
4-hop
```

不只看总体平均，因为总体平均可能被 2-hop 占多数所主导。

### 10.4 胜平负

报告：

```text
B 优于 A 的题数
A/B 相同的题数
A 优于 B 的题数
```

### 10.5 多重比较

主结论只围绕：

```text
R@10
AE@10
```

其它指标、hop 分层、RRF 权重和 reranker 作为次要或探索性结果，不把它们全部包装成“显著提升”。

## 11. 通过、停止和扩展规则

### 11.1 进入主实验前

必须满足：

- 环境依赖完整；
- 单元测试通过；
- 原 BM25 基线行为不变；
- 100 题 sample 固定；
- Qwen JSON 可解析；
- 候选与 gold 不发生 prompt 泄漏；
- embedding cache 校验通过。

### 11.2 进入 H=1 后

满足任一条件可以继续：

```text
Hybrid 初始入口召回提高
或
Hybrid H0 的 R@10/AE@10 提高
或
Hybrid 新增候选被 Qwen/seed 实际利用
```

### 11.3 进入 H=2

需要：

```text
B1-H1 在 R@10 或 AE@10 上优于 A1-H1
```

并且改善不能完全集中在单一 hop。

### 11.4 进入全量 1,000 题

建议同时满足：

- B1-H1 的配对均值差为正；
- 至少一个主指标的 95% bootstrap 区间不跨 0；
- AE@10 没有明显下降；
- 成本增加有明确上限；
- 机制控制排除了“只是多判断”的解释。

如果 100 题区间仍然很宽，应把 1,000 题视作降低不确定性的确认实验，而不是把它当作重新选参数的机会。

### 11.5 停止条件

满足任一条件应停止并转错误分析：

- Qwen 评分覆盖不足；
- 大量静默解析失败；
- Hybrid 新增的几乎全是无关候选；
- H1 后主指标无收益且 seed precision 下降；
- 原 BM25 基线无法复现；
- sample/corpus/index 发生不一致。

## 12. 实验运行顺序

推荐顺序：

```text
1. E0 环境与接口
2. E1 初始召回
3. E2 Qwen 校准
4. E3 A/B H0/H1
5. E6 Qwen 稳定性
6. E4 H2
7. E5 机制控制
8. E7 跨数据集
9. E8 Reranker
10. E9 全量
11. E10 性能成本
```

实际判断顺序：

```text
E0 失败
  → 停止

E1 没有入口收益
  → 先看噪声；若噪声高，不进入大规模扩展

E2 不稳定
  → 停止主实验并修评分器

E3 无收益
  → 分析 no-entry/wrong-seed/wrong-bridge

E3 有收益
  → 做 E6、E4、E5

机制解释成立且成本可接受
  → 做 E7/E9

基础对照完成
  → 才做 Reranker 和参数敏感性
```

## 13. 输出目录与记录

建议：

```text
output\experiments\
  E0-preflight\
  E1-initial-retrieval\
  E2-scorer-calibration\
  E3-main-ab\
  E4-two-hop-confirmation\
  E5-mechanism-controls\
  E6-qwen-stability\
  E7-generalization\
  E8-reranker\
  E9-full-musique\
  E10-cost-latency\
```

每个实验目录至少包含：

```text
config.json
details.jsonl
metrics.json
summary.md
```

主实验还应包含：

```text
paired-comparison.json
error-analysis.jsonl
seed-analysis.json
cost.json
```

## 14. 实验前检查清单

### 代码

- [ ] `numpy` 已安装；
- [ ] `pyproject.toml` 已更新；
- [ ] 原测试通过；
- [ ] BM25 默认路径不变；
- [ ] initial retriever 只在 hop 0 生效；
- [ ] checkpoint 可恢复。

### 数据

- [ ] `musique.json` SHA256 固定；
- [ ] `musique_corpus.json` SHA256 固定；
- [ ] 100 题 ID 文件固定；
- [ ] BM25 索引签名固定；
- [ ] 每道题都有 gold evidence。

### Embedding

- [ ] 模型名和维度记录；
- [ ] 11,656 个 chunk 全部有向量；
- [ ] query/passage 使用同一模型；
- [ ] L2 归一化标记正确；
- [ ] cache 可被 manifest 校验。

### Qwen

- [ ] 实际模型名记录；
- [ ] prompt hash 固定；
- [ ] temperature=0；
- [ ] 返回 JSON 格式固定；
- [ ] 不包含 gold、answer、supporting flag；
- [ ] 失败不会静默变成 0 分；
- [ ] batch size 在 A/B 两组一致。

### 运行

- [ ] 固定随机种子；
- [ ] 独立重复使用独立请求缓存；
- [ ] API Key 不出现在日志；
- [ ] 记录并发；
- [ ] 记录实际成本与延迟。

## 15. 第一轮建议实际执行范围

为了尽快得到可解释结果，第一次正式研究建议只做：

```text
E0 + E1 + E2 + E3 + E6
```

即：

1. 10 题 smoke；
2. 初始 BM25/Dense/Hybrid 比较；
3. 20 题 Qwen 校准；
4. 100 题 A/B 的 H=0 与 H=1；
5. 20 题 × 3 次 Qwen 稳定性重复。

暂不运行：

```text
H=2
budget matching
跨数据集
全量 1,000 题
Reranker
RRF 权重搜索
```

这样可以先用较小成本回答最重要的问题：

> BGE 新增的候选，能否被 Qwen 识别并最终转化成多跳证据收益？

如果答案是否定的，H=2、全量和 Reranker 都不会解决根本问题。

## 16. 最终结论将如何写

实验报告分成四层，避免过度解读：

### 第一层：候选生成

回答：

```text
BGE 是否给 BM25 增加了新的 gold entry？
新增的是证据还是噪声？
```

### 第二层：Qwen 过滤

回答：

```text
Qwen 能否识别这些新增证据？
评分是否稳定？
是否有大量主题相关但非必要证据？
```

### 第三层：SAG2 扩展

回答：

```text
Hybrid 的初始收益是否能穿过 seed 和 term expansion？
两轮扩展是否仍有边际收益？
四跳问题是否改善更多？
```

### 第四层：外部有效性

回答：

```text
结论是否在 2Wiki/HotpotQA 出现？
是否值得扩展到 1,000 题？
成本和延迟是否可以接受？
```

只有在四层证据都成立时，才能写成：

> 在本实验环境下，BM25+BGE 混合初始召回改善了 SAG2-style 多跳证据检索。

如果只有 E1 的初始检索改善，则最多写成：

> BGE 提高了初始候选中的 gold passage 数量，但没有证据表明该收益能转化为最终多跳证据覆盖。
