# SAG2 混合检索缩小版实验计划

## 1. 项目目标

在现有 SAG2 Benchmark 基础上，增加 BGE-M3 向量召回和本地 Qwen 证据判断，验证以下问题：

> 在相同题目、语料、JEV 替代模型、候选预算和扩展规则下，BM25+BGE-M3 混合初始召回是否比纯 BM25 初始召回获得更高的多跳证据召回？

第一阶段只做缩小版，不追求完整复现论文全部实验。

目标规模：

- 数据集：MuSiQue；
- 问题数：100 题；
- Corpus：11,656 个 source passages；
- 扩展轮数：H=0 和 H=1；
- 初始候选数：100；
- 返回与评测：R@1、R@2、R@5、R@10、AE@5、AE@10。

## 2. 当前已有资源

### 2.1 数据

本地数据目录：

```text
D:\learn\知识库\RAG\SAG2\dataset
```

已核对：

```text
musique.json          1,000 个问题
musique_corpus.json   11,656 个 passages
```

数据规模与 SAG2 论文中的 MuSiQue benchmark 一致。

### 2.2 BM25 索引

已有索引：

```text
D:\learn\知识库\RAG\SAG2\SAG-Benchmark\sag2\.cache\sag2\musique.sqlite
```

已有运行摘要显示：

```text
question_count = 1000
document_count = 11656
chunk_count = 11656
status = indexed
```

第一版实验不需要重新下载 MuSiQue，也不需要重建 BM25 索引。

### 2.3 代码

已有：

```text
D:\learn\知识库\RAG\SAG2\SAG-Benchmark\sag2
```

已包含：

- SQLite FTS5 BM25；
- SAG2 主检索循环；
- JEV 客户端协议；
- 扩展 seed 与 term 选择；
- 最终 top-10 多样性排序；
- 指标与结果输出；
- Local reranker 扩展；
- Agentic SAG2 扩展；
- 测试和历史结果。

### 2.4 模型服务

#### BGE-M3 Embedding

```text
base_url: http://192.168.2.108:8013/v1
model: bge-m3:latest
```

预期接口：

```text
POST /v1/embeddings
```

用途：

- query embedding；
- passage embedding；
- 全 corpus 向量召回。

#### BGE Reranker

```text
base_url: http://192.168.2.108:8014/v1
model: bge-reranker-v2-m3
rerank endpoint: /v1/rerank
```

用途：

- 对已经召回的候选进行 cross-encoder 精排；
- 不负责全 corpus 初始召回。

#### Qwen 证据判断模型

```text
base_url: http://192.168.2.69:9099/v1
model: qwen3.5-397b-a17b
```

预期接口：

```text
POST /v1/chat/completions
```

用途：

- 替代论文中的 JEV；
- 判断候选是否为完整答案链中的必要事实；
- 为 SAG2 排序、seed 选择和扩展提供 0-1 分数。

API Key 不写入本计划、代码或 Git。

### 2.5 Python 环境

指定使用：

```text
D:\learn\.venv
```

运行时应优先使用：

```text
D:\learn\.venv\Scripts\python.exe
```

### 2.6 系统工具

已确认：

```text
uv     已安装
git    已安装
Python 3.12 已安装
```

当前普通 PowerShell 中未找到 `cl`。如果已有 SQLite 扩展可以复用，这不是立即阻塞项；如果重新编译失败，则需要 Visual Studio Build Tools 的 C++ 工作负载。

## 3. 当前仍然缺少什么

### 3.1 环境配置

缺少：

```text
SAG-Benchmark\sag2\.env
```

需要增加不含真实密钥的配置模板，例如：

```dotenv
EMBEDDING_BASE_URL=http://192.168.2.108:8013/v1
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_API_KEY=

RERANK_BASE_URL=http://192.168.2.108:8014/v1
RERANK_MODEL=bge-reranker-v2-m3

EVIDENCE_BASE_URL=http://192.168.2.69:9099/v1
EVIDENCE_MODEL=qwen3.5-397b-a17b
EVIDENCE_API_KEY=
```

真实密钥仅通过本地 `.env` 或环境变量提供。

### 3.2 BGE-M3 Embedding 客户端

缺少：

- OpenAI 兼容 `/v1/embeddings` 客户端；
- 批量请求；
- 超时与重试；
- 向量维度和数量校验；
- L2 normalization；
- 失败记录。

### 3.3 Passage Embedding 缓存

需要生成：

```text
.cache/sag2/musique-bge-m3.npy
.cache/sag2/musique-bge-m3.ids.json
.cache/sag2/musique-bge-m3.manifest.json
```

缓存应保存：

- 向量；
- passage/document ID 顺序；
- embedding 模型名称；
- corpus SHA256；
- 向量维度；
- 请求参数；
- 生成时间。

如果使用 1024 维 float32：

```text
11,656 × 1024 × 4 bytes ≈ 47.8 MB
```

### 3.4 Dense Retriever

缺少：

- query embedding；
- 余弦相似度检索；
- top-k；
- passage ID 与原 corpus 的映射；
- 对重复 passage 的去重。

第一版不需要向量数据库，使用 NumPy 矩阵点积即可。

### 3.5 Hybrid Retriever

缺少：

- BM25 top-k 与 BGE top-k；
- RRF 融合；
- 去重；
- 融合后固定返回 100 个候选；
- 保留两路原始排名供日志分析。

建议第一版使用 RRF，不直接相加 BM25 原始分数和 cosine：

```text
RRF score =
    1 / (60 + BM25_rank)
    +
    1 / (60 + BGE_rank)
```

### 3.6 本地 Qwen 证据判断适配器

现有 `JevClient` 使用 TypeSafe `/systemone` 批量决策协议，不能直接把普通 Chat API 地址替进去。

需要新增：

```text
QwenEvidenceScorer.score(query, candidates, evidence)
```

输出接口要与 `JevClient.score()` 保持一致：

```text
候选 passage
+ jev_score
+ jev_model
+ usage
```

评分要求：

- 使用 SAG2 原始 `CRITERION`；
- 原始问题与检索上下文一并输入；
- 不使用 gold evidence 或答案；
- 输出 `{id, score}`；
- `score ∈ [0,1]`；
- 温度设为 0；
- 尽量一次判断多个候选；
- 所有 A/B 组使用完全相同的提示词和解析逻辑。

注意：

> 这是“本地 LLM 替代 JEV”，不是原论文 JEV 的严格复现。

### 3.7 A/B 实验运行脚本

缺少：

```text
run_hybrid_compare.py
```

至少要支持：

```text
--retrieval-mode bm25
--retrieval-mode hybrid
--dataset-dir
--limit
--offset
--candidate-k
--seed-k
--expansions
--qwen-base-url
--qwen-model
--embedding-base-url
--embedding-model
```

### 3.8 100 题分层子集

缺少固定题集文件，建议：

```text
dataset/musique_100_for_ab.json
```

建议按 hop 分层：

```text
2-hop: 40
3-hop: 30
4-hop: 30
```

记录固定随机 seed，确保 A/B 组使用完全相同的题目。

### 3.9 测试与契约检查

尚需验证：

- BGE-M3 是否支持 batch；
- embedding 返回维度；
- 是否已经归一化；
- 超长 passage 处理；
- Qwen 是否支持 Chat Completions；
- Qwen 是否能稳定输出 JSON；
- `temperature=0` 是否生效；
- Qwen 上下文是否容纳 evidence + candidate；
- reranker 分数范围和返回结构；
- 并发限制。

## 4. 实验设计

### 4.1 主对照

#### A 组：BM25 + Qwen 证据判断

```text
Question
  ↓
SQLite FTS5 BM25 top-100
  ↓
Qwen evidence scoring
  ↓
SAG2 seed/term expansion
  ↓
Qwen re-scoring
```

#### B 组：BM25 + BGE-M3 + Qwen 证据判断

```text
Question
  ├── BM25 top-100
  └── BGE-M3 top-100
        ↓
       RRF
        ↓
     top-100
        ↓
Qwen evidence scoring
        ↓
SAG2 seed/term expansion
        ↓
Qwen re-scoring
```

除初始候选生成外，其余流程完全一致。

### 4.2 实验矩阵

| 组别 | 初始召回 | 证据判断 | 扩展 |
|---|---|---|---|
| A0 | BM25 | 无 | 无 |
| A1-H0 | BM25 | Qwen | 0 |
| A1-H1 | BM25 | Qwen | 1 |
| B1-H0 | BM25+BGE-M3 RRF | Qwen | 0 |
| B1-H1 | BM25+BGE-M3 RRF | Qwen | 1 |

第二轮扩展先不跑。只有第一轮证明有收益后，再增加：

```text
A1-H2
B1-H2
```

### 4.3 可选 Reranker 对照

只在主实验完成后增加：

```text
A2：BM25 top-100 → BGE Reranker → Qwen
B2：Hybrid top-100 → BGE Reranker → Qwen
```

不能只在 Hybrid 组增加 reranker，否则无法区分收益来源。

Reranker 的原始分数不能直接与 Qwen 的 JEV 分数相加。第一版应只用于排序，再由 Qwen 做必要证据判断。

## 5. 评价指标

### 5.1 原有指标

```text
Recall@1
Recall@2
Recall@5
Recall@10
AllEvidence@5
AllEvidence@10
```

### 5.2 入口诊断

```text
InitialEntryRecall
    初始 top-100 是否至少有一个 gold supporting passage

InitialFullRecall
    初始 top-100 是否已包含全部 gold passages

HybridAddedEvidence
    BGE 为初始池新增了多少 gold passage

HybridNoise
    BGE 新增候选中有多少不是 gold passage
```

### 5.3 SAG2 过程诊断

```text
SeedPrecision
    expansion seeds 中 gold evidence 的比例

BridgeRecovery
    正确后续证据有多少来自扩展检索

No-entry
    初始池没有可用的中间入口

Wrong-seed
    Qwen 选择了错误证据作为 seed

No-bridge
    seed 中没有可用的桥接词

Wrong-bridge
    扩展词指向错误主题

Incomplete
    找到了部分证据但没有凑齐完整答案链
```

### 5.4 成本与性能

```text
embedding 请求数
embedding 构建时间
向量缓存大小
Qwen 请求数
passage judgments
输入/输出 token
每题延迟
失败和重试次数
```

## 6. 代码实施计划

### 模块 1

```text
scripts/embedding_client.py
```

负责调用 BGE-M3 的 `/v1/embeddings`。

### 模块 2

```text
scripts/build_passage_embeddings.py
```

负责：

- 读取本地 MuSiQue corpus；
- 批量调用 BGE-M3；
- 写入向量缓存；
- 保存 corpus hash 和模型元数据。

### 模块 3

```text
scripts/hybrid_retriever.py
```

负责：

- BGE dense top-k；
- BM25 top-k；
- RRF；
- 去重；
- 固定候选预算。

### 模块 4

```text
scripts/qwen_evidence_scorer.py
```

负责：

- 调用本地 Qwen Chat；
- 按 SAG2 criterion 判断候选；
- 解析 JSON；
- 提供与 JevClient 相同的 `score()` 接口；
- 保存原始请求和响应。

### 模块 5

```text
scripts/run_hybrid_compare.py
```

负责：

- 选择 A/B 组；
- 保持参数一致；
- 运行 100 题；
- 汇总指标；
- 保存逐题候选、seed、term、分数和错误。

### 模块 6

```text
scripts/make_dev_subset.py
```

负责按 hop 分层抽取 100 题并保存固定 ID 列表。

## 7. 分阶段里程碑

### 阶段 0：环境与接口检查

- 使用 `D:\learn\.venv`；
- 检查 aiohttp、numpy、jieba、python-dotenv；
- 测试 BGE embedding；
- 测试 Qwen chat；
- 测试 BGE rerank；
- 确认 token、向量维度和 JSON 输出。

完成标准：

```text
三个本地服务都能返回稳定格式，不包含真实密钥日志。
```

### 阶段 1：BGE embedding 缓存

- 生成 11,656 个 passage embeddings；
- 校验数量和对应 ID；
- 固定缓存版本。

完成标准：

```text
向量行数 = 11,656
ID 数 = 11,656
所有向量维度一致
```

### 阶段 2：Hybrid Retriever

- BM25 top-100；
- BGE top-100；
- RRF；
- 输出 top-100。

完成标准：

```text
每组候选数固定为 100
无重复 passage
保存 BM25 rank、BGE rank、RRF score
```

### 阶段 3：Qwen 证据判断

- 使用论文 SAG2 criterion；
- 输出 0-1 分数；
- 批量评分；
- 记录 token 和请求。

完成标准：

```text
所有候选都有有效分数
JSON 解析成功
不使用 gold labels
```

### 阶段 4：十题 smoke test

```text
10 题
A：BM25+Qwen H=0
B：Hybrid+Qwen H=0
```

目的：

- 验证完整链路；
- 检查候选和日志；
- 估算 100 题耗时。

### 阶段 5：100 题正式缩小实验

运行：

```text
A0
A1-H0
A1-H1
B0
B1-H0
B1-H1
```

完成标准：

- 全部题目有结果或明确错误；
- 指标可按题配对；
- 输出可复现配置；
- 结果不混用不同参数。

### 阶段 6：Reranker 可选实验

如果 Hybrid 初始召回有明显收益，再运行：

```text
BM25 → BGE Reranker → Qwen
Hybrid → BGE Reranker → Qwen
```

### 阶段 7：分析报告

至少回答：

1. BGE 是否提高了初始入口召回？
2. BGE 新增候选是否最终转化为 gold evidence？
3. Hybrid 是否提高了 R@10 和 AE@10？
4. Hybrid 是否增加了 Qwen 判断成本？
5. 改善主要出现在 2-hop、3-hop 还是 4-hop？
6. 哪些失败属于 no-entry、wrong-seed 或 wrong-bridge？

## 8. 验收标准

缩小版实验完成需要满足：

- 100 题 A/B 组完全配对；
- 始终使用同一个 Qwen prompt 和 temperature；
- BM25 与 Hybrid 初始候选预算相同；
- 不向 Qwen 暴露答案或 gold evidence；
- 向量缓存包含 corpus hash；
- 每道题保存候选、seed、扩展词、分数和错误；
- 结果可以按题配对计算 bootstrap 区间；
- 运行中断后可以恢复或至少不会覆盖已有结果；
- API Key 不进入代码、日志或 Git。

## 9. 风险与应对

### 风险 1：Qwen 判断不稳定

应对：

- temperature=0；
- 固定 JSON 输出；
- 10 题 smoke test；
- 少量题目重复运行测量方差；
- 必要时改用点对点评分，减少一次请求中的候选数量。

### 风险 2：Qwen 协议不支持严格 JSON

应对：

- 增加宽松 JSON 提取器；
- 失败时重试；
- 仍失败则记录 `judge_parse_failed`，不要静默给 0 分。

### 风险 3：BGE 请求慢或 batch 限制

应对：

- passage embedding 只做一次并缓存；
- 查询时只 embed question；
- 分批处理并保存成功结果；
- 控制并发。

### 风险 4：RRF 稀释 BM25 精确命中

应对：

- 第一版固定 RRFs；
- 保存两路原始排名；
- 分析 gold passage 是否只来自一路；
- 不提前调大量融合权重。

### 风险 5：单位不一致

必须保持：

- 使用同一 source passage；
- 使用同一 question ID；
- 使用同一 gold evidence；
- 使用同一 dedup 规则；
- 使用同一评测脚本。

### 风险 6：API Key 暴露

已经在本轮对话中出现过真实密钥。实验完成后应轮换。

## 10. 最终目标产物

```text
D:\learn\知识库\RAG\SAG2\
  doc\
    SAG2_Hybrid_Experiment_Plan.md
    SAG2_Hybrid_Experiment_Report.md
  dataset\
    musique.json
    musique_corpus.json
    musique_100_for_ab.json
  SAG-Benchmark\sag2\
    .cache\sag2\
      musique.sqlite
      musique-bge-m3.npy
      musique-bge-m3.ids.json
    output\hybrid-ab\
      A1-H0\
      A1-H1\
      B1-H0\
      B1-H1\
```

## 11. 结论

当前已具备数据和 BM25 基线，主要新增工作是：

```text
BGE-M3 embedding
  +
BM25/BGE RRF
  +
Qwen evidence scorer
  +
paired A/B runner
```

不需要先重造整个 SAG2。最合理的路径是保持现有 BM25、扩展算法和后处理不变，只替换“初始候选生成”和“证据判断模型”，用受控实验衡量混合召回是否真正改善多跳证据覆盖。

