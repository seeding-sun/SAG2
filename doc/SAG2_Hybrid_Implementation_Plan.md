# SAG2 混合检索实验实现方案

## 1. 文档目的

本文给出在现有 SAG2 Benchmark 代码上实现“纯 BM25 初始召回”与“BM25+BGE-M3 混合初始召回”对照实验的工程方案。

本文重点回答四件事：

1. 现有代码已经具备什么能力；
2. 还缺哪些模块，每个模块的输入、输出和接口是什么；
3. 如何以最小改动接入 `search_sag2()`，同时保持原 BM25 基线可复现；
4. 如何保证 A/B 两组除初始召回外完全一致。

本文是实施规范，不替代已有实验计划：

```text
D:\learn\知识库\RAG\SAG2\doc\SAG2_Hybrid_Experiment_Plan.md
```

本阶段只实现受控缩小实验，不修改论文原始算法，不直接承诺“混合检索一定更好”。

## 2. 实验问题与边界

核心研究问题：

> 在相同题目、语料、证据判断模型、候选预算、seed 规则、扩展轮次和评测规则下，BM25+BGE-M3 的混合初始召回是否比纯 BM25 初始召回提高多跳证据覆盖？

主对照只有两组：

```text
A：BM25 top-100            → Qwen 证据判断 → 原 SAG2 扩展与排序
B：BM25+BGE RRF top-100    → Qwen 证据判断 → 原 SAG2 扩展与排序
```

主实验第一轮只跑：

```text
H=0
H=1
```

以下内容不属于第一阶段主实验：

- BGE Reranker；
- 修改扩展词算法；
- 修改 seed 数量；
- 修改 DPP；
- 修改最终候选数；
- 调整 RRF 权重做参数搜索。

否则无法判断收益究竟来自初始召回、精排还是算法参数。

## 3. 现有资产

### 3.1 数据

数据目录：

```text
D:\learn\知识库\RAG\SAG2\dataset
```

已确认：

```text
musique.json          1,000 道题
musique_corpus.json   11,656 个 source passages
```

题目 hop 分布：

```text
2-hop   518
3-hop   316
4-hop   166
```

固定 100 题子集建议按：

```text
2-hop  40
3-hop  30
4-hop  30
```

分层抽样。

### 3.2 BM25 索引

已有 SQLite FTS5 索引：

```text
D:\learn\知识库\RAG\SAG2\SAG-Benchmark\sag2\.cache\sag2\musique.sqlite
```

现有统计：

```text
question_count = 1000
document_count = 11656
chunk_count = 11656
status = indexed
```

当前语料恰好是一段一个 chunk，因此 `chunk_count == document_count`。实现仍应坚持“向量与 BM25 使用同一个 chunk 单位”，避免以后长文档被切分时产生不可比问题。

不需要重新下载数据，也不需要重建当前 BM25 索引。

### 3.3 现有代码能力

关键文件：

```text
scripts/sag2.py
scripts/run_sag2_benchmark.py
scripts/model_api.py
scripts/benchmark_data.py
scripts/local_reranker.py
```

已经具备：

- SQLite FTS5/BM25；
- `CorpusIndex`；
- SAG2 主循环；
- JEV 批量评分协议；
- expansion seeds；
- 扩展词选择；
- 扩展检索；
- 共享上下文；
- DPP top-10；
- Recall 和 AllEvidence 指标；
- checkpoint/resume；
- HTTP 重试、响应缓存、Token 统计和密钥脱敏；
- 本地 reranker 参考实现。

### 3.4 模型服务

BGE-M3 Embedding：

```text
base_url: http://192.168.2.108:8013/v1
model: bge-m3:latest
endpoint: /embeddings
```

BGE Reranker：

```text
base_url: http://192.168.2.108:8014/v1
model: bge-reranker-v2-m3
endpoint: /rerank
```

Qwen 证据判断：

```text
base_url: http://192.168.2.69:9099/v1
model: qwen3.5-122b-a10b
endpoint: /chat/completions
```

真实 API Key 只能通过本地 `.env` 或环境变量注入，禁止写入代码、文档、日志、manifest 或 Git。

### 3.5 Python 环境

指定解释器：

```text
D:\learn\.venv\Scripts\python.exe
```

项目依赖目前未声明 NumPy。实施时需要：

- 先检查目标环境是否已有 NumPy；
- 如果没有，在目标环境中安装；
- 在 `pyproject.toml` 中增加 `numpy` 依赖。

## 4. 当前缺口

需要新增：

```text
scripts/embedding_client.py
scripts/build_passage_embeddings.py
scripts/vector_retriever.py
scripts/hybrid_retriever.py
scripts/qwen_evidence_scorer.py
scripts/make_musique_ab_subset.py
scripts/run_hybrid_compare.py
scripts/analyze_hybrid_results.py
```

第二阶段可选：

```text
scripts/bge_rerank_client.py
```

需要最小修改：

```text
scripts/sag2.py
```

需要配置：

```text
SAG-Benchmark\sag2\.env
```

## 5. 总体架构

```text
MuSiQue questions
       │
       ├────────────── A route ──────────────┐
       │                                     │
       │                         BM25 top-100 │
       │                                     │
       └────────────── B route ──────────────┤
                                             │
                         BM25 top-100 ───────┤
                         BGE-M3 top-100 ─────┤
                                             ▼
                                        RRF fusion
                                             │
                                          top-100
                                             │
                                             ▼
                                   Qwen evidence scorer
                                             │
                                             ▼
                              original SAG2 seed/expansion
                                             │
                                             ▼
                                  DPP + final top-k
                                             │
                                             ▼
                                R@k / AE@k / diagnostics
```

实现原则：

1. A/B 两组使用完全相同的题目、corpus、Qwen、prompt、温度、候选预算、seed、扩展词和最大轮次；
2. 只替换第一轮候选生成；
3. 从第二轮开始，两组仍调用原来的 `CorpusIndex.retrieve_expansion()`；
4. 主实验不叠加 Reranker；
5. 原始 BM25 路径必须保持默认行为不变。

## 6. 检索单位与公平性

### 6.1 统一使用 `chunks.id`

BM25 当前返回 `chunks.id`。向量检索也必须返回同一个 `chunks.id`，而不是另造 passage 编号。

Embedding 构建脚本应从 SQLite `chunks` 表按 `id` 顺序读取：

```sql
SELECT id, title, heading, content, document_id, source_id
FROM chunks
ORDER BY id
```

默认嵌入文本：

```text
title
heading
content
```

其中 `heading` 为空时跳过。

这样可以保证：

- BM25 与向量使用同一个检索单位；
- RRF 可以直接按 `chunks.id` 融合；
- 候选经过 Qwen 和 SAG2 时字段结构一致；
- 评测仍使用原有 `full_doc` 单位；
- 不需要把 vector passage 再映射回 BM25 chunk。

### 6.2 固定参数

主实验固定：

```text
candidate_k = 100
seed_k = 10
expansion_term_limit = 20
max_hops = 0 或 1
RRF k = 60
embedding batch size = 32
Qwen temperature = 0
```

所有 A/B 组必须使用同一组值。

## 7. 新增配置

增加 `.env.example` 配置模板；真实 `.env` 不进入 Git：

```dotenv
SAG_EMBEDDING_BASE_URL=http://192.168.2.108:8013/v1
SAG_EMBEDDING_MODEL=bge-m3:latest
SAG_EMBEDDING_API_KEY=
SAG_EMBEDDING_BATCH_SIZE=32
SAG_EMBEDDING_DIMENSION=1024

SAG_EVIDENCE_BASE_URL=http://192.168.2.69:9099/v1
SAG_EVIDENCE_MODEL=qwen3.5-122b-a10b
SAG_EVIDENCE_API_KEY=
SAG_EVIDENCE_BATCH_SIZE=10

SAG_RERANK_BASE_URL=http://192.168.2.108:8014/v1
SAG_RERANK_MODEL=bge-reranker-v2-m3
SAG_RERANK_API_KEY=
```

如果 embedding 服务要求任意非空 Key，只在本机 `.env` 填写。日志中统一替换为 `<REDACTED>`。

## 8. 模块设计

### 8.1 `embedding_client.py`

职责：

- 调用 OpenAI 兼容的 `/v1/embeddings`；
- 支持 query 和 passage 批量嵌入；
- 校验返回数量、索引、维度与数值合法性；
- 可选 L2 normalization；
- 超时、重试和响应缓存；
- 返回实际模型名与 Token usage。

建议接口：

```python
class EmbeddingClient:
    async def embed(
        self,
        texts: list[str],
        *,
        input_type: str = "passage",
    ) -> tuple[np.ndarray, dict]:
        ...

    async def embed_query(self, text: str) -> tuple[np.ndarray, dict]:
        ...
```

请求示例：

```json
{
  "model": "bge-m3:latest",
  "input": ["text 1", "text 2"],
  "encoding_format": "float"
}
```

必要约束：

- 单批最大条数来自配置；
- 不静默截断 passage；
- 返回向量行数必须等于输入条数；
- `data[i].index` 必须覆盖 `0..n-1`；
- 所有向量维度一致；
- 不含 NaN/Inf；
- 归一化后必须验证范数接近 1；
- 缓存键不能包含 API Key。

### 8.2 `build_passage_embeddings.py`

职责：

- 打开现有 BM25 索引；
- 按 `chunks.id` 读取待嵌入文本；
- 分批调用 BGE-M3；
- 保存可复用和可校验的 embedding cache；
- 中断后可以依靠 HTTP 响应缓存继续。

建议产物：

```text
.cache/sag2/musique-bge-m3.npy
.cache/sag2/musique-bge-m3.ids.json
.cache/sag2/musique-bge-m3.manifest.json
```

缓存规模估算：

```text
11,656 × 1024 × float32(4 bytes) ≈ 47.8 MB
```

`ids.json` 只保存与 `.npy` 行号一一对应的 `chunks.id`。

`manifest.json` 至少包含：

```json
{
  "schema_version": 1,
  "dataset": "musique",
  "corpus_sha256": "...",
  "chunk_policy": "desktop-sentences-v1",
  "chunk_chars": 2200,
  "retrieval_unit": "chunk",
  "embedding_model": "bge-m3:latest",
  "embedding_base_url": "http://192.168.2.108:8013/v1",
  "dimension": 1024,
  "count": 11656,
  "normalized": true,
  "batch_size": 32,
  "text_template": "title\\nheading\\ncontent",
  "created_at": "ISO-8601",
  "code_sha256": "..."
}
```

写入要求：

- 先写临时文件，再原子替换；
- 最终加载时重新检查 corpus hash、chunk policy、模型和维度；
- 任一不一致则拒绝加载，不允许“凑合使用旧向量”。

### 8.3 `vector_retriever.py`

职责：

- 加载 embedding cache；
- 对问题做 query embedding；
- 使用余弦相似度或归一化后的点积；
- 返回 top-k 的原始 `chunks.id`；
- 从 `CorpusIndex` 取回完整候选字段；
- 遵守 `excluded` 集合。

建议接口：

```python
class VectorRetriever:
    async def search(
        self,
        index: "CorpusIndex",
        query: str,
        *,
        limit: int,
        excluded: set[int] | None = None,
    ) -> tuple[list[dict], dict]:
        ...
```

输出的候选必须至少包含：

```text
id
document_id
title
heading
content
full_doc
source_id
dense_score
dense_rank
retrieval_route = "dense"
```

第一版直接使用 NumPy 矩阵乘法和 `argpartition`，不引入向量数据库。

由于全语料只有 11,656 个 chunk，单机内存检索足够，Faiss/Milvus 不是第一阶段依赖。

### 8.4 `hybrid_retriever.py`

职责：

- 获取 BM25 top-N；
- 获取 BGE top-N；
- 按 `chunks.id` 去重；
- 使用 RRF 融合；
- 修复单路漏召回；
- 返回固定 top-100，并保留两路原始排名。

建议接口：

```python
class HybridRetriever:
    name = "hybrid"

    async def search(
        self,
        index: "CorpusIndex",
        query: str,
        *,
        excluded: set[int],
        limit: int,
    ) -> list[dict]:
        ...
```

第一版默认两路各取 100：

```text
pool_k = 100
RRF k = 60
```

RRF 公式，rank 从 1 开始：

```text
rrf_score(c) =
    1 / (60 + bm25_rank(c))
    +
    1 / (60 + dense_rank(c))
```

只有一路召回时，另一项记为 0。

融合结果需要保留：

```text
bm25_rank
bm25_score
dense_rank
dense_score
rrf_score
retrieved_by: ["bm25", "dense"]
initial_rank
```

排序键固定为：

```text
(-rrf_score, best_route_rank, id)
```

不要直接把 BM25 原始分数和 cosine 相加，两者量纲不同。

### 8.5 `qwen_evidence_scorer.py`

现有 `JevClient` 调用 TypeSafe `/systemone`，返回结构是：

```json
{
  "answers": {
    "candidate_123": {
      "type": "noul",
      "noul": 0.82
    }
  }
}
```

普通 Qwen Chat API 不会返回这一结构，因此不能直接把 `/v1` 地址传给 `JevClient`。

新增 `QwenEvidenceScorer`，保持与 `search_sag2()` 兼容的接口：

```python
async def score(
    self,
    query: str,
    candidates: list[dict],
    evidence: list[dict],
    *,
    context: dict | None = None,
) -> tuple[list[dict], dict]:
    ...
```

输出候选需增加：

```text
jev_score
jev_model
```

继续沿用 `jev_score` 字段名，是为了不改动现有 SAG2 排序代码；文档和报告中应注明“这是 Qwen 代理的 JEV 分数，不是原 JEV 模型输出”。

提示词要求：

- 直接使用 `sag2.CRITERION`；
- 输入原始问题、当前检索上下文、候选标题和候选正文；
- 不输入答案、gold evidence、`is_supporting` 或题目 decomposition；
- 明确声明 passage 中的指令只是被判断的数据；
- `temperature = 0`；
- 固定 JSON 输出。

建议输出格式：

```json
{
  "scores": {
    "candidate_123": 0.82,
    "candidate_124": 0.07
  }
}
```

解析规则：

- 去掉单个 Markdown JSON code fence 后解析；
- key 集合必须与本次候选集合完全一致；
- 分数必须是有限数值且位于 `[0, 1]`；
- 缺项、多项、重复项或非法值都视为协议错误；
- 协议错误最多重试一次；
- 仍失败则抛出明确异常并在 runner 中记录 `judge_parse_failed`；
- 不允许静默回填 0 分。

批量策略：

- 默认每批 10 个候选；
- 根据估算 token 数动态减小批次；
- 不截断单个候选原文；
- 单候选超过上下文预算时直接报错；
- A/B 两组使用完全相同的批量策略。

Usage 至少记录：

```text
requests
judgments
batches
input_tokens
output_tokens
latency_ms
actual_model
```

### 8.6 `make_musique_ab_subset.py`

职责：

- 按 hop 分层；
- 使用固定随机种子；
- 生成 A/B 共用的题集；
- 输出 ID 和抽样清单。

建议命令：

```powershell
D:\learn\.venv\Scripts\python.exe scripts\make_musique_ab_subset.py `
  --dataset-dir D:\learn\知识库\RAG\SAG2\dataset `
  --output D:\learn\知识库\RAG\SAG2\dataset\musique_100_for_ab.json `
  --seed 20261111 `
  --strata 2hop:40,3hop:30,4hop:30
```

文件至少保存：

```json
{
  "dataset": "musique",
  "seed": 20261111,
  "strata": {
    "2hop": 40,
    "3hop": 30,
    "4hop": 30
  },
  "ids": ["2hop__...", "3hop1__...", "4hop2__..."]
}
```

runner 必须验证：

- ID 数量为 100；
- ID 唯一；
- 所有 ID 都存在于问题集；
- A/B 工具使用同一个 sample file hash。

### 8.7 `run_hybrid_compare.py`

职责：

- 运行单组 A 或 B；
- 读取固定 sample file；
- 复用现有数据加载、指标和输出格式；
- 保存初始候选、评分、seed、扩展词和错误；
- 支持中断恢复。

建议 CLI：

```text
--retrieval-mode bm25|hybrid
--dataset-dir
--sample-file
--output-dir
--candidate-k 100
--seed-k 10
--max-hops 0|1
--expansion-term-limit 20
--embedding-base-url
--embedding-model
--evidence-base-url
--evidence-model
--evidence-batch-size 10
--concurrency
--resume
--preflight
```

运行流：

1. 加载并验证固定的 100 题；
2. 打开同一个 BM25 索引；
3. 构造 A 或 B 的 initial retriever；
4. 调用 `search_sag2(initial_retriever=...)`；
5. 保存完整 trace；
6. 用原有 `unique_documents()` 和 `recall_metrics()` 计算指标；
7. 写入 checkpoint 和 summary。

建议输出：

```text
output\hybrid-ab\
  paired_sample.json
  A1-H0\
    config.json
    details.jsonl
    metrics.json
    search_results.json
    requests\
  A1-H1\
  B1-H0\
  B1-H1\
  comparison_report.json
```

每题详情至少保留：

```text
question_index
dataset_sample_id
question
initial_candidate_ids
initial_candidate_docs
initial_bm25_rank
initial_dense_rank
initial_rrf_score
seed_ids
frontier_ids
expansion_terms
final_candidate_ids
retrieved_docs
metrics
all_evidence_metrics
judge_usage
elapsed_ms
error_type
error_message
```

### 8.8 `analyze_hybrid_results.py`

职责：

- 按 `dataset_sample_id` 配对 A/B 结果；
- 计算均值差值；
- 计算配对 bootstrap 区间；
- 输出分层和错误分布；
- 生成最终实验报告所需表格。

不要只比较总均值，还要输出：

- 每题的 A/B 差值；
- 2/3/4-hop 分层差值；
- 胜/平/负题数；
- 失败类型变化。

### 8.9 `bge_rerank_client.py`（第二阶段）

仅支持：

```text
POST /v1/rerank
```

用途：

- 对已召回的 100 个候选做 cross-encoder 精排；
- 不能替代全 corpus 向量召回；
- 不能只在 B 组使用它。

第二阶段如启用 Reranker，必须做成新的完整对照：

```text
A2：BM25 → Reranker → Qwen
B2：Hybrid → Reranker → Qwen
```

## 9. `sag2.py` 最小改造

### 9.1 新增 initial retriever 协议

增加可选参数：

```python
async def search_sag2(
    index: CorpusIndex,
    client: "EvidenceScorer",
    query: str,
    *,
    candidate_k: int = 100,
    seed_k: int = 10,
    max_hops: int = 2,
    expansion_term_limit: int = 20,
    context: dict | None = None,
    initial_retriever: "InitialRetriever | None" = None,
) -> dict:
```

协议：

```python
class InitialRetriever(Protocol):
    name: str

    async def search(
        self,
        index: CorpusIndex,
        query: str,
        *,
        excluded: set[int],
        limit: int,
    ) -> list[dict]:
        ...
```

### 9.2 仅替换 hop 0

原逻辑：

```python
expression = index.match_query(query)
hits = index.retrieve(expression, seen, candidate_k)
```

新逻辑分两种：

```text
initial_retriever is None
    → 保持原 BM25 行为

initial_retriever is not None
    → hop 0 调用 initial_retriever
    → hop >= 1 继续调用 index.retrieve_expansion()
```

注意：使用 Hybrid 时，不能因为 BM25 查询表达式为空就提前终止。即使 BM25 无命中，Dense 仍可能召回有效候选。

扩展轮：

```text
hop >= 1
  → 不调用 initial_retriever
  → 继续使用原 SAG2 expansion
```

### 9.3 增加 trace

首轮 trace 至少增加：

```text
retriever: "bm25" | "hybrid"
initial_rank
bm25_rank
dense_rank
rrf_score
```

每次 JEV/Scorer trace 增加：

```text
frontier_ids
seed_ids
```

字段只增加，不删除，保持旧输出可读。

### 9.4 保持默认兼容

必须满足：

- `initial_retriever=None` 时排序、trace 结构和结果与改造前一致；
- 原有 `tests/test_sag2.py` 和 reference fixtures 全部通过；
- 现有 `run_sag2_benchmark.py` 不需要修改即可继续跑原 BM25 实验。

## 10. 初始召回与 SAG2 的连接

### 10.1 A 组

```python
result = await search_sag2(
    index,
    scorer,
    question,
    candidate_k=100,
    seed_k=10,
    max_hops=hops,
    expansion_term_limit=20,
)
```

### 10.2 B 组

```python
hybrid = HybridRetriever(
    bm25_index=index,
    vector_retriever=vector_retriever,
    pool_k=100,
    rrf_k=60,
)

result = await search_sag2(
    index,
    scorer,
    question,
    candidate_k=100,
    seed_k=10,
    max_hops=hops,
    expansion_term_limit=20,
    initial_retriever=hybrid,
)
```

### 10.3 `fusion_score`

`search_sag2()` 当前会把本轮顺序转成：

```text
fusion_score = 1 / (61 + rank)
```

第一阶段保留该逻辑。Hybrid 的原始 RRF 分数只用于生成初始顺序和诊断，不直接改变后续 SAG2 融合规则。

这意味着实验变量严格限定为：

```text
initial candidate set + initial candidate order
```

## 11. 指标与诊断

### 11.1 主指标

```text
Recall@1
Recall@2
Recall@5
Recall@10
AllEvidence@5
AllEvidence@10
```

### 11.2 初始召回指标

```text
InitialEntryRecall
    初始 top-100 至少包含一个 gold passage 的题目比例

InitialFullRecall
    初始 top-100 包含全部 gold passages 的题目比例

HybridAddedEvidence
    B 初始集合相对 A 新增的 gold passage 数量

HybridNoise
    B 初始集合相对 A 新增的候选数
    以及其中非 gold passage 的比例
```

### 11.3 过程指标

```text
SeedPrecision
BridgeRecovery
```

定义：

```text
SeedPrecision
    SAG2 选作下一轮 seed 的候选中
    属于 gold evidence 的比例

BridgeRecovery
    最终 gold evidence 中
    首次由 hop >= 1 扩展召回的占比或数量
```

### 11.4 错误分类

```text
no-entry
wrong-seed
no-bridge
wrong-bridge
incomplete
judge-parse-failed
model-request-failed
```

分类必须基于 trace 和 gold 计算，不能人工猜测后覆盖原始结果。

### 11.5 成本

```text
embedding 请求次数
embedding 构建时间
query embedding 次数
向量缓存大小
Qwen 请求次数
passage judgments
input/output tokens
每题延迟
失败和重试次数
```

## 12. 实验矩阵

第一轮主实验：

| 组别 | 初始召回 | 证据判断 | 扩展 |
|---|---|---|---|
| A1-H0 | BM25 | Qwen | 0 |
| A1-H1 | BM25 | Qwen | 1 |
| B1-H0 | BM25+BGE RRF | Qwen | 0 |
| B1-H1 | BM25+BGE RRF | Qwen | 1 |

BM25 纯检索基线 A0 可以从 A1 的 `initial_candidate_ids` 离线计算，不需要额外调用 Qwen。

建议运行顺序：

```text
A1-H0
B1-H0
A1-H1
B1-H1
```

先完成 H=0，再决定是否值得继续 H=1。

## 13. 分阶段实施

### 阶段 0：接口 smoke

检查：

- `/v1/embeddings` 的返回格式；
- batch 最大条数；
- Qwen `/chat/completions` 是否支持 `response_format`；
- Qwen 是否能稳定输出合法 JSON；
- reranker 的返回字段和分数范围；
- 三个服务的实际模型名和上下文限制。

验收：

```text
每条测试都产生脱敏后的日志
失败时可定位到服务、HTTP 状态和响应摘要
```

### 阶段 1：依赖与配置

- 检查 `numpy`；
- 增加 `.env.example`；
- 增加 `.env` 加载；
- 更新 `pyproject.toml`；
- 确认 `.env`、`.cache` 和请求日志不会进入 Git。

### 阶段 2：Embedding cache

- 生成 11,656 行向量；
- 验证 ID 与 `chunks.id` 对齐；
- 验证维度、有限值、归一化和 corpus hash；
- 做 10 道 query embedding 检索抽查。

验收：

```text
vector_count == index.total
len(ids) == vector_count
ids 唯一
vector.shape[1] 固定
```

### 阶段 3：Vector/Hybrid retriever

- 实现 dense top-100；
- 实现 BM25 top-100；
- 实现 RRF top-100；
- 验证候选无重复；
- 验证 `excluded` 生效；
- 验证两路排名字段完整。

### 阶段 4：Qwen scorer

- 接入现有 `CRITERION`；
- 批量判断；
- 严格解析 JSON；
- 记录 usage；
- 检查 prompt 中不存在 gold 信息。

### 阶段 5：SAG2 接线

- 增加 `initial_retriever`；
- hop 0 使用注入 retriever；
- hop >= 1 使用原 expansion；
- 保持默认 BM25 行为；
- 增加 seed/frontier trace。

### 阶段 6：测试和 10 题 smoke

先跑：

```text
10 题
A1-H0
B1-H0
```

检查：

- 两组题目 ID 完全一致；
- 候选数一致；
- Qwen prompt 和 temperature 一致；
- 每题有初始候选、seed、trace 和 usage；
- 估算 100 题耗时和调用成本。

### 阶段 7：100 题主实验

运行：

```text
A1-H0
B1-H0
A1-H1
B1-H1
```

每组输出独立目录，禁止覆盖。

### 阶段 8：可选 Reranker

只有主实验存在明确收益后再开始。

### 阶段 9：分析报告

至少回答：

1. BGE 是否提高 `InitialEntryRecall`；
2. BGE 是否提高 `InitialFullRecall`；
3. Hybrid 新增候选中有多少最终成为 gold evidence；
4. H=0/H=1 下 R@10 和 AE@10 如何变化；
5. Qwen 判断成本和延迟增加多少；
6. 改善/退化主要集中在哪个 hop；
7. 失败从 `no-entry` 转向了 `wrong-seed` 还是 `no-bridge`。

## 14. 测试清单

### 14.1 单元测试

```text
test_embedding_client.py
test_vector_retriever.py
test_hybrid_retriever.py
test_qwen_evidence_scorer.py
test_hybrid_runner.py
```

重点用例：

- embedding 返回行数错误；
- embedding 维度不一致；
- 向量非有限值；
- 缓存 manifest 与 corpus hash 不一致；
- excluded ID 被错误返回；
- RRF 并列排序不稳定；
- BM25 单路命中；
- Dense 单路命中；
- 两路重复 ID 被重复计分；
- Qwen 返回缺项、多项、字符串分数；
- Qwen 请求中出现 gold 字段；
- A/B sample ID 不一致；
- checkpoint 题目身份变化。

### 14.2 回归测试

必须运行：

```powershell
Set-Location D:\learn\知识库\RAG\SAG2\SAG-Benchmark\sag2
D:\learn\.venv\Scripts\python.exe -m pytest tests -q
```

核心断言：

- 原 `search_sag2()` 在 `initial_retriever=None` 时行为不变；
- 原 paper reference fixture 通过；
- 原 `run_sag2_benchmark.py` 可继续 index-only 运行。

### 14.3 集成测试

- 真实 BGE embedding 小批量；
- 真实 Qwen 10 候选评分；
- 10 题 A/B smoke；
- 中断后 resume；
- 同一题重复运行命中响应缓存；
- 密钥不出现在生成文件中。

## 15. 日志、缓存与恢复

### 15.1 日志

每个请求至少记录：

```text
service
endpoint
request_hash
model
status
retry_count
latency_ms
usage
```

禁止记录：

```text
Authorization header
真实 API Key
包含密钥的异常字符串
```

### 15.2 响应缓存

可复用现有 `model_api.RequestJournal` 的思路：

- 以 endpoint + payload 的 SHA256 作为键；
- 成功后保存 response；
- 重跑相同请求直接复用；
- 失败 attempt 单独保存；
- 缓存不含认证头。

### 15.3 Runner checkpoint

- 使用 `details.jsonl` 逐题追加；
- 每题完成立即 flush；
- 启动时按 `dataset_sample_id` 去重；
- 发现重复 ID 直接报错；
- 配置或代码 hash 变化时拒绝恢复旧输出。

## 16. 运行命令草案

以下命令仅作为实施后的调用模板。

进入项目：

```powershell
Set-Location D:\learn\知识库\RAG\SAG2\SAG-Benchmark\sag2
```

生成固定题集：

```powershell
D:\learn\.venv\Scripts\python.exe scripts\make_musique_ab_subset.py `
  --dataset-dir D:\learn\知识库\RAG\SAG2\dataset `
  --output D:\learn\知识库\RAG\SAG2\dataset\musique_100_for_ab.json `
  --seed 20261111 `
  --strata 2hop:40,3hop:30,4hop:30
```

构建 embedding cache：

```powershell
D:\learn\.venv\Scripts\python.exe scripts\build_passage_embeddings.py `
  --dataset-dir D:\learn\知识库\RAG\SAG2\dataset `
  --index .cache\sag2\musique.sqlite `
  --cache-dir .cache\sag2 `
  --batch-size 32
```

运行 A：

```powershell
D:\learn\.venv\Scripts\python.exe scripts\run_hybrid_compare.py `
  --retrieval-mode bm25 `
  --dataset-dir D:\learn\知识库\RAG\SAG2\dataset `
  --sample-file D:\learn\知识库\RAG\SAG2\dataset\musique_100_for_ab.json `
  --output-dir output\hybrid-ab\A1-H1 `
  --max-hops 1 `
  --candidate-k 100 `
  --seed-k 10
```

运行 B：

```powershell
D:\learn\.venv\Scripts\python.exe scripts\run_hybrid_compare.py `
  --retrieval-mode hybrid `
  --dataset-dir D:\learn\知识库\RAG\SAG2\dataset `
  --sample-file D:\learn\知识库\RAG\SAG2\dataset\musique_100_for_ab.json `
  --output-dir output\hybrid-ab\B1-H1 `
  --max-hops 1 `
  --candidate-k 100 `
  --seed-k 10
```

H=0 使用 `--max-hops 0`，其余参数不得变化。

## 17. 风险与应对

### 风险 1：检索单位不一致

应对：

- 向量直接使用 `chunks.id`；
- manifest 绑定 chunk policy 和 chunk chars；
- runner 断言所有候选 ID 存在于当前索引。

### 风险 2：Qwen 不稳定

应对：

- temperature=0；
- 固定 JSON schema；
- 批量大小固定；
- smoke test；
- 解析失败显式报错；
- 可重复运行少量题估算方差。

### 风险 3：A/B 后续流程发生漂移

应对：

- 只允许 hop 0 使用不同 retriever；
- hop >= 1 强制调用同一 `retrieve_expansion()`；
- config 中保存 seed、term limit、max hops 和 prompt hash；
- analyzer 按题配对。

### 风险 4：RRF 稀释精确命中

应对：

- 第一版不改权重；
- 保留两路原始 rank；
- 统计 gold passage 来自 BM25、Dense 还是两者；
- 后续再做权重敏感性实验。

### 风险 5：100 题结论不稳定

应对：

- 固定题集；
- 配对比较；
- 报告均值差和 bootstrap 区间；
- 按 hop 分层；
- 不把探索性结果写成论文结论。

### 风险 6：API Key 泄漏

应对：

- 只从环境变量读取；
- 请求日志脱敏；
- manifest 不保存 Key；
- 提交前扫描 `.env`、日志和输出；
- 实验完成后轮换已经暴露过的 Key。

## 18. 验收标准

实现完成必须同时满足：

1. 原 BM25 路径默认行为不变；
2. 原测试全部通过；
3. embedding cache 数量、ID、维度、hash 全部校验通过；
4. Vector/Hybrid 每组候选固定为 100 且无重复；
5. A/B 两组使用完全相同的 100 个题目 ID；
6. A/B 两组使用相同 Qwen prompt、temperature、batch size；
7. 主实验不启用 Reranker；
8. 每题保存初始候选、seed、frontier、扩展词、最终候选和 usage；
9. 解析失败不会静默变成 0 分；
10. 中断后可以恢复，且不会覆盖已完成结果；
11. 输出文件不包含真实 API Key；
12. 最终报告能按题配对，并给出分层指标和失败类型。

## 19. 最终交付物

代码：

```text
scripts/embedding_client.py
scripts/build_passage_embeddings.py
scripts/vector_retriever.py
scripts/hybrid_retriever.py
scripts/qwen_evidence_scorer.py
scripts/make_musique_ab_subset.py
scripts/run_hybrid_compare.py
scripts/analyze_hybrid_results.py
```

修改：

```text
scripts/sag2.py
pyproject.toml
.env.example
```

数据与缓存：

```text
D:\learn\知识库\RAG\SAG2\dataset\musique_100_for_ab.json
D:\learn\知识库\RAG\SAG2\SAG-Benchmark\sag2\.cache\sag2\musique-bge-m3.npy
D:\learn\知识库\RAG\SAG2\SAG-Benchmark\sag2\.cache\sag2\musique-bge-m3.ids.json
D:\learn\知识库\RAG\SAG2\SAG-Benchmark\sag2\.cache\sag2\musique-bge-m3.manifest.json
```

实验结果：

```text
output\hybrid-ab\A1-H0\
output\hybrid-ab\A1-H1\
output\hybrid-ab\B1-H0\
output\hybrid-ab\B1-H1\
output\hybrid-ab\comparison_report.json
```

文档：

```text
doc\SAG2_Hybrid_Experiment_Report.md
```

## 20. 结论

最稳妥的实现路径是：

```text
保留原 SAG2
  +
新增向量召回与 RRF
  +
新增 Qwen 证据评分适配器
  +
只替换第一轮候选生成
```

不应先改扩展算法或重写主循环。只要首轮 retriever 注入点、统一 chunk 单位、固定题集和严格参数冻结做到位，这个实验就能清楚回答“混合召回究竟改善了入口覆盖，还是只增加了无关噪声”。
