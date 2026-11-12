# SAG2 实验现状与服务器执行计划

日期：2026-11-12

## 1. 已完成的工作

### 1.1 代码实现

已完成并推送到 GitHub：

```text
SAG-Benchmark/sag2/scripts/embedding_client.py
SAG-Benchmark/sag2/scripts/build_passage_embeddings.py
SAG-Benchmark/sag2/scripts/vector_retriever.py
SAG-Benchmark/sag2/scripts/hybrid_retriever.py
SAG-Benchmark/sag2/scripts/qwen_evidence_scorer.py
SAG-Benchmark/sag2/scripts/make_musique_ab_subset.py
SAG-Benchmark/sag2/scripts/run_initial_retrieval_eval.py
SAG-Benchmark/sag2/scripts/run_hybrid_compare.py
SAG-Benchmark/sag2/scripts/run_hybrid_suite.py
SAG-Benchmark/sag2/scripts/analyze_hybrid_results.py
```

`search_sag2()` 已支持首轮 retriever 注入：

```text
hop 0：BM25 / Dense / Hybrid
hop >= 1：继续使用原始 SAG2 扩展
```

### 1.2 数据和缓存

已确认：

```text
MuSiQue questions：1,000
MuSiQue corpus：11,656 passages
BM25 chunks：11,656
固定 A/B/C 100 题：40 个 2-hop、30 个 3-hop、30 个 4-hop
固定 smoke 10 题：4 个 2-hop、3 个 3-hop、3 个 4-hop
```

本地已构建：

```text
BGE-M3 embedding：11,656 × 1024
embedding model：bge-m3:latest
```

### 1.3 模型

论文原版 JEV 未使用。当前实验使用：

```text
请求模型：qwen3.5-122b-a10b
实际返回模型：qwen3.5-122b-a10b
temperature：0
```

必须把这个模型视为 Qwen evidence scorer，而不是论文 JEV 的严格复现。

## 2. 已完成的实验结果

### 2.1 100 题初始召回，不调用 Qwen

| 方法 | R@5 | R@10 | AE@10 | InitialFullRecall |
|---|---:|---:|---:|---:|
| BM25 | 43.42% | 50.67% | 14.00% | 37.00% |
| BGE Dense | 50.00% | 59.50% | 25.00% | 62.00% |
| BM25+BGE 等权 RRF | 52.67% | 58.00% | 22.00% | 55.00% |

结论：

- BGE 单独召回优于 BM25；
- Hybrid 的 R@5 最高；
- Hybrid 的 R@10 低于 BGE-only；
- 等权 RRF 可能稀释 BGE 的候选顺序。

Hybrid 相对 BM25 新增 3,732 个候选，其中 33 个是 gold evidence，平均噪声率约 99.06%。

### 2.2 10 题 H=0 和 H=1

| 组别 | R@5 | R@10 | AE@10 |
|---|---:|---:|---:|
| A：BM25 H=0 | 60.00% | 60.00% | 20.00% |
| B：Hybrid H=0 | 49.17% | 61.67% | 20.00% |
| C：BGE-only H=0 | 61.67% | 70.00% | 40.00% |
| A：BM25 H=1 | 52.50% | 57.50% | 20.00% |
| B：Hybrid H=1 | 49.17% | 49.17% | 10.00% |
| C：BGE-only H=1 | 60.83% | 65.83% | 40.00% |

### 2.3 阶段诊断

10 题上的阶段指标：

| 配置 | 初始 R@5/R@10 | Qwen后 R@5/R@10 | 最终 R@5/R@10 |
|---|---|---|---|
| BM25 H=0 | 40.83 / 53.33 | 60.00 / 60.00 | 60.00 / 60.00 |
| Hybrid H=0 | 40.83 / 56.67 | 49.17 / 61.67 | 49.17 / 61.67 |
| BM25 H=1 | 40.83 / 53.33 | — | 52.50 / 57.50 |
| Hybrid H=1 | 40.83 / 56.67 | — | 49.17 / 49.17 |

Qwen 对 gold 和非 gold 的平均分：

| 配置 | gold 平均分 | 非 gold 平均分 |
|---|---:|---:|
| BM25 H=0 | 0.6364 | 0.0061 |
| Hybrid H=0 | 0.3846 | 0.0041 |
| BM25 H=1 | 0.6190 | 0.0078 |
| Hybrid H=1 | 0.5227 | 0.0049 |

Seed precision：

```text
BM25 H=1：18%
Hybrid H=1：17%
```

诊断结论：

1. Hybrid 初始 top-10 增加了 gold 覆盖；
2. Qwen 对 Hybrid 中的 gold 平均打分更低；
3. 扩展 seed 质量很低，错误 seed 会传播扩展错误；
4. DPP 在 10 题中没有改变 R@5/R@10；
5. 当前下降主要来自 Qwen 重排与错误 seed，而不是 DPP；
6. BGE-only 是当前最有希望的候选方案。

## 3. 当前判断

可以支持的表述：

> 在固定 10 题 smoke 中，BGE-only+Qwen 明显优于 BM25 和等权 RRF，但样本不足，需要 100 题确认。

不能支持的表述：

> BM25 和 BGE 的所有混合方法都无效。

当前没有测试：

- 偏向 BGE 的加权 RRF；
- BGE top 70 + BM25 top 30 的配额融合；
- Dense-first、BM25 仅补召回的方案；
- 100 题上的统计确认。

## 4. 下一阶段计划

### 阶段 S0：服务器预检

1. 拉取 GitHub 最新代码；
2. 安装 Python 依赖；
3. 配置 `.env`；
4. 确认 BGE 和 Qwen 服务；
5. 构建 BM25 索引和 BGE embedding cache。

### 阶段 S1：100 题 H=0

依次运行：

```text
A1-H0：BM25+Qwen
B1-H0：Hybrid+Qwen
C1-H0：BGE-only+Qwen
```

目的：

- 确认初筛收益能否穿过 Qwen；
- 判断等权 RRF 是否确实稀释 BGE；
- 不进行参数调优。

### 阶段 S2：阶段分析

输出：

- A/B/C 的 R@1/2/5/10；
- AE@5/10；
- 配对 Bootstrap；
- gold/non-gold Qwen 分数；
- seed precision；
- 失败题列表。

停止条件：

- 如果 C 不超过 A，暂停 H=1；
- 如果 C 明显超过 A，进入 H=1；
- 不在同一 100 题上反复调 RRF 权重后作为最终结果。

### 阶段 S3：100 题 H=1

只在 S1/S2 通过后运行：

```text
A1-H1
B1-H1
C1-H1
```

目的：

- 判断 BGE-only 的初始收益能否被 SAG2 扩展放大；
- 检查 seed precision 和 bridge recovery；
- 比较 Qwen 扩展对三种初始召回的影响。

### 阶段 S4：RRF 改进实验

只有确认 C 优于 B 后再做：

```text
等权 RRF
偏 BGE RRF：1:2
偏 BGE RRF：1:3
BGE 配额 + BM25 补召回
```

必须使用独立调参集，避免在最终 100 题上过拟合。

### 阶段 S5：扩展到 1,000 题

条件：

- 100 题方向一致；
- 配对区间支持收益；
- 成本可接受；
- seed 质量没有明显恶化。

## 5. 服务器运行方式

### 5.1 拉取代码

```bash
git clone https://github.com/seeding-sun/SAG2.git
cd SAG2/SAG-Benchmark/sag2
```

后续更新：

```bash
git pull --ff-only
```

### 5.2 安装依赖

```bash
uv sync --frozen --dev
```

### 5.3 配置环境

```bash
cp .env.example .env
```

在服务器本地编辑 `.env`：

```dotenv
SAG_EMBEDDING_BASE_URL=http://192.168.2.108:8013/v1
SAG_EMBEDDING_MODEL=bge-m3:latest
SAG_EMBEDDING_API_KEY=

SAG_EVIDENCE_BASE_URL=http://192.168.2.69:9099/v1
SAG_EVIDENCE_MODEL=qwen3.5-122b-a10b
SAG_EVIDENCE_API_KEY=服务器本地填写
```

`.env` 已被 Git 忽略，不能提交。

### 5.4 100 题 H=0

从仓库根目录执行：

```bash
cd SAG-Benchmark/sag2
uv run python scripts/run_hybrid_suite.py \
  --dataset-dir ../../dataset \
  --sample-file ../../dataset/musique_100_for_ab.json \
  --output-root ../../output/abc-100/H0 \
  --arms bm25 dense hybrid \
  --max-hops 0 \
  --candidate-k 100 \
  --seed-k 10 \
  --concurrency 4 \
  --evidence-concurrency 1
```

该脚本会：

1. 检查 BM25 索引；
2. 检查 BGE cache；
3. 缺少时自动构建；
4. 依次运行 A、C、B；
5. 使用 `details.jsonl` checkpoint 支持恢复。

### 5.5 100 题 H=1

H=0 通过后：

```bash
uv run python scripts/run_hybrid_suite.py \
  --dataset-dir ../../dataset \
  --sample-file ../../dataset/musique_100_for_ab.json \
  --output-root ../../output/abc-100/H1 \
  --arms bm25 dense hybrid \
  --max-hops 1 \
  --candidate-k 100 \
  --seed-k 10 \
  --concurrency 4 \
  --evidence-concurrency 1
```

### 5.6 生成配对分析

```bash
uv run python scripts/analyze_hybrid_results.py \
  --a-dir ../../output/abc-100/H0/A1-H0 \
  --b-dir ../../output/abc-100/H0/C1-H0 \
  --output ../../output/abc-100/H0/comparison-CvsA.json

uv run python scripts/analyze_hybrid_results.py \
  --a-dir ../../output/abc-100/H0/B1-H0 \
  --b-dir ../../output/abc-100/H0/C1-H0 \
  --output ../../output/abc-100/H0/comparison-CvsB.json
```

H=1 将路径中的 `H0` 改为 `H1`。

## 6. 预计耗时

基于当前实际 Qwen 服务的 10 题 smoke：

```text
单个 H=0 实验臂：约 9.5 分钟 / 10 题
三个 H=0 实验臂：约 28.5 分钟 / 10 题
估算 100 题：约 4.75 小时

单个 H=1 实验臂：约 20.3 分钟 / 10 题
三个 H=1 实验臂：约 61 分钟 / 10 题
估算 100 题：约 10 小时
```

服务器并发提高后可能更快，但三组必须使用相同并发和 batch 参数。

## 7. GitHub 更新原则

代码、脚本、固定题集、实验方案和汇总报告提交到 GitHub。

以下内容不提交：

```text
.env
.cache/
SQLite 索引
BGE .npy 缓存
Qwen 原始请求缓存
output/ 下的逐题大文件
```

服务器跑完后，应把以下内容复制成小型汇总文件再更新 GitHub：

```text
metrics.json
comparison-*.json
comparison-*.md
实验报告 Markdown
```

