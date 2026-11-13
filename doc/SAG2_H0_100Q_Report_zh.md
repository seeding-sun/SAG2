# SAG2 混合检索实验 —— 100 题 H=0 阶段报告

日期：2026-11-12
执行环境：Linux，4×A800 80GB，Python 3.12.11
数据：MuSiQue 1,000 题，corpus 11,656 passages，固定 100 题分层子集
代码版本：`e075ee3`（main）

---

## 1. 实验目标

本次实验不对标论文原始数值（论文使用托管 JEV 模型），而是回答一个具体问题：

> **把 SAG2 的首轮召回从 BM25 换成 BGE-M3（或 BGE+BM25 融合），
> 新增的候选能否被 Qwen 证据评分器识别，并最终转化为多跳证据检索收益？**

实验设计遵循"只改一个变量"原则：仅替换 hop 0 的初始召回器，其余全部冻结。

```text
A 臂：BM25 top-100                 → Qwen 证据评分 → SAG2 流程
B 臂：BM25 + BGE-M3 等权 RRF top-100 → Qwen 证据评分 → SAG2 流程
C 臂：BGE-M3 dense top-100          → Qwen 证据评分 → SAG2 流程
```

冻结参数：

```text
candidate_k        100
seed_k             10
max_hops           0        （本轮只做初始评分，不扩展）
top_k              10
expansion_term_limit 20
embedding model    bge-m3:latest  (1024 维)
evidence model     qwen3.5-122b-a10b
temperature        0
response_format    json_object
concurrency        4
evidence_concurrency 1
```

---

## 2. 环境预检（E0）

| 检查项 | 要求 | 实测结果 | 结论 |
|---|---|---|---|
| Python | ≥3.11 | uv 安装 3.12.11（系统自带 3.10.12） | 通过 |
| 依赖安装 | `uv sync --frozen --dev` | 成功 | 通过 |
| 单元测试 | 全部通过 | **106 passed**（58s） | 通过 |
| 代码风格 | ruff | All checks passed | 通过 |
| SQLite FTS5 | 必须可用 | SQLite 3.37.2，FTS5 建表成功 | 通过 |
| BM25 索引 | 11,656 chunks | 11,656 chunks / 27 MB / 建库 7.3s | 通过 |
| BGE-M3 embedding | 1024 维 | 11,656 × 1024 / 47.8 MB / 构建 32s | 通过 |
| BGE 服务 | 模型名匹配 | `192.168.2.108:8013`，返回 `bge-m3:latest` | 通过 |
| Qwen 服务 | 模型名匹配 + 严格 JSON | `192.168.2.69:9099`，`qwen3.5-122b-a10b` | 通过 |
| GPU | 第一阶段不需要 | 4×A800 80GB（仅重排实验需要） | 富余 |
| 磁盘 | ≥2 GB | `/data` 剩余 18 TB | 通过 |

> 说明：`verify_paper_results.py` / `verify_agentic_results.py` 依赖
> `2wikimultihopqa.json`、`hotpotqa.json`，本仓库未包含这两个数据集，
> 因此论文数值复现校验无法运行。这不影响 MuSiQue 实验。

---

## 3. E1：100 题初始召回（不调用 Qwen）

目的：把"候选生成"与"证据判断"解耦，先确认 BGE 是否真的补足了 BM25 的入口。

| 方法 | R@1 | R@2 | R@5 | R@10 | AE@5 | AE@10 | 全证据入口召回 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 28.08 | 34.17 | 43.42 | 50.67 | 11.00 | 14.00 | 37.00 |
| BGE-M3 dense | 25.83 | 36.25 | 50.00 | **59.50** | 15.00 | **25.00** | **62.00** |
| BM25+BGE 等权 RRF | 25.50 | 36.50 | **52.67** | 58.00 | 18.00 | 22.00 | 55.00 |

Hybrid 相对 BM25 的增量：

```text
新增候选      3,730
其中 gold       33
平均噪声率   99.06%
```

**本结果与作者在 Windows 上的历史结果逐位一致**，证明数据、索引、题集、
chunk 策略、RRF 参数全部对齐，实验可复现。

判读：

- BGE 单独召回的 R@10 与全证据入口召回均最优（+8.83pp / +25pp）；
- 等权 RRF 的 R@5 最高，但 R@10 反而回落，说明融合把 BGE 的头部候选打散了；
- 三路入口召回（entry recall）均为 100%，即"至少命中一条 gold"没有问题，
  瓶颈在"命中全部 gold"。

---

## 4. E3：100 题 H=0 主实验（含 Qwen 证据评分）

### 4.1 主结果

| 臂 | R@1 | R@2 | R@5 | R@10 | AE@5 | AE@10 | 初始 R@10 | 初始全证据 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: BM25+Qwen | 30.83 | 44.00 | 53.08 | 57.67 | 15.00 | 19.00 | 50.67 | 37.00 |
| B: RRF+Qwen | 31.08 | 44.08 | 58.67 | 62.33 | 25.00 | 30.00 | 58.00 | 55.00 |
| **C: BGE+Qwen** | **30.92** | **45.92** | **61.25** | **65.83** | **27.00** | **32.00** | 59.50 | 62.00 |

### 4.2 配对 Bootstrap 检验（2000 次重采样，单位=配对题目）

**C − A（BGE 对 BM25）**

| 指标 | 均值 | 95% CI | 显著 |
|---|---:|---|---|
| Recall@1 | +0.08 pp | [−3.17, +3.50] | 否 |
| Recall@2 | +1.92 pp | [−3.83, +7.58] | 否 |
| **Recall@5** | **+8.17 pp** | **[+2.92, +13.75]** | **是** |
| **Recall@10** | **+8.17 pp** | **[+4.17, +12.50]** | **是** |
| **AllEvidence@5** | **+12.00 pp** | **[+5.00, +19.00]** | **是** |
| **AllEvidence@10** | **+13.00 pp** | **[+5.00, +22.00]** | **是** |

逐题 R@10：**C 赢 35 / 平 47 / 输 18**

**B − A（等权 RRF 对 BM25）**

| 指标 | 均值 | 95% CI | 显著 |
|---|---:|---|---|
| **Recall@5** | **+5.58 pp** | **[+1.83, +9.67]** | **是** |
| **Recall@10** | **+4.67 pp** | **[+0.75, +8.58]** | **是** |
| **AllEvidence@5** | **+10.00 pp** | **[+4.00, +17.00]** | **是** |
| **AllEvidence@10** | **+11.00 pp** | **[+4.00, +19.00]** | **是** |

逐题 R@10：**B 赢 33 / 平 47 / 输 20**

**C − B（BGE 对等权 RRF）**

| 指标 | 均值 | 95% CI | 显著 |
|---|---:|---|---|
| Recall@5 | +2.58 pp | [−2.33, +7.42] | 否 |
| Recall@10 | +3.50 pp | [−0.00, +7.42] | 否（临界） |
| AllEvidence@10 | +2.00 pp | [−4.00, +8.00] | 否 |

逐题 R@10：**C 赢 22 / 平 63 / 输 15**

### 4.3 分 hop 拆解（40 题 2-hop / 30 题 3-hop / 30 题 4-hop）

| hop | n | C−A R@10 | C−A AE@10 | B−A R@10 | B−A AE@10 |
|---|---:|---:|---:|---:|---:|
| 2-hop | 40 | **+11.25 pp** | **+22.5 pp** | +3.75 pp | +12.5 pp |
| 3-hop | 30 | +8.89 pp | +13.3 pp | **+8.89 pp** | **+20.0 pp** |
| **4-hop** | 30 | **+3.33 pp** | **+0.0 pp** | +1.67 pp | **+0.0 pp** |

三条臂在 4-hop 上的全证据命中率完全相同（AE@10 = **3.33%**，即 30 题中仅 1 题命中；AE@5 = 0%）。

### 4.4 成本

| 臂 | Qwen 判定次数 | 输入 token | 输出 token | 单题均耗时 | 单臂总耗时 |
|---|---:|---:|---:|---:|---:|
| A | 10,000 | 1,557,487 | — | 140.6 s | 58.8 min |
| B | 10,000 | 1,710,597 | — | 124.0 s | 51.9 min |
| C | 10,000 | 1,869,784 | — | 139.3 s | 58.3 min |

三臂判定次数**完全对等**（各 10,000 次），输入 token 差异来自候选文本长度（C 比 A 多 20%）。
人工成本没有增加，只是评分上下文的 token 变多。

---

## 5. 结论

### 5.1 核心问题已得到答案：BGE 的收益穿透了 Qwen 重排

| 对比 | 初始 R@10 差距 | 最终 R@10 差距 | 保留率 |
|---|---:|---:|---:|
| C − A | +8.83 pp | +8.16 pp | **92%** |
| B − A | +7.33 pp | +4.67 pp | 64% |

作者在 10 题冒烟中观察到"Qwen 对 Hybrid 的 gold 候选打分更低"（0.3846 vs 0.6364），
并据此担心增量证据会被评分器过滤掉。**在 100 题上，这个担忧对 BGE-only 不成立**：
入口收益几乎无损地传递到了最终 top-10。

### 5.2 需要修正作者基于 10 题得出的结论

作者 10 题报告的表述是：

> "C 的 R@10 增加 16.67 个百分点 …… 当前等权 RRF 会把 BGE 原本较强的候选顺序打散，
> 引入 BM25 的噪声。"

100 题结果不支持这个强度：

- C 相对 B 仅高 **+3.50 pp**，置信区间下界触零（[−0.00, +7.42]），**统计上不显著**；
- B 相对 A 是**显著正向**的（R@10 +4.67 pp，AE@10 +11 pp）。

因此正确表述应为：

> **等权 RRF 是一个有效的融合方案，显著优于 BM25；它略逊于 BGE-only，
> 但两者差距在 100 题上与零无法区分。** 10 题的 16.67 pp 是小样本波动。

### 5.3 4-hop 完全没有获得收益：本轮最重要的发现

| 4-hop（n=30） | A: BM25 | B: RRF | C: BGE |
|---|---:|---:|---:|
| Recall@10 | 40.00 | 41.67 | 43.33 |
| **AllEvidence@10** | **3.33** | **3.33** | **3.33** |

（30 题中仅 1 题命中全证据，AE@5 为 0%。）

对比 2-hop 上 BGE 带来的 +22.5 pp 全证据提升，4-hop 的提升为 **0.0 pp**。

推论：

1. BGE 改善的是**入口层**（第一跳能否找到）；
2. 4-hop 的瓶颈**不在入口**，而在 seed 选择与词项扩展机制；
3. 因此"换更强的首轮召回器"这一条路径，**无法单独解决深跳问题**。

### 5.4 门控判定

作者预设的推进条件是：

> "如果 C 不超过 A，暂停 H=1；如果 C 明显超过 A，进入 H=1。"

C 显著超过 A（R@10 +8.17 pp，CI 下界 +4.17）。**门控通过，允许进入 H=1。**

4-hop 的零收益同时给出了 H=1 的明确可证伪假设：

> 如果加入一轮扩展后，4-hop 的 AllEvidence@10 仍然不动，
> 则瓶颈在 seed/扩展机制本身，而不在召回层。

---

## 6. 结论的边界（不可越界表述）

本轮使用 **Qwen (`qwen3.5-122b-a10b`) 替代论文的托管 JEV 评分器**，
二者不是同一个模型。因此：

**可以写：**

> 在本实验环境下，把 SAG2 的首轮召回从 BM25 换成 BGE-M3 dense，
> 使 100 题 MuSiQue 上的最终 Recall@10 提升 8.17 个百分点、
> AllEvidence@10 提升 13 个百分点，且收益几乎无损地穿过了 Qwen 重排。

**不能写：**

- ❌ 复现了论文的 BM25 54.41 → SAG2 H=2 86.62 的结果；
- ❌ BM25 与 BGE 的所有混合方法都无效；
- ❌ 该结论可推广到 2WikiMultiHopQA / HotpotQA。

**尚未实验：**

- H=1 / H=2 扩展轮次；
- 偏 BGE 的加权 RRF（1:2、1:3）、配额融合；
- seed precision 与错误 seed 传播的定量分析；
- Qwen 评分稳定性独立重复；
- 1,000 题全量；
- BGE Reranker 对照。

---

## 7. 运行效率标定（H=1 的加速依据）

H=0 三臂耗时 2 小时 49 分。为了压缩 H=1 的时间，先对 Qwen 服务做了并发标定，
再用真实 10 题验证了两种加速手段。

### 7.1 Qwen 证据评分服务的并发上限

用真实记录的 evidence prompt（约 1,982 input tokens）回放压测：

| 并发 | 吞吐 (req/s) | 延迟 p50 | 延迟 max | 失败 |
|---:|---:|---:|---:|---:|
| 4 | 0.347 | 10.94 s | 13.67 s | 0 |
| 8 | 0.457 | 12.79 s | 20.14 s | 0 |
| **12** | **0.624** | 14.17 s | 26.23 s | 0 |
| 16 | 0.636 | 23.62 s | 26.93 s | 0 |
| 24 | 0.621 | 35.16 s | 41.75 s | 0 |
| 32 | 0.625 | 46.13 s | 56.73 s | 0 |

**结论：服务端吞吐在并发 12 处饱和（约 0.63 req/s），继续加并发只增延迟、不增吞吐。**

这带来两个直接推论：

1. **三臂并行没有收益**。把三臂同时跑（各并发 4，合计 12）与顺序跑（单臂并发 12），
   总吞吐完全相同。并行只会增加日志与 checkpoint 的复杂度。
2. **真正的加速手段是把单臂并发从 4 提到 12。**

### 7.2 真实 10 题验证

| 配置 | 10 题耗时 | R@5 | R@10 | AE@10 |
|---|---:|---:|---:|---:|
| concurrency=4 | 360.7 s | 57.50 | 60.83 | 30.00 |
| concurrency=12 | **158.2 s** | 57.50 | 60.83 | 30.00 |

**加速 2.28 倍，且指标逐位相同**，说明并发不影响结果的确定性。

### 7.3 复用 hop-0 的响应缓存

缓存键为 `sha256(endpoint + payload)`。H=1 的首轮请求（hop 0）与 H=0 的请求
在候选、问题、prompt、batch 分组上完全一致，因此可以命中已有缓存。

做法：把 H=0 各臂的 `requests/` 目录复制到 H=1 对应目录，再以 `--resume` 启动。

实测（10 题 C 臂 H=1，预置 hop-0 缓存）：

```text
hop 0 的 10 个 batch  latency 全部为 0 ms   ← 全部命中缓存
hop 1 的 11 个 batch  真实调用
总耗时 324.8 s，判定数 2,100（H=0 为 1,000）
```

即 **H=1 在判定数翻倍的情况下，耗时反而略低于 H=0**。

### 7.4 H=1 的时间估算

| 方式 | 真实请求数 | 预估耗时 |
|---|---:|---:|
| 原方案 concurrency=4，无缓存 | 6,300 | 约 6.3 h |
| concurrency=12 | 6,300 | 约 2.8 h |
| **concurrency=12 + 复用 hop-0 缓存** | **3,300** | **约 2.0 h** |

H=1 启动命令（三臂顺序，预置缓存）：

```bash
cd SAG-Benchmark/sag2
for a in A B C; do
  # A1-H0/C1-H0/B1-H0 -> A1-H1/C1-H1/B1-H1
  src=../../output/abc-100/H0/${a}1-H0
  dst=../../output/abc-100/H1/${a}1-H1
  mkdir -p "$dst" && cp -r "$src/requests" "$dst/"
done

uv run python scripts/run_hybrid_suite.py \
  --dataset-dir ../../dataset --sample-file ../../dataset/musique_100_for_ab.json \
  --output-root ../../output/abc-100/H1 --arms bm25 dense hybrid --max-hops 1 \
  --candidate-k 100 --seed-k 10 --concurrency 12 --evidence-concurrency 1 \
  --skip-prepare --resume
```

> 注意：`--concurrency` 同时决定题级 worker 数与 aiohttp 连接池上限；
> `--evidence-concurrency` 控制单题内部 batch 的并行度。100 题时题级并发 12
> 已足够填满服务端，无需提高 `evidence-concurrency`。

---

## 8. 产物位置

```text
output/abc-100/H0/
  A1-H0/{config.json,details.jsonl,metrics.json,requests/}
  B1-H0/{config.json,details.jsonl,metrics.json,requests/}
  C1-H0/{config.json,details.jsonl,metrics.json,requests/}
  comparison-CvsA.json / .md
  comparison-BvsA.json / .md
  comparison-CvsB.json / .md

output/E1-initial-retrieval/{config.json,details.jsonl,metrics.json}

SAG-Benchmark/sag2/.cache/sag2/
  musique.sqlite              BM25 FTS5 索引
  musique-bge-m3.npy          passage embedding 缓存
  musique-bge-m3.manifest.json
  request-cache/              Qwen 响应缓存（用于断点续跑）
```

`details.jsonl` 每题包含：`trace`（各阶段候选与分数）、`jev_usage`（逐批 token 与延迟）、
`initial_candidates`（含 bm25_rank / dense_rank / rrf_score）、`retrieved_source_ids`。

---

## 9. 复现命令

```bash
cd SAG-Benchmark/sag2
uv sync --frozen --dev
cp .env.example .env      # 填写 SAG_EMBEDDING_* 与 SAG_EVIDENCE_*

# 索引与 embedding 缓存
uv run python scripts/run_sag2_benchmark.py --datasets musique \
  --dataset-dir ../../dataset --index-only --index-workers 8
uv run python scripts/build_passage_embeddings.py --dataset musique \
  --dataset-dir ../../dataset --index .cache/sag2/musique.sqlite \
  --cache-dir .cache/sag2 --batch-size 32 --concurrency 2

# E1 初始召回
uv run python scripts/run_initial_retrieval_eval.py \
  --dataset-dir ../../dataset --sample-file ../../dataset/musique_100_for_ab.json \
  --cache-dir .cache/sag2 --output-dir ../../output/E1-initial-retrieval

# E3 三臂 H=0
uv run python scripts/run_hybrid_suite.py \
  --dataset-dir ../../dataset --sample-file ../../dataset/musique_100_for_ab.json \
  --output-root ../../output/abc-100/H0 --arms bm25 dense hybrid --max-hops 0 \
  --candidate-k 100 --seed-k 10 --concurrency 12 --evidence-concurrency 1

# 配对分析
uv run python scripts/analyze_hybrid_results.py \
  --a-dir ../../output/abc-100/H0/A1-H0 \
  --b-dir ../../output/abc-100/H0/C1-H0 \
  --output ../../output/abc-100/H0/comparison-CvsA.json
```
