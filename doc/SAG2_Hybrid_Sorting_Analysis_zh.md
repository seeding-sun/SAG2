# SAG2 Hybrid 排序与失败诊断记录

日期：2026-11-11

## 1. 初始混合排序公式

Hybrid 初始召回使用 reciprocal rank fusion，当前采用等权配置：

```text
rrf_k = 60
bm25_weight = 1.0
dense_weight = 1.0
pool_k = 100

RRF(c) =
    1.0 / (60 + BM25Rank(c))
    +
    1.0 / (60 + DenseRank(c))
```

BM25 和 BGE 各取 100 个候选，按 chunk ID 去重后计算 RRF，排序得到初始 top-100。

SAG2 随后把初始排名转换成：

```text
fusion_score = 1 / (61 + initial_rank)
```

该分数只用于 Qwen 分数相同时的次级排序。

最终排序链路为：

```text
初始 RRF 顺序
→ Qwen 证据分数降序
→ fusion_score 降序
→ indexed ID 升序
→ DPP 重排前 10
```

因此，最终结果不是直接把 BM25 原始分数和 BGE 余弦分数相加，也不是直接用 RRF 分数决定最终顺序。Qwen 分数是第一排序键。

## 2. 分阶段结果

固定 10 题上，初始 top-5 的召回完全相同，Hybrid 的初始 top-10 更高。

| 配置 | 阶段 | R@5 | R@10 |
|---|---|---:|---:|
| BM25 H=0 | 初始候选 | 40.83% | 53.33% |
| Hybrid H=0 | 初始候选 | 40.83% | 56.67% |
| BM25 H=0 | Qwen 后 | 60.00% | 60.00% |
| Hybrid H=0 | Qwen 后 | 49.17% | 61.67% |
| BM25 H=1 | 初始候选 | 40.83% | 53.33% |
| Hybrid H=1 | 初始候选 | 40.83% | 56.67% |
| BM25 H=1 | Qwen 与扩展后 | 52.50% | 57.50% |
| Hybrid H=1 | Qwen 与扩展后 | 49.17% | 49.17% |

在这 10 题上，DPP 前后的 R@5/R@10 没有变化。因此当前下降不是由 DPP 多样性排序造成的。

## 3. Qwen 对 gold 与噪声的评分

| 配置 | gold 数量 | gold 平均分 | 非 gold 数量 | 非 gold 平均分 |
|---|---:|---:|---:|---:|
| BM25 H=0 | 22 | 0.6364 | 978 | 0.0061 |
| Hybrid H=0 | 26 | 0.3846 | 974 | 0.0041 |
| BM25 H=1 | 42 | 0.6190 | 2,058 | 0.0078 |
| Hybrid H=1 | 44 | 0.5227 | 2,048 | 0.0049 |

Hybrid 候选池中的 gold 数量更多，但 Qwen 给 gold 的平均分更低。该现象可能来自模型重定向、批次组成和模型判断不稳定，需要进一步用独立重复实验区分。

## 4. 扩展 seed 质量

| 配置 | seed 数 | gold seed 数 | Seed precision |
|---|---:|---:|---:|
| BM25 H=1 | 100 | 18 | 18% |
| Hybrid H=1 | 100 | 17 | 17% |

两组 seed precision 都较低。Hybrid 没有把初始候选中的额外 gold 转化成更可靠的扩展起点，错误 seed 会进一步造成扩展词漂移。

## 5. 典型胜负题

Hybrid H=1 明显低于 BM25 H=1 的题：

```text
3hop1__64957_87694_64412
BM25 R@10 = 0.667, Hybrid R@10 = 0.333

4hop1__161810_583746_457883_650651
BM25 R@10 = 1.000, Hybrid R@10 = 0.250

4hop1__343844_17130_70784_61381
BM25 R@10 = 0.500, Hybrid R@10 = 0.250
```

Hybrid 胜出的题：

```text
4hop1__525129_315334_131926_87157
BM25 R@10 = 0.250, Hybrid R@10 = 0.750
```

## 6. 结论

1. 等权 RRF 确实提高了初始 top-10 中的 gold 覆盖。
2. Hybrid 初始 top-5 没有提高，说明新增收益主要集中在第 6 至第 10 名。
3. Qwen 重排后，Hybrid 的前 5 条质量下降；一轮扩展后，前 10 条也下降。
4. 当前主要问题不是 RRF 公式失效，而是 Qwen 评分和 seed 选择没有利用混合候选的增量证据。
5. 下一步应增加 BGE-only+Qwen 对照，并检查错误 seed 与错误桥接词。

## 7. 数据位置

```text
output/hybrid-smoke/A1-H0/details.jsonl
output/hybrid-smoke/B1-H0/details.jsonl
output/hybrid-smoke/A1-H1/details.jsonl
output/hybrid-smoke/B1-H1/details.jsonl
output/hybrid-smoke/comparison-H0.json
output/hybrid-smoke/comparison-H1.json
```

每题 `details.jsonl` 中重点字段：

```text
initial_candidates
trace[].candidate_ids
trace[].scores
trace[].seed_ids
retrieved_docs
metrics
all_evidence_metrics
```

Qwen 原始请求与响应：

```text
output/hybrid-smoke/<arm>/requests/<question_id>/qwen/qwen_evidence/
```
