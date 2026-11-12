# SAG2 实验工程记录：代码改动、事故与实际问题

日期：2026-11-12
仓库：`/data/machine_learning/cmh/知识库/SAG2`，基线提交 `e075ee3`（main）
配套文档：`SAG2_H0_100Q_Report_zh.md`、`SAG2_H1_100Q_Report_zh.md`、`SAG2_Expansion_Effectiveness_Diagnosis_zh.md`

本文件只记录**工程侧**的事：改了哪些代码、为什么改、遇到哪些实际故障、怎么排查和修复、哪些问题还没解决。实验结论不在本文档里。

---

## 0. 一览

| # | 事项 | 类型 | 严重度 | 状态 |
|---|---|---|---|---|
| 1 | Qwen 候选 ID 漏一位数字导致 C1-H1 在 58/100 崩溃 | 故障 | **高（中断实验）** | 已修复 + 已审计 |
| 2 | H=0 报告把 hop 分层差值误读成水平值（4-hop AE@10） | 数据错误 | **高（对外数字）** | 已更正 |
| 3 | `verify_paper_results.py` 因缺 `2wikimultihopqa.json` 报 FileNotFoundError | 环境/数据缺失 | 低（不影响实验） | 已定性，不修 |
| 4 | 并发标定：Qwen 在 concurrency=12 饱和，"三臂并行"无效 | 性能假设被推翻 | 中（影响排期） | 已定性并采用 |
| 5 | H=1 的 hop-0 复用 H=0 请求缓存，中途中止后靠缓存续跑 | 工程实践 | 中（可行性关键） | 已验证 |
| 6 | 三臂 H=1 实际跑在两个不同代码版本上 | 可比性风险 | 中 | 已量化，判定不影响 |
| 7 | 诊断时 localhost 端口探测造成短暂误判 | 误判 | 低 | 已澄清 |
| 8 | 代码改动未提交 git | 待办 | 中 | 待你决定 |
| 9 | 只读诊断时实例化 `CorpusIndex` 把 BM25 索引清空 | 故障 | **高（可毁数据）** | 已重建 + 已逐位校验 |
| 10 | kill 错进程 + 请求未设 `max_tokens` 把 122B 实例打瘫 40 分钟 | 故障 | **高（中断他人服务）** | 已定位根因 + 已加输出上限 |

---

## 1. 代码改动清单

### 1.1 改了什么

```
 M SAG-Benchmark/sag2/scripts/qwen_evidence_scorer.py     +50 / -7
 M SAG-Benchmark/sag2/tests/test_qwen_evidence_scorer.py  +27
?? SAG-Benchmark/sag2/scripts/diagnostics/                （4 个新文件）
?? dataset/musique_smoke10.json                           （10 题冒烟子集）
?? doc/SAG2_H0_100Q_Report_zh.md
?? doc/SAG2_H1_100Q_Report_zh.md
?? doc/SAG2_Expansion_Effectiveness_Diagnosis_zh.md
?? doc/SAG2_Engineering_Incidents_and_Code_Changes_zh.md  （本文件）
```

### 1.2 明确没有改的部分（重要）

以下文件**一行未改**，全部为上游原样：

- `scripts/sag2.py` —— BM25 检索、`expansion_terms` 选词、`retrieve_expansion`、hop-1 池构造、DPP 重排
- `scripts/run_sag2_benchmark.py` —— 运行器、指标计算、`details.jsonl` 写出
- `scripts/hybrid_retriever.py`、`scripts/vector_retriever.py` —— RRF 与 dense 召回
- `config.json`、`tests/` 下其余测试

也就是说：**所有实验结论都建立在论文官方检索逻辑之上，唯一被修改的是"替代 JEV 的评分器协议解析层"。** 这一点在写论文或对外汇报时必须说清楚。

---

## 2. 代码改动 1：候选 ID 单字符删除的窄范围修复

### 2.1 改动的动机

`qwen_evidence_scorer.parse_scores` 原本要求返回的 `scores` 键集合与候选集合**严格相等**，否则抛 `QwenProtocolError`。这个严格性是对的（防止错位打分），但一次真实运行证明它会因为模型的一个字符笔误而中断整条实验。

### 2.2 改动内容

新增两个函数，并在 `parse_scores` 的等值校验失败时尝试修复：

```python
def _is_one_char_deletion(shorter: str, longer: str) -> bool:
    """Return True when `shorter` equals `longer` with exactly one character removed."""
    if len(longer) != len(shorter) + 1:
        return False
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))
```

```python
def repair_deleted_digit_key(scores, expected_keys):
    missing = sorted(expected_keys - set(scores))
    extra = sorted(set(scores) - expected_keys)
    if len(missing) != 1 or len(extra) != 1:
        return scores, []
    wanted, emitted = missing[0], extra[0]
    prefix = "candidate_"
    if not (wanted.startswith(prefix) and emitted.startswith(prefix)):
        return scores, []
    if not _is_one_char_deletion(emitted[len(prefix):], wanted[len(prefix):]):
        return scores, []
    repaired = {(wanted if key == emitted else key): value for key, value in scores.items()}
    return repaired, [(emitted, wanted)]
```

修复生效必须**同时**满足：

1. 恰好 1 个期望键缺失；
2. 恰好 1 个多余键；
3. 两个键都以 `candidate_` 开头；
4. 多余键是缺失键**删掉恰好一个字符**得到的。

任何一条不满足就保持原样，让调用方继续抛错。

### 2.3 关键设计决定：为什么不做更宽的修复

排查中确认模型还可能犯**数字替换**类错误（见 §3.6，A1-H1 出现过 `7376→7377`）。我**故意没有**把替换类错误纳入修复，原因：

- 替换类错误的"正确目标"不唯一，无法像删除那样用结构唯一性证明；
- 一旦放宽，错位打分可能静默发生，而错位打分对检索指标是**污染级**的，比崩溃危险得多。

宁可保留一次可观测的崩溃，也不要引入一次不可观测的错配。

### 2.4 审计链路

修复不静默发生：`_parse` 返回的 batch 记录新增字段

```json
{"repaired_keys": ["candidate_1017->candidate_10197"]}
```

写入 `details.jsonl` 的两处：`trace[<jev stage>].batches[i].repaired_keys` 与 `jev_usage.batches[i].repaired_keys`。逐题可追查。

### 2.5 测试

`tests/test_qwen_evidence_scorer.py` 新增 4 个用例：

| 用例 | 期望 |
|---|---|
| `test_parse_scores_repairs_one_deleted_digit_in_candidate_key` | 正例：`candidate_1017` → `candidate_10197`，并记录 repairs |
| 反例 1：`{}`（只有一个缺失键，无多余键） | 抛错 |
| 反例 2：`candidate_9999`（不是删一个字符的关系） | 抛错 |
| 反例 3：两个期望键都缺失（歧义） | 抛错 |

全量测试：**110 passed**（改动前 106）。`ruff check scripts tests` 全过。

---

## 3. 事故 1：C1-H1 在 58/100 崩溃

### 3.1 时间线

| 时间（UTC） | 事件 |
|---|---|
| 06:53 | A1-H1 完成（100/100） |
| ~07:5x | C1-H1 启动，跑到 58/100 时抛出异常终止 |
| 08:0x | 定位并修复，C1-H1 从 58 续跑到 100 |
| 08:11 | C1-H1 完成 |
| 08:52 | B1-H1 完成 |

### 3.2 现象

```text
QwenProtocolError: Qwen score keys do not match candidates;
missing=['candidate_10197'], extra=['candidate_1017']
```

题目：`3hop1__497845_629431_64412`，hop 1 第 11 个 batch（0-based `batches[10]`）。

### 3.3 三层排查

| 步骤 | 做法 | 结果 |
|---|---|---|
| 1 | 删掉该 batch 的缓存重跑 | **3/3 确定性复现**（temperature=0） |
| 2 | 走官方重试路径（attempt 2 换 system prompt 要求修正 JSON） | **3/3 仍然复现** |
| 3 | 单独只问 `candidate_10197`、或改变候选顺序 | **正常输出** |

结论：不是数据问题、不是缓存脏数据、也不是"格式不对"能靠提示词纠正的问题，而是**位置相关的模型笔误**——特定 10 个候选组成的特定批次会稳定漏抄一位数字。

### 3.4 为什么"删缓存重跑"这类常规手段全部无效

`temperature=0` + 同一 endpoint + 同一 payload ⇒ 服务端返回是确定性的。所以任何"再试一次"的做法都只是在重复同一个错误。这也是为什么必须改解析层，而不是加重试。

### 3.5 修复后的续跑

修复后从 58 继续，hop-0 全部命中缓存：

```text
Completed 60/100 in 20.3s
3hop1__497845_629431_64412
  hop 1 repaired: ['candidate_1017->candidate_10197']   lat_ms 0
```

该题 `Recall@10 = 33.3%`，成功通过。

### 3.6 全量审计结果（判断修复是否污染结果）

扫描全部 6 个 run 的请求记录与 `details.jsonl`：

| run | 审计记录数 | 无法解析的响应 | 非空 `repaired_keys` |
|---|---:|---:|---:|
| A1-H0 | 1,000 | 0 | 0 |
| B1-H0 | 1,000 | 0 | 0 |
| C1-H0 | 1,000 | 0 | 0 |
| A1-H1 | 2,081 | 0 | 0 |
| B1-H1 | 2,082 | 0 | 0 |
| C1-H1 | 2,088 | 0 | **2** |

两点结论：

1. **崩溃前已完成的 58 题中，无效响应数为 0** ⇒ 补丁在这些题上根本不触发，新旧代码结果逐位相同，C1-H1 的 58 题与续跑的 42 题可以合并比较。
2. `C1-H1` 的 2 次修复全部属于同一道题（`3hop1__497845_629431_64412`），且都在同一个 batch 的两次尝试上。

另有一条不触发补丁的历史问题：**A1-H1 有 1 条数字"替换"类错误（`7376→7377`）**，属于官方重试成功的路径，补丁按设计不认这种关系，因此该条从未被静默修复。这条记录保留在审计日志里。

---

## 4. 事故 2：H=0 报告里的 4-hop AE@10 数字误读

### 4.1 现象

H=0 阶段口头汇报时说过"4-hop AE@10 三条臂都是 10.0%"。

### 4.2 根因

把 `hop_strata` 里的 **差值字段（mean_delta = 0.0）** 当成了 **水平值**读。实际水平值是 **3.33%（1/30）**，AE@5 为 0%。

### 4.3 修复

`doc/SAG2_H0_100Q_Report_zh.md` 已改为 3.33%（1/30）。

### 4.4 教训（已纳入后续做法）

`hop_strata` 里同时存在 `mean_delta` 与绝对水平两种量，字段名不直观。后续所有分层数字改为**直接由 `details.jsonl` 重算绝对水平**，不再从汇总 json 里读差值，避免同类误读。这也是后来写 `scripts/diagnostics/` 的直接动机之一。

---

## 5. 事故 3：`verify_paper_results.py` 报 FileNotFoundError

### 5.1 现象

```text
FileNotFoundError: [Errno 2] No such file or directory:
'/data/machine_learning/cmh/知识库/SAG2/dataset/2wikimultihopqa.json'
```

调用链：`verify_paper_results.py` → `BenchmarkData.load_samples()` → `scripts/benchmark_data.py:36`。

### 5.2 定性

**不是环境问题，是数据缺失。** 仓库的 `dataset/` 只有 MuSiQue：

```
dataset/
  musique.json              12 MB，1,000 题
  musique_corpus.json      6.0 MB，11,656 passages
  musique_100_for_ab.json  100 题固定子集（seed=20261111）
  musique_smoke10.json     10 题冒烟子集
```

脚本默认要跑三个数据集（`2wikimultihopqa`、`hotpotqa`、`musique`），另外两个从未随仓库提供。

### 5.3 处理决定

**不修脚本、不伪造数据。** 该脚本的用途是"逐位复现论文三数据集数字"，本实验只做 MuSiQue，且目标是"替代 JEV 下的机制验证"而非"复现论文表格"。强行让它跑通只会产生误导。

替代的一致性验证改用 `scripts/verify_paper_results.py --datasets musique`（如支持）或直接看 E1 初始召回是否与作者 Windows 结果逐位一致 —— 后者已在 H=0 阶段通过。

### 5.4 影响

无。所有正式实验都通过 `run_sag2_benchmark.py` 显式指定 `--datasets musique` 运行，从未依赖该脚本。

---

## 6. 实际问题 4：并发标定推翻了"三臂并行"

### 6.1 初始假设

H=0 三臂串行约 3 小时；很自然地想"三臂并行"把时间压到 1 小时。

### 6.2 实测压测

用真实 evidence 评分 prompt 回放：

| 并发 | 吞吐 (req/s) | 延迟 p50 |
|---:|---:|---:|
| 4 | 0.347 | 10.9 s |
| 8 | 0.457 | 12.8 s |
| **12** | **0.624** | 14.2 s |
| 16 | 0.636 | 23.6 s |
| 24 | 0.621 | 35.2 s |
| 32 | 0.625 | 46.1 s |

### 6.3 结论

服务端在 **concurrency=12** 处饱和（约 0.63 req/s）；加到 32 吞吐不涨，延迟从 14 s 涨到 46 s。

所以**三臂并行是错的**：把 12 并发拆给 3 个臂，与把 12 并发给 1 个臂，总吞吐相同、总时间相同，只是多出日志与 checkpoint 的复杂度与互相干扰风险。正确做法是**串行三臂 + 单臂内 concurrency=12**。

### 6.4 实际采用的加速手段

| 手段 | 效果 |
|---|---|
| 并发 4 → 12 | 实测 10 题从 360.7 s 降到 158.3 s（**2.28×**） |
| H=1 复用 H=0 的 hop-0 请求缓存 | 每题 10 个 batch 延迟为 0，H=1 输入 token 从估 4× 降到实际约 4× 但请求数只增 ~1.08× |
| 串行而非并行 | 避免无效复杂度 |

H=1 三臂实际用时约 1.7–2.2 小时，对比"并发 4 + 无缓存"的估算 6.3 小时，**省约 4 小时**。

---

## 7. 实际问题 5：H=1 全靠请求级缓存才能续跑

### 7.1 机制

请求日志按 `sha256(endpoint + payload)` 落在：

```
output/abc-100/H1/<arm>/requests/<qid>/qwen/qwen_evidence/<hash>.json
```

每题每个 hop 的每个 batch 一条，内含 `endpoint`、`request`、`raw_response`、`response`。

### 7.2 为什么这是 H=1 可行的前提

H=1 的第 0 跳与 H=0 完全同参（只差 `max_hops`）。因此把 H=0 的 `requests/` 目录复制到 H=1 后，**hop-0 的 1,000 条请求全部命中，延迟为 0**：

```text
2hop__21103_16254    hop0=10 (zero=10)  hop1=11 (real=11)  judgments=210
```

这带来两个直接好处：

1. **成本**：H=1 不需要重跑 hop-0 的 100 万 input token × 3 臂；
2. **可比性**：H=0 与 H=1 的 hop-0 评分**逐位相同**，因此两者差值可以干净地归因到扩展轮，而不是评分抖动。

### 7.3 副作用与应对

- 崩溃后续跑只重算缺失部分（§3.5 的 `Completed 60/100 in 20.3s`）；
- 缓存命中不会掩盖代码改动：`config.json` 记录了 `qwen_scorer_sha256`，可事后核对某次 run 用的是哪个版本（见 §8）。

---

## 8. 遗留问题（重要）：三臂 H=1 跑在两个代码版本上

### 8.1 事实

`config.json` 里的 `qwen_scorer_sha256`：

| run | scorer sha256（前 16 位） | 代码版本 |
|---|---|---|
| A1-H0 | `8c3d0ba7b5a7ffea` | 原始 |
| B1-H0 | `8c3d0ba7b5a7ffea` | 原始 |
| C1-H0 | `8c3d0ba7b5a7ffea` | 原始 |
| **A1-H1** | `8c3d0ba7b5a7ffea` | **原始** |
| **B1-H1** | **`9f2bef7590e261a7`** | **已打补丁** |
| **C1-H1** | **`9f2bef7590e261a7`** | **已打补丁** |

（`git show HEAD:...` 得到的原始文件 hash 为 `8c3d0ba7`；当前工作区为 `9f2bef75`。）

### 8.2 为什么我认为仍然可比

补丁是**条件触发**的窄修复，只有在"1 缺失 + 1 多余 + 单字符删除"同时成立时才改变行为。实测触发次数：

- A1-H1：0 次
- B1-H1：0 次
- C1-H1：2 次（同一道题）

即三臂共有 1 道题（`3hop1__497845_629431_64412`，属 C 臂）的解析行为不同，其余 299 道完全一致。对 100 题量级的指标影响 ≤ 1 题，且方向明确（修复让它从崩溃变为可评分）。

### 8.3 但这仍然是一个应该修掉的卫生问题

**建议**：正式对外前，用当前版本把 A1-H1 与 B1-H1 重跑一遍（hop-0 全命中缓存，成本主要是 hop-1，约 40 分钟/臂），让三臂的 `qwen_scorer_sha256` 统一。这样 `config.json` 自证可比，不需要依赖本节的论证。

---

## 9. 误判记录：localhost 端口探测

诊断阶段探测 BGE 服务时，`127.0.0.1:8013` 无响应，一度怀疑 embedding 服务挂了。实际：

- 配置与 manifest 用的是 `http://192.168.2.108:8013/v1`，**该地址正常**；
- 本机 `127.0.0.1:8012` 也活着，但那是另一个服务实例，与实验无关。

结论：**实验从未受影响**，只是我探测了错误的地址。记录在此是为了避免后人重复这个弯路：核对服务是否可用，应以 `config.json` / manifest 里记录的 `base_url` 为准，不要用 localhost 猜。

---

## 10. 事故 4：只读诊断把 BM25 索引清空

### 10.1 现象

为了验证"扩展能否重新捞回掉出种子的 gold"，我在一个临时脚本里直接实例化索引对象做检索：

```python
index = S.CorpusIndex(ROOT/'.cache/sag2/musique.sqlite', corpus, 'probe-x')
```

期望只是"连上现成索引查几条"，实际执行后报错：

```text
File "/…/scripts/sag2.py", line 263, in documents
    title, content = doc["title"].strip(), doc["text"]
KeyError: 'text'
```

紧接着所有查询都返回空。事后查库：

```text
chunks      : 0
chunk_fts   : 0
metadata    : [('["probe-x",2200,5,null,"desktop-sentences-v1","plain-text-default"]',)]
```

**索引被清空了。**

### 10.2 根因

`CorpusIndex.__init__` 是"惰性建库"语义，不是"只读连接"：它把 `[fingerprint, chunk_chars, INDEX_SCHEMA, source_dir, CHUNK_POLICY, …]` 算成一个 signature 与 `metadata` 里存的值比对，**不等就 `DROP TABLE chunks / chunk_fts / source_files` 然后拿传入的 `corpus` 重建**（`scripts/sag2.py:215-250`）。

我在探测脚本里：

1. 把 `fingerprint` 传成了假的 `'probe-x'` → signature 必然不等 → 触发重建；
2. 传入的 `corpus` 只塞了 `id/title/content/full_doc` 字段，缺 `documents()` 需要的 `text` → 重建到一半抛 `KeyError`。

两件事叠加的结果：旧的表已经被 DROP，新的没建成，留下一个空库。

**这是纯粹的自伤，与实验代码无关。** 正式跑批不会踩到，因为 `run_sag2_benchmark.py` 传入的 fingerprint 与 `.cache` 里建库时的一致，signature 命中，全程走"不重建"分支。

### 10.3 影响边界

| 对象 | 是否受影响 |
|---|---|
| `output/abc-100/**` 的 `details.jsonl` / 汇总 json | **否**，全部是已落盘的历史结果，事故未触碰 |
| `doc/` 下所有报告 | 否 |
| Qwen 评分服务的请求缓存 | 否（独立目录） |
| `.cache/sag2/musique.sqlite` | **是，被清空** |

即：**没有任何实验结论依赖被毁的数据**，损失的是"以后继续做检索实验的能力"，直到重建为止。重建是纯计算、无外网依赖，成本约 7 秒。

### 10.4 恢复

```bash
cd /data/machine_learning/cmh/知识库/SAG2/SAG-Benchmark/sag2
.venv/bin/python scripts/run_sag2_benchmark.py \
  --datasets musique --dataset-dir ../../dataset --index-only --index-workers 8
# musique: 11656 段原文 / 11656 个索引片段 / 1000 题
# real 0m7.206s
```

### 10.5 校验（关键：要证明"与事故前逐位一致"，而不是"能跑"）

| 检查项 | 结果 |
|---|---|
| `chunks` 行数 | 11,656（与建库声明一致） |
| `chunk_fts` 行数 | 11,656 |
| `id == document_id` 是否恒成立 | 是（0 例外），说明没重复插入 |
| `MATCH 'hasselhoff'` | 3 条，与事故前一致 |
| `diagnose_expansion.py --arm A1 --hop H1` | seed precision 158/1000=15.80%；lost 34；2hop 分桶 `solved 19 / no-bridge 16 / lost-in-rank 5` —— **与事故前逐位相同** |
| `diagnose_ranking.py` 自校验 | 100/100 一致 |

行数一致是必要条件但不充分（chunk 边界若漂移，行数可能仍然相同），所以额外用诊断脚本的**汇总数字**做了一次交叉验证：这些数字由 `details.jsonl` 与索引联合重算，若索引内容有变，seed precision 与漏斗必然对不上。

### 10.6 教训（已纳入后续做法）

1. **`CorpusIndex(...)` 是写操作，不是读操作。** 只想查数据时，用 `sqlite3.connect(f"file:{path}?mode=ro", uri=True)` + `load_rank_extension(db)`，绝不再实例化索引对象。
2. 确实需要索引对象做实验时，**先把 `.sqlite` 复制到 `/tmp`** 再在副本上实例化，fingerprint 与 `corpus` 都按真实值传。
3. 任何"探测脚本"在跑之前先想清楚：**它会不会写。** 本次是为了回答一个只需要只读查询的问题，却动用了会重建库的入口。
4. 事故后修复必须**证明等价**，不能只说"重跑通了"。本次用 6 项交叉校验（含 3 组与历史结果逐位比对）确认无漂移。

---

## 11. 新增的诊断脚本

为把"从汇总 json 读数"改成"从 `details.jsonl` 重算"，新增 4 个脚本（`ruff check` 全过）：

| 脚本 | 作用 | 是否调用模型 |
|---|---|---|
| `scripts/diagnostics/diagnose_expansion.py` | gold 漏斗、seed 精度、逐题失败归因 | 否 |
| `scripts/diagnostics/diagnose_scorer.py` | 评分分布、扩展词来源与命中率 | 否 |
| `scripts/diagnostics/diagnose_ranking.py` | 四策略对比、α 扫描、累积池反事实 | 否 |
| `scripts/diagnostics/probe_scorer.py` | 重放指定批次，对比模型与判据 | **是** |

其中 `diagnose_ranking.py` 附带一个**关键的自校验**：用记录的 trace 重放排序，若结果与 `retrieved_source_ids` 不一致就说明复现逻辑错了。当前 H=0/H=1 三臂全部 **100/100 一致**，这是所有反事实数字可信的前提。

D1 轮又补了两个：

| 脚本 | 作用 | 是否调用模型 |
|---|---|---|
| `scripts/diagnostics/diagnose_grades.py` | 档位直方图、种子精度、分组 Recall，含集合级自校验 | 否 |
| `scripts/diagnostics/compare_arms.py` | 两臂配对 Bootstrap + 符号检验 + Holm 校正 + 分 hop 拆解 | 否 |

`diagnose_grades.py` 的自校验比之前更严：`rank_dpp` 只重排 `candidate_k` 前缀、不改变集合，所以重放出的 top-10 **集合**必须与 `retrieved_docs` 反查的集合完全相等。三个 run（A1-20 / D1-b10 / D1-b100）全部 20/20 通过。

---

## 12. 待办

| 项 | 说明 |
|---|---|
| **代码未提交 git** | 已在 D1 轮解决：`7fcf236` 提交 ID 修复，`d6bc3f9` 提交分级刻度，`dbfce93` 提交输出上限 |
| 统一三臂代码版本 | 见 §8.3 |
| `verify_paper_results.py` | 若将来要跑 2Wiki/HotpotQA，需补齐数据集；否则建议在脚本里显式提示"仅支持 musique" |
| 探针结果存档 | `probe_scorer.py` 目前只输出到 `/tmp`，建议落一份到 `output/` 便于对比 |
| `batch=100` 是否继续 | 见 §13.6，建议只保留 `batch=10` 为主臂 |

---

## 13. 事故 5：请求未设 `max_tokens`，把 122B 实例打瘫 40 分钟

这是本轮最严重的事故，而且**影响面超出自己的实验**——它是共享网关。必须完整记录。

### 13.1 时间线

| 时刻 | 事件 |
|---|---|
| 22:14 | `D1B-H1`（`batch=100`）启动，与 `D1-H1` 并行 |
| 22:40 | 某题（`2hop__223823_29905`）单请求超时，重试 |
| 23:00 | 第 4 次重试也超时（每次恰好 300s），共耗 20 分钟 |
| 23:00 | `RuntimeError` 向上抛出，`TaskGroup` 终止整个 run；已完成的 21 题安全落盘 |
| 23:05 | 第一次续跑失败：漏了 `--resume` |
| 23:10 | 第二次续跑失败：改了 `--evidence-concurrency`，而它在 `config.json` 里，哈希校验拒绝 |
| 23:13 | 第三次续跑启动，但**只 kill 了 bash 包装进程，python 子进程仍在跑** |
| 23:20 | 两个孤儿进程持续打 100-候选请求，服务开始全面超时 |
| 23:32 | 清掉所有进程后 20 分钟，**1 token 请求仍超时** |
| 23:35 | 定位到只有 `qwen3.5-122b-a10b` 不响应，`qwen3-30b-a3b` 0.1s 正常 |
| 23:44 | 服务自行排空恢复，HTTP 200 / 0.2s |

### 13.2 现象与三个错误假设

现象：服务端在**完全没有客户端负载**的情况下，GPU 利用率仍有 76–90%，`/v1/models` 2ms 返回但任何生成都超时。

我先后假设过：

1. **"服务被打死了，需要重启"** —— 部分错误。HTTP 层一直活着，是推理槽被占。
2. **"缓存队列积压，等排空就好"** —— 这个对了，但当时无法区分"会排空"和"永久卡死"，所以只能等。
3. **"容器重启就能解决"** —— 能解决，但代价是打断共享网关上的其他人。最终没走到这一步。

### 13.3 根因：两个缺陷叠加

**缺陷 A：请求从来不设 `max_tokens`。**

查 `output/grade-100/H1/D1B-H1/requests/**/*.json`：

```text
payload keys: ['messages', 'model', 'response_format', 'temperature']
max_tokens:          <未设置>
max_completion_tokens: <未设置>
```

而容器启动参数是：

```text
vllm serve /root/Qwen3.5-122B-A10B-GPTQ-Int4 --port 8042 \
  --tensor-parallel-size 4 --max-num-seqs 8 --max-model-len 262144 \
  --gpu-memory-utilization 0.45 --no-enable-prefix-caching --enforce-eager
```

`--max-model-len 262144` + 不设上限 = 一旦模型进入重复循环，单个请求可以生成几十万 token。实测正常请求只需 ~13 token/候选（100 候选 → 1319 输出 token），所以正常运行看不出问题，只在退化时爆炸。

**缺陷 B：kill 错了进程层级。**

`run_hybrid_compare.py` 通过 `nohup ... &` 启动时，`$!` 拿到的是 **bash 包装进程**，真正的 python 是它的子进程。`kill` 父 shell 不会带走 python。结果是我以为"已停止"，实际还在往服务打请求——这直接导致清空客户端后服务仍不恢复，让我误判为"服务卡死"。

验证：

```bash
pgrep -af 'run_hybrid_compare'
```

事故时确实列出两个 python 进程，命令行还是我几十分钟前启动的那两条。

### 13.4 修复

**A. 加输出上限（`dbfce93`）**

`QwenEvidenceScorer` 新增 `max_output_tokens_per_candidate`，上限 = `256 + 值 × 候选数`。

关键设计：**默认 0 = 不设上限**。这样 A1/B1/C1 的历史 payload 保持逐字节一致，请求日志仍能按哈希重放，历史结果的可复现性不受影响。设为 32 时，10 候选 → 576 token，对比实测需要的 ~130 token，留 4 倍余量。

上限同时写入两处便于审计：批次 detail 的 `max_tokens`，以及 `config.json`。

**B. 加健康探测（`run_when_healthy.sh`）**

先发一次真实 completion 探测，只有 HTTP 200 才启动实验。这样服务没恢复时不会拿重试去撞墙（每次重试 5 分钟 × 4 次 = 20 分钟白烧）。

**C. 正确的停止方式**

```bash
pkill -f 'run_hybrid_compare'     # 按命令行匹配，杀干净所有层级
pgrep -af 'run_hybrid_compare'    # 必须确认为空再继续
```

### 13.5 教训

1. **共享服务上，任何"我以为停了"都必须用 `pgrep` 证明，而不是 `kill` 返回 0。**
2. **调用外部模型必须设输出上限。** 不设上限的请求在正常时看不出代价，只在退化时变成炸弹，而这个炸弹落在共享资源上。
3. **`resume` 的配置校验会把"调参"变成"不能续跑"。** `evidence_concurrency` 进了 config 哈希，想降并发只能换目录或删 config。事故中我因此浪费了两轮。
4. **"精确 300 秒超时"是关键线索。** 服务端慢会是浮动的超时；每次都恰好等于客户端 timeout，说明请求根本没开始返回。

### 13.6 对 `batch=100` 的结论

`batch=100` 在这个服务上**工程上不可行**：

- 单请求要模型输出 100 个 grade，接近 `--max-num-seqs 8` 下的长尾服务边界；
- 一次抖动吃掉 20 分钟并带崩整轮（`TaskGroup` 无隔离）；
- 它唯一的动机是省请求数（60 vs 420），而省下的 46% 输入 token 换不来科学结论。

本轮决定：**`D1-H1`（`batch=10`）作为主臂**，因为它与 A1-H1 的 batch 配置完全相同，是唯一能干净隔离"刻度改动"的配置。`D1B-H1` 的 27 题结果保留在 `output/grade-100/H1/D1B-H1/`，仅作为"该服务下不可行"的证据，不进主结论。

---

## 14. 校验命令

```bash
cd /data/machine_learning/cmh/知识库/SAG2/SAG-Benchmark/sag2

# 单元测试（应 129 passed）
.venv/bin/python -m pytest -q

# 静态检查
.venv/bin/python -m ruff check scripts tests

# 排序复现自校验（应输出 100/100）
.venv/bin/python scripts/diagnostics/diagnose_ranking.py --all

# 检查每次 run 用的代码版本
for f in ../../output/abc-100/H*/?1-H*/config.json; do
  python3 -c "import json,sys;d=json.load(open('$f'));print('$f'.split('/')[-2],d['qwen_scorer_sha256'][:16])"
done
```
