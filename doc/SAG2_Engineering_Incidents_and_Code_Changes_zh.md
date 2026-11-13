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
| 11 | Qwen 整条漏判一个候选，`D1-H1` 跑 11 题后整轮崩溃 | 故障 | **高（中断实验）** | 已修复（缺键补 0） |
| 12 | 统计脚本用截断文本反查 doc id，导致"池内/池外"拆分失真 | 数据错误 | **高（会误导决策）** | 已改用干净 id 字段并自校验 |
| 13 | `D1-H1` 全程零重试但墙钟慢 3.2 倍 | 配置取舍 | 低（非方法开销） | 已定性，本轮起恢复并发 |
| 14 | **网关静默替换模型**：`qwen3.5-122b-a10b` 实际返回 `qwen3.8-27b`，污染 `D1-H1` 后 14 题 | 故障 | **高（污染结果 + 阻塞后续）** | 已定位 + 已修正结论；**122B 目前不可达** |

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

## 14. 事故 6：模型整条漏判一个候选，跑 11 题后整轮崩溃

### 现象

`D1-H1`（100 题，`batch=10`）2026-11-12 23:44 启动，00:09 崩溃，`details.jsonl` 停在 11/100。日志尾部：

```
qwen_evidence_scorer.QwenProtocolError:
Qwen score protocol failed after retry:
Qwen score keys do not match candidates; missing=['candidate_7623'], extra=[]
```

`TaskGroup` 遇到 `QwenProtocolError` 直接解散，整轮退出——和事故 5 一样，**单题失败带走整轮**。

### 根因

不是网络抖动，是**模型回应里整条漏掉了 `candidate_7623`**：10 个候选只回了 9 个 grade，`finish_reason=stop`，输出 105 token（未截断）。

对全部已落盘响应的审计：

| run | 响应数 | key 覆盖不全 | 频率 |
| --- | --- | --- | --- |
| `D1-H1`（本次，grade 模式） | 254 | 2（同一题两次重试） | 0.8% |
| `A1-H1`（100 题，binary 模式） | 2081 | 1 | 0.05% |

两个结论：

1. **漏判是模型的固有失误**，与 6 档刻度无关（A1 也有，只是那一次恰好"少一个 + 多一个"被删位修复救回）。
2. **原重试机制对这类错误完全无效**：请求日志把 HTTP 200 的响应体按 payload 哈希落盘，重试读缓存拿到同一份内容，必然同样失败——所以日志里是"failed after retry"，实际上只尝试了一次。

按 0.5% 量级估算，100 题约 4000 个请求，**必然撞上多次**。不修就无法无人值守。

### 修复：缺键补 0，并把补全写进 trace

`qwen_evidence_scorer.py`：

- `_unwrap_keyed_object(..., missing=None, fill_missing=None)`：删位修复后如果仍然只有"缺"、没有"多"，且 `fill_missing` 非 `None`，则对缺失的 key 补该值，并把补齐的 id 追加进 `missing`（落到 batch 的 `imputed_keys`）。
- 补全只接受**纯缺失**：一旦出现模型凭空编造的 key（`extra` 非空），整份信封不可信，仍然抛错。
- 新增构造参数 `fill_missing_score: float | None = None`，**默认 `None` 保持历史严格语义**（二值/grade 两条路径都零回归）。grade 模式下必须给刻度锚点（`0/2/.../10`），binary 模式下必须在 `[0,1]`，构造期校验。
- 聚合：`usage["imputed_judgments"]`，`usage["fill_missing_score"]`，每个 batch 的 `imputed_keys`。

`sag2.py`：`usage` 显式聚合新增 `imputed_judgments`；`run_hybrid_compare.py`：新增 `--fill-missing-score` 并写入 `config.json`。

**为什么补 0 而不是重试或丢弃**：

- 补 0 语义诚实——"模型没有对这条给出任何正面证据"，与 rubric 的 `0`（不相关）同向，且**永远不会伪造种子**（0 < `grade_threshold=0.4`）。
- 重试已被证明无效（缓存命中同一份坏响应）；要真正重试得做请求日志失效化，会破坏"响应不可变"这一可复现前提。
- 丢弃该候选会让候选池变 9 条，破坏与 A1 的可比性。

代价量化：本次只有 `2hop__359312_18803` 的 1 个候选被补 0（1/4000 量级），对 Recall 的影响远小于噪声。

### 处理与代价

- 补全策略进入 `config.json`，与既有 config 冲突，故旧 config 存档为 `doc/artifacts/D1-H1-config-before-imputation.json` 后重写。
- 中途又改了一次 `sag2.py`（仅为把 `imputed_judgments` 聚合进 `jev_usage`），因此 `sag2_code_sha256` 变化，手工同步 config 并回填已写 15 行的该字段（该值可由 batch 的 `imputed_keys` 逐条复算，非估计值）。
- 重跑 15 题靠请求缓存秒级重放，实际只损失 00:09–08:55 的无人值守时间（约 8.7 小时），不是 8.7 小时的算力。

**教训**：长跑任务里，凡是"模型返回结构"的断言都必须有可审计的降级路径；`TaskGroup` 与协议错误的组合等于"一题崩全轮"。这两条在事故 5 之后已经出现过一次，仍然只修了 token 上限、没修协议严格性。

---

## 15. 事故 7（诊断侧）：用截断文本反查 doc id，把"池内/池外"拆分算错

### 15.1 现象

在拆解"第一轮漏掉的 gold 最终去哪了"时，第一版脚本用 `initial_docs`（chunk **文本**）做反查：

```python
t2i = {}   # full_doc 文本 -> document_id
for did, full in db.execute("select document_id, full_doc from chunks"):
    t2i.setdefault(full, did)
i10 = set(t2i[d] for d in row['initial_docs'][:10])
```

它一度给出"扩展轮平均新增 97.9 个候选"，与 `hop-1` 池子平均只有 106.3 条自相矛盾——新增 98 条意味着第一轮只留下 8 条，与"5 个 BM25 保护项 + 种子"的机制不符。

### 15.2 根因

`initial_docs` 存的是**落库时的 chunk 文本**，而 `chunks.full_doc` 是原始段落。两者在长段落上因 `chunk_chars=2200` 截断而不相等，`t2i.get()` 大量返回 `None`，再被列表推导静默丢弃。

**最危险的地方是它不报错**：丢几条不会抛异常，只是让分母变小、比例失真。抽查 3 题时 `docs映射` 显示 10/10，掩盖了这个问题（短段落能对上）。

### 15.3 修复：改用干净 id 字段，并强制自校验

`details.jsonl` 里本来就有现成的 id 字段，不需要反查：

| 用途 | 正确字段 |
|---|---|
| 第一轮 100 候选 | `initial_candidate_ids` |
| 第一轮 top-10 | `initial_candidate_ids[:10]` |
| hop-1 池 | `trace` 中 `stage="jev", hop=1` 的 `scores[].id` |
| 最终 top-10 | `retrieved_source_ids`（注意：与 `retrieved_docs` 的顺序**不一致**，不可混用） |

改完后两个自校验全部通过：

```text
A1 自校验: final R@10 不符 0/100, initial R@10 不符 0/100
D1 自校验: final R@10 不符 0/100, initial R@10 不符 0/100
两臂 initial_candidate_ids 逐题相同: 100/100
```

其中"两臂 `initial_candidate_ids` 逐题相同"本身就是一个强证据：两臂的第一轮是**同一份结果**，所以后续差异只能来自扩展轮。

### 15.4 教训

凡是"从文本反查 id"的统计脚本，都必须带一条**重算指标与 `metrics.json` 对账**的自校验，且误差阈值设到 `1e-9`，不通过就不许出数。文本映射这类静默丢数据的路径，抽查几题是发现不了的。

---

## 16. 事故 8（配置）：`D1-H1` 全程零重试，但墙钟慢 3.2 倍

### 16.1 事实

| | A1-H1 | D1-H1 |
|---|---|---|
| 单题计算量 `mean_query_ms` | 237,080 | 241,882（+2.0%） |
| 墙钟 | 2,088 s | 6,672 s（3.2×） |
| `--concurrency`（题级） | 12 | 3 |
| 重试次数 | — | **0** |

### 16.2 定性

这不是方法开销，是**为共享实例主动让路**：122B 是多人共用的，空载时 GPU 利用率仍 76–90%，容器只有 8 个序列槽。当时把题级并发压到 3，在飞请求峰值 3，留 5 个槽给别人。

代价：100 题从 ~35 分钟变成 ~111 分钟。收益：全程零超时、零重试、零崩溃（对比同期 `batch=100` 臂四次 300s 超时并把实例打瘫）。

### 16.3 决定

单题计算量已证明与 A1 相同（+2.0%），说明 `--concurrency 3` 只是排队、没有额外开销。后续实验把并发恢复到中间档（`--concurrency 6`）：在飞峰值 6，仍留 2 个槽，墙钟约减半。

### 16.4 汇报纪律

**不同并发的两个臂不能比墙钟。** 唯一可比口径是 `mean_query_ms`（单题计算量）与 token 数。任何"本轮比上轮快/慢"的表述都必须注明并发配置。

---

## 17. 环境与运行参数（`D1-H1` 100 题，2026-11-13）

这一节记录复现所需的完整环境，便于后续实验逐项对照。

### 17.1 主机与服务

```text
主机：Linux 6.8.0-138-generic（共享 GPU 服务器）
工作目录：/data/machine_learning/cmh/知识库/SAG2
SAG2 代码：SAG-Benchmark/sag2
Python：项目内 .venv
```

| 服务 | 地址 | 说明 |
|---|---|---|
| Qwen 证据判定 | `qwen3.5-122b-a10b` | 共享 122B 实例，容器 `--max-num-seqs 8` |
| Embedding | `bge-m3:latest`，1024 维 | 仅向量/混合臂需要，本轮 BM25 未用 |
| BM25 索引 | `.cache/sag2/musique.sqlite` | 11,656 段落 / 11,656 片段 |

### 17.2 数据

| | |
|---|---|
| 样本 | `dataset/musique_100_for_ab.json`，100 题（2hop 40 / 3hop 30 / 4hop 30） |
| `sample_sha256` | `971109fac5d9ea1947f95b66682f61b02277391d67aec7e666ad529f5a23d93d` |
| 语料 | MuSiQue 11,656 段落，`corpus_sha256` = `73157a03…` |

每题 gold 段落数 **2–4 个（均值 2.90，合计 290）**，不是 10 个。

### 17.3 代码版本（可追溯）

| 文件 | SHA256 前 16 位 |
|---|---|
| `sag2.py` | `04661a0a7cd708ef` |
| `run_hybrid_compare.py` | `c1471733eee81449` |
| `qwen_evidence_scorer.py` | `709961128b2b14b8` |

三项均记录在 `output/grade-100/H1/A6-H1-qwen3.5-122b-a10b/config.json`。

### 17.4 完整命令行

```bash
cd SAG-Benchmark/sag2
.venv/bin/python scripts/run_hybrid_compare.py \
  --arm D1-H1 \
  --scoring-scale grade --grade-threshold 0.4 --fill-missing-score 0 \
  --candidate-k 100 --seed-k 10 --max-hops 1 --top-k 10 \
  --evidence-batch-size 10 \
  --concurrency 3 --evidence-concurrency 1 \
  --max-output-tokens-per-candidate 32 --timeout 300 \
  --sample ../../dataset/musique_100_for_ab.json \
  --output ../../output/grade-100/H1/A6-H1-qwen3.5-122b-a10b
```

### 17.5 实测运行特征

| 指标 | 值 |
|---|---|
| 墙钟 | 6,671.6 s（约 1 小时 51 分） |
| 判定次数 | 20,520 |
| 请求数 | 2,094 |
| 输入 / 输出 token | 6,489,275 / 243,714 |
| 每批输出 token | 均值 116.4，最大 148（上限 576 从未逼近） |
| 每批延迟 | 中位 11.4 s，p99 33.7 s，最大 50.2 s |
| 超时 | 0 |
| 重试 | 0 |
| 补全判定 | 2（0.010%） |
| 早停题 | 1（`2hop__359312_18803`，`eligible_count=0`） |

---

---

## 19. 事故 9（最重要）：网关静默替换模型，`D1-H1` 最后 14 题由 27B 判定

### 19.1 发现过程

2026-11-13 11:45 准备启动下一臂时做服务健康检查，健康检查本身通过（HTTP 200、113 ms），但**响应体里的 `model` 字段是 `qwen3.8-27b`，而请求的是 `qwen3.5-122b-a10b`**。

顺着这条线回查基准臂，发现 `D1-H1` 的 `batches[].model` 存在两种取值：

```text
A1-H1: {'qwen3.5-122b-a10b': 2080}                      # 干净
D1-H1: {'qwen3.5-122b-a10b': 1834, 'qwen3.8-27b': 260}  # 混入 12.42%
```

### 19.2 波及范围（可精确定位）

按每题响应缓存的 mtime 还原完成顺序，切换是一个**硬切点**：

```text
11-13 10:52:30  122B   0/22   4hop1__73810_…
11-13 10:54:04   27b   5/22   4hop1__788226_…   <- 切换
11-13 10:54:12   27b   8/22
11-13 10:54:42   27b  16/22
11-13 10:55:24   27b  21/21   （此后全部 100%）
```

| | |
|---|---|
| 受影响题数 | **14 / 100**（`details.jsonl` 第 86–99 行） |
| 受影响跳数 | **全部是 4hop**（这 14 题正好是 4hop 的尾部） |
| 受影响批次 | 260 / 2,094（12.42%） |
| 完全干净的题 | 86 / 100 |
| 切换时刻 | **2026-11-13 10:54:04** |

**坏运气叠加**：模型退化恰好落在最容易暴露问题的 4hop 尾部。如果不查，会得出"4hop 收益更薄"的结论，而部分原因是模型被换掉了。

### 19.3 当前服务状态：122B 已不可达

网关**不按请求的 `model` 字段路由**，大模型被静默映射到小模型：

| 请求 | 实际返回 |
|---|---|
| `qwen3.5-122b-a10b` | **`qwen3.8-27b`** ❌ |
| `qwen3.5-397b-a17b` | **`qwen3.6-27b`** ❌ |
| `qwen3-235b-a22b` | **`qwen2:7b`** ❌ |
| `qwen3-30b-a3b` | **`qwen2:7b`** ❌ |
| `qwen2.5:32b` | **`qwen2:7b`** ❌ |
| `qwen3.6-27b` | `qwen3.6-27b` ✅ |
| `qwen3.8-27b` | `qwen3.8-27b` ✅ |
| `gemma4-26b-a4b` | `gemma4-26b-a4b` ✅ |

`/v1/models` 仍然**列出** `qwen3.5-122b-a10b`，但请求它返回的是 27B。也就是说：模型清单是静态的，实际路由取决于后端哪些实例在线。

**后果：在当前服务状态下无法跑出与 A1-H1/D1-H1 可比的臂。** 任何新实验都会静默变成"27B 模型实验"。

### 19.4 对已有结论的影响：结论成立，且更强

只用 **86 个 100% 由 122B 判定的题**重算：

| K | A1-H1 | D1-H1 | 差值 | 95% CI | p | Holm |
|---|---:|---:|---:|---|---:|---:|
| R@1 | 0.3072 | 0.3052 | −0.0019 | [−0.0446, +0.0407] | 0.943 | 0.943 |
| R@2 | 0.4845 | 0.5126 | +0.0281 | [−0.0184, +0.0746] | 0.249 | 0.498 |
| **R@5** | 0.5688 | 0.6676 | **+0.0988** | [+0.0455, +0.1570] | <0.001 | **0.002** |
| **R@10** | 0.6076 | 0.7267 | **+0.1192** | [+0.0630, +0.1773] | <0.001 | **0.002** |

符号检验：**29 胜 / 6 负 / 51 平，p = 1.17×10⁻⁴**。

**剔除污染后 R@10 的差值从 +0.1100 抬到 +0.1192**——污染是往低估的方向偏的，所以原始 0.6925 是保守值。

4hop 在干净子集上（n=16）仍是 +0.0469，**"4hop 收益最薄"的判断没有被推翻**，但它现在只建立在 16 题上。

### 19.5 处理方案

1. **不重跑 86 题**：它们 100% 干净，重跑只是浪费。
2. **只重跑 14 题**：等 122B 恢复后，删除这 14 题的响应缓存（否则会命中 27b 的旧响应），重跑后拼回 `details.jsonl`，得到干净 100 题。
3. **加一道守卫**：每次 run 结束（以及诊断时）检查 `batches[].model` 是否与 `config.evidence_model` 一致，不一致就拒绝出结论。

### 19.6 教训

1. **健康检查不能只看 HTTP 状态码。** 200 + 低延迟完全可能来自一个被换掉的模型。必须校验响应体里**实际返回的 model 名字**。
2. **`config.json` 记录的 `evidence_model` 是"我请求了什么"，不是"实际用了什么"。** 只有 `batches[].model` 才是事实。这两者必须交叉核对。
3. **共享服务的模型可用性会在长跑中途变化。** `D1-H1` 跑了 1 小时 51 分，切换发生在第 3.5 小时附近（跨了重启缓存重放）。长跑必须记录每批的模型名并做一致性断言。
4. 这是**静默失败**的典型：没有报错、没有超时、指标看起来完全正常，只有翻日志才能发现。

---

## 20. 事故 10（协议 / 编排 / 续跑）：一次键名错位引发的三层故障

### 20.1 事实

`D1B2-H1-qwen3.8-27b` 跑到 **39/100** 时整个 TaskGroup 崩掉：

```text
qwen_evidence_scorer.QwenProtocolError: Qwen score protocol failed after retry:
Qwen score keys do not match candidates; missing=['candidate_11486'], extra=['candidate_11483']
```

模型把 `candidate_11486` 的**最后一位写错**（6 → 3），产出 `candidate_11483`。

与此同时，链脚本却打印了 `=== chain complete ===`，看上去像全部成功。

### 20.2 定性：三个独立缺陷叠加

**A. 协议层：既有修复只覆盖"少一位"，没覆盖"错一位"**

`repair_deleted_digit_key` 的判据是 `_is_one_char_deletion`，即 emitted 必须等于 wanted 删除一个字符。`11483` 与 `11486` 长度相同、只差最后一位，属于**替换**而非**删除**，因此不匹配，直接抛错。而 `fill_missing_score` 只在"纯遗漏、无多余键"时才生效，这里是"遗漏一个 + 多出一个"，所以两条兜底路径都走不到。

**B. 编排层：链脚本不看返回码**

```bash
run_arm D1B2-H1-qwen3.8-27b ...     # 返回值被丢弃
echo "=== chain complete ==="
```

`run_arm` 内部 `return $rc`，但调用处没有 `|| exit 1`。所以某一臂彻底失败时，脚本仍会打印"完成"。**这是比崩溃本身更危险的缺陷**——如果没有主动核对题数，这条链会被当成"全绿"记录下来。

**C. 续跑层：代码指纹哈希守卫拦死恢复**

修好 scorer 后，`config.json` 里的 `qwen_scorer_sha256` 与当前文件不一致，`--resume` 直接拒绝：

```text
ValueError: Resume configuration differs from .../D1B2-H1-qwen3.8-27b/config.json
```

守卫本身是对的（防止在语义变更后拼接两个不同实现的结果），但当时**没有任何合法的越权通道**，导致"崩溃 → 修复 → 无法续跑 → 只能重跑 100 题"。

### 20.3 影响

**没有损失已算完的工作**：checkpoint 保住了 39 题，续跑只补 61 题。总代价约 40 分钟（修复 + 重跑）。

### 20.4 修复

1. **协议层：支持单字符替换修复**。新增 `_is_one_char_substitution`，与删除合并为 `_digit_distance_1`。
2. **协议层：加歧义守卫**。若错键与**多个**候选都只差一位，拒绝猜测（例如批次内同时存在 `11486` 和 `11489`，而模型写了 `11483`）。此时仍抛错。
3. **协议层：允许丢弃单个游离键**。在 `fill_missing_score` 模式下，真不存在的键不再阻断整轮：丢弃该键、按 `fill_missing` 补遗漏项，并把丢弃的键记入 `batches[].dropped_keys` 与汇总 `dropped_judgments`。**判决值因此不会被伪造**（丢弃意味着该候选被保守降到下限），只是不再中断长跑。
4. **协议层：比例守卫**。若一批里超过 20% 的键不属于本批（最少容忍 1 个），说明模型答的是另一个批次，仍然抛错——避免"整批静默补 0"把 run 变成空壳。
5. **编排层：`|| exit 1`**。任一臂失败即终止链，杜绝谎报完成。
6. **续跑层：`--accept-code-change "理由"`**。仅当差异键**全部**属于代码指纹（`sag2_code_sha256` / `runner_code_sha256` / `qwen_scorer_sha256`）时放行；检索或评分语义相关的键一旦变化仍然硬拒。放行时把变更前后哈希与理由写入 `config.json` 的 `resume_code_changes`，保证工件可审计。

单元测试 **137 → 142**，新增 5 条（替换位修复、歧义拒绝、单游离键丢弃并记录、整批错乱拒绝，以及 §20.6 的阈值空操作提示）：

```bash
test_repairs_a_single_mistyped_digit_when_the_target_is_unambiguous
test_refuses_a_mistyped_digit_that_is_equally_close_to_two_candidates
test_imputation_drops_one_invented_key_and_records_it
test_imputation_refuses_wholesale_key_corruption
```

### 20.5 教训

1. **"失败"和"看起来成功"必须能区分。** 链式脚本不检查子进程返回码，等于把运维安全网拆了。任何 `run_arm` 式封装都要 `|| exit 1`。
2. **长跑的崩溃点不是随机分布的。** 这次错误发生在第 39 题，前 38 题一个错键都没有——说明该故障是低频事件，**跑 100 题几乎必然会触发一次**。低频协议异常必须在协议层消化，不能指望"足够幸运"。
3. **兜底策略要标记来源，而不是掩盖。** 补 0 分和丢弃键都是降级，必须写进产物（`imputed_keys` / `dropped_keys`），否则事后无法判断某题的分数是模型给的还是系统填的。
4. **守卫要留合法出口。** 只拦不放的守卫在真实故障中会变成二次伤害；正确做法是放行 + 强制说明原因 + 留痕。
5. **不要用"猜"去修复模型输出。** 替换位修复只有在目标唯一时才允许；一旦可能对应多个候选，宁可抛错也不能随便赋值——一次误判会污染该题全部后续推理。

### 20.6 附带发现（方法论）：离散档位让"阈值调参"出现空操作

`GRADE_RUBRIC` 的锚点是 `{0, 2, 4, 6, 8, 10}`，`jev_score = grade / 10`，因此 `grade_threshold` 只在区间边界上有区分力：

| 设定 | 实际选中档位 | 是否有效 |
|---|---|---|
| 0.0 | ≥ 0 | 全放行 |
| 0.2 | ≥ 2 | 有效 |
| **0.3** | **≥ 4** | **与 0.4 完全等价（空操作）** |
| 0.4 | ≥ 4 | 当前默认 |
| 0.5 | ≥ 6 | 与 0.6 等价 |
| 0.6 | ≥ 6 | 有效 |

原本计划的"扫描 τ∈{0.2, 0.3, 0.4}"里有三分之一是空操作。这类陷阱的危险在于它**不会报错**：跑完得到一个和 0.4 一模一样的数字，很容易被解读成"重标定无效"。

**已在代码里加提示**：`run_hybrid_compare.validate_args` 在 grade 模式下检查阈值是否落在档位间隙，若是则 `logging.warning` 指出它等价于哪个边界值。新增 2 条测试覆盖该提示的触发与不触发。

---

## 21. 校验命令

```bash
cd /data/machine_learning/cmh/知识库/SAG2/SAG-Benchmark/sag2

# 单元测试（应 142 passed）
.venv/bin/python -m pytest -q

# 静态检查
.venv/bin/python -m ruff check scripts tests

# 模型一致性守卫（必须做：网关会静默换模型，见 §19）
.venv/bin/python -m scripts.diagnostics.check_model_consistency ../../output/grade-100/H1/A6-H1-qwen3.5-122b-a10b
# 污染时退出码 1，并输出干净子集指标；干净时退出码 0

# 批量审计所有 run
for d in ../../output/abc-100/H*/* ../../output/grade-100/H*/*; do
  [ -f "$d/details.jsonl" ] || continue
  .venv/bin/python -m scripts.diagnostics.check_model_consistency "$d" --json \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['run'],d['model_mix'])"
done

# 排序复现自校验（应输出 100/100）
.venv/bin/python scripts/diagnostics/diagnose_ranking.py --all

# 检查续跑是否跨过代码变更（应无输出；有输出说明该臂跨版本拼接）
grep -l resume_code_changes ../../output/*/H*/*/config.json 2>/dev/null

# 检查协议降级量（补 0 / 丢弃键都不应为大数）
python3 -c "
import json,glob
for f in sorted(glob.glob('../../output/*/H*/*/metrics.json')):
    d=json.load(open(f))
    print(f.split('/')[-2], d.get('judgments'), d.get('imputed_judgments'), d.get('dropped_judgments'))"

# 检查每次 run 用的代码版本
for f in ../../output/abc-100/H*/?1-H*/config.json; do
  python3 -c "import json,sys;d=json.load(open('$f'));print('$f'.split('/')[-2],d['qwen_scorer_sha256'][:16])"
done
```
