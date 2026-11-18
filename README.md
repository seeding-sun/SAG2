# SAG2

本仓库是 **SAG2 多跳问答检索实验** 的工作副本：在保留上游 SAG-Benchmark 代码的基础上，完成了 Hybrid 检索、分级证据评分、Multi-Query / Dual 扩展、跨数据集与 BridgeRAG 对齐等一整套本地实验，并沉淀方案、结论与可复跑脚本。

- 上游基准与历史结果：`SAG-Benchmark/`
- 实验方案与报告：`doc/`
- MuSiQue / Hotpot / 2Wiki 等数据子集：`dataset/`
- 本地大结果目录：`output/`（体积大，默认不进 Git；指标与结论在 `doc/` 归档）

---

## 我们做了什么

1. **Hybrid 初检**  
   实现并跑通 **BM25 vs BM25+BGE-M3（RRF）** 对照：只换初检器，证据评分、候选预算、扩展与评测协议固定。

2. **证据评分改造（核心）**  
   - 从 0/1 二值改为 **六档 rubric（0/2/4/6/8/10）**  
   - **低于 4 分不当种子、不填充**，避免 0 分段落污染扩展查询  
   - 验证：「把已有证据用对」比单纯「多检索」更值钱

3. **判定模型与工程守卫**  
   - 统一主用 `qwen3.8-27b`（网关曾对 122B 静默替换，不可靠）  
   - 增加 **模型一致性硬断言** 与事后审计脚本，防止跑完才发现模型被换

4. **Multi-Query / Dual 与 BridgeRAG**  
   - Dense + Multi-Query ± Dual 的 n100 矩阵  
   - 与论文结构 BridgeRAG 的 **对齐冒烟（同语料、同模型）**

5. **跨数据集**  
   - MuSiQue / HotpotQA / 2Wiki 上的 H1 等对照实验与报告

6. **臂命名与可复现**  
   - 统一臂名：`{A|B|C}{2|6}-H{n}-{模型}`（BM25 / Hybrid / Dense × 二值/六档 × 扩展轮数）  
   - 规范见 `doc/SAG2_Arm_Naming_zh.md`

---

## 验证了什么（结论摘要）

| 结论 | 证据 |
| --- | --- |
| 六档 + 不填充有效 | MuSiQue 100 题：相对二值，R@5 **+0.0925**、R@10 **+0.1100**（Holm 显著）；填充种子席位 83.9%→0%，种子精度约翻倍。见 `doc/SAG2_D1_Grade_Scale_100Q_Report_zh.md` |
| 判定用 27B 可迁移且更稳 | 同配置下 27B 相对 122B 有提升；后续实验以 27B 为匹配基线。见 `doc/SAG2_27B_Judge_Transfer_100Q_Report_zh.md` |
| Dual 在六档上有增益 | n100：`C6-MQ-H2-dual` H2 R@10 **0.8358** vs 无 dual **0.7883**（约 +4.8pp）；二值上 dual 反而略降。见 `doc/SAG2_n100_MQ_Dual_and_BridgeRAG_Smoke_Results_zh.md` |
| 对齐冒烟：我们的 dual 强于「论文结构+我们模型」的 BridgeRAG | 30 题：dual R@10 **0.922** vs BridgeRAG-ours **0.814**（题级 R@5：8↑/2↓/20 平）。见同上与 `doc/SAG2_BridgeRAG_Smoke30_GATE_zh.md` |
| Hybrid 初检单独换检索器 | 10 题冒烟与完整计划见 `doc/SAG2_Hybrid_*` |

更完整的时间线与服务器计划：`doc/SAG2_Experiment_Status_and_Server_Plan_zh.md`。  
工程事故与代码改动：`doc/SAG2_Engineering_Incidents_and_Code_Changes_zh.md`。

---

## 方案是什么

固定评测协议，做 **受控单变量** 实验：

```text
初检（BM25 / Hybrid RRF / Dense）
  → 证据评分（二值 或 六档+不填充）
  → 种子选择 / 扩展（H0/H1/H2，可选 Multi-Query / Dual）
  → 排序与 R@K 评测
```

臂名约定：

| 字段 | 含义 |
| --- | --- |
| A / B / C | BM25 / Hybrid / Dense |
| 2 / 6 | 二值 / 六档 |
| H0 / H1 / H2 | 扩展轮数 |
| 末段 | `evidence_model` slug，如 `qwen3.8-27b` |

主路径依赖本地 OpenAI 兼容服务：

- Embedding：`bge-m3`
- Evidence judge：`qwen3.8-27b`（推荐开启 strict model）
- 可选 rerank：`bge-reranker-v2-m3`

论文里的托管 JEV 结果仅作外部对照；**本地 Qwen chat 判定 ≠ 论文 JEV**。

---

## 仓库结构

```text
dataset/                 # 题集与语料（含 musique_100_for_ab 等固定子集）
doc/                     # 方案、报告、阶段结论（中文为主）
output/                  # 本地跑数产物（通常 gitignore）
SAG-Benchmark/
  sag2/                  # 可运行实现：scripts / tests / prompts
    scripts/             # 基准、Hybrid、MQ/Dual、诊断与链式脚本
    docs/hybrid-experiment.md
  sag/                   # 上游相关代码
scritp/                  # 少量辅助脚本（历史目录名）
```

---

## 怎么用

### 1. 环境

建议在 `SAG-Benchmark/sag2` 下用 Python 3.11+ 与 `uv`：

```bash
cd SAG-Benchmark/sag2
uv sync --frozen
```

离线自检：

```bash
uv run pytest -q
uv run ruff check scripts tests
uv run python scripts/verify_paper_results.py
```

### 2. 配置

```bash
cp .env.example .env
# 只在本地 .env 填密钥与 endpoint，不要提交
```

至少配置：

```dotenv
SAG_EMBEDDING_BASE_URL=...
SAG_EMBEDDING_MODEL=bge-m3:latest
SAG_EVIDENCE_BASE_URL=...
SAG_EVIDENCE_MODEL=qwen3.8-27b
SAG_EVIDENCE_API_KEY=...
```

### 3. 建索引 / 写 embedding 缓存

根目录 `dataset/` 可被脚本通过 `--dataset-dir` 引用。Hybrid 流程详见：

- [Hybrid experiment guide](SAG-Benchmark/sag2/docs/hybrid-experiment.md)

典型步骤：建 BM25 SQLite 索引 → 构建 BGE-M3 passage embedding 缓存 → 成对跑 A/B 臂 → 分析 `results/` 或 `output/`。

### 4. 跑实验（示例）

在 `SAG-Benchmark/sag2` 下：

```bash
# Hybrid 对照套件（见脚本帮助与 hybrid-experiment.md）
uv run python scripts/run_hybrid_suite.py --help

# 基准入口
uv run python scripts/run_sag2_benchmark.py --help

# 链式 100 题等：见 scripts/run_*_chain.sh、diagnostics/
```

固定 100 题子集（若需重建）：

```bash
uv run python scripts/make_musique_ab_subset.py \
  --dataset-dir ../../dataset \
  --output ../../dataset/musique_100_for_ab.json \
  --seed 20261111 \
  --strata 2hop:40,3hop:30,4hop:30
```

### 5. 读结果

- 指标与阶段结论：优先看 `doc/*Report*.md`、`doc/*Results*.md`、`doc/*Conclusion*.md`
- 原始 details：本地 `output/<run>/`（未入库时请向维护者要归档包）

---

## 关键文档索引

| 文档 | 内容 |
| --- | --- |
| `doc/SAG2_Experiment_Status_and_Server_Plan_zh.md` | 总现状与执行计划 |
| `doc/SAG2_Arm_Naming_zh.md` | 臂命名规范 |
| `doc/SAG2_Hybrid_Experiment_Plan.md` / `SAG2_Hybrid_Implementation_Plan.md` | Hybrid 方案与实现 |
| `doc/SAG2_D1_Grade_Scale_100Q_Report_zh.md` | 六档评分 100 题正式报告 |
| `doc/SAG2_27B_Judge_Transfer_100Q_Report_zh.md` | 27B 判定迁移 |
| `doc/SAG2_n100_MQ_Dual_and_BridgeRAG_Smoke_Results_zh.md` | MQ±Dual n100 与 BridgeRAG 冒烟 |
| `doc/SAG2_Engineering_Incidents_and_Code_Changes_zh.md` | 工程事故与代码变更 |
| `SAG-Benchmark/README.md` / `README-CN.md` | 上游基准说明 |
| `SAG-Benchmark/sag2/README.md` | sag2 实现说明 |

---

## 数据与许可

基准实现、prompts 与部分历史结果来自 SAG-Benchmark。MuSiQue 等数据放入 `dataset/` 是为了让文档中的实验可直接复跑。再分发或商用前请核对各组件许可证与上游声明。

密钥只放本地 `.env`，勿提交。
