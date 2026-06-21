# PaiSmart RAG 评测体系 (Context & Quality)

一个可复现的检索增强生成（RAG）质量评测工具。它把"上下文质量"变成**可度量、可回归、可进 CI**的指标，覆盖检索质量、延迟和幻觉三个维度。

> 设计目标：不是"跑通一个 RAG"，而是把它当成一个**需要被持续度量的质量系统**——这正是 Context & Quality Engineer 的核心职责。

## 度量的指标

| 维度 | 指标 | 含义 |
| --- | --- | --- |
| 检索 | **Recall@k** (hit-rate) | 前 k 个结果中是否命中相关文档。RAG 里最关键——LLM 只要拿到一个对的 chunk 就能作答 |
| 检索 | **MRR** | 第一个相关文档的平均倒数排名，衡量"排得够不够靠前" |
| 检索 | **nDCG@k** | 带排名折扣的增益，奖励把相关文档排在更前面 |
| 延迟 | **p50 / p95 / p99** | 检索接口端到端延迟分位数 |
| 生成 | **幻觉率** | 回答中存在参考信息无法支撑的论断，或对库外问题强行作答的比例 |
| 生成 | **拒答正确率** | 对知识库外的问题，是否正确回答"暂无相关信息" |
| 生成 | **groundedness** | LLM-as-judge 对回答"有据可依"程度的 0~1 打分 |

## 目录结构

```
eval/
├── corpus/                  # 教育领域种子文档（上传到知识库）
├── datasets/education_qa.jsonl   # gold set：问题 + 相关文档 + 参考答案（含 4 条库外问题测拒答）
├── src/
│   ├── metrics.py           # 纯函数指标实现（Recall/MRR/nDCG/分位数）
│   ├── clients.py           # 后端检索客户端 + OpenAI 兼容 LLM 客户端
│   ├── prompts.py           # 与生产环境 DeepSeekClient 一致的 prompt 拼装
│   ├── judge.py             # 拒答检测 + LLM-as-judge 幻觉评审
│   ├── evaluate.py          # 编排：检索→(可选)生成→评审→报告→门禁
│   └── report.py            # 输出 JSON + Markdown 报告
├── fixtures/                # 录制的检索结果，供离线/CI 确定性评测
├── tests/test_metrics.py    # 指标数学的单元测试（无需任何基础设施，跑在 CI）
├── reports/                 # 评测报告输出
└── config.example.yaml      # 配置 + 回归门禁阈值
```

## 两种运行模式

### 1. 离线模式（CI 用，确定性，无需基础设施）
基于 `fixtures/` 里录制的检索结果计算全部指标，用于守护指标计算逻辑和回归门禁：

```bash
pip install -r requirements.txt
python src/evaluate.py --mode offline
```

### 2. 在线模式（产出简历用的真实数字）
对**正在运行**的后端跑真实评测。需要先 `docker-compose up` 起服务、用 `scripts/seed_corpus.py` 把 `corpus/` 灌进知识库，再：

```bash
cp config.example.yaml config.yaml      # 填入账号；export DEEPSEEK_API_KEY=...
python src/evaluate.py --mode live --generate --judge
```

- `--generate`：用与生产一致的 prompt 调 LLM 生成回答
- `--judge`：启用 LLM-as-judge 逐句核验回答是否有据可依

> 真实的 Recall/MRR/幻觉率应来自在线模式。仓库里 `reports/` 中的样例报告基于 `fixtures/`，仅用于演示流程，**不应直接当作真实指标**。

## 回归门禁（shift-left）

`config.yaml` 的 `gates` 定义阈值，`evaluate.py` 在任一门禁不达标时**以非零退出码结束**——检索/质量回归会像单测失败一样让 CI 变红。这就是把质量左移到提交环节的做法。

## 根因分析

每次运行都会在报告末尾列出**失败用例**（未召回 / 幻觉），并在 `reports/latest.json` 里保留每题的排名列表、命中情况、延迟和回答，便于定位是分块、embedding、检索权重还是 prompt 的问题。
