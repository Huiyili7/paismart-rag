# PaiSmart 面试演示手册 (Context & Quality)

一份可照着走的演示流程 + 讲解话术。目标：把"我做了一个 RAG"升级成"我把 RAG 当成一个**可度量、有回归门禁、能定位问题**的质量系统"。

---

## 0. 演示前准备（建议提前在本机跑通一次）

```bash
# 1) 起依赖服务（ES / MySQL / Kafka / Redis / MinIO）
cd docs && docker-compose up -d

# 2) 起后端
mvn spring-boot:run

# 3) 起前端
cd frontend && pnpm install && pnpm dev
```

> 若网络/时间紧张，演示重点可只放在 **评测体系 + CI**（不依赖前端），这部分最打动 Context & Quality 岗位。

---

## 1. 灌入评测语料（教育领域知识库）

```bash
cd eval
pip install -r requirements.txt
export PAISMART_USER=admin PAISMART_PASS=admin123
python scripts/seed_corpus.py --base-url http://localhost:8081
# 异步索引，等 ~10-30s
```

---

## 2. 跑真实评测（产出简历用的数字）⭐ 核心环节

```bash
cp config.example.yaml config.yaml      # 填账号
export DEEPSEEK_API_KEY=sk-...           # 用于答案生成 + LLM 评审
python src/evaluate.py --mode live --generate --judge \
    --note "live run on seeded education KB"
```

会输出并写入 `reports/latest.md`：Recall@1/3/5、MRR、nDCG、检索延迟 p50/p95/p99、幻觉率、拒答正确率、groundedness，以及**失败用例表**。

**话术**：
> "我没有只看‘能不能答’，而是建了一个 20 题的 gold set，把检索质量量化成 Recall@5、MRR、nDCG，把生成质量量化成幻觉率和拒答正确率。这是我刚刚跑出来的真实报告——注意这里有失败用例表，是我做根因分析的入口。"

---

## 3. 离线评测 + 回归门禁（shift-left 演示）

```bash
python src/evaluate.py --mode offline      # 基于录制 fixtures，确定性
echo "exit code: $?"                        # 门禁不达标则非零
```

**话术**：
> "评测不只是出报告，还接了回归门禁：Recall@5、MRR、幻觉率任一跌破阈值，脚本就以非零退出码失败。这样它能像单测一样卡在 CI 里——质量左移到提交环节。"

---

## 4. 对话演示（含反幻觉「关键镜头」）

在前端聊天界面依次问：

| # | 问题 | 期望表现 | 讲解点 |
| --- | --- | --- | --- |
| 1 | 什么是过拟合？ | 基于文档作答，**句末带 (来源#: 文件名)** | grounding + 引用 |
| 2 | TCP 和 UDP 有什么区别？ | 命中 computer_networks 文档 | 混合检索召回 |
| 3 | 你好呀 | 直接寒暄，**不触发检索** | Function Calling 路由 |
| 4 | **公司 Q4 销售额是多少？** | **回答「暂无相关信息」** | ⭐ 反幻觉关键镜头 |

**话术**（第 4 问）：
> "这是我最想展示的一幕：库里没有的内容，系统不会编，而是按 system prompt 的规则老实拒答。幻觉控制在我看来首先是工程约束——prompt 强制引用来源 + 信息不足必须拒答 + 低温度，然后才用评测去量化它。"

---

## 5. 展示 CI（GitHub Actions）

打开 `.github/workflows/ci.yml` 与仓库 Actions 页：
- `backend`：编译全工程 + 跑无基础设施依赖的单元测试（含 `QueryRouterServiceTest`）。
- `rag-eval`：跑指标单测 + 离线评测门禁，并上传评测报告 artifact。

**话术**：
> "单元测试和检索评测都进了 CI。检索评测在 CI 里跑录制 fixtures 保证确定性，真实指标在本地连 ES 跑出来。"

---

## 6. 技术深挖速查（被追问时的弹药）

| 主题 | 关键事实（代码出处） |
| --- | --- |
| 混合检索 | KNN 召回窗口 `topK×30` + BM25 rescore（KNN 权重 0.2 / BM25 1.0，operator=AND）；向量失败降级纯文本（`HybridSearchService`） |
| 分块 | 父块 1MB 流式防 OOM；子块 段落→句子→HanLP 分词→字符 四级，`chunk-size=512`（`ParseService`） |
| Embedding | DashScope `text-embedding-v4`，2048 维（`application.yml`） |
| 权限 | userId / public / orgTag 直接下推到 ES bool filter，检索时隔离多租户（`HybridSearchService`） |
| 反幻觉 | system 规则：先结论后据、句末引用、信息不足拒答、prompt-injection 防护；`temperature=0.3`（`application.yml` + `DeepSeekClient`） |
| 路由 | Function Calling 决定 知识库检索 / 通用问答，并把问题改写成独立查询；失败安全回退检索（`QueryRouterService`） |
| 记忆 | Redis 存会话，保留最近 20 条、7 天 TTL（`ChatHandler`） |

---

## 7. 诚实边界（被问到要能答）

- 真实指标必须来自 `--mode live`；仓库内 `reports/` 的样例来自录制 fixtures，仅演示流程。
- `@SpringBootTest` 集成测试需 ES/MySQL/Kafka/Redis，未进入轻量 CI，本地或独立集成流水线运行。
- 项目最初基于学习教程，我的增量是：评测体系、回归门禁与 CI、Function Calling 路由、检索参数与权限设计的理解与改造。
