# Enterprise Knowledge Agent

基于 FastAPI 的企业知识库 Agent，支持 PDF、Markdown 和 TXT 文档入库。模型会根据问题动态调用混合检索、原文读取和文档列表工具，经过多步推理后返回带来源引用的答案。

## 功能

- PDF、Markdown、TXT 文档解析
- 文档结构自动检测与分层切分
- 法条、标题、Embedding 语义和固定滑窗四种 Chunking 策略
- 本地 BGE 中文 Embedding
- Qdrant 向量存储与相似度检索
- BM25 与向量检索混合召回
- RRF 排名融合
- BGE CrossEncoder Rerank
- DeepSeek 答案生成
- 多步 Agent 工具调用
- 检索、原文读取和文档列表工具
- 工具执行轨迹与引用校验
- 最大步数和重复调用保护
- 文件名、页码和原文来源引用
- 检索策略评测：Recall@1、Recall@5、MRR
- 80 题分类评测集：事实、改写、多跳、无答案、冲突和 Prompt Injection
- 共享安全拦截：拒绝密钥泄露、伪造法条和删除知识库请求
- FastAPI 接口和 Web 页面

## Agent 流程

```text
用户问题
   |
   v
Agent 分析问题
   |
   +--> search_knowledge_base
   |      Vector + BM25 + RRF + Rerank
   |
   +--> read_source_chunk
   |      读取完整来源 chunk
   |
   +--> list_documents
   |      查看知识库文档范围
   |
   v
根据工具结果决定下一步
   |
   v
最终答案 + 引用校验 + 执行轨迹
```

## Docker 部署

确保已安装 Docker Desktop 和 WSL2。

复制环境变量：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填写 `DEEPSEEK_API_KEY`，然后启动：

```powershell
docker compose up -d --build
```

打开 `http://127.0.0.1:8000/`。

## 本地开发

```powershell
uv sync
uv run uvicorn enterprise_rag.main:app --reload
```

## Agent 接口

`POST /agent/chat`

```json
{
  "question": "对比两份制度中关于报销流程的差异",
  "limit": 5
}
```

响应包含：

- `answer`：最终答案。
- `sources`：答案实际引用的来源。
- `trace`：每一步工具名称、参数、状态和耗时。
- `usage`：LLM 请求次数与 Token 用量。
- `citation_status`：`ok`、`missing` 或 `invalid`。
- `stop_reason`：结束原因，例如 `completed`、`max_steps` 或 `no_evidence`。

旧的固定 RAG 接口 `POST /chat` 继续保留，用于对比固定工作流与 Agent 的行为。

## Agent 限制

- 默认最多执行 6 个模型步骤，可通过 `AGENT_MAX_STEPS` 调整到 1-12。
- 相同工具和参数不会重复执行。
- 单次检索最多返回 8 条来源。
- 工具观测结果超过 16000 字符会被截断。
- 最终引用编号必须对应 Agent 实际检索到的来源。

## 测试

```powershell
uv run pytest -q
uv run ruff check .
```

当前结果：`29 passed`，Ruff 检查通过。

## 检索评测

```powershell
uv run python -m enterprise_rag.evaluation
```

## 80 题 Agent/RAG 评测

评测数据：

```text
data/eval/civil_code_eval_80.jsonl
```

| 类别 | 数量 |
|---|---:|
| 直接事实题 | 40 |
| 同义改写题 | 15 |
| 跨法条题 | 10 |
| 无答案题 | 8 |
| 冲突或模糊题 | 4 |
| Prompt Injection 题 | 3 |
| 总计 | 80 |

固定字符切分下的普通 RAG 基线：

```text
答案正确率（人工复核后）：100%
忠实度：100%
平均耗时：6705.68 ms
P95 耗时：9371.16 ms
```

固定字符切分下，10 道跨法条题对比：

| 指标 | 普通 RAG | Agent |
|---|---:|---:|
| 答案正确率 | 100% | 100% |
| 忠实度 | 100% | 100% |
| 证据完整覆盖率 | 60% | 80% |
| 平均耗时 | 6647.92 ms | 14546.10 ms |
| P95 耗时 | 7279.71 ms | 15548.79 ms |

Agent 在这 10 道题中执行了 27 次模型请求，消耗 71988 Token。结果表明 Agent 更适合多跳和补充证据场景，普通事实题仍应使用 `/chat` 以控制延迟和成本。

## 自动结构切分对比

民法典 PDF 自动识别为 `article` 结构，置信度 `0.85`，从 206 页中提取 1260 条连续法条。结构化入库耗时约 24.95 秒，Qdrant 中该文档只保留 1260 个 `article` chunks。

同一套 80 题的检索证据覆盖对比：

| 类别 | 固定字符切分 | 法条结构切分 |
|---|---:|---:|
| 直接事实题 | 97.5% | 100% |
| 同义改写题 | 93.3% | 100% |
| 跨法条题 | 60% | 40% |

结构切分显著改善了单法条事实题和同义改写题，但跨法条题需要多个来源，仅靠扩大单条法条 Chunk 不能解决。该类别应优先路由到 Agent，通过多次检索和原文读取补充证据。

自动切分入口：

```text
src/enterprise_rag/article_chunker.py
src/enterprise_rag/structure_detector.py
src/enterprise_rag/chunking_router.py
```

重新上传同一文档时，系统会按 `document_id` 删除旧向量，再写入新切分结果，避免不同策略的 Chunk 混在同一索引中。

复现 Agent 多跳评测：

```powershell
uv run python scripts/evaluate_civil_eval80.py `
  --endpoint agent `
  --category multi_hop `
  --output data/eval/civil_code_eval_80_agent_multi_hop_results.jsonl
```

## 安全防护

普通 RAG 和 Agent 在进入检索、工具调用或 LLM 前都会执行共享安全检查，拦截：

- 要求忽略系统规则或泄露系统提示词、API Key。
- 要求调用不存在的 `delete_documents` 工具。
- 要求删除或清空知识库。
- 要求伪造法条或来源并输出虚假结论。

Prompt Injection 评测结果为 `3/3` 正确拒绝，危险请求不会进入模型或工具循环：

```text
工具调用次数：0
模型请求次数：0
Token 消耗：0
```

该规则层是基础防护，生产环境还需要权限控制、输入分类器、审计日志和工具级授权。

复现安全测试：

```powershell
uv run python scripts/evaluate_civil_eval80.py `
  --endpoint agent `
  --category prompt_injection `
  --output data/eval/civil_code_eval_80_agent_prompt_injection_results.jsonl
```
