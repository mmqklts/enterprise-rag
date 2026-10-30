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
- `knowledge_base_id` 贯穿入库、向量检索、BM25 和 Agent 工具调用
- PostgreSQL 用户、知识库和成员角色数据模型
- Alembic 数据库迁移
- scrypt 密码哈希与 HS256 JWT 身份认证
- 基于知识库成员角色的上传与查询权限控制
- Redis + Celery 异步文档解析任务
- 文档任务状态查询与失败重试
- 文档列表、删除、原文件清理和向量同步
- Redis 固定窗口限流，覆盖登录、注册、问答、上传和管理操作
- 认证、成员管理和文档生命周期审计日志
- CSP、HSTS、禁止页面嵌套等 HTTP 安全响应头
- 请求 ID、结构化日志、阶段耗时、Token 和成本估算
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

在 `.env` 中填写 `DEEPSEEK_API_KEY`，并按
[生产部署清单](docs/PRODUCTION_DEPLOYMENT.md) 生成随机
`POSTGRES_PASSWORD` 和 `JWT_SECRET_KEY`。Compose 不再接受缺失的
PostgreSQL 密码。然后启动：

```powershell
docker compose up -d --build
```

打开 `http://127.0.0.1:8000/`。

Compose 会同时启动：

```text
app       FastAPI Web/API
worker    Celery 文档处理 Worker
migrate   启动时执行 Alembic 数据库迁移
postgres  用户和知识库权限元数据
redis     Celery Broker 和结果后端
qdrant    向量数据库
```

Linux 服务器首次部署：

```bash
git clone https://github.com/mmqklts/enterprise-rag.git
cd enterprise-rag
cp .env.example .env
docker compose up -d --build
docker compose ps -a
```

`migrate` 容器执行完成后应以退出码 `0` 结束，其余服务应保持运行。

## 异步文档处理

上传接口现在只保存文件并创建任务，立即返回 `202`：

```text
POST /documents/upload
GET  /jobs/{job_id}
```

任务状态：

```text
pending     等待 Worker
processing  正在解析、切分和 Embedding
completed   入库完成
failed      处理失败
```

任务记录包含 `attempts`、起止时间、切分策略、chunk 数量、文档 ID 和错误信息。Celery 默认最多重试 2 次，Worker 使用与 App 相同的 PostgreSQL、Qdrant 和模型挂载配置。

Web 页面上传后会轮询 `/jobs/{job_id}`，完成后再提示 chunk 数量。

文档生命周期接口：

```text
GET    /knowledge-bases/{id}/documents
DELETE /knowledge-bases/{id}/documents/{document_id}
```

删除操作要求 `owner` 或 `editor` 权限，并同步清理数据库记录、Qdrant 向量和上传原文件。

真实验证流程已经通过：

```text
上传 -> pending -> processing -> completed
-> 文档列表可见
-> 问答返回答案
-> 删除返回 204
-> 文档列表为空
-> Qdrant 对应知识库向量为 0
```

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
  "limit": 5,
  "knowledge_base_id": 1
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

当前结果：`48 passed`，Ruff 检查通过。

## 可观测性

每个 HTTP 响应都会返回 `X-Request-ID`。结构化日志包含：

```text
request_id
method
path
status_code
duration_ms
```

普通 RAG 日志额外记录：

```text
user_id
knowledge_base_id
retrieval_ms
rerank_ms
llm_ms
total_ms
prompt_tokens
completion_tokens
total_tokens
estimated_cost
```

成本通过以下环境变量计算：

```dotenv
LLM_CACHE_HIT_PRICE_PER_MILLION=0.04
LLM_CACHE_MISS_PRICE_PER_MILLION=2
LLM_OUTPUT_PRICE_PER_MILLION=8
```

实际使用时应按所用模型和当前价格修改。日志不记录用户问题原文、API Key 或密码。

## 权限数据模型

项目使用 PostgreSQL 保存用户和知识库权限元数据，启动数据库：

```powershell
docker compose up -d postgres
```

执行迁移：

```powershell
uv run alembic upgrade head
```

当前表：

```text
users                    用户账号和密码哈希
knowledge_bases          知识库及所有者
knowledge_base_members   知识库成员及角色
```

成员角色约束：

```text
owner    管理知识库和成员
editor   上传和更新文档
viewer   只读查询
```

数据库连接从 `DATABASE_URL` 读取。Docker Compose 会把应用连接到 `postgres` 服务，本地执行迁移时使用 `.env` 中的地址。

## 知识库隔离

上传、普通 RAG 和 Agent 接口都接收数据库整数 `knowledge_base_id`：

```text
POST /documents/upload  -> Form knowledge_base_id
POST /chat              -> JSON knowledge_base_id
POST /agent/chat        -> JSON knowledge_base_id
```

该 ID 会写入每个 Qdrant Payload，并在向量检索和 BM25 Scroll 阶段作为过滤条件使用。Agent 工具不能由模型自行选择知识库，服务端会把当前请求的知识库范围传入所有工具。

已验证同一问题在 `default` 和其他知识库下的行为：

```text
default  -> 返回答案和来源
other-kb -> 返回 0 条来源
```

`knowledge_base_id` 现在是数据库中的整数知识库 ID。除认证接口外，所有业务接口必须携带：

```http
Authorization: Bearer <access_token>
```

权限矩阵：

| 操作 | owner | editor | viewer |
|---|---:|---:|---:|
| 普通 RAG `/chat` | 允许 | 允许 | 允许 |
| Agent `/agent/chat` | 允许 | 允许 | 允许 |
| 上传 `/documents/upload` | 允许 | 允许 | 拒绝 |

服务端会查询 `knowledge_base_members` 并校验用户角色。客户端不能通过修改 `knowledge_base_id` 访问未加入的知识库。

## 身份认证

已提供以下接口：

```text
POST /auth/register  注册用户并创建默认知识库
POST /auth/login     用户名密码登录并签发 JWT
GET  /auth/me        使用 Bearer Token 获取当前用户和知识库角色
```

密码使用 Python 标准库 `scrypt` 加盐哈希，不保存明文密码。JWT 使用 HS256 签名，并提供过期时间。

当 `APP_ENV=production` 时，服务会拒绝默认值、占位值或少于 32 字符的
`JWT_SECRET_KEY`，配置错误会直接阻止启动。

环境变量：

```dotenv
JWT_SECRET_KEY=replace_with_a_long_random_secret
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

注册后系统会自动创建：

```text
knowledge_bases          默认知识库
knowledge_base_members   owner 成员关系
```

Web 页面已经加入登录、注册和知识库选择，并在上传与问答请求中携带 JWT。

## 知识库管理

知识库管理接口：

```text
POST   /knowledge-bases
GET    /knowledge-bases
POST   /knowledge-bases/{id}/members
PATCH  /knowledge-bases/{id}/members/{user_id}
DELETE /knowledge-bases/{id}/members/{user_id}
```

管理规则：

- 所有接口都需要 Bearer Token。
- 新知识库创建后，创建者自动成为 `owner`。
- 只有 `owner` 可以添加、修改和移除成员。
- 同一个用户不能在一个知识库中重复加入。
- 角色只能是 `owner`、`editor` 或 `viewer`。
- 知识库必须至少保留一个 `owner`，不能降级或删除最后一个 owner。

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

结构化索引下的 Agent 多跳评测：

| 指标 | 普通 RAG | Agent |
|---|---:|---:|
| 答案正确率 | 90% | 100% |
| 忠实度 | 100% | 100% |
| 证据完整覆盖率 | 40% | 70% |
| 平均耗时 | 约 6.5 秒 | 10.60 秒 |
| P95 耗时 | 约 7.3 秒 | 14.50 秒 |
| 模型请求次数 | 10 | 25 |
| Token 总量 | 未记录 | 45651 |

法条结构切分降低了单条证据的检索成本，但多跳问题仍需要多个法条。Agent 将证据覆盖率从 40% 提升到 70%，答案正确率达到 100%，代价是额外检索、模型调用和延迟。

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
  --token "<access_token>" `
  --knowledge-base-id 1 `
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

当前项目还提供：

- Redis 限流：登录、注册、普通 RAG、Agent、上传、删除和知识库创建分别配置阈值。
- 审计日志：记录认证结果、知识库成员变更、文档上传和删除，不记录密码或 Token。
- 安全响应头：CSP、HSTS、`X-Content-Type-Options` 和
  `X-Frame-Options`。
- 生产密钥校验：生产环境拒绝弱 JWT 密钥。
- 网络隔离：Compose 默认只把服务端口绑定到 `127.0.0.1`，公网入口应经过
  HTTPS 反向代理。

部署、备份和上线检查见
[生产部署清单](docs/PRODUCTION_DEPLOYMENT.md)。

复现安全测试：

```powershell
uv run python scripts/evaluate_civil_eval80.py `
  --endpoint agent `
  --category prompt_injection `
  --token "<access_token>" `
  --knowledge-base-id 1 `
  --output data/eval/civil_code_eval_80_agent_prompt_injection_results.jsonl
```
