# Enterprise RAG Knowledge Base

基于 FastAPI 的企业知识库问答系统，支持 PDF、Markdown 和 TXT 文档入库，并通过混合检索、Rerank 和 LLM 返回带来源引用的答案。

## 功能

- PDF、Markdown、TXT 文档解析
- 文本切分与重叠窗口
- 本地 BGE 中文 Embedding
- Qdrant 向量存储与相似度检索
- BM25 与向量检索混合召回
- RRF 排名融合
- BGE CrossEncoder Rerank
- DeepSeek 答案生成
- 文件名、页码和原文来源引用
- 检索策略评测：Recall@1、Recall@5、MRR
- FastAPI 接口和 Web 页面

## 处理流程

```text
PDF / Markdown / TXT
        ↓
文档解析
        ↓
文本切分
        ↓
BGE Embedding
        ↓
Qdrant 向量存储
        ↓
Vector + BM25 混合检索
        ↓
RRF 排名融合
        ↓
BGE Rerank
        ↓
DeepSeek 生成答案
        ↓
答案 + 来源引用

## Docker 部署

确保已安装 Docker Desktop 和 WSL2。

复制环境变量：

```powershell
Copy-Item .env.example .env