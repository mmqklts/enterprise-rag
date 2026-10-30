# 生产部署清单

这份清单面向单机 Docker Compose 部署。当前项目适合秋招演示和中小规模内部知识库，不应直接宣称为大规模生产平台。

## 1. 生成密钥

在服务器项目目录执行：

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

分别生成 `POSTGRES_PASSWORD` 和 `JWT_SECRET_KEY`，写入 `.env`。不要复用、不要把 `.env` 提交到 Git。

`DATABASE_URL` 中的密码必须与 `POSTGRES_PASSWORD` 一致。上传文件、数据库卷和模型目录都需要持久化和备份。

## 2. 启动服务

```bash
docker compose up -d --build
docker compose ps -a
curl -fsS http://127.0.0.1:8000/health
```

正常情况下 `migrate` 退出码为 `0`，`app`、`worker`、`postgres`、`redis` 和 `qdrant` 正常运行。PostgreSQL、Redis 和 Qdrant 端口只绑定到 `127.0.0.1`，不要直接暴露到公网。

## 3. 配置 HTTPS

推荐使用 Caddy 或 Nginx 作为反向代理。示例见 `deploy/Caddyfile.example`。

```bash
sudo cp deploy/Caddyfile.example /etc/caddy/Caddyfile
sudo sed -i 's/localhost/你的域名/g' /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

代理需要设置：

```text
X-Forwarded-Proto: https
X-Forwarded-For: 客户端 IP
```

只开放 `80/443`。应用端口保持 `127.0.0.1:8000` 绑定。

## 4. 数据备份

PostgreSQL：

```bash
docker compose exec -T postgres pg_dump -U rag enterprise_rag > backup_$(date +%F).sql
```

Qdrant：

```bash
docker compose exec -T qdrant tar -czf - /qdrant/storage > qdrant_$(date +%F).tar.gz
```

上传原文件：

```bash
tar -czf uploads_$(date +%F).tar.gz data/uploads
```

至少定期执行恢复演练。只有备份成功、但未验证恢复，不算完成备份。

## 5. 上线检查

- `APP_ENV=production`，`JWT_SECRET_KEY` 至少 32 个随机字符。
- `POSTGRES_PASSWORD` 是随机值，`.env` 权限仅服务器用户可读。
- `RATE_LIMIT_ENABLED=1`，Redis 可用；`RATE_LIMIT_FAIL_OPEN=0`。
- 域名启用 HTTPS，HTTP 自动跳转 HTTPS。
- 只开放 `80/443`，数据库和向量库不对公网开放。
- 配置 PostgreSQL、Qdrant 和上传文件的定时备份。
- 监控容器状态、错误日志、P95 延迟、Token 成本和限流次数。
- 定期轮换 DeepSeek API Key 和 JWT 密钥；轮换 JWT 密钥会使旧 Token 失效。

## 6. 当前边界

- 限流基于 Redis，单机 Compose 可直接使用；多实例部署可以共享同一 Redis。
- 审计日志目前输出到标准输出，生产环境应接入 Loki、ELK 或云日志服务。
- 尚未实现邮箱验证、密码重置、刷新 Token、账号锁定和多因素认证。
- 尚未实现对象存储、扫描 PDF OCR、跨区域高可用和自动灾备。
