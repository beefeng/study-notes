# I3 AI Functions 实验工程

实验通过 seekdb embedded 执行原生 SQL `AI_EMBED`、`AI_PROMPT` 和 `AI_COMPLETE`。模型请求由 seekdb 根据逻辑模型和端点配置发出，直接访问 `.env` 中配置的真实服务；工程不启动或实现任何模型 HTTP 服务。

## 配置与运行

`.env` 保存本地地址和密钥，已被 `.gitignore` 排除。工程不会把密钥写入报告或日志。

```bash
cd "Easy Data x AI/experiment/task"
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python i3_ai_functions_lab.py
```

必需配置：`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`、`EMBEDDING_ENDPOINT`、`EMBEDDING_API_KEY`、`EMBEDDING_MODEL`。可选 Rerank 配置：`RERANK_ENDPOINT`、`RERANK_API_KEY`、`RERANK_MODEL`、`RERANK_PROVIDER`。`AI_RERANK` 需要支持 Rerank 请求协议的端点；聊天补全接口不能默认当作 Rerank 接口，因此当前未配置时实验会跳过此步骤并在报告标明。

## 实验流程

建立 12 条评论并配置 seekdb 逻辑模型/端点；用 `AI_EMBED` 生成向量，SQL 做租户和状态过滤及向量召回；配置了 Rerank 端点时用 `AI_RERANK` 精排；再通过 `AI_PROMPT` 和 `AI_COMPLETE` 生成 JSON 摘要，校验后写回。另执行事务回滚演示，输出函数调用数和客户端观测 P95。报告写入 `results/latest.json`。

本版本不模拟 HTTP 429 或伪造模型响应；故障场景需通过真实 Provider/网关演练验证。成本需结合真实 Provider 的 usage 和价格核算。
