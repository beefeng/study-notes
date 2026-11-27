"""Run I3 SQL AI Functions directly against the model endpoints configured in .env."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import pylibseekdb
import pymysql

ROOT = Path(__file__).resolve().parent
REVIEWS = (
    (1, 7, "退款到账很快，客服解释清楚，整体体验不错。", "billing"),
    (2, 7, "申请退款三天了还没有进度，查询入口不明显。", "billing"),
    (3, 7, "客服回复及时，帮我解决了账号登录问题。", "service"),
    (4, 7, "物流延迟两天，客服主动跟进并说明原因。", "delivery"),
    (5, 7, "退款申请需要填写订单编号，处理流程比较清楚。", "billing"),
    (6, 7, "商品质量符合描述，安装也很简单。", "product"),
    (7, 7, "退款成功了，但是到账时间比预期长。", "billing"),
    (8, 7, "客服没有提前说明退款规则，沟通了好几次。", "service"),
    (9, 7, "包装完整，发货速度很快。", "delivery"),
    (10, 7, "申请退款后能看到每一步处理状态，体验很好。", "billing"),
    (11, 7, "商品和页面描述一致，暂时没有质量问题。", "product"),
    (12, 7, "客服处理退款问题很耐心，但退款到账较慢。", "billing"),
)


def _load_env() -> dict[str, str]:
    """Load KEY=VALUE settings from .env without printing secrets."""
    values: dict[str, str] = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    values.update({key: value for key, value in os.environ.items() if value})
    return values


def _required(config: dict[str, str], key: str) -> str:
    """Return one required environment setting or raise a safe actionable error."""
    value = config.get(key, "").strip()
    if not value:
        raise ValueError(f"Missing {key}; configure it in this directory's .env")
    return value


def validate_summary(raw: str) -> dict[str, Any]:
    """Parse AI_COMPLETE output and enforce the lab's summary contract."""
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("summary"), str):
        raise ValueError("AI_COMPLETE must return a JSON object with summary")
    if value.get("sentiment") not in {"positive", "negative", "mixed"}:
        raise ValueError("AI_COMPLETE returned unsupported sentiment")
    if not isinstance(value.get("review_count"), int) or value["review_count"] < 0:
        raise ValueError("AI_COMPLETE returned invalid review_count")
    return value


def _register_model(cursor: pymysql.cursors.Cursor, name: str, kind: str, model: str,
                    endpoint: str, key: str, provider: str) -> None:
    """Register a logical seekdb model and map it to its real HTTP endpoint."""
    model_config = json.dumps({"type": kind, "model_name": model})
    endpoint_config = json.dumps({"ai_model_name": name, "scope": "all", "url": endpoint,
                                  "access_key": key, "provider": provider,
                                  "request_model_name": model})
    cursor.execute("CALL DBMS_AI_SERVICE.CREATE_AI_MODEL(%s, %s)", (name, model_config))
    cursor.execute("CALL DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT(%s, %s)",
                   (name + "_endpoint", endpoint_config))


def _setup(cursor: pymysql.cursors.Cursor, config: dict[str, str]) -> None:
    """Register models and create isolated review/task tables in seekdb."""
    embed_url = _required(config, "EMBEDDING_ENDPOINT")
    llm_base = _required(config, "LLM_BASE_URL").rstrip("/")
    _register_model(cursor, "i3_embedding", "dense_embedding", _required(config, "EMBEDDING_MODEL"),
                    embed_url, _required(config, "EMBEDDING_API_KEY"), "OPENAI")
    _register_model(cursor, "i3_completion", "completion", _required(config, "LLM_MODEL"),
                    llm_base + "/chat/completions", _required(config, "LLM_API_KEY"), "OPENAI")
    if config.get("RERANK_ENDPOINT"):
        _register_model(cursor, "i3_rerank", "rerank", _required(config, "RERANK_MODEL"),
                        config["RERANK_ENDPOINT"], _required(config, "RERANK_API_KEY"),
                        config.get("RERANK_PROVIDER", "SILICONFLOW"))
    cursor.execute("CREATE TABLE i3_reviews (review_id INT PRIMARY KEY, tenant_id INT NOT NULL, category VARCHAR(32) NOT NULL, content TEXT NOT NULL, status VARCHAR(20) NOT NULL, input_hash VARCHAR(64) NOT NULL, embedding VECTOR(1024), embedding_version VARCHAR(64), summary_text TEXT NULL)")
    cursor.execute("CREATE TABLE i3_tasks (task_id VARCHAR(64) PRIMARY KEY, review_id INT NOT NULL, status VARCHAR(24) NOT NULL, attempt_count INT NOT NULL, error_message VARCHAR(256) NULL)")
    rows = [(rid, tenant, category, text, "approved", hashlib.sha256(text.encode()).hexdigest())
            for rid, tenant, text, category in REVIEWS]
    cursor.executemany("INSERT INTO i3_reviews (review_id, tenant_id, category, content, status, input_hash) VALUES (%s,%s,%s,%s,%s,%s)", rows)


def run_experiment(work_dir: Path) -> dict[str, Any]:
    """Run embedding, retrieval, optional reranking, completion and rollback via seekdb SQL."""
    config = _load_env()
    work_dir.mkdir(parents=True, exist_ok=True)
    database_dir = work_dir / "seekdb"
    if database_dir.exists() and any(database_dir.iterdir()):
        raise FileExistsError(f"Database directory must be empty: {database_dir}")
    with closing(pylibseekdb.open(str(database_dir))) as runtime:
        with pymysql.connect(database="test", autocommit=True, **runtime.connection_options()) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT VERSION()")
                version = str(cursor.fetchone()[0])
                _setup(cursor, config)
                durations: list[float] = []
                for rid, _tenant, _text, _category in REVIEWS:
                    start = time.perf_counter()
                    cursor.execute("UPDATE i3_reviews SET embedding=AI_EMBED('i3_embedding',content), embedding_version=%s WHERE review_id=%s",
                                   (config["EMBEDDING_MODEL"], rid))
                    durations.append((time.perf_counter() - start) * 1000)
                query = "退款进度与客服处理"
                cursor.execute("SET @i3_query_embedding=AI_EMBED('i3_embedding',%s)", (query,))
                cursor.execute("SELECT review_id,category,content,L2_DISTANCE(embedding,@i3_query_embedding) AS distance FROM i3_reviews WHERE tenant_id=7 AND status='approved' ORDER BY distance LIMIT 6")
                candidates = [{"review_id": row[0], "category": row[1], "content": row[2], "distance": float(row[3])} for row in cursor.fetchall()]
                if config.get("RERANK_ENDPOINT"):
                    cursor.execute("SELECT AI_RERANK('i3_rerank',%s,%s,'content')", (query, json.dumps(candidates, ensure_ascii=False)))
                    candidates = json.loads(cursor.fetchone()[0])
                cursor.execute("SELECT AI_PROMPT(%s,%s)", ("仅返回一个合法JSON对象，不要Markdown代码块，字段必须为summary字符串、sentiment取positive/negative/mixed、review_count非负整数。根据以下评论总结：{0}", json.dumps(candidates[:3], ensure_ascii=False)))
                prompt = json.loads(cursor.fetchone()[0])
                start = time.perf_counter()
                cursor.execute("SELECT AI_COMPLETE('i3_completion',%s)", (json.dumps(prompt, ensure_ascii=False),))
                summary = validate_summary(str(cursor.fetchone()[0]))
                durations.append((time.perf_counter() - start) * 1000)
                cursor.execute("UPDATE i3_reviews SET summary_text=%s WHERE review_id=1", (json.dumps(summary, ensure_ascii=False),))
                connection.begin()
                cursor.execute("UPDATE i3_reviews SET summary_text=AI_COMPLETE('i3_completion',%s) WHERE review_id=12", (json.dumps({"template": "Summarize {0}", "args": ["rollback test"]}),))
                connection.rollback()
                cursor.execute("SELECT summary_text FROM i3_reviews WHERE review_id=12")
                rolled_back = cursor.fetchone()[0] is None
    result = {"database_version": version, "provider": "real endpoints configured in .env; requests issued by seekdb AI Functions",
              "review_count": len(REVIEWS), "function_calls": {"AI_EMBED": len(REVIEWS) + 1, "AI_PROMPT": 1,
              "AI_COMPLETE": 2, "AI_RERANK": int(bool(config.get("RERANK_ENDPOINT")))},
              "rerank_configured": bool(config.get("RERANK_ENDPOINT")),
              "top_review_ids": [row.get("review_id") for row in candidates[:3]], "summary": summary,
              "transaction_demo": {"rolled_back_local_write": rolled_back},
              "latency_ms": {"samples": len(durations), "p95": sorted(durations)[math.ceil(.95 * len(durations))-1]}}
    report = ROOT / "results" / "latest.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    """Run one isolated seekdb experiment and print its sanitized report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.work_dir:
        report = run_experiment(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="i3-ai-functions-") as temp:
            report = run_experiment(Path(temp))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
