import json
import statistics
import time
from math import ceil
from urllib.request import Request, urlopen

API_URL = "http://127.0.0.1:8000/chat"
QUESTION = "请假需要提前多久申请？"
REQUESTS = 10


def ask(question: str) -> dict:
    payload = json.dumps(
        {"question": question, "limit": 5},
        ensure_ascii=False,
    ).encode("utf-8")

    request = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = ceil(percent / 100 * len(ordered)) - 1
    return ordered[max(index, 0)]


def print_metric(label: str, values: list[float]) -> None:
    print(
        f"{label}: "
        f"平均 {statistics.mean(values):.2f}，"
        f"P50 {percentile(values, 50):.2f}，"
        f"P95 {percentile(values, 95):.2f}"
    )


def main() -> None:
    print("预热请求...")
    ask(QUESTION)

    wall_times: list[float] = []
    retrieval_times: list[float] = []
    rerank_times: list[float] = []
    llm_times: list[float] = []
    server_total_times: list[float] = []
    prompt_tokens: list[float] = []
    completion_tokens: list[float] = []
    total_tokens: list[float] = []
    cache_hit_tokens: list[float] = []
    cache_miss_tokens: list[float] = []

    for index in range(1, REQUESTS + 1):
        started = time.perf_counter()
        result = ask(QUESTION)
        wall_ms = (time.perf_counter() - started) * 1000

        metrics = result["metrics"]
        wall_times.append(wall_ms)
        retrieval_times.append(metrics["retrieval_ms"])
        rerank_times.append(metrics["rerank_ms"])
        llm_times.append(metrics["llm_ms"])
        server_total_times.append(metrics["total_ms"])
        prompt_tokens.append(metrics["prompt_tokens"])
        completion_tokens.append(metrics["completion_tokens"])
        total_tokens.append(metrics["total_tokens"])
        cache_hit_tokens.append(metrics["cache_hit_tokens"])
        cache_miss_tokens.append(metrics["cache_miss_tokens"])

        print(
            f"第 {index} 次: 端到端 {wall_ms:.2f} ms,"
            f"检索 {metrics['retrieval_ms']:.2f} ms,"
            f"Rerank {metrics['rerank_ms']:.2f} ms,"
            f"LLM {metrics['llm_ms']:.2f} ms,"
            f"Token {metrics['total_tokens']}"
            f"缓存命中 {metrics['cache_hit_tokens']},"
            f"缓存未命中 {metrics['cache_miss_tokens']}"
        )

    print()
    print_metric("端到端耗时", wall_times)
    print_metric("服务端总耗时", server_total_times)
    print_metric("检索耗时", retrieval_times)
    print_metric("Rerank 耗时", rerank_times)
    print_metric("LLM 耗时", llm_times)
    print()
    print(f"Prompt Token 总量: {sum(prompt_tokens):.0f}")
    print(f"Completion Token 总量: {sum(completion_tokens):.0f}")
    print(f"Token 总量: {sum(total_tokens):.0f}")
    print(f"单次平均 Token: {statistics.mean(total_tokens):.2f}")
    print(f"缓存命中 Token 总量: {sum(cache_hit_tokens):.0f}")
    print(f"缓存未命中 Token 总量: {sum(cache_miss_tokens):.0f}")


if __name__ == "__main__":
    main()