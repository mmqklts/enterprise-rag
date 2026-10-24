import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from enterprise_rag.llm_client import DeepSeekClient

API_URL = "http://127.0.0.1:8000/chat"
QUESTIONS_PATH = Path("data/eval/civil_code_questions.jsonl")
RESULTS_PATH = Path("data/eval/civil_code_results.jsonl")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value)


def ask(question: str) -> dict[str, Any]:
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


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Judge 未返回 JSON：{text}")

    return json.loads(cleaned[start : end + 1])


def build_judge_prompt(
    question: str,
    reference_answer: str,
    generated_answer: str,
    sources: list[dict[str, Any]],
) -> str:
    source_text = "\n\n".join(
        (
            f"[{source['index']}] {source['filename']} "
            f"第 {source['page']} 页\n{source['text']}"
        )
        for source in sources
    )

    return f"""
你是严格的 RAG 答案评测员。
请根据问题、参考答案和检索来源评估生成答案。
只输出 JSON，不要输出 Markdown 或额外说明。

判断标准：
1. answer_correct：生成答案与参考答案语义一致，并且没有遗漏问题直接询问的关键事实。
2. faithful：生成答案中的所有事实陈述都能被检索来源支持。
3. citation_correct：答案中的引用编号确实支持对应结论。
只判断问题直接询问的范围，不要因为答案没有补充问题未询问的附加规定而判错。

输出格式：
{{
  "answer_correct": true,
  "faithful": true,
  "citation_correct": true,
  "reason": "简短原因"
}}

问题：
{question}

参考答案：
{reference_answer}

生成答案：
{generated_answer}

检索来源：
{source_text}
"""


def citation_metrics(
    answer: str,
    sources: list[dict[str, Any]],
    expected_filename: str,
    expected_text: str,
) -> tuple[bool, bool, float]:
    citations = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]

    if not citations:
        return False, False, 0.0

    valid_indices = all(1 <= index <= len(sources) for index in citations)
    correct_citations = 0

    for index in set(citations):
        if not 1 <= index <= len(sources):
            continue

        source = sources[index - 1]
        if (
            source.get("filename") == expected_filename
            and normalize_text(expected_text)
            in normalize_text(str(source.get("text", "")))
        ):
            correct_citations += 1

    accuracy = correct_citations / len(set(citations))
    return True, valid_indices, accuracy


def load_questions() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    questions = load_questions()
    judge = DeepSeekClient()
    results: list[dict[str, Any]] = []

    for position, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"[{position}/{len(questions)}] {question}")

        response = ask(question)
        answer = str(response["answer"])
        sources = response["sources"]

        judge_result = parse_json_object(
            judge.generate(
                build_judge_prompt(
                    question=question,
                    reference_answer=item["reference_answer"],
                    generated_answer=answer,
                    sources=sources,
                )
            ).content
        )

        citation_present, citation_valid, citation_source_accuracy = (
            citation_metrics(
                answer=answer,
                sources=sources,
                expected_filename=item["expected_filename"],
                expected_text=item["expected_text"],
            )
        )
        retrieval_hit = any(
            source.get("filename") == item["expected_filename"]
            and normalize_text(item["expected_text"])
            in normalize_text(str(source.get("text", "")))
            for source in sources
        )

        result = {
            **item,
            "answer": answer,
            "sources": sources,
            "metrics": response["metrics"],
            "answer_correct": bool(judge_result["answer_correct"]),
            "faithful": bool(judge_result["faithful"]),
            "citation_correct_judge": bool(
                judge_result["citation_correct"]
            ),
            "citation_present": citation_present,
            "citation_valid": citation_valid,
            "retrieval_hit": retrieval_hit,
            "citation_source_accuracy": citation_source_accuracy,
            "judge_reason": str(judge_result.get("reason", "")),
        }
        results.append(result)
        time.sleep(0.2)

    RESULTS_PATH.write_text(
        "".join(
            json.dumps(result, ensure_ascii=False) + "\n"
            for result in results
        ),
        encoding="utf-8",
    )

    count = len(results)
    print()
    print(f"答案正确率：{sum(r['answer_correct'] for r in results) / count:.3f}")
    print(f"忠实度：{sum(r['faithful'] for r in results) / count:.3f}")
    print(
        "引用存在率："
        f"{sum(r['citation_present'] for r in results) / count:.3f}"
    )
    print(
        "引用编号有效率："
        f"{sum(r['citation_valid'] for r in results) / count:.3f}"
    )
    print(
        "引用来源命中率："
        f"{sum(r['citation_source_accuracy'] for r in results) / count:.3f}"
    )
    print(
        "检索命中率："
        f"{sum(r['retrieval_hit'] for r in results) / count:.3f}"
    )
    print(
        "引用准确率（Judge）："
        f"{sum(r['citation_correct_judge'] for r in results) / count:.3f}"
    )
    print(f"结果文件：{RESULTS_PATH}")


if __name__ == "__main__":
    main()
