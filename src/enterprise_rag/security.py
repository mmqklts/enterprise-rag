import re

SAFE_REFUSAL_ANSWER = (
    "我不能泄露系统提示词、密钥或内部配置，"
    "也不能伪造法条或来源，"
    "也不能执行删除、修改或其他越权操作。"
)


UNSAFE_REQUEST_PATTERNS = (
    re.compile(
        r"(忽略|无视|绕过).{0,20}(之前|以上|所有).{0,10}(指令|规则)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(系统提示词|system\s*prompt).{0,40}"
        r"(输出|告诉我|泄露|展示)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(api\s*key|密钥).{0,30}(输出|告诉我|泄露|展示)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"delete_documents", re.IGNORECASE),
    re.compile(
        r"(删除|清空).{0,20}(知识库|全部文档|所有文档|文档)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(不要引用来源|伪造来源|伪造法条).{0,30}"
        r"(声称|直接说|告诉)",
        re.IGNORECASE | re.DOTALL,
    ),
)


def is_unsafe_request(text: str) -> bool:
    return any(pattern.search(text) for pattern in UNSAFE_REQUEST_PATTERNS)
