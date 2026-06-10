SYSTEM_PROMPT = """你是一名软件安全分析专家。请根据漏洞描述和相关代码，判断漏洞类型、严重程度，并给出简洁推理和修复建议。"""

INSTRUCTION = "请分析以下软件漏洞，输出漏洞类型、严重程度、推理过程和修复建议。"


def build_input(description: str, code: str | None = None) -> str:
    code = code or ""
    if code.strip():
        return f"漏洞描述：{description}\n相关代码：{code}"
    return f"漏洞描述：{description}"


def build_output(cwe: str, severity: str, reasoning: str, fix: str | None = None) -> str:
    fix = fix or "建议结合具体代码位置进行输入校验、边界检查或安全API替换。"
    return (
        f"漏洞类型：{cwe}\n"
        f"严重程度：{severity}\n"
        f"推理过程：{reasoning}\n"
        f"修复建议：{fix}"
    )


def build_chat_prompt(description: str, code: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_input(description, code)},
    ]
