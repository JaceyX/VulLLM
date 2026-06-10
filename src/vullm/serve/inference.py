import os
from openai import OpenAI

from src.vullm.prompts import build_chat_prompt


def mock_analyze(description: str, code: str | None = None, memory_cases: list[dict] | None = None) -> dict:
    text = f"{description}\n{code or ''}".lower()

    if "sql" in text or "query" in text:
        cwe, severity = "CWE-89", "CRITICAL"
        reasoning = "用户输入影响SQL语句结构，存在SQL注入风险。"
        fix = "使用参数化查询或预编译语句。"
    elif "html" in text or "script" in text or "document.write" in text:
        cwe, severity = "CWE-79", "HIGH"
        reasoning = "用户可控输入进入HTML上下文，可能导致跨站脚本攻击。"
        fix = "进行输出编码、输入校验并配置CSP。"
    elif "strcpy" in text or "buffer" in text:
        cwe, severity = "CWE-120", "HIGH"
        reasoning = "未检查缓冲区长度可能导致内存越界写入。"
        fix = "使用边界检查和安全拷贝函数。"
    elif "md5" in text:
        cwe, severity = "CWE-327", "MEDIUM"
        reasoning = "使用弱加密算法可能导致密码或敏感信息被破解。"
        fix = "使用Argon2、bcrypt或scrypt等安全哈希算法。"
    else:
        cwe, severity = "UNKNOWN", "MEDIUM"
        reasoning = "需要结合更多上下文判断漏洞类型和影响范围。"
        fix = "补充代码上下文并进行安全审计。"

    return {
        "cwe": cwe,
        "severity": severity,
        "reasoning": reasoning,
        "fix": fix,
        "memory_cases": memory_cases or [],
    }


def openai_compatible_analyze(
    description: str,
    code: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    client = OpenAI(
        base_url=base_url or os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"),
        api_key=api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
    )
    model = model or os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    response = client.chat.completions.create(
        model=model,
        messages=build_chat_prompt(description, code),
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content
