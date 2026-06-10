"""scripts/collect_data.py - 手动抓取漏洞数据集

数据源：
  - NVD CVE Feeds: https://nvd.nist.gov/vuln/data-feeds  (美国政府公开数据，public domain)
  - CVEfixes:      https://github.com/secureIT-project/CVEfixes  (MIT License)

子命令：
  nvd            下载 NVD feeds（默认近 5 年）
  cvefixes       浅克隆 CVEfixes 仓库
  parse-nvd      解析 NVD feeds 为 JSONL
  parse-cvefixes 解析 CVEfixes 为 JSONL（含 code/fix）
  join           按 CVE ID 合并 NVD + CVEfixes
  reasoning      用 LLM 为每条数据生成中文 reasoning + fix_summary
  all            跑完全流程

示例：
  # 完整流程：抓 2022-2025 的 NVD + 克隆 CVEfixes + 合并
  python scripts/collect_data.py all --years 2022 2023 2024 2025 --limit 2000

  # 分步
  python scripts/collect_data.py nvd --years 2022 2023 2024 2025
  python scripts/collect_data.py cvefixes
  python scripts/collect_data.py parse-nvd --years 2022 2023 2024 2025
  python scripts/collect_data.py parse-cvefixes
  python scripts/collect_data.py join --limit 2000

  # 跑完后，下一步用项目自带的数据集脚本归一化为指令数据：
  python scripts/build_dataset.py \
      --input  data/processed/joined.jsonl \
      --output data/processed/train.jsonl
"""

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from openai import OpenAI

# ===== 路径配置 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
NVD_DIR = RAW_DIR / "nvd"
CVEFIXES_DIR = RAW_DIR / "cvefixes"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ===== 远端 =====
NVD_FEED_URL = "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz"
CVEFIXES_REPO = "https://github.com/secureIT-project/CVEfixes.git"


# ===== 工具函数 =====
def cvss_to_severity(score: float | None) -> str:
    """CVSS 分数 → LOW / MEDIUM / HIGH / CRITICAL。"""
    if score is None:
        return "UNKNOWN"
    if score >= 9.0:
        return "CRITICAL"
    if score >= 7.0:
        return "HIGH"
    if score >= 4.0:
        return "MEDIUM"
    if score > 0.0:
        return "LOW"
    return "UNKNOWN"


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ===== 1. NVD 下载 =====
def cmd_nvd(args) -> None:
    NVD_DIR.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        json_path = NVD_DIR / f"nvdcve-1.1-{year}.json"
        if json_path.exists():
            print(f"  [skip] {json_path.name} 已存在")
            continue
        gz_path = NVD_DIR / f"nvdcve-1.1-{year}.json.gz"
        url = NVD_FEED_URL.format(year=year)
        print(f"  [get]  {url}")
        try:
            urllib.request.urlretrieve(url, gz_path)
        except Exception as e:
            print(f"  [err]  下载失败: {e}", file=sys.stderr)
            continue
        # 解压
        with gzip.open(gz_path, "rb") as f_in, open(json_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        size_mb = json_path.stat().st_size / 1024 / 1024
        print(f"  [ok]   {json_path.name}  ({size_mb:.1f} MB)")


# ===== 2. CVEfixes 克隆 =====
def cmd_cvefixes(args) -> None:
    if CVEFIXES_DIR.exists() and (CVEFIXES_DIR / ".git").exists():
        print(f"  [skip] {CVEFIXES_DIR} 已存在")
        return
    CVEFIXES_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [clone] {CVEFIXES_REPO}  (depth=1)")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", CVEFIXES_REPO, str(CVEFIXES_DIR)],
            check=True,
        )
        print(f"  [ok]   {CVEFIXES_DIR}")
    except FileNotFoundError:
        print("  [err]  git 未安装或不在 PATH 中（Windows 需先安装 Git for Windows）",
              file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"  [err]  克隆失败: {e}", file=sys.stderr)


# ===== 3. NVD 解析 =====
def parse_nvd_file(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("CVE_Items", []):
        cve = item.get("cve", {})
        cve_id = cve.get("CVE_data_meta", {}).get("ID", "")

        # description (英文优先)
        desc = ""
        for d in cve.get("description", {}).get("description_data", []):
            if d.get("lang") == "en":
                desc = d.get("value", "").strip()
                break
        if not desc:
            continue

        # CWE
        cwe = "UNKNOWN"
        for w in cve.get("weaknesses", {}).get("description", []):
            if w.get("lang") == "en":
                val = w.get("value", "")
                if val.startswith("CWE-"):
                    cwe = val
                    break

        # severity (CVSS v3 优先)
        impact = item.get("impact", {})
        severity = "UNKNOWN"
        score = None
        if "baseMetricV3" in impact:
            v3 = impact["baseMetricV3"]
            severity = v3.get("baseSeverity", "UNKNOWN")
            score = v3.get("cvssV3", {}).get("baseScore")
        elif "baseMetricV2" in impact:
            v2 = impact["baseMetricV2"]
            severity = v2.get("severity", "UNKNOWN")
            score = v2.get("cvssV2", {}).get("baseScore")
        if severity == "UNKNOWN" and score is not None:
            severity = cvss_to_severity(score)

        yield {
            "id": cve_id,
            "description": desc,
            "cwe": cwe,
            "severity": severity,
        }


def cmd_parse_nvd(args) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "nvd_parsed.jsonl"
    total = 0
    with open(out, "w", encoding="utf-8") as fout:
        for year in args.years:
            src = NVD_DIR / f"nvdcve-1.1-{year}.json"
            if not src.exists():
                print(f"  [skip] {src.name} 不存在，请先跑 nvd")
                continue
            n = 0
            for record in parse_nvd_file(src):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
            total += n
            print(f"  [ok]   {src.name} → {n} 条")
    print(f"  [done] {out}  (共 {total} 条)")


# ===== 4. CVEfixes 解析 =====
def find_cvefixes_csv(repo_dir: Path) -> Path | None:
    """在 CVEfixes 仓库里找聚合 CSV。兼容多个版本。"""
    candidates = list(repo_dir.rglob("*.csv"))
    if not candidates:
        return None
    # 优先文件名包含 data / extract / final / summary
    priority_keys = ["data", "extract", "final", "summary", "all"]
    for key in priority_keys:
        for p in candidates:
            if key in p.name.lower():
                return p
    return candidates[0]


def parse_cvefixes_csv(csv_path: Path) -> Iterable[dict]:
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cve_id = (row.get("cve_id") or row.get("cve") or row.get("CVE_ID") or "").strip()
            if not cve_id:
                continue
            # CWE 字段可能是列表 / 多值，取第一个
            cwe_raw = (row.get("cwe_id") or row.get("cwe") or row.get("CWE_ID") or "")
            cwe = "UNKNOWN"
            for sep in [";", ",", " ", "|"]:
                if sep in cwe_raw:
                    cwe = cwe_raw.split(sep)[0].strip()
                    break
            else:
                cwe = cwe_raw.strip() or "UNKNOWN"
            if not cwe.startswith("CWE-"):
                cwe = "UNKNOWN"
            # code / fix 截断避免过长
            code = (row.get("func_before") or row.get("code") or "")[:4000]
            fix = (row.get("func_after") or row.get("fix") or "")[:4000]
            yield {
                "id": cve_id,
                "code": code,
                "fix": fix,
                "cwe": cwe,
                "language": (row.get("lang") or row.get("language") or "unknown"),
            }


def cmd_parse_cvefixes(args) -> None:
    if not CVEFIXES_DIR.exists():
        print(f"  [err]  {CVEFIXES_DIR} 不存在，请先跑 cvefixes", file=sys.stderr)
        return
    csv_path = find_cvefixes_csv(CVEFIXES_DIR)
    if csv_path is None:
        print(f"  [err]  {CVEFIXES_DIR} 中未找到 CSV，可能需要先在仓库内执行 data_extractor",
              file=sys.stderr)
        return
    print(f"  [read] {csv_path}")
    out = PROCESSED_DIR / "cvefixes_parsed.jsonl"
    n = write_jsonl(parse_cvefixes_csv(csv_path), out)
    print(f"  [ok]   {out}  ({n} 条)")


# ===== 5. 合并 =====
def cmd_join(args) -> None:
    nvd_path = PROCESSED_DIR / "nvd_parsed.jsonl"
    cf_path = PROCESSED_DIR / "cvefixes_parsed.jsonl"

    if not nvd_path.exists():
        print(f"  [err]  {nvd_path} 不存在，请先跑 parse-nvd", file=sys.stderr)
        return

    cf_map: dict[str, dict] = {}
    if cf_path.exists():
        with open(cf_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cid = rec.get("id", "")
                if cid and cid not in cf_map:
                    cf_map[cid] = rec
        print(f"  [load] {cf_path}  ({len(cf_map)} 条带 code/fix)")
    else:
        print(f"  [warn] {cf_path} 不存在，仅用 NVD 数据")

    out = PROCESSED_DIR / "joined.jsonl"
    count = 0
    with open(nvd_path, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("id", "")
            cf = cf_map.get(cid, {})
            merged = {
                "id": cid,
                "description": rec.get("description", ""),
                "cwe": rec.get("cwe", "UNKNOWN"),
                "severity": rec.get("severity", "UNKNOWN"),
                "code": cf.get("code", ""),
                "fix": cf.get("fix", ""),
                "language": cf.get("language", ""),
            }
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    with_code = sum(1 for cid, _ in cf_map.items() if cid in {l["id"] for l in []})  # 占位
    print(f"  [ok]   {out}  (共 {count} 条，其中含 code 的 ≈ {min(count, len(cf_map))} 条)")


# ===== 6. 一键 =====
def cmd_all(args) -> None:
    print("=" * 60)
    print(f"年份范围: {args.years}")
    print(f"上限条数: {args.limit or '不限制'}")
    print("=" * 60)

    print("\n[1/5] 下载 NVD feeds ...")
    cmd_nvd(args)

    print("\n[2/5] 克隆 CVEfixes ...")
    cmd_cvefixes(args)

    print("\n[3/5] 解析 NVD ...")
    cmd_parse_nvd(args)

    print("\n[4/5] 解析 CVEfixes ...")
    cmd_parse_cvefixes(args)

    print("\n[5/5] 合并 NVD + CVEfixes ...")
    cmd_join(args)

    print("\n" + "=" * 60)
    print("[done] 全部数据已就绪")
    print("=" * 60)
    print("下一步：")
    print(f"  python scripts/build_dataset.py \\")
    print(f"      --input  {PROCESSED_DIR / 'joined.jsonl'} \\")
    print(f"      --output {PROCESSED_DIR / 'train.jsonl'}")


# ===== 7. Reasoning 生成 =====
REASONING_PROMPT = """你是一名软件安全分析专家。下面给出一条已知漏洞的信息，请生成两段简短的中文分析，用于训练漏洞分析大模型。

【漏洞描述】
{description}

【相关代码（节选）】
{code}

【修复代码（节选）】
{fix}

【已确认的标签】
- CWE 类型：{cwe}
- 严重程度：{severity}

请输出严格的 JSON（不要加任何 markdown 标记或额外说明文字）：
{{
  "reasoning": "1~2 句话，说明该漏洞被归为此 CWE 类型的根本原因，以及为何对应此严重程度",
  "fix_summary": "1~2 句话，基于修复代码概括修复思路（不重复原始代码原文）"
}}"""


def build_reasoning_messages(record: dict) -> list[dict]:
    """为单条记录构造 LLM 提示消息。"""
    user = REASONING_PROMPT.format(
        description=(record.get("description", "") or "(无)").strip(),
        code=(record.get("code", "") or "(无)").strip()[:2000],
        fix=(record.get("fix", "") or "(无)").strip()[:2000],
        cwe=record.get("cwe", "UNKNOWN"),
        severity=record.get("severity", "UNKNOWN"),
    )
    return [
        {"role": "system", "content": "你是软件安全分析专家，只输出严格 JSON。"},
        {"role": "user", "content": user},
    ]


def call_reasoning_api(
    client: OpenAI,
    model: str,
    record: dict,
    max_retries: int,
) -> tuple[str, str, int, int]:
    """调用 LLM 生成 (reasoning, fix_summary)，返回 token 计数。"""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=build_reasoning_messages(record),
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "{}").strip()
            data = json.loads(content)
            in_tok = getattr(resp.usage, "prompt_tokens", 0) or 0
            out_tok = getattr(resp.usage, "completion_tokens", 0) or 0
            return (
                data.get("reasoning", "").strip(),
                data.get("fix_summary", "").strip(),
                int(in_tok),
                int(out_tok),
            )
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed after {max_retries} retries: {last_err}")


def cmd_reasoning(args) -> None:
    """用 LLM 批量为 joined.jsonl 生成 reasoning + fix_summary。"""
    src = PROCESSED_DIR / args.input
    dst = PROCESSED_DIR / args.output

    if not src.exists():
        print(f"  [err]  {src} 不存在，请先跑 join", file=sys.stderr)
        return

    api_key = args.api_key or os.getenv("VULLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  [err]  需要 --api-key 或环境变量 VULLM_API_KEY / OPENAI_API_KEY",
              file=sys.stderr)
        return

    base_url = args.base_url or os.getenv("VULLM_BASE_URL", "https://api.openai.com/v1")
    client = OpenAI(base_url=base_url, api_key=api_key)

    # 断点续传
    done_ids: set[str] = set()
    if dst.exists() and args.resume:
        with open(dst, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("id"):
                        done_ids.add(rec["id"])
                except json.JSONDecodeError:
                    continue
        if done_ids:
            print(f"  [resume] {dst.name} 已完成 {len(done_ids)} 条，将跳过")

    # 加载源
    records: list[dict] = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id") and rec["id"] not in done_ids:
                records.append(rec)

    if args.limit:
        records = records[:args.limit]

    if not records:
        print("  [done] 没有需要处理的记录")
        return

    print(f"  [start] model={args.model}  base_url={base_url}")
    print(f"  [start] 待处理 {len(records)} 条  并发 {args.concurrency}  输出 {dst.name}")

    out_lock = threading.Lock()
    counters = {"ok": 0, "fail": 0, "in_tok": 0, "out_tok": 0}

    def process_one(rec: dict) -> None:
        try:
            r, fs, in_tok, out_tok = call_reasoning_api(
                client, args.model, rec, args.max_retries
            )
            rec["reasoning"] = r
            rec["fix_summary"] = fs
            with out_lock:
                counters["ok"] += 1
                counters["in_tok"] += in_tok
                counters["out_tok"] += out_tok
        except Exception as e:
            rec["reasoning"] = ""
            rec["fix_summary"] = ""
            with out_lock:
                counters["fail"] += 1
        finally:
            with out_lock:
                with open(dst, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(process_one, rec) for rec in records]
        for i, _ in enumerate(as_completed(futures), 1):
            if i % 10 == 0 or i == len(futures):
                print(f"  [{i}/{len(futures)}] ok={counters['ok']} fail={counters['fail']} "
                      f"in_tok={counters['in_tok']} out_tok={counters['out_tok']}")

    # 成本估算（按 gpt-4o-mini 公开价，1.5/0.6 USD / 1M tokens）
    in_cost = counters["in_tok"] / 1_000_000 * 0.15
    out_cost = counters["out_tok"] / 1_000_000 * 0.60
    print("\n  [done] ok={ok}  fail={fail}".format(**counters))
    print(f"  [cost] (按 gpt-4o-mini 价) in=${in_cost:.4f}  out=${out_cost:.4f}  "
          f"total=${in_cost + out_cost:.4f}")
    print(f"  [output] {dst}")
    print("  [next]  python scripts/build_dataset.py \\")
    print(f"             --input  {dst} \\")
    print(f"             --output {PROCESSED_DIR / 'train.jsonl'}")


# ===== main =====
def main() -> None:
    parser = argparse.ArgumentParser(
        description="手动抓取漏洞数据集 (NVD + CVEfixes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    common_years = argparse.ArgumentParser(add_help=False)
    common_years.add_argument(
        "--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025],
        help="要抓取的 NVD 年份",
    )

    p_nvd = sub.add_parser("nvd", parents=[common_years], help="下载 NVD feeds")
    p_nvd.set_defaults(func=cmd_nvd)

    p_cf = sub.add_parser("cvefixes", help="浅克隆 CVEfixes 仓库")
    p_cf.set_defaults(func=cmd_cvefixes)

    p_pn = sub.add_parser("parse-nvd", parents=[common_years], help="解析 NVD → JSONL")
    p_pn.set_defaults(func=cmd_parse_nvd)

    p_pf = sub.add_parser("parse-cvefixes", help="解析 CVEfixes → JSONL")
    p_pf.set_defaults(func=cmd_parse_cvefixes)

    p_j = sub.add_parser("join", help="合并 NVD + CVEfixes")
    p_j.add_argument("--limit", type=int, default=0, help="最多写入多少条 (0=全部)")
    p_j.set_defaults(func=cmd_join)

    p_r = sub.add_parser("reasoning", help="用 LLM 生成 reasoning + fix_summary")
    p_r.add_argument("--input", default="joined.jsonl",
                     help="输入 JSONL 名称（位于 data/processed/ 下）")
    p_r.add_argument("--output", default="joined_with_reasoning.jsonl",
                     help="输出 JSONL 名称（位于 data/processed/ 下）")
    p_r.add_argument("--api-key", default="",
                     help="API key（也可用环境变量 VULLM_API_KEY / OPENAI_API_KEY）")
    p_r.add_argument("--base-url", default="",
                     help="OpenAI 兼容端点（默认 https://api.openai.com/v1）")
    p_r.add_argument("--model", default="gpt-4o-mini",
                     help="模型名（也支持 deepseek-chat / qwen-plus 等）")
    p_r.add_argument("--concurrency", type=int, default=8, help="并发请求数")
    p_r.add_argument("--max-retries", type=int, default=3, help="单条失败重试次数")
    p_r.add_argument("--limit", type=int, default=0, help="最多处理多少条 (0=全部)")
    p_r.add_argument("--resume", action="store_true", help="断点续传，跳过已完成的记录")
    p_r.set_defaults(func=cmd_reasoning)

    p_all = sub.add_parser("all", parents=[common_years], help="一键跑完全流程")
    p_all.add_argument("--limit", type=int, default=0, help="最多写入多少条 (0=全部)")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
