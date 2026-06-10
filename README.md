# VulLLM

> 基于 Qwen / DeepSeek 的大模型漏洞分析与推理系统
> Vulnerability Analysis & Reasoning System powered by Large Language Models

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![Framework](https://img.shields.io/badge/Transformers-4.43%2B-orange)](#)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FQLoRA-yellow)](#)
[![vLLM](https://img.shields.io/badge/vLLM-OpenAI%20Compatible-purple)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
  - [1. 数据集构建](#1-数据集构建)
  - [2. 数据抓取与 LLM 推理增强](#2-数据抓取与-llm-推理增强)
  - [3. LoRA / QLoRA 训练](#3-lora--qlora-训练)
  - [4. 评测](#4-评测)
  - [5. 检索记忆构建](#5-检索记忆构建)
  - [6. FastAPI 推理服务](#6-fastapi-推理服务)
  - [7. vLLM 部署](#7-vllm-部署)
- [核心模块](#核心模块)
- [样例数据](#样例数据)
- [局限性 / 后续工作](#局限性--后续工作)
- [简历表达](#简历表达)
- [许可](#许可)

## 项目概述

VulLLM 是一个面向软件安全领域的 LLM 应用系统，专注于**漏洞类型识别、严重程度评估与推理解释**。系统以 Qwen2.5 为基座模型，结合 NVD、BigVul 等公开漏洞数据集进行 LoRA / QLoRA 微调，并通过历史漏洞案例检索增强推理能力，最终以 vLLM + FastAPI 对外提供服务。

## 核心特性

| 能力 | 说明 |
| --- | --- |
| 数据集构建 | 兼容 NVD / BigVul / 自定义 JSONL，自动归一化为指令微调数据 |
| 数据抓取 | 一键下载 NVD feeds、克隆 CVEfixes 仓库、按 CVE ID 合并为训练样本 |
| LLM 推理增强 | 调用任意 OpenAI 兼容端点（OpenAI / DeepSeek / DashScope / vLLM）为每条数据补全中文 `reasoning` + `fix_summary`，内置并发、重试、续传与成本统计 |
| 参数高效微调 | Transformers + PEFT + TRL，支持 LoRA / QLoRA（4-bit NF4 + double quant） |
| 多任务输出 | 单次推理同时输出 CWE 类型、严重程度、推理过程、修复建议 |
| 自动化评测 | Accuracy / Macro-F1 / per-class report，预留 GPT Judge 接口 |
| 检索增强 | 基于历史漏洞案例库的 Top-K 检索，作为推理上下文 |
| 在线服务 | FastAPI + vLLM OpenAI-Compatible API，支持 Mock / 真实推理切换 |

## 系统架构

```
+-------------------+     +-------------------+     +-----------------------+
| NVD / BigVul /    | --> | 数据归一化         | --> | data/processed/       |
| 自定义 JSONL      |     | 指令数据构建       |     |  {train,eval}.jsonl   |
+-------------------+     +-------------------+     +-----------+-----------+
                                                                |
                                                                v
                                          +---------------------------------+
                                          | QLoRA 微调 (Qwen2.5)            |
                                          | Transformers + PEFT + TRL       |
                                          +-----------------+---------------+
                                                            |
                                                            v
                                          +---------------------------------+
                                          | outputs/qwen_vullm_lora/        |
                                          +-----------------+---------------+
                                                            |
+--------------------+     +---------------------------+  |
| 漏洞描述 / 代码    | --> |  FastAPI /analyze         | <+
+--------------------+     |  - Reasoning Memory (RAG) |
                           |  - mock_analyze           |     +--------------------+
                           |  - vllm_analyze           | --> | reasoning_memory   |
                           +-------------+-------------+     | .json              |
                                         |                   +--------------------+
                                         v
                           +---------------------------+
                           | 评测 / 输出                |
                           | Accuracy / Macro-F1       |
                           +---------------------------+
```

## 目录结构

```
VulLLM/
├── configs/
│   └── train_qwen_lora.yaml        # QLoRA 训练超参
├── data/
│   ├── sample/                      # 示例数据（4 条覆盖 XSS / SQLi / BufferOverflow / WeakHash）
│   ├── raw/                         # 原始 NVD / BigVul 占位
│   └── processed/                   # 处理后指令数据
├── outputs/                         # 训练产物 / 记忆库 / 评测结果
├── scripts/                         # 命令行入口
│   ├── build_dataset.py
│   ├── build_memory.py
│   ├── train_lora.py
│   ├── evaluate.py
│   ├── predict_mock.py
│   ├── collect_data.py               # NVD / CVEfixes 抓取 + LLM 推理补全
│   └── vllm_client.py
├── src/vullm/
│   ├── prompts.py                                  # 中文 ChatML prompt 模板
│   ├── data/build_instruction_dataset.py
│   ├── eval/metrics.py
│   ├── memory/reasoning_memory.py
│   ├── serve/{fastapi_app.py, inference.py}
│   └── train/train_lora.py
└── requirements.txt
```

## 环境要求

| 用途 | 推荐配置 |
| --- | --- |
| Python | 3.10+ |
| 训练（QLoRA 4-bit） | Linux + NVIDIA GPU（≥ 16 GB 显存可微调 7B），需安装对应版本 CUDA / PyTorch / bitsandbytes |
| 推理 | CPU 可跑 1.5B 量化版（GGUF / llama.cpp）；GPU 推荐 vLLM |
| 评测 / FastAPI / 检索 | 仅需 scikit-learn + fastapi + openai SDK，无 GPU 依赖 |

> 当前仓库默认在 `configs/train_qwen_lora.yaml` 中使用 `Qwen2.5-1.5B-Instruct` + 4-bit QLoRA，目的是在 16 GB 显卡上可直接复现。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建指令数据集
python scripts/build_dataset.py \
  --input  data/sample/vul_sample.jsonl \
  --output data/processed/train.jsonl

# 3. 构建检索记忆
python scripts/build_memory.py \
  --input  data/sample/vul_sample.jsonl \
  --output outputs/reasoning_memory.json

# 4. 用 Mock 推理批量预测
python scripts/predict_mock.py \
  --input  data/sample/eval_sample.jsonl \
  --output outputs/predictions.jsonl

# 5. 评测
python scripts/evaluate.py \
  --pred outputs/predictions.jsonl \
  --gold data/sample/eval_sample.jsonl
```

执行完成后可在 `outputs/eval_result.json` 查看 CWE / 严重程度的 Accuracy 与 Macro-F1。

## 使用指南

### 1. 数据集构建

输入 JSONL 每行一条漏洞记录，字段命名兼容多种来源：

| 归一化字段 | 兼容输入字段 |
| --- | --- |
| `description` | `description` / `summary` / `cve_description` |
| `code` | `code` / `func_before` / `vulnerable_code` |
| `cwe` | `cwe` / `cwe_id` / `cwe_id_label` |
| `severity` | `severity` / `cvss_severity` / `label_severity` |
| `reasoning` / `fix` | `reasoning` / `fix` / `patch` / `recommendation` |

输出 schema：

```json
{
  "id": "CVE-XXXX-XXXX",
  "instruction": "请分析以下软件漏洞，输出漏洞类型、严重程度、推理过程和修复建议。",
  "input": "漏洞描述：...\n相关代码：...",
  "output": "漏洞类型：CWE-XX\n严重程度：...\n推理过程：...\n修复建议：...",
  "cwe": "CWE-XX",
  "severity": "HIGH"
}
```

### 2. 数据抓取与 LLM 推理增强

`scripts/collect_data.py` 提供一站式数据获取 + 推理补全，子命令一览：

| 子命令 | 用途 |
| --- | --- |
| `nvd` | 下载 NVD 年度 CVE feeds（美国政府公开数据） |
| `cvefixes` | 浅克隆 CVEfixes 仓库（含 vulnerable code + fix） |
| `parse-nvd` / `parse-cvefixes` | 解析为统一 JSONL |
| `join` | 按 CVE ID 合并 NVD + CVEfixes |
| `reasoning` | **用 LLM 为每条数据生成中文 `reasoning` + `fix_summary`** |
| `all` | 一键跑完前 5 步 |

一键流程：

```bash
# 1. 抓数据 + 合并（输出 data/processed/joined.jsonl）
python scripts/collect_data.py all --years 2022 2023 2024 2025 --limit 2000

# 2. 配置 API key（任选一种接入方式）
export VULLM_API_KEY=sk-...                                          # OpenAI
# export VULLM_BASE_URL=https://api.deepseek.com                     # 切到 DeepSeek
# export VULLM_MODEL=deepseek-chat
# export VULLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1   # 切到 DashScope Qwen
# export VULLM_MODEL=qwen-plus

# 3. 批量生成中文推理 + 修复总结（断点续传、并发 16）
python scripts/collect_data.py reasoning --resume --concurrency 16

# 4. 归一化为指令数据
python scripts/build_dataset.py \
  --input  data/processed/joined_with_reasoning.jsonl \
  --output data/processed/train.jsonl
```

`reasoning` 子命令支持任意 OpenAI 兼容端点（OpenAI / DeepSeek / DashScope / 本地 vLLM / Ollama），内置并发、重试、断点续传与 token 成本统计。`joined_with_reasoning.jsonl` 会显式带 `reasoning` 字段，下游 `build_dataset.py` 优先使用真实推理而非模板兜底。

> 不需要联网抓数据时，直接使用样例数据 + `build_dataset.py` 即可跑通整条流程。

参考成本（gpt-4o-mini 公开价）：

| 数据量 | 估算费用 |
| --- | --- |
| 1,000 条 | ~$0.17 |
| 5,000 条 | ~$0.83 |
| 20,000 条 | ~$3.30 |

### 3. LoRA / QLoRA 训练

```bash
python scripts/train_lora.py --config configs/train_qwen_lora.yaml
```

配置要点（详见 `configs/train_qwen_lora.yaml`）：

- 基座模型：`Qwen/Qwen2.5-1.5B-Instruct`（可换 3B / 7B）
- 量化：4-bit NF4 + double quant + bfloat16
- LoRA：`r=16, alpha=32, dropout=0.05`，target 为 Q / K / V / O / gate / up / down 投影
- 训练：1 epoch，batch size 1 × 8 梯度累积，lr 2e-4，warmup 3%

> 若机器无 NVIDIA GPU，可跳过训练，直接阅读代码与运行数据构建 / 评测 / FastAPI 部分。

### 4. 评测

```bash
# 真实预测 vs 真实标签
python scripts/evaluate.py \
  --pred outputs/predictions.jsonl \
  --gold data/processed/eval.jsonl

# 项目自带的 sample 评测
python scripts/evaluate.py \
  --pred data/sample/predictions_sample.jsonl \
  --gold data/sample/eval_sample.jsonl
```

评测脚本会从 `prediction` 文本中用正则抽取出 `漏洞类型` / `严重程度`，计算 Accuracy 与 Macro-F1，并输出 per-class report。预留位置可扩展为 GPT Judge 对推理过程打分。

### 5. 检索记忆构建

```bash
python scripts/build_memory.py \
  --input  data/sample/vul_sample.jsonl \
  --output outputs/reasoning_memory.json
```

生成的 `reasoning_memory.json` 会在 FastAPI 启动时被加载（`outputs/reasoning_memory.json` 存在则启用），通过词袋 + 余弦相似度做 Top-K 案例检索。

### 6. FastAPI 推理服务

```bash
uvicorn src.vullm.serve.fastapi_app:app --reload
```

请求示例：

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "User input is rendered into HTML without sanitization.",
    "code": "document.write(location.hash)",
    "use_memory": true,
    "use_vllm": false,
    "top_k": 3
  }'
```

返回示例：

```json
{
  "mode": "mock",
  "cwe": "CWE-79",
  "severity": "HIGH",
  "reasoning": "用户可控输入进入HTML上下文，可能导致跨站脚本攻击。",
  "fix": "进行输出编码、输入校验并配置CSP。",
  "memory_cases": [
    {"score": 0.83, "id": "...", "description": "...", ...}
  ]
}
```

### 7. vLLM 部署

启动 vLLM 服务（LoRA 权重请按 vLLM 当前版本支持方式加载 adapter）：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001
```

通过 OpenAI-Compatible 接口调用：

```bash
python scripts/vllm_client.py \
  --base-url http://127.0.0.1:8001/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --description "应用未对用户输入做校验就拼入 SQL 查询"
```

或在 FastAPI 请求中设置 `"use_vllm": true`，自动走 `openai_compatible_analyze`。

## 核心模块

| 模块 | 路径 | 职责 |
| --- | --- | --- |
| Prompt 模板 | `src/vullm/prompts.py` | 中文 system prompt、ChatML 拼装、输入输出格式化 |
| 数据归一化 | `src/vullm/data/build_instruction_dataset.py` | 多源字段名兼容、train/eval 切分、JSONL 读写 |
| 训练 | `src/vullm/train/train_lora.py` | QLoRA + PEFT + TRL SFTTrainer，Qwen ChatML 模板 |
| 评测 | `src/vullm/eval/metrics.py` | 正则抽取 + sklearn 指标 + per-class report |
| 检索记忆 | `src/vullm/memory/reasoning_memory.py` | 词袋 + 余弦相似度，零外部向量库依赖 |
| 数据抓取 / LLM 增强 | `scripts/collect_data.py` | NVD + CVEfixes 抓取、按 CVE ID 合并、调用 LLM 批量生成中文 `reasoning` / `fix_summary`，内置并发、重试、续传 |
| 推理 | `src/vullm/serve/inference.py` | `mock_analyze` 关键词兜底 + `openai_compatible_analyze` 真实推理 |
| 服务 | `src/vullm/serve/fastapi_app.py` | `/analyze` 端点：use_memory / use_vllm 开关 |

## 样例数据

`data/sample/vul_sample.jsonl` 内置 4 条典型漏洞覆盖：

| ID | 漏洞类型 | 严重程度 |
| --- | --- | --- |
| CVE-SAMPLE-0001 | CWE-79 (XSS) | HIGH |
| CVE-SAMPLE-0002 | CWE-89 (SQL Injection) | CRITICAL |
| CVE-SAMPLE-0003 | CWE-120 (Buffer Overflow) | HIGH |
| CVE-SAMPLE-0004 | CWE-327 (Weak Hash) | MEDIUM |

可直接用于构建数据集、检索记忆与端到端流程验证。

## 局限性 / 后续工作

- 检索记忆当前为词袋 + 余弦，召回质量有限，可替换为 sentence-transformers / BGE 向量检索；
- 推理过程质量受 `reasoning` 子命令所选 LLM 影响，可结合 GPT Judge 对生成结果做二次筛选；
- 训练侧未配置 packing / flash-attn，长序列可能 OOM；
- 缺少单元测试与 CI；
- 入口脚本中 `from src.xxx` 隐含 `PYTHONPATH=.`，发布为包时建议加 `pyproject.toml`。

## 简历表达

VulLLM——基于Qwen的大模型漏洞分析与推理系统

- 面向软件漏洞类型识别与严重程度评估任务，基于Qwen2.5构建漏洞分析大模型，实现漏洞分类、风险评估及推理解释能力；
- 基于NVD、BigVul等公开漏洞数据集构建指令微调数据集，设计"漏洞描述→漏洞类型→严重程度→推理过程"的多任务训练样本；
- 基于LoRA/QLoRA实现参数高效微调，采用Accuracy、Macro-F1等指标评估模型性能，并与零样本大模型进行对比；
- 构建漏洞推理知识库，通过历史漏洞案例检索增强模型推理能力；
- 基于vLLM与FastAPI实现在线推理服务化部署；
- 设计 LLM 辅助的推理数据增强流程，弥补 NVD 等公开数据缺少推理过程的短板，生成高质量「漏洞描述 + 代码 + 推理 + 修复」四元组训练样本。

## 许可

MIT License
