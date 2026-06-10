from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from src.vullm.memory.reasoning_memory import ReasoningMemory
from src.vullm.serve.inference import mock_analyze, openai_compatible_analyze


class AnalyzeRequest(BaseModel):
    description: str
    code: str | None = None
    use_vllm: bool = False
    use_memory: bool = True
    top_k: int = 3


app = FastAPI(title="VulLLM API", version="0.1.0")

memory_path = Path("outputs/reasoning_memory.json")
memory = ReasoningMemory.from_json(memory_path) if memory_path.exists() else None


@app.get("/")
def health_check():
    return {"message": "VulLLM API is running"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    memory_cases = []
    if req.use_memory and memory is not None:
        memory_cases = memory.search(f"{req.description}\n{req.code or ''}", top_k=req.top_k)

    if req.use_vllm:
        prediction = openai_compatible_analyze(req.description, req.code)
        return {
            "mode": "vllm",
            "prediction": prediction,
            "memory_cases": memory_cases,
        }

    result = mock_analyze(req.description, req.code, memory_cases=memory_cases)
    return {
        "mode": "mock",
        **result,
    }
