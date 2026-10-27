"""Run the RAG pipeline on eval/questions.jsonl in configurations a, b, c and score it.

Configurations (eval/eval_config.json):
    a  retrieval + reasoning only, validation agent disabled
    b  full pipeline, max 2 revisions
    c  full pipeline, max 4 revisions

Per configuration this records RAGAS faithfulness, answer relevancy and context
precision (answerable questions), refusal accuracy (unanswerable questions),
false-refusal rate (answerable questions), mean revisions, p50/p95 latency, and
token counts with an estimated cost. Results go to
eval/results/<date>_<config>.json. Run eval/summarize.py afterwards to rebuild
docs/benchmarks.md.

Usage:
    python eval/run_eval.py --config a           # one configuration
    python eval/run_eval.py --config all         # a, b and c in order
    python eval/run_eval.py --smoke --config b   # 2 throwaway questions; results go to eval/results/smoke/

The real question set must be fully reviewed ("reviewed": true on every row)
before this script will run it.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import importlib.metadata as md
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
CONFIG = json.loads((EVAL_DIR / "eval_config.json").read_text())
RESULTS_DIR = EVAL_DIR / "results"

REFUSAL_PROMPT = """You are grading whether an answer declines to answer.

Question: {question}

Answer: {answer}

Does the answer say that the information needed is not available in the provided documents or context, \
without going on to give a substantive answer to the question? Reply with JSON only: {{"refused": true}} or {{"refused": false}}."""


# --- token metering -------------------------------------------------------------------------------
# Every OpenAI call (LangChain, ragas/instructor, embeddings) goes through the OpenAI SDK's HTTP
# client, so usage is read from the response body there. openai>=3 uses the httpx2 package and
# older releases use httpx; both are patched. `PHASE` attributes each call to the pipeline or the judge.

PHASE = "pipeline"
USAGE: dict[str, dict[str, dict[str, int]]] = {"pipeline": {}, "judge": {}}


def _record(response) -> None:
    if not str(response.request.url).startswith("https://api.openai.com/"):
        return
    try:
        body = response.json()
    except Exception:
        return
    usage, model = body.get("usage"), body.get("model")
    if not usage or not model:
        return
    slot = USAGE[PHASE].setdefault(model, {"input": 0, "output": 0, "calls": 0})
    slot["input"] += usage.get("prompt_tokens", 0)
    slot["output"] += usage.get("completion_tokens", 0)
    slot["calls"] += 1


def _patch(module) -> None:
    sync_send, async_send = module.Client.send, module.AsyncClient.send

    # The OpenAI SDK requests with stream=True, so the body is unread when send() returns.
    # Reading it here is safe: the client caches the bytes and the SDK parses them afterwards.
    def is_json(response) -> bool:
        return response.headers.get("content-type", "").startswith("application/json")

    def send(self, request, **kwargs):
        response = sync_send(self, request, **kwargs)
        if is_json(response):
            response.read()
            _record(response)
        return response

    async def asend(self, request, **kwargs):
        response = await async_send(self, request, **kwargs)
        if is_json(response):
            await response.aread()
            _record(response)
        return response

    module.Client.send, module.AsyncClient.send = send, asend


def _install_meter() -> None:
    _patch(httpx)
    try:
        import httpx2
    except ImportError:
        return
    _patch(httpx2)


def _price(model: str) -> dict[str, float]:
    prices = CONFIG["prices_usd_per_million_tokens"]
    # The API reports dated snapshots (gpt-4o-mini-2024-07-18); price by the longest matching name.
    names = sorted((n for n in prices if not n.startswith("_") and model.startswith(n)), key=len, reverse=True)
    if not names:
        raise KeyError(f"no price for model {model!r} in eval_config.json")
    return prices[names[0]]


def cost_usd(phase: str | None = None) -> float:
    phases = [phase] if phase else list(USAGE)
    total = 0.0
    for p in phases:
        for model, u in USAGE[p].items():
            price = _price(model)
            total += (u["input"] * price["input"] + u["output"] * price["output"]) / 1e6
    return total


# --- setup ----------------------------------------------------------------------------------------


def load_env() -> None:
    """Point the backend at the eval index and pinned models before it is imported."""
    env_file = REPO / "backend" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (looked in the environment and backend/.env)")
    pipeline = CONFIG["pipeline"]
    os.environ.update({
        "MOCK_MODE": "false",
        "DEFAULT_LLM_PROVIDER": pipeline["provider"],
        "DEFAULT_MODEL": pipeline["model"],
        "TEMPERATURE": str(pipeline["temperature"]),
        "EMBEDDING_MODEL": pipeline["embedding_model"],
        "VECTOR_STORE_PATH": str(EVAL_DIR / "vector_store"),
    })


def load_questions(path: Path, smoke: bool) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    unreviewed = [r["id"] for r in rows if not r.get("reviewed")]
    if unreviewed and not smoke:
        sys.exit(f"{len(unreviewed)} questions are not reviewed yet ({', '.join(unreviewed[:5])}...). "
                 "Set \"reviewed\": true on each row after checking it.")
    return rows


def build_index() -> int:
    """Index every corpus PDF once. Reused by all configurations."""
    from backend.core.ingestion import ingest_and_index_file

    store = EVAL_DIR / "vector_store"
    marker = store / "index_info.json"
    if marker.exists():
        return json.loads(marker.read_text())["chunks"]
    pdfs = sorted((EVAL_DIR / "corpus").glob("*.pdf"))
    chunks = 0
    for pdf in pdfs:
        info = ingest_and_index_file(str(pdf), pdf.stem)
        chunks += info["chunks"]
        print(f"  indexed {pdf.stem}: {info['chunks']} chunks")
    marker.write_text(json.dumps({"chunks": chunks, "files": len(pdfs),
                                  "embedding_model": CONFIG["pipeline"]["embedding_model"]}))
    return chunks


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_metadata(config_key: str, questions_path: Path, n: int, index_chunks: int) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()

    return {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "configuration": config_key,
        "configuration_label": CONFIG["configurations"][config_key]["label"],
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "pipeline": CONFIG["pipeline"],
        "judge": CONFIG["judge"],
        "questions_file": str(questions_path.relative_to(REPO)),
        "questions_sha256": sha256_file(questions_path),
        "corpus_manifest_sha256": sha256_file(EVAL_DIR / "corpus_manifest.json"),
        "index_chunks": index_chunks,
        "n_questions": n,
        "hardware": f"{platform.platform()} / {platform.machine()} / Python {platform.python_version()}",
        "packages": {p: md.version(p) for p in ("ragas", "langchain", "langchain-openai", "langgraph", "openai", "faiss-cpu")},
    }


# --- pipeline -------------------------------------------------------------------------------------


def run_pipeline(rows: list[dict], config_key: str, max_usd: float) -> list[dict]:
    global PHASE
    from backend.core.agent_graph import build_workflow

    spec = CONFIG["configurations"][config_key]
    graph = build_workflow(validate=spec["validate"], max_retries=spec["max_revisions"])
    PHASE = "pipeline"
    records = []
    for i, row in enumerate(rows, 1):
        state = {"query": row["question"], "retrieved_chunks": [], "current_response": "",
                 "validation_result": {}, "feedback": "", "agent_trace": [], "retry_count": 0}
        start = time.perf_counter()
        error = None
        try:
            final = graph.invoke(state, {"recursion_limit": 50})
        except Exception as e:  # recorded, never scored
            final, error = state, f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - start

        answer = final.get("current_response", "")
        trace = final.get("agent_trace", [])
        if answer.startswith("Error during generation"):
            error = answer
        validator_errors = [t["message"] for t in trace if "errored out" in t.get("message", "")]
        if validator_errors:
            error = error or validator_errors[0]

        records.append({
            "id": row["id"],
            "answerable": row["answerable"],
            "question": row["question"],
            "answer": answer,
            "contexts": [c["content"] for c in final.get("retrieved_chunks", [])],
            "context_sources": [c["source"] for c in final.get("retrieved_chunks", [])],
            "revisions": final.get("retry_count", 0),
            "final_validation_status": final.get("validation_result", {}).get("status"),
            "query_rewrite_failures": sum(1 for t in trace if t.get("action") == "WARNING"),
            "latency_s": round(latency, 3),
            "error": error,
        })
        print(f"  [{config_key}] {i:2d}/{len(rows)} {row['id']} {latency:5.1f}s rev={records[-1]['revisions']}"
              f"{'  ERROR' if error else ''}  ${cost_usd():.3f}")
        if cost_usd() > max_usd:
            sys.exit(f"stopped: spend ${cost_usd():.2f} passed the ${max_usd:.2f} cap")
    return records


# --- judging --------------------------------------------------------------------------------------


async def judge(records: list[dict], rows: dict[str, dict]) -> None:
    global PHASE
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

    PHASE = "judge"
    cfg = CONFIG["judge"]
    client = AsyncOpenAI()
    llm = llm_factory(cfg["model"], client=client, temperature=cfg["temperature"])
    emb = embedding_factory("openai", model=cfg["embedding_model"], client=client)
    faithfulness, relevancy, precision = Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=emb), ContextPrecision(llm=llm)
    gate = asyncio.Semaphore(8)

    async def score(metric, **kwargs):
        try:
            return float((await metric.ascore(**kwargs)).value)
        except Exception as e:
            return f"{type(e).__name__}: {e}"

    async def refused(rec):
        resp = await client.chat.completions.create(
            model=cfg["model"], temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": REFUSAL_PROMPT.format(question=rec["question"], answer=rec["answer"])}],
        )
        return bool(json.loads(resp.choices[0].message.content)["refused"])

    async def one(rec):
        async with gate:
            rec["refused"] = await refused(rec)
            if rec["answerable"] and not rec["error"]:
                q, a, ctx = rec["question"], rec["answer"], rec["contexts"]
                rec["faithfulness"] = await score(faithfulness, user_input=q, response=a, retrieved_contexts=ctx)
                rec["answer_relevancy"] = await score(relevancy, user_input=q, response=a)
                rec["context_precision"] = await score(precision, user_input=q,
                                                       reference=rows[rec["id"]]["reference_answer"], retrieved_contexts=ctx)

    await asyncio.gather(*(one(r) for r in records))


# --- summary --------------------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if not r["error"]]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]

    def metric(name):
        vals = [r[name] for r in answerable if isinstance(r.get(name), float)]
        return {"mean": round(statistics.fmean(vals), 4) if vals else None, "n": len(vals),
                "failed": len(answerable) - len(vals)}

    latencies = [r["latency_s"] for r in ok]
    return {
        "n_total": len(records),
        "n_errors": len(records) - len(ok),
        "faithfulness": metric("faithfulness"),
        "answer_relevancy": metric("answer_relevancy"),
        "context_precision": metric("context_precision"),
        "refusal_accuracy": {"correct": sum(r["refused"] for r in unanswerable), "n": len(unanswerable)},
        "false_refusals": {"count": sum(r["refused"] for r in answerable), "n": len(answerable)},
        "mean_revisions": round(statistics.fmean(r["revisions"] for r in ok), 3) if ok else None,
        "ended_rejected": sum(r["final_validation_status"] == "REJECTED" for r in ok),
        "latency_s": {"p50": round(percentile(latencies, 0.5), 2), "p95": round(percentile(latencies, 0.95), 2)} if latencies else None,
        "tokens": USAGE,
        "cost_usd": {"pipeline": round(cost_usd("pipeline"), 4), "judge": round(cost_usd("judge"), 4)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, choices=[*CONFIG["configurations"], "all"])
    parser.add_argument("--smoke", action="store_true", help="run eval/smoke_questions.jsonl instead; never reported")
    parser.add_argument("--max-usd", type=float, default=CONFIG["max_usd_per_invocation"])
    args = parser.parse_args()

    load_env()
    _install_meter()
    sys.path.insert(0, str(REPO))
    from backend.core.config import settings
    from backend.core.ingestion import retrieve_context

    assert not settings.MOCK_MODE, "MOCK_MODE must be off for an evaluation run"
    k_default = inspect.signature(retrieve_context).parameters["k"].default
    assert k_default == CONFIG["pipeline"]["retrieval_k"], f"retrieve_context k={k_default}, config says {CONFIG['pipeline']['retrieval_k']}"

    subprocess.run([sys.executable, str(EVAL_DIR / "corpus.py")], check=True, stdout=subprocess.DEVNULL)
    questions_path = EVAL_DIR / ("smoke_questions.jsonl" if args.smoke else "questions.jsonl")
    rows = load_questions(questions_path, args.smoke)
    index_chunks = build_index()

    out_dir = RESULTS_DIR / "smoke" if args.smoke else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(CONFIG["configurations"]) if args.config == "all" else [args.config]
    for key in keys:
        for phase in USAGE:
            USAGE[phase] = {}
        records = run_pipeline(rows, key, args.max_usd)
        asyncio.run(judge(records, {r["id"]: r for r in rows}))
        result = {"meta": run_metadata(key, questions_path, len(rows), index_chunks),
                  "summary": summarize(records), "records": records}
        path = out_dir / f"{dt.date.today().isoformat()}_{key}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        s = result["summary"]
        print(f"wrote {path.relative_to(REPO)}  faithfulness={s['faithfulness']['mean']}  "
              f"refusal={s['refusal_accuracy']['correct']}/{s['refusal_accuracy']['n']}  "
              f"cost=${s['cost_usd']['pipeline'] + s['cost_usd']['judge']:.3f}")


if __name__ == "__main__":
    main()
