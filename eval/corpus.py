"""Download the evaluation corpus and check it against eval/corpus_manifest.json.

The corpus is 20 public arXiv papers on retrieval-augmented generation and LLM
agents. PDFs are written to eval/corpus/ (gitignored); only the manifest is
committed. The manifest pins each paper to a versioned arXiv URL and records
the SHA-256 of the bytes, so every evaluation run can name the exact corpus.

Usage:
    python eval/corpus.py            # download from the manifest and verify hashes
    python eval/corpus.py --pin      # resolve latest arXiv versions and rewrite the manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
MANIFEST = EVAL_DIR / "corpus_manifest.json"

ARXIV_IDS = [
    "2005.11401",  # Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    "2004.04906",  # Karpukhin et al., Dense Passage Retrieval
    "2002.08909",  # Guu et al., REALM
    "2007.01282",  # Izacard & Grave, Fusion-in-Decoder
    "2112.04426",  # Borgeaud et al., RETRO
    "2212.10496",  # Gao et al., HyDE
    "2305.06983",  # Jiang et al., FLARE (Active RAG)
    "2310.11511",  # Asai et al., Self-RAG
    "2401.15884",  # Yan et al., Corrective RAG
    "2403.14403",  # Jeong et al., Adaptive-RAG
    "2307.03172",  # Liu et al., Lost in the Middle
    "2309.15217",  # Es et al., RAGAS
    "2305.14251",  # Min et al., FActScore
    "2312.10997",  # Gao et al., RAG survey
    "2210.03629",  # Yao et al., ReAct
    "2303.11366",  # Shinn et al., Reflexion
    "2303.17651",  # Madaan et al., Self-Refine
    "2302.04761",  # Schick et al., Toolformer
    "2212.14024",  # Khattab et al., Demonstrate-Search-Predict
    "2310.06117",  # Zheng et al., Step-Back Prompting
]

USER_AGENT = "Agentic-Research-System eval corpus fetcher"
ATOM = "{http://www.w3.org/2005/Atom}"


def _ssl_context() -> ssl.SSLContext:
    # python.org macOS builds ship without a CA bundle; certifi supplies one.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp:
        return resp.read()


def pin_versions() -> list[dict]:
    """Ask the arXiv API for each paper's current version and title."""
    feed = _get("http://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(ARXIV_IDS))
    entries = {}
    for entry in ET.fromstring(feed).iter(f"{ATOM}entry"):
        versioned = entry.find(f"{ATOM}id").text.rsplit("/", 1)[-1]  # e.g. 2005.11401v4
        base = versioned.split("v")[0]
        title = " ".join(entry.find(f"{ATOM}title").text.split())
        entries[base] = {"arxiv_id": versioned, "title": title}
    missing = [i for i in ARXIV_IDS if i not in entries]
    if missing:
        sys.exit(f"arXiv API returned no entry for: {missing}")
    return [
        {**entries[i], "url": f"https://arxiv.org/pdf/{entries[i]['arxiv_id']}", "file": f"{i}.pdf"}
        for i in ARXIV_IDS
    ]


def fetch(papers: list[dict], record_hashes: bool) -> None:
    CORPUS_DIR.mkdir(exist_ok=True)
    for paper in papers:
        path = CORPUS_DIR / paper["file"]
        if not path.exists():
            print(f"  get   {paper['arxiv_id']}  {paper['title'][:60]}")
            path.write_bytes(_get(paper["url"]))
            time.sleep(3)  # arXiv asks automated clients to wait between requests
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if record_hashes:
            paper["sha256"] = digest
            paper["bytes"] = path.stat().st_size
        elif digest != paper["sha256"]:
            sys.exit(f"{path.name}: sha256 {digest} does not match manifest {paper['sha256']}")
        print(f"  ok    {paper['file']}  {digest[:12]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pin", action="store_true", help="resolve versions and rewrite the manifest")
    args = parser.parse_args()

    if args.pin:
        papers = pin_versions()
        fetch(papers, record_hashes=True)
        MANIFEST.write_text(json.dumps(papers, indent=2) + "\n")
        print(f"wrote {MANIFEST.relative_to(EVAL_DIR.parent)} ({len(papers)} papers)")
    else:
        if not MANIFEST.exists():
            sys.exit("no manifest; run with --pin first")
        fetch(json.loads(MANIFEST.read_text()), record_hashes=False)
        print("all hashes match the manifest")


if __name__ == "__main__":
    main()
