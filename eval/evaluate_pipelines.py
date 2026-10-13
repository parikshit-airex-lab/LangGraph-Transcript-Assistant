"""
End-to-end retrieval comparison: LangChain/Chroma pipeline vs the from-scratch
NumPy pipeline (rag_annotated.py).

Both pipelines receive the SAME raw document and are allowed to chunk it their
own way, because chunking is part of what is being compared. Gold labels are
therefore computed per pipeline: a chunk is gold if it contains the question's
answer span. This is the only labelling scheme that stays fair when the two
systems produce different chunk boundaries.

  langchain : RecursiveCharacterTextSplitter(1000, 200) -> HuggingFaceEmbeddings
              -> Chroma (HNSW), k=4. Exactly the app's path.
  numpy     : rag_annotated.split(800, 150) -> rag_annotated.embed
              -> matrix @ q -> argsort. Exactly the from-scratch path; the
              functions are imported, not reimplemented.

Usage
  python evaluate_pipelines.py --backend numpy
  python evaluate_pipelines.py --backend langchain --backend numpy --out pipelines.json
"""

import argparse
import json
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
K_VALUES = (1, 3, 4, 5, 10)
DEFAULT_K = 4


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_inputs(questions_path: str, raw_path: str):
    with open(questions_path, encoding="utf-8") as f:
        spec = json.load(f)
    with open(raw_path, encoding="utf-8") as f:
        raw = f.read()
    return raw, spec["questions"]


def assign_gold(chunk_texts, questions):
    """A chunk is gold for a question if it contains that question's answer span."""
    normalised = [normalise(t) for t in chunk_texts]
    resolved = []
    for q in questions:
        span = normalise(q["answer_span"])
        gold = [i for i, t in enumerate(normalised) if span in t]
        if not gold:
            raise SystemExit(
                f"Question {q['id']} lost its gold chunk under this chunking: "
                f"{q['answer_span']!r}. The span was split across a boundary."
            )
        resolved.append({"id": q["id"], "question": q["question"], "gold": gold})
    return resolved


# ------------------------------------------------------------------ pipelines

class LangChainPipeline:
    name = "LangChain + Chroma (1000/200, HNSW)"
    chunk_config = "RecursiveCharacterTextSplitter(1000, 200)"

    def __init__(self, raw: str, size: int = 1000, overlap: int = 200):
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        self.chunk_texts = splitter.split_text(raw)
        self.chunk_config = f"RecursiveCharacterTextSplitter({size}, {overlap})"
        self.name = f"LangChain + Chroma ({size}/{overlap}, HNSW)"

        # Constructed BEFORE the timer: the NumPy pipeline loads its encoder at
        # import time, so timing the model load here would compare cold start
        # against warm and overstate the framework's indexing cost.
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        t0 = time.perf_counter()
        docs = [
            Document(page_content=t, metadata={"chunk_id": i})
            for i, t in enumerate(self.chunk_texts)
        ]
        self.store = Chroma.from_documents(documents=docs, embedding=embeddings)
        self.index_seconds = time.perf_counter() - t0

    def search(self, query: str, k: int):
        hits = self.store.similarity_search(query, k=k)
        return [h.metadata["chunk_id"] for h in hits]


class NumpyPipeline:
    """Drives rag_annotated.py's own split/embed/retrieve, not a reimplementation."""

    name = "NumPy from-scratch (800/150, brute force)"
    chunk_config = "rag_annotated.split(size=800, overlap=150)"

    def __init__(self, raw: str, size: int = 800, overlap: int = 150):
        sys.path.insert(0, REPO)
        os.makedirs(os.path.join(os.getcwd(), "docs"), exist_ok=True)  # module-level load()
        import rag_annotated as rag

        self.rag = rag
        self.chunk_texts = rag.split(raw, size=size, overlap=overlap)
        self.chunk_config = f"rag_annotated.split(size={size}, overlap={overlap})"
        self.name = f"NumPy from-scratch ({size}/{overlap}, brute force)"

        t0 = time.perf_counter()
        # Rebind the module's "vector database" (the two aligned collections its
        # own docstring describes) to this corpus, then use its retrieve() as-is.
        rag.chunks = [{"text": t, "source": "ai_infra_transcript.txt"}
                      for t in self.chunk_texts]
        rag.matrix = rag.embed(self.chunk_texts)
        self.index_seconds = time.perf_counter() - t0

    def search(self, query: str, k: int):
        hits = self.rag.retrieve(query, k=k)
        lookup = {id(c): i for i, c in enumerate(self.rag.chunks)}
        return [lookup[id(record)] for record, _score in hits]


def build(kind: str, raw: str, size=None, overlap=None):
    cls = {"langchain": LangChainPipeline, "numpy": NumpyPipeline}[kind]
    if size is None:
        return cls(raw)
    return cls(raw, size=size, overlap=overlap)


# -------------------------------------------------------------------- metrics

def evaluate(pipeline, questions, repeats=3):
    resolved = assign_gold(pipeline.chunk_texts, questions)
    max_k = max(K_VALUES)
    rows, latencies = [], []

    pipeline.search("warm up", max_k)

    for q in resolved:
        gold = set(q["gold"])
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            retrieved = pipeline.search(q["question"], max_k)
            timings.append((time.perf_counter() - start) * 1000)
        latencies.append(min(timings))

        ranks = [i for i, cid in enumerate(retrieved) if cid in gold]
        rows.append({
            "id": q["id"], "question": q["question"],
            "gold": sorted(gold), "retrieved": [int(c) for c in retrieved],
            "first_gold_rank": (ranks[0] + 1) if ranks else None,
            "hit": {k: any(r < k for r in ranks) for k in K_VALUES},
            "recall": {k: len([c for c in retrieved[:k] if c in gold]) / len(gold)
                       for k in K_VALUES},
        })

    n = len(rows)
    return {
        "pipeline": pipeline.name,
        "chunking": pipeline.chunk_config,
        "chunks": len(pipeline.chunk_texts),
        "mean_chunk_chars": sum(len(t) for t in pipeline.chunk_texts) / len(pipeline.chunk_texts),
        "index_seconds": pipeline.index_seconds,
        "gold_per_question": sum(len(r["gold"]) for r in rows) / n,
        "questions": n,
        "hit_at": {k: sum(r["hit"][k] for r in rows) / n for k in K_VALUES},
        "recall_at": {k: sum(r["recall"][k] for r in rows) / n for k in K_VALUES},
        "mrr": sum(1.0 / r["first_gold_rank"] if r["first_gold_rank"] else 0.0
                   for r in rows) / n,
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[max(0, int(round(0.95 * len(latencies))) - 1)],
        },
        "misses": [r for r in rows if not r["hit"][DEFAULT_K]],
        "per_question": rows,
    }


def report(results):
    head = (f"{'pipeline':44}{'chunks':>7}{'hit@1':>7}{'hit@4':>7}{'hit@10':>7}"
            f"{'MRR':>7}{'mean ms':>9}{'p95 ms':>8}{'index s':>9}")
    print("\n" + head)
    print("-" * len(head))
    for r in results:
        print(f"{r['pipeline']:44}{r['chunks']:>7}{r['hit_at'][1]:>7.3f}"
              f"{r['hit_at'][4]:>7.3f}{r['hit_at'][10]:>7.3f}{r['mrr']:>7.3f}"
              f"{r['latency_ms']['mean']:>9.1f}{r['latency_ms']['p95']:>8.1f}"
              f"{r['index_seconds']:>9.1f}")

    for r in results:
        print(f"\n{r['pipeline']}: {len(r['misses'])} miss(es) at k={DEFAULT_K}, "
              f"{r['mean_chunk_chars']:.0f} chars/chunk, "
              f"{r['gold_per_question']:.2f} gold/question")
        for m in r["misses"]:
            print(f"  Q{m['id']:>2} gold={m['gold']} got={m['retrieved'][:DEFAULT_K]}  {m['question']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", action="append",
                        choices=["langchain", "numpy"], default=None)
    parser.add_argument("--questions", default=os.path.join(HERE, "eval_questions.json"))
    parser.add_argument("--raw", default=os.path.join(HERE, "ai_infra_transcript.txt"))
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="override both pipelines' chunk size (ablation)")
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    raw, questions = load_inputs(args.questions, args.raw)
    print(f"{len(raw)} characters of raw transcript, {len(questions)} questions")

    results = [evaluate(build(kind, raw, args.chunk_size, args.chunk_overlap),
                        questions, args.repeats)
               for kind in (args.backend or ["langchain", "numpy"])]
    report(results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
