"""
Retrieval evaluation harness for the transcript RAG pipeline.

Runs the SAME question set against different retriever backends so the
LangChain/Chroma pipeline and the from-scratch NumPy pipeline can be compared
on identical inputs, identical gold labels and identical metrics.

Backends
  --backend brute    sentence-transformers encode + exact NumPy cosine search.
                     Reference implementation; no Chroma, no LangChain.
  --backend chroma   the real app stack: langchain_huggingface.HuggingFaceEmbeddings
                     + langchain_community.vectorstores.Chroma (HNSW), k=4 as in
                     Transcript_RAG_backend.build_rag_graph.
  --backend numpy    your from-scratch implementation, loaded from --numpy-module.
                     The module must expose either
                         build_index(chunks: list[str]) -> index
                         search(index, query: str, k: int) -> list[int]   # chunk indices
                     or a class NumpyRAG with .index(chunks) and .search(query, k).

Usage
  python evaluate_rag.py --backend brute
  python evaluate_rag.py --backend chroma
  python evaluate_rag.py --backend numpy --numpy-module numpy_rag
  python evaluate_rag.py --backend brute --backend chroma --out results.json
"""

import argparse
import importlib
import json
import os
import re
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "all-MiniLM-L6-v2"          # same model id as the app
DEFAULT_K = 4                            # same k as vectorstore.as_retriever(search_kwargs={"k": 4})
K_VALUES = (1, 3, 4, 5, 10)


# ---------------------------------------------------------------- data loading

def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_corpus_and_questions(questions_path: str):
    with open(questions_path, encoding="utf-8") as f:
        spec = json.load(f)

    corpus_path = os.path.join(os.path.dirname(questions_path), spec["corpus"])
    with open(corpus_path, encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    normalised = [normalise(t) for t in texts]

    questions = []
    for q in spec["questions"]:
        span = normalise(q["answer_span"])
        gold = [i for i, t in enumerate(normalised) if span in t]
        if not gold:
            raise SystemExit(
                f"Question {q['id']} has no gold chunk: answer_span "
                f"{q['answer_span']!r} was not found anywhere in the corpus."
            )
        questions.append({"id": q["id"], "question": q["question"], "gold": gold})

    return texts, questions


# ------------------------------------------------------------------- backends

class BruteForceBackend:
    """Exact cosine search over the same embeddings the app uses."""

    name = "brute (MiniLM + exact NumPy cosine)"

    def __init__(self, texts):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME)
        # all-MiniLM-L6-v2 ships a Normalize module, so vectors are unit length
        # and a dot product IS the cosine similarity.
        self.matrix = np.asarray(self.model.encode(texts, batch_size=32))

    def search(self, query: str, k: int):
        vector = np.asarray(self.model.encode([query]))[0]
        scores = self.matrix @ vector
        return list(np.argsort(-scores)[:k])


class ChromaBackend:
    """The production path: LangChain HuggingFaceEmbeddings + Chroma HNSW."""

    name = "chroma (app stack: HuggingFaceEmbeddings + Chroma HNSW)"

    def __init__(self, texts):
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings

        self.texts = texts
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        docs = [
            Document(page_content=t, metadata={"chunk_id": i})
            for i, t in enumerate(texts)
        ]
        self.store = Chroma.from_documents(documents=docs, embedding=embeddings)

    def search(self, query: str, k: int):
        hits = self.store.similarity_search(query, k=k)
        return [h.metadata["chunk_id"] for h in hits]


class NumpyModuleBackend:
    """Adapter for the from-scratch implementation kept outside this repo."""

    def __init__(self, texts, module_name: str):
        self.name = f"numpy (from-scratch: {module_name})"
        sys.path.insert(0, os.getcwd())
        module = importlib.import_module(module_name)

        if hasattr(module, "build_index") and hasattr(module, "search"):
            self.index = module.build_index(texts)
            self._search = lambda q, k: module.search(self.index, q, k)
        elif hasattr(module, "NumpyRAG"):
            rag = module.NumpyRAG()
            rag.index(texts)
            self._search = lambda q, k: rag.search(q, k)
        else:
            raise SystemExit(
                f"{module_name} must expose build_index/search or a NumpyRAG class. "
                "See the docstring at the top of this file."
            )

    def search(self, query: str, k: int):
        return list(self._search(query, k))


def build_backend(kind: str, texts, numpy_module: str | None):
    if kind == "brute":
        return BruteForceBackend(texts)
    if kind == "chroma":
        return ChromaBackend(texts)
    if kind == "numpy":
        if not numpy_module:
            raise SystemExit("--backend numpy requires --numpy-module")
        return NumpyModuleBackend(texts, numpy_module)
    raise SystemExit(f"unknown backend {kind}")


# -------------------------------------------------------------------- metrics

def evaluate(backend, questions, k_values=K_VALUES, repeats=3):
    max_k = max(k_values)
    per_question = []
    latencies = []

    backend.search("warm up the model and the index", max_k)  # exclude cold start

    for q in questions:
        gold = set(q["gold"])

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            retrieved = backend.search(q["question"], max_k)
            timings.append((time.perf_counter() - start) * 1000)
        latencies.append(min(timings))  # min of repeats = least noisy estimate

        ranks = [i for i, cid in enumerate(retrieved) if cid in gold]
        per_question.append({
            "id": q["id"],
            "question": q["question"],
            "gold": sorted(gold),
            "retrieved": [int(c) for c in retrieved],
            "first_gold_rank": (ranks[0] + 1) if ranks else None,
            "hit": {k: any(r < k for r in ranks) for k in k_values},
            "recall": {
                k: len([c for c in retrieved[:k] if c in gold]) / len(gold)
                for k in k_values
            },
        })

    n = len(per_question)
    reciprocal = [
        1.0 / r["first_gold_rank"] if r["first_gold_rank"] else 0.0
        for r in per_question
    ]

    return {
        "backend": backend.name,
        "questions": n,
        "hit_at": {k: sum(r["hit"][k] for r in per_question) / n for k in k_values},
        "recall_at": {k: sum(r["recall"][k] for r in per_question) / n for k in k_values},
        "mrr": sum(reciprocal) / n,
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[max(0, int(round(0.95 * len(latencies))) - 1)],
            "max": max(latencies),
        },
        "misses": [r for r in per_question if not r["hit"][DEFAULT_K]],
        "per_question": per_question,
    }


def print_table(results):
    header = f"{'backend':52} {'hit@1':>7} {'hit@4':>7} {'hit@10':>7} {'MRR':>7} {'mean ms':>9} {'p95 ms':>8}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['backend']:52} "
            f"{r['hit_at'][1]:>7.3f} {r['hit_at'][4]:>7.3f} {r['hit_at'][10]:>7.3f} "
            f"{r['mrr']:>7.3f} {r['latency_ms']['mean']:>9.1f} {r['latency_ms']['p95']:>8.1f}"
        )

    for r in results:
        print(f"\n{r['backend']}: {len(r['misses'])} miss(es) at k={DEFAULT_K}")
        for m in r["misses"]:
            print(f"  Q{m['id']:>2} gold={m['gold']} got={m['retrieved'][:DEFAULT_K]}  {m['question']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", action="append", default=None,
                        choices=["brute", "chroma", "numpy"])
    parser.add_argument("--numpy-module", default=None)
    parser.add_argument("--questions", default=os.path.join(HERE, "eval_questions.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    backends = args.backend or ["brute"]
    texts, questions = load_corpus_and_questions(args.questions)
    print(f"{len(texts)} chunks, {len(questions)} questions, "
          f"{sum(len(q['gold']) for q in questions) / len(questions):.2f} gold chunks/question")

    results = []
    for kind in backends:
        backend = build_backend(kind, texts, args.numpy_module)
        results.append(evaluate(backend, questions, repeats=args.repeats))

    print_table(results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
