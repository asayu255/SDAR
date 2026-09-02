import json
import threading
import time
import warnings
from typing import List, Optional, Union
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs):
    """Fetch the documents at these row numbers, in this order.

    One gather, not one lookup per row. ``corpus`` is an Arrow-backed
    datasets.Dataset, and ``corpus[i]`` walks its whole indexing machinery and
    builds a fresh Python dict every time -- so a turn that retrieves top-3 for
    44 queries paid 132 of those. Measured end to end, a batched retrieval took
    292 ms per turn against ~20-40 ms of encoder and FAISS: nearly all of the
    rest was this loop. ``corpus[list]`` is a single take over the table.

    The return shape is unchanged: one dict per requested row, in the order
    asked for, duplicates included (different queries do retrieve the same
    passage).
    """
    idxs = [int(idx) for idx in doc_idxs]
    if not idxs:
        return []
    columns = corpus[idxs]
    names = list(columns.keys())
    return [{name: columns[name][position] for name in names} for position in range(len(idxs))]


# One request at a time on the GPU.
#
# FastAPI runs a synchronous endpoint in its worker threadpool, so this server
# will happily execute forty /retrieve calls at once against ONE sharded FAISS
# index and ONE encoder. Two things go wrong. The activations of n concurrent
# encodes coexist, which is why a single request of 384 queries was refused with
# a 500 on a box where 383 was served -- the ceiling was not the request, it was
# the request plus whatever else was in flight. And a GpuIndex built by
# index_cpu_to_all_gpus shares its resources across the shards; searching it from
# several threads at once is not something faiss promises to survive, and the
# symptom when it does not is that nothing returns at all -- eleven requests
# timed out together after 600 s with the server still accepting connections.
#
# Serialising costs nothing that was real. The GPU executes these one at a time
# regardless; running them concurrently only multiplied the memory and removed
# the guarantee. Document loading stays outside the lock, because it is Arrow and
# host memory and it is the part that genuinely overlaps.
_GPU = threading.Lock()


def load_model(model_path: str, use_fp16: bool = False):
    # model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}" for query in query_list
                ]

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!")
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results = [
                {
                    "title": content.split("\n")[0].strip('"'),
                    "text": "\n".join(content.split("\n")[1:]),
                    "contents": content,
                }
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        with _GPU:
            query_emb = self.encoder.encode(query)
            scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        spent = {"gpu_wait": 0.0, "encode": 0.0, "faiss": 0.0, "load": 0.0}
        for start_idx in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            mark = time.perf_counter()
            with _GPU:
                spent["gpu_wait"] += time.perf_counter() - mark
                mark = time.perf_counter()
                batch_emb = self.encoder.encode(query_batch)
                spent["encode"] += time.perf_counter() - mark

                mark = time.perf_counter()
                batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
                spent["faiss"] += time.perf_counter() - mark

            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()
            flat_idxs = sum(batch_idxs, [])
            mark = time.perf_counter()
            batch_results = load_docs(self.corpus, flat_idxs)
            spent["load"] += time.perf_counter() - mark
            # chunk them back
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]
            results.extend(batch_results)
            scores.extend(batch_scores)
        # Where the time went, per request -- which with a batched client is once
        # per rollout turn. The encoder and the FAISS scan are bounded by what the
        # hardware can do; the document lookup is not, and attributing between
        # them by argument is how a quarter of a second per turn stayed invisible.
        print(
            f"[retrieve] {len(query_list):4d} queries  topk {num}  "
            f"gpu_wait {1000 * spent['gpu_wait']:7.1f} ms  "
            f"encode {1000 * spent['encode']:6.1f} ms  "
            f"faiss {1000 * spent['faiss']:6.1f} ms  "
            f"load_docs {1000 * spent['load']:6.1f} ms",
            flush=True,
        )
        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    # A list as well as a string, because the index is Flat: a search reads the
    # whole 32 GB of embeddings regardless of how many queries it is given, so
    # 126 separate requests read it 126 times and one request with 126 queries
    # reads it once. Measured against this server, an unloaded single query is
    # 80 ms and 126 concurrent ones take 7.5 s -- a 93x inflation that is the
    # index being re-read, not the server being slow.
    #
    # A plain string still behaves exactly as before, so an un-upgraded client
    # keeps working and no restart has to be coordinated with one.
    query: Union[str, List[str]]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

# How many /retrieve calls are being served right now. A client that retries
# without bound turns a slow server into a queue and the queue into the reason
# it is slow, and from the client side that is indistinguishable from a server
# that has stopped answering -- eleven requests timing out together at 600 s
# said nothing about how many were in front of them. One number per request is
# enough to tell the two apart.
_inflight = 0
_inflight_lock = threading.Lock()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts one query or a list of them and performs retrieval.
    Input format:
    {
      "query": "What is Python?",                     # or ["What is Python?", ...]
      "topk": 3,
      "return_scores": true
    }
    The response is {"result": [...]} with one entry per query, in order -- for a
    single string that is the one-element list it has always been.
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    queries = [request.query] if isinstance(request.query, str) else list(request.query)
    started = time.perf_counter()

    global _inflight
    with _inflight_lock:
        _inflight += 1
        concurrent = _inflight
    try:
        # batch_search even for one query. DenseRetriever._batch_search encodes
        # the whole list in one forward pass and hands FAISS one (n, dim) matrix,
        # which for a Flat index is one pass over the embeddings instead of n
        # passes. It chunks by retrieval_batch_size, so a longer list only makes
        # it read the index fewer times.
        #
        # That chunking is NOT a guarantee that any request can be served: a
        # chunk is encoded and searched on a GPU that also holds the index, and
        # one that does not fit raises out of here as a bare 500. Measured on
        # this server at retrieval_batch_size=512, 383 queries in a request were
        # served and 384 were not -- and that ceiling was free memory over the
        # cost of a query with every other in-flight request counted against it,
        # which is what _GPU above now bounds. The client finds what is left of
        # it by halving a request that keeps drawing 5xx; see _Coalescer._send.
        if request.return_scores:
            results, scores = retriever.batch_search(queries, num=request.topk, return_score=True)
        else:
            results = retriever.batch_search(queries, num=request.topk, return_score=False)
            scores = None
    finally:
        with _inflight_lock:
            _inflight -= 1

    served = time.perf_counter()

    # One entry per query, in the order they were sent -- which for a single
    # string is the one-element list the old response already was.
    resp = []
    for position, documents in enumerate(results):
        if scores is not None:
            resp.append(
                # float(): numpy float32 is not JSON serialisable
                [{"document": doc, "score": float(score)} for doc, score in zip(documents, scores[position])]
            )
        else:
            resp.append(documents)
    # The two _batch_search cannot see: its own total as the endpoint measures it,
    # and the reshaping into the response.
    print(
        f"[retrieve]              search {1000 * (served - started):6.1f} ms  "
        f"format {1000 * (time.perf_counter() - served):5.1f} ms  "
        f"inflight {concurrent:3d}",
        flush=True,
    )
    return {"result": resp}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument(
        "--index_path", type=str, default="~/data/searchR1/e5_Flat.index", help="Corpus indexing file."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="~/data/searchR1/wiki-18.jsonl",
        help="Local corpus file.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument(
        "--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model."
    )
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI server on.")

    args = parser.parse_args()

    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method=args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        # How many queries one encoder pass and one FAISS search handle. Reached
        # from /retrieve now that it accepts a list, so it is the cap on how much
        # of the index re-reading a batched client can amortise away.
        retrieval_batch_size=512,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)

    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=args.port)
