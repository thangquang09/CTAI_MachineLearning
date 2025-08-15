import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from underthesea import word_tokenize

json_path = "/home/thangquang/CODE/CTAI_MachineLearning/data/ise-dsc01-train_new_preprocessed.json"
with open(json_path, "r", encoding="utf8") as fh:
    data = json.load(fh)

dataset = [ex for ex in data if ex["verdict"] != "NEI"]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def calc_metrics(ranks, k_list=(1, 3, 5)):
    """ranks: vị trí (bắt đầu từ 1) của evidence. None nếu không tìm thấy."""
    hit1 = 1 if ranks is not None and ranks == 1 else 0
    mrr = 1.0 / ranks if ranks is not None else 0.0
    recalls = {k: 1 if ranks is not None and ranks <= k else 0 for k in k_list}
    return hit1, mrr, recalls

def tokenize(sent: str):
    """Tách từ, trả list token (giữ dấu _)."""
    return word_tokenize(sent, format="text").split()

def hash_ctx(sents):
    return hashlib.md5(" ".join(sents).encode()).hexdigest()

def calc_single_metrics(rank):
    """Trả hit@1, mrr, recall3, recall5 cho 1 query."""
    hit1 = 1 if rank == 1 else 0
    mrr = 1.0 / rank if rank else 0.0
    rec3 = 1 if rank and rank <= 3 else 0
    rec5 = 1 if rank and rank <= 5 else 0
    return hit1, mrr, rec3, rec5



metrics_sum = {"hit1": 0, "mrr": 0, "recall@3": 0, "recall@5": 0}

dataset_raw = dataset

bert = SentenceTransformer(MODEL_NAME)

context_cache = {}      # cid -> dict(sents, bm25, emb, sent2id)
claims = []             # giữ thứ tự với dataset_idx
dataset = []            # bản ghi kèm cid

print("Pre-computing context cache ...")
for item in tqdm(dataset_raw):
    cid = hash_ctx(item['context'])
    if cid not in context_cache:
        sents = [s.strip() for s in item['context']]
        tokenized_sents = [tokenize(s) for s in sents]
        bm25 = BM25Okapi(tokenized_sents)
        emb = bert.encode(sents,
                          batch_size=128,
                          convert_to_tensor=True,
                          normalize_embeddings=True)
        # map sentence text -> index for O(1) lookup when finding gold
        sent2id = {s: i for i, s in enumerate(sents)}
        context_cache[cid] = {"sents": sents, "bm25": bm25, "emb": emb, "sent2id": sent2id}
    dataset.append({**item, "cid": cid})
    claims.append(item['claim'].strip())

print("Encoding all claims ...")
claim_embs = bert.encode(claims,
                         batch_size=128,
                         convert_to_tensor=True,
                         normalize_embeddings=True)

# Precompute tokenized claims for BM25 scoring to avoid repeated tokenization
tokenized_claims = [tokenize(c) for c in claims]

# --------------------- 2. HÀM CHẠY CHO 1 CLAIM -------------------
def evaluate_idx(idx: int):
    ex = dataset[idx]
    cid = ex["cid"]
    ctx = context_cache[cid]
    gold_sent = ex["evidence"].strip()

    # ---- BM25 scoring using pre-tokenized claim ------------------
    q_tokens = tokenized_claims[idx]
    scores = ctx["bm25"].get_scores(q_tokens)

    # Fast rank computation: get gold sentence index if present, then
    # count number of sentences with strictly higher score.
    gold_id = ctx["sent2id"].get(gold_sent)
    if gold_id is None:
        rank = None
    else:
        gold_score = scores[gold_id]
        # rank is 1 + number of scores strictly greater than gold_score
        rank = int(1 + np.sum(scores > gold_score))

    return calc_single_metrics(rank)

# --------------------- 3. CHẠY ĐA LUỒNG --------------------------
def main():
    num_workers = min(16, (os.cpu_count() or 1) * 2)
    metrics_sum = np.zeros(4, dtype=float)    # [hit1, mrr, rec3, rec5]

    print(f"Evaluating with {num_workers} threads ...")
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(evaluate_idx, i) for i in range(len(dataset))]
        for fut in tqdm(as_completed(futures), total=len(dataset)):
            metrics_sum += np.array(fut.result())

    N = len(dataset)
    print("\n============= BASELINE (Thread) =============")
    print(f"Queries evaluated : {N}")
    print(f"Hit@1             : {metrics_sum[0]/N:.4f}")
    print(f"MRR               : {metrics_sum[1]/N:.4f}")
    print(f"Recall@3          : {metrics_sum[2]/N:.4f}")
    print(f"Recall@5          : {metrics_sum[3]/N:.4f}")


if __name__ == "__main__":
    main()