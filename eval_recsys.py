# -*- coding: utf-8 -*-
import requests
import itertools
import json
import math
import os
import random
from collections import defaultdict
from statistics import mean

BASE = "http://127.0.0.1:8058"
CATALOG_SIZE = 1682
TOP_K_DEFAULT = 10
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 12345

# ========== Helpers ==========
def safe_float(x, default=0.0):
    if x is None:
        return default
    try:
        return float(x)
    except (ValueError, TypeError):
        return default

def get_movie_id(rec):
    mid = rec.get("movie_id")
    return None if mid is None else str(mid)

def _get_json(url, params=None):
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 404:
        raise FileNotFoundError(f"404 Not Found: {url}")
    r.raise_for_status()
    return r.json()

def fetch_user_ids():
    r = requests.get(f"{BASE}/users")
    r.raise_for_status()
    return r.json().get("users", [])

def fetch_user_prefs(user_id):
    pr = requests.get(f"{BASE}/users/{user_id}")
    if pr.status_code == 200:
        return pr.json().get("preferences", {}) or {}
    return {}

# Torna sia risultati che meta (per bandit: diagnostica)
def fetch_recs(user_id, method="constraint", **kwargs):
    if method == "constraint":
        url = f"{BASE}/recommendations/{user_id}"
        params = {"top_k": kwargs.get("top_k", TOP_K_DEFAULT)}
        resp = _get_json(url, params)
        return {
            "results": resp.get("results", []),
            "meta": {"status": resp.get("status", "ok"), "epsilon": None, "diagnostics": None}
        }
    elif method == "bandit":
        url1 = f"{BASE}/recommendations_bandit/{user_id}"
        params = {
            "top_k": kwargs.get("top_k", TOP_K_DEFAULT),
            "epsilon": kwargs.get("epsilon", 0.35),
            "candidate_pool": kwargs.get("candidate_pool", 160),
            "explore_extra": kwargs.get("explore_extra", 400),
            "seed": kwargs.get("seed", 42),
        }
        try:
            resp = _get_json(url1, params)
        except FileNotFoundError:
            url2 = f"{BASE}/bandit/{user_id}"
            resp = _get_json(url2, params)
        return {
            "results": resp.get("results", []),
            "meta": {
                "status": resp.get("status", "ok"),
                "epsilon": resp.get("epsilon"),
                "diagnostics": resp.get("diagnostics", {})
            }
        }
    else:
        raise ValueError(f"Metodo non supportato: {method}")

# ========== Ground truth (opzionale) ==========
# relevant.json (facoltativo):
# { "Utente1": ["568","53",...], "Utente2": [...] }
def load_relevant_local():
    path = "relevant.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {uid: {str(x) for x in ids} for uid, ids in data.items()}
    return {}

# ========== Metriche ==========
def is_genre_flag(k, v):
    return (
        k not in ("unknown", "movie_id", "score", "novel", "pick_strategy", "novelty_reason", "movie_title") and
        ((v == 1) or (v is True) or (isinstance(v, str) and v == "1"))
    )

def accuracy(results):
    if not results: return 0.0
    return sum(1 for r in results if safe_float(r.get("score"), 0.0) >= 2.0) / len(results)

def partial_accuracy(results):
    if not results: return 0.0
    return sum(1 for r in results if safe_float(r.get("score"), 0.0) >= 1.0) / len(results)

def diversity_ild(results):
    G = []
    for r in results:
        genres = {k for k, v in r.items() if is_genre_flag(k, v)}
        if genres: G.append(genres)
    if len(G) < 2: return 0.0
    pairs = list(itertools.combinations(G, 2))
    if not pairs: return 0.0
    def jacc(a, b):
        inter = len(a & b); union = len(a | b)
        return inter / (union or 1)
    sims = [jacc(a, b) for a, b in pairs]
    return 1.0 - (sum(sims) / len(sims))

def serendipity(results, prefs):
    if not results: return 0.0
    desiderati = set(prefs.get("generi_desiderati", []) or [])
    if not desiderati: return 0.0
    novel = 0
    for r in results:
        genres = {k for k, v in r.items() if is_genre_flag(k, v)}
        if genres.isdisjoint(desiderati):
            novel += 1
    return novel / len(results)

def coverage(all_results):
    rec_ids = {get_movie_id(r) for rs in all_results for r in rs if get_movie_id(r) is not None}
    return len(rec_ids) / CATALOG_SIZE

def personalization_jaccard(all_users_results):
    if len(all_users_results) < 2: return 0.0
    sims = []
    for a, b in itertools.combinations(all_users_results, 2):
        setA = {get_movie_id(r) for r in a if get_movie_id(r) is not None}
        setB = {get_movie_id(r) for r in b if get_movie_id(r) is not None}
        inter = len(setA & setB); union = len(setA | setB)
        sims.append(inter / (union or 1))
    return 1 - (sum(sims) / len(sims))

# proxy di “rilevanza” se non c’è GT
def proxy_relevance_from_prefs(rec, prefs):
    desiderati = set(prefs.get("generi_desiderati", []) or [])
    vietati = set(prefs.get("generi_vietati", []) or [])
    genres = {k for k, v in rec.items() if is_genre_flag(k, v)}
    if vietati and not genres.isdisjoint(vietati): return 0
    if desiderati and genres.intersection(desiderati): return 1
    return 0

def precision_at_k(results, relevant_ids=None, prefs=None, k=TOP_K_DEFAULT):
    if not results: return 0.0
    recs = results[:k]
    hits = 0
    for r in recs:
        mid = get_movie_id(r)
        if relevant_ids is not None:
            hits += 1 if (mid and mid in relevant_ids) else 0
        else:
            hits += 1 if (prefs and proxy_relevance_from_prefs(r, prefs) == 1) else 0
    return hits / max(1, len(recs))

def recall_at_k(results, relevant_ids=None, prefs=None, k=TOP_K_DEFAULT):
    if relevant_ids is None or not relevant_ids: return 0.0
    recs = results[:k]
    hits = sum(1 for r in recs if (get_movie_id(r) in relevant_ids))
    return hits / len(relevant_ids)

def ndcg_at_k(results, relevant_ids=None, prefs=None, k=TOP_K_DEFAULT):
    if not results: return 0.0
    def rel(r):
        if relevant_ids is not None:
            mid = get_movie_id(r)
            return 1 if (mid and mid in relevant_ids) else 0
        else:
            return 1 if (prefs and proxy_relevance_from_prefs(r, prefs) == 1) else 0
    recs = results[:k]
    dcg = 0.0
    for idx, r in enumerate(recs, start=1):
        g = rel(r); dcg += g if idx == 1 else g / math.log2(idx)
    gains_sorted = sorted([rel(r) for r in recs], reverse=True)
    idcg = 0.0
    for idx, g in enumerate(gains_sorted, start=1):
        idcg += g if idx == 1 else g / math.log2(idx)
    return 0.0 if idcg == 0 else dcg / idcg

def ci_bootstrap(values, alpha=0.05, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
    if not values: return (0.0, 0.0, 0.0)
    rnd = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(B):
        sample = [values[rnd.randrange(n)] for _ in range(n)]
        boots.append(mean(sample))
    boots.sort()
    low = boots[int((alpha/2)*B)]
    high = boots[int((1 - alpha/2)*B)-1]
    return (round(mean(values), 3), round(low, 3), round(high, 3))

# ========== valutazione per utente ==========
def eval_user(user_id, prefs, relevant_ids, top_k=TOP_K_DEFAULT, epsilon=0.35, candidate_pool=160, explore_extra=400, seed=42):
    # constraint
    c_resp = fetch_recs(user_id, method="constraint", top_k=top_k)
    c_results = c_resp["results"]

    # bandit
    b_resp = fetch_recs(
        user_id,
        method="bandit",
        top_k=top_k,
        epsilon=epsilon,
        candidate_pool=candidate_pool,
        explore_extra=explore_extra,
        seed=seed
    )
    b_results = b_resp["results"]
    b_meta = b_resp.get("meta", {}) or {}
    b_diag = b_meta.get("diagnostics") or {}

    rows = []

    for method, results, meta in (
        ("constraint", c_results, {"epsilon": None, "diagnostics": None}),
        ("bandit", b_results, {"epsilon": b_meta.get("epsilon"), "diagnostics": b_diag}),
    ):
        ild_val = diversity_ild(results)
        row = {
            "user_id": user_id,
            "method": method,
            "n": len(results),
            "accuracy": accuracy(results),
            "partial_accuracy": partial_accuracy(results),
            "ild": ild_val,
            "diversity": ild_val,  # retro-compatibilità: diversity = ILD
            "serendipity": serendipity(results, prefs),
            "precision@k": precision_at_k(results, relevant_ids, prefs, k=top_k),
            "recall@k": recall_at_k(results, relevant_ids, prefs, k=top_k),
            "ndcg@k": ndcg_at_k(results, relevant_ids, prefs, k=top_k),
            "movie_ids": [r.get("movie_id") for r in results if r.get("movie_id")],
        }
        # aggiungi diagnostica bandit se presente
        if method == "bandit" and meta.get("diagnostics") is not None:
            d = meta["diagnostics"]
            row["bandit_meta"] = {
                "epsilon": meta.get("epsilon"),
                "exploit_pool_size": d.get("exploit_pool_size"),
                "explore_pool_size": d.get("explore_pool_size"),
                "explore_ratio": d.get("explore_ratio"),
                "novel_count": d.get("novel_count")
            }
        rows.append(row)

    return rows

# ========== main ==========
def main():
    users = fetch_user_ids()
    if not users:
        print("Nessun utente trovato")
        return

    prefs_map = {uid: fetch_user_prefs(uid) for uid in users}
    relevant_local = load_relevant_local()

    all_rows = []
    for uid in users:
        try:
            relevant_ids = relevant_local.get(uid)  # None se manca: useremo proxy
            all_rows.extend(
                eval_user(
                    uid,
                    prefs_map.get(uid, {}),
                    relevant_ids,
                    top_k=TOP_K_DEFAULT,
                    epsilon=0.35,
                    candidate_pool=160,
                    explore_extra=400,
                    seed=42
                )
            )
        except Exception as e:
            print(f"Errore {uid}: {e}")

    # aggregazione
    agg = defaultdict(lambda: defaultdict(list))
    per_method_results = defaultdict(list)
    for r in all_rows:
        m = r["method"]
        for k in ("accuracy","partial_accuracy","ild","serendipity","precision@k","recall@k","ndcg@k"):
            agg[m][k].append(r[k])
        per_method_results[m].append([{"movie_id": mid} for mid in r["movie_ids"]])

    aggregate = {}
    for method, vals in agg.items():
        out_m = {}
        for metric_key in ("precision@k","recall@k","ndcg@k","ild","serendipity","accuracy","partial_accuracy"):
            mu, lo, hi = ci_bootstrap(vals[metric_key], alpha=0.05, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
            out_m[metric_key] = {"mean": mu, "ci95": [lo, hi]}
        out_m["coverage"] = round(coverage(per_method_results[method]), 3)
        out_m["personalization_jaccard"] = round(personalization_jaccard(per_method_results[method]), 3)
        aggregate[method] = out_m

    out = {
        "per_user": all_rows,
        "aggregate": aggregate,
        "settings": {
            "top_k": TOP_K_DEFAULT,
            "bootstrap_B": BOOTSTRAP_B,
            "catalog_size": CATALOG_SIZE
        }
    }
    with open("results.json","w",encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Risultati salvati in results.json")

if __name__ == "__main__":
    main()
