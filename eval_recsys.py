#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Valutazione sistema di raccomandazione (backend + utils invariati).

Metriche:
- Accuracy@K:   quota item che rispettano TUTTI i vincoli "hard"
                (no generi vietati, min_release_year, runtime entro tolleranza se definito).
- PartialAccuracy@K: media, sui K item, della percentuale di vincoli soddisfatti (3 check: forbidden, year, runtime).
- Diversity@K:
    * avg Jaccard distance sui vettori di generi binari
    * director diversity = registi unici / K
    * release year variance (proxy copertura temporale)
- Serendipity@K: media su item di [ relevance * unexpectedness ],
    dove relevance = min-max(hybrid_score) su lista;
          unexpectedness = 1 - similarity
          similarity = cf_score normalizzato in [0,1] tramite (cf+1)/2 (clipped)
                       (fallback: cosine similarity sui generi col "liked_movie").

Per la lista "sampled" lo script ripete N run (seed diversi) e aggrega media/std.
Salva un unico JSON con risultati per utente e aggregati globali.
"""

from __future__ import annotations
import argparse, json, math, time, itertools, statistics, os
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests


# -----------------------------
# Utility di I/O e normalizzazioni
# -----------------------------
def read_movies(movies_csv: str) -> pd.DataFrame:
    df = pd.read_csv(movies_csv)
    # parsing colonne note
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    if "runtime" in df.columns:
        df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    if "awards" in df.columns:
        df["awards"] = pd.to_numeric(df["awards"], errors="coerce").fillna(0)

    # genere: insieme standard (adattabile a dataset 1682 film MovieLens 100k)
    KNOWN_GENRES = [
        "unknown","Action","Adventure","Animation","Children","Comedy","Crime","Documentary",
        "Drama","Fantasy","Film_noir","Horror","Musical","Mystery","Romance","Sci_fi",
        "Thriller","War","Western"
    ]
    genres = [g for g in KNOWN_GENRES if g in df.columns]
    # campi base
    for col in ["movie_id","movie_title","director"]:
        if col not in df.columns:
            raise ValueError(f"Colonna mancante nel CSV: {col}")

    return df, genres


def minmax_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    mn, mx = float(s.min()), float(s.max())
    if not math.isfinite(mn) or not math.isfinite(mx) or mx - mn <= 1e-12:
        # tutti uguali → tutti 0.0
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def norm_cf_to_similarity(cf: float) -> float:
    # cf_score atteso in [-1,1]; normalizzo a [0,1]
    if cf is None or not math.isfinite(cf):
        return None
    return float(np.clip((cf + 1.0) / 2.0, 0.0, 1.0))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# -----------------------------
# Vincoli & feature helper
# -----------------------------
def movie_year(row: pd.Series) -> Optional[int]:
    if "release_date" not in row or pd.isna(row["release_date"]):
        return None
    try:
        return int(pd.to_datetime(row["release_date"]).year)
    except Exception:
        return None


def genres_vector(row: pd.Series, genre_cols: List[str]) -> np.ndarray:
    vals = []
    for g in genre_cols:
        v = row.get(g, 0)
        try:
            vals.append(int(v))
        except Exception:
            vals.append(0)
    return np.array(vals, dtype=int)


def jaccard_distance(x: np.ndarray, y: np.ndarray) -> float:
    inter = np.logical_and(x == 1, y == 1).sum()
    union = np.logical_or(x == 1, y == 1).sum()
    if union == 0:
        return 0.0
    return 1.0 - (inter / union)


def check_constraints(row: pd.Series, prefs: Dict[str, Any], genre_cols: List[str]) -> Tuple[bool, float]:
    """
    Ritorna:
      - passed_all (bool)
      - partial_score ∈ {0, 1/3, 2/3, 1} (media dei 3 check)
    Check:
      1) NO generi vietati
      2) anno >= min_release_year (se anno mancante → fail)
      3) runtime entro preferred_runtime ± tolleranza (se preferred non definito → considerato satisfied)
    """
    # 1) forbidden genres
    forbidden = set(prefs.get("generi_vietati", []) or [])
    forb_cols = [g for g in forbidden if g in genre_cols]
    forb_ok = True
    if forb_cols:
        present = sum(int(row.get(g, 0) or 0) for g in forb_cols)
        forb_ok = (present == 0)

    # 2) min year
    min_year = prefs.get("min_release_year", None)
    y = movie_year(row)
    year_ok = True
    if min_year is not None:
        year_ok = (y is not None and int(y) >= int(min_year))

    # 3) runtime range
    pr = prefs.get("preferred_runtime", None)
    tol = int(prefs.get("tolleranza_runtime", 0) or 0)
    runtime_ok = True
    if pr is not None:
        rt = row.get("runtime", None)
        if pd.isna(rt):
            runtime_ok = False
        else:
            runtime_ok = (abs(float(rt) - float(pr)) <= tol)

    flags = [forb_ok, year_ok, runtime_ok]
    partial = sum(1.0 if f else 0.0 for f in flags) / 3.0
    passed_all = all(flags)
    return passed_all, partial


# -----------------------------
# Chiamate API backend
# -----------------------------
def api_get_users(base_url: str) -> List[str]:
    r = requests.get(f"{base_url}/users", timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("users", [])


def api_get_user_prefs(base_url: str, user_id: str) -> Dict[str, Any]:
    r = requests.get(f"{base_url}/users/{user_id}", timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"Preferenze utente non trovate per {user_id}: {data}")
    return data.get("preferences", {})


def api_get_recs(base_url: str, user_id: str, alpha: float, temperature: float, seed: Optional[int]) -> Dict[str, Any]:
    params = {"alpha": alpha, "temperature": temperature}
    if seed is not None:
        params["seed"] = seed
    r = requests.get(f"{base_url}/recommendations_hybrid/{user_id}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


# -----------------------------
# Metriche su una lista (DataFrame)
# -----------------------------
def evaluate_list(
    df_list: pd.DataFrame,
    prefs: Dict[str, Any],
    movies_df: pd.DataFrame,
    genre_cols: List[str],
    liked_movie_title: Optional[str]
) -> Dict[str, Any]:
    """
    df_list: deve contenere almeno movie_id, movie_title, hybrid_score; cf_score opzionale.
    """
    # Uniamo metadati (genere, director, runtime, release_date) via movie_id
    if "movie_id" not in df_list.columns:
        raise ValueError("df_list deve avere 'movie_id'")
    df_merge = df_list.merge(
        movies_df[["movie_id", "release_date", "runtime", "director"] + genre_cols],
        on="movie_id", how="left"
    )

    K = len(df_merge)
    if K == 0:
        return {
            "K": 0, "accuracy": 0.0, "partial_accuracy": 0.0,
            "diversity": {"jaccard_avg": 0.0, "director_diversity": 0.0, "year_variance": 0.0},
            "serendipity": 0.0
        }

    # --- Accuracy & Partial ---
    flags_all = []
    partials = []
    for _, row in df_merge.iterrows():
        passed, p = check_constraints(row, prefs, genre_cols)
        flags_all.append(1.0 if passed else 0.0)
        partials.append(p)
    accuracy = float(np.mean(flags_all))
    partial_acc = float(np.mean(partials))

    # --- Diversity ---
    # Generi: Jaccard medio su tutte le coppie
    G = np.stack([genres_vector(row, genre_cols) for _, row in df_merge.iterrows()], axis=0)
    if K >= 2:
        dists = []
        for i, j in itertools.combinations(range(K), 2):
            dists.append(jaccard_distance(G[i], G[j]))
        jaccard_avg = float(np.mean(dists)) if dists else 0.0
    else:
        jaccard_avg = 0.0

    # Registi: quota unici
    uniq_dir = df_merge["director"].fillna("UNK").nunique()
    director_div = float(uniq_dir) / float(K)

    # Varianza anno
    years = [movie_year(row) for _, row in df_merge.iterrows()]
    years_num = [y for y in years if y is not None]
    year_var = float(np.var(years_num)) if len(years_num) >= 2 else 0.0

    diversity = {
        "jaccard_avg": jaccard_avg,
        "director_diversity": director_div,
        "year_variance": year_var,
    }

    # --- Serendipity ---
    # relevance: min-max(hybrid_score)
    if "hybrid_score" in df_merge.columns:
        rel = minmax_series(df_merge["hybrid_score"]).values
    else:
        # fallback su prob se presente
        rel = minmax_series(df_merge.get("prob", pd.Series(np.ones(K)))).values

    # unexpectedness: 1 - similarity
    # similarity da cf_score se disponibile; altrimenti cosine sui generi col liked_movie
    sim = np.zeros(K, dtype=float)

    if "cf_score" in df_merge.columns and df_merge["cf_score"].notna().any():
        cf = df_merge["cf_score"].astype(float).fillna(0.0).values
        sim = np.array([norm_cf_to_similarity(x) if x is not None else 0.0 for x in cf], dtype=float)
    else:
        # fallback: cosine con il vettore generi del liked_movie
        if liked_movie_title:
            liked_row = movies_df.loc[movies_df["movie_title"] == liked_movie_title]
            if not liked_row.empty:
                v_like = genres_vector(liked_row.iloc[0], genre_cols)
                for idx in range(K):
                    sim[idx] = cosine_sim(G[idx], v_like)
            else:
                sim[:] = 0.0
        else:
            sim[:] = 0.0

    sim = np.clip(sim, 0.0, 1.0)
    unexpected = 1.0 - sim
    ser_item = rel * unexpected
    serendipity = float(np.mean(ser_item))

    return {
        "K": K,
        "accuracy": accuracy,
        "partial_accuracy": partial_acc,
        "diversity": diversity,
        "serendipity": serendipity
    }


# -----------------------------
# Valutazione per utente
# -----------------------------
def evaluate_user(
    base_url: str,
    user_id: str,
    alpha: float,
    temperature: float,
    runs: int,
    movies_df: pd.DataFrame,
    genre_cols: List[str],
) -> Dict[str, Any]:
    prefs = api_get_user_prefs(base_url, user_id)
    liked = prefs.get("liked_movie")

    # --- deterministico
    data = api_get_recs(base_url, user_id, alpha=alpha, temperature=temperature, seed=None)
    top_det = pd.DataFrame(data.get("top_deterministic", []))
    det_metrics = evaluate_list(top_det, prefs, movies_df, genre_cols, liked)

    # --- sampled (stocastico) ripetuto
    sampled_metrics = []
    for run_seed in range(runs):
        data_s = api_get_recs(base_url, user_id, alpha=alpha, temperature=temperature, seed=run_seed)
        sampled = pd.DataFrame(data_s.get("sampled", []))
        m = evaluate_list(sampled, prefs, movies_df, genre_cols, liked)
        sampled_metrics.append(m)

    # aggrego sampled
    def agg_metric(path: List[str], default=0.0) -> Tuple[float, float]:
        vals = []
        for m in sampled_metrics:
            v = m
            for p in path:
                v = v.get(p, {})
            if isinstance(v, dict):
                # se chiediamo una foglia e troviamo dict, non è la foglia
                continue
            vals.append(float(v))
        if not vals:
            return default, 0.0
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    sampled_agg = {
        "accuracy": _tuple_to_dict(*agg_metric(["accuracy"])),
        "partial_accuracy": _tuple_to_dict(*agg_metric(["partial_accuracy"])),
        "serendipity": _tuple_to_dict(*agg_metric(["serendipity"])),
        "diversity": {
            "jaccard_avg": _tuple_to_dict(*agg_metric(["diversity", "jaccard_avg"])),
            "director_diversity": _tuple_to_dict(*agg_metric(["diversity", "director_diversity"])),
            "year_variance": _tuple_to_dict(*agg_metric(["diversity", "year_variance"])),
        },
        "runs": runs,
    }

    return {
        "user_id": user_id,
        "deterministic": det_metrics,
        "sampled": sampled_agg,
        "params": {"alpha": alpha, "temperature": temperature},
    }


def _tuple_to_dict(mean: float, std: float) -> Dict[str, float]:
    return {"mean": float(mean), "std": float(std)}


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Valutazione recommender (backend invariato)")
    ap.add_argument("--base-url", type=str, default="http://localhost:8000", help="Base URL FastAPI backend")
    ap.add_argument("--movies-file", type=str, required=True, help="Path al CSV dei film (1682 righe nel tuo set)")
    ap.add_argument("--alpha", type=float, default=0.5, help="Peso CF nella somma ibrida (pass-through all'endpoint)")
    ap.add_argument("--temperature", type=float, default=0.7, help="Temperatura softmax (pass-through all'endpoint)")
    ap.add_argument("--runs", type=int, default=20, help="Ripetizioni per la lista 'sampled'")
    ap.add_argument("--output", type=str, default="results", help="Cartella output")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Carico film + colonne genere
    movies_df, genre_cols = read_movies(args.movies_file)

    # Elenco utenti via API
    user_ids = api_get_users(args.base_url)
    if not user_ids:
        raise RuntimeError("Nessun utente trovato via /users")

    # Valuto
    per_user = []
    for uid in user_ids:
        try:
            res = evaluate_user(
                base_url=args.base_url,
                user_id=uid,
                alpha=args.alpha,
                temperature=args.temperature,
                runs=args.runs,
                movies_df=movies_df,
                genre_cols=genre_cols,
            )
            per_user.append(res)
            print(f"[OK] {uid}")
        except Exception as e:
            print(f"[ERRORE] {uid}: {e}")

    # Aggregati globali (media semplice sui deterministic e sulle mean dei sampled)
    def agg_over_users(key_path: List[str], is_sampled: bool) -> float:
        vals = []
        for u in per_user:
            v = u["sampled"] if is_sampled else u["deterministic"]
            tmp = v
            for k in key_path:
                tmp = tmp[k]
            # sampled: leaf è {"mean":..,"std":..} → prendiamo "mean"
            if is_sampled and isinstance(tmp, dict) and "mean" in tmp:
                vals.append(float(tmp["mean"]))
            elif not is_sampled and isinstance(tmp, (int, float)):
                vals.append(float(tmp))
        return float(np.mean(vals)) if vals else 0.0

    global_agg = {
        "deterministic": {
            "accuracy": agg_over_users(["accuracy"], False),
            "partial_accuracy": agg_over_users(["partial_accuracy"], False),
            "serendipity": agg_over_users(["serendipity"], False),
            "diversity": {
                "jaccard_avg": agg_over_users(["diversity", "jaccard_avg"], False),
                "director_diversity": agg_over_users(["diversity", "director_diversity"], False),
                "year_variance": agg_over_users(["diversity", "year_variance"], False),
            },
        },
        "sampled": {
            "accuracy": agg_over_users(["accuracy"], True),
            "partial_accuracy": agg_over_users(["partial_accuracy"], True),
            "serendipity": agg_over_users(["serendipity"], True),
            "diversity": {
                "jaccard_avg": agg_over_users(["diversity", "jaccard_avg"], True),
                "director_diversity": agg_over_users(["diversity", "director_diversity"], True),
                "year_variance": agg_over_users(["diversity", "year_variance"], True),
            },
        },
    }

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "params": {
            "alpha": args.alpha,
            "temperature": args.temperature,
            "K": 10,  # coerente con backend attuale; se cambi TOP_K, non serve toccare lo script
            "runs": args.runs,
            "base_url": args.base_url,
            "movies_file": args.movies_file,
        },
        "users_evaluated": [u["user_id"] for u in per_user],
        "results_per_user": per_user,
        "global_aggregates": global_agg,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output, f"eval_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSalvato: {out_path}")


if __name__ == "__main__":
    main()