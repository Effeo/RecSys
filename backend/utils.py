import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Optional
from config import *
from sklearn.decomposition import TruncatedSVD


# ========================
# Helper
# ========================
def load_users() -> Dict[str, Any]:
    """
    Carica il dizionario degli utenti (utente_id -> informazioni utente)
    dal file JSON specificato in USERS_FILE.

    Se il file non esiste, restituisce un dizionario vuoto.
    """
    if USERS_FILE.exists():
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users: Dict[str, Any]):
    """
    Salva il dizionario degli utenti (utente_id -> informazioni utente)
    nel file JSON specificato in USERS_FILE.

    Se il file non esiste, viene creato.

    :param users: dizionario degli utenti
    """
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

def clean_results(df_in: pd.DataFrame) -> list[dict]:
    """
    Pulisce i risultati della raccomandazione per renderli
    più facilmente serializzabili in JSON.

    :param df_in: dataframe con le colonne da pulire
    :return: lista di dizionari, ciascuno corrispondente ad una riga di df_in
    """
    df2 = df_in.copy()
    # NaN -> None
    df2 = df2.replace({np.nan: None})
    # Timestamp -> stringa
    for col in df2.select_dtypes(include=["datetime64[ns]"]).columns:
        df2[col] = df2[col].dt.strftime("%Y-%m-%d")
    return df2.to_dict(orient="records")

# ========================
# Raccomandazioni content-based per utente
# ========================
def recommend_movies(df_movies: pd.DataFrame, utente_id: str, pref: dict, top_k: int = TOP_K):
    df_f = df_movies.copy()
    score = pd.Series(0.0, index=df_f.index, dtype=float)

    # Generi desiderati (bonus)
    if pref.get("generi_desiderati"):
        cols = [g for g in pref["generi_desiderati"] if g in df_f.columns]
        if cols:
            score += df_f[cols].fillna(0).sum(axis=1)

    # Premi (bonus)
    if pref.get("prefer_award_winning", False):
        awarded = (df_f["awards"].fillna(0) > 0).astype(int)
        score += awarded * AWARD_WEIGHT

    # Registi graditi (bonus)
    if pref.get("favorite_directors"):
        liked_dir = df_f.get("director", pd.Series(index=df_f.index)).isin(pref["favorite_directors"]).fillna(False).astype(int)
        score += liked_dir * DIRECTOR_WEIGHT

    # Runtime: bonus dentro tolleranza, malus fuori
    if pref.get("preferred_runtime") is not None:
        tol = max(0, pref.get("tolleranza_runtime", 15))
        runtime = pd.to_numeric(df_f.get("runtime"), errors="coerce")
        delta = (runtime - pref["preferred_runtime"]).abs()

        inside = (delta <= tol).fillna(False).astype(int)
        score += inside * RUNTIME_WEIGHT

        excess = (delta - tol).clip(lower=0).fillna(0)
        denom = tol if tol > 0 else 1
        scale = (excess / denom).clip(0, 3)
        score -= scale * RUNTIME_OUTSIDE_MALUS

        if MISSING_RUNTIME_MALUS > 0:
            score -= runtime.isna().astype(int) * MISSING_RUNTIME_MALUS

    # Generi vietati (malus)
    if pref.get("generi_vietati"):
        cols_forbidden = [g for g in pref["generi_vietati"] if g in df_f.columns]
        if cols_forbidden:
            present_forbidden = df_f[cols_forbidden].fillna(0).sum(axis=1)
            score -= present_forbidden * FORBIDDEN_GENRE_MALUS

    # Anno: malus sotto soglia (nessun drop)
    if pref.get("min_release_year") is not None:
        years = df_f["release_date"].dt.year
        score -= years.isna().astype(float) * MISSING_YEAR_MALUS
        diff = (pref["min_release_year"] - years.fillna(pref["min_release_year"])).clip(lower=0)
        score -= diff * YEAR_BELOW_MALUS_PER_YEAR

    df_f["score"] = score.fillna(0.0)

    # Ordina per score constraint (il taglio top_k lo faremo dopo sul campionamento)
    recs = df_f.sort_values(by="score", ascending=False)

    # Colonne di output suggerite
    cols_out = ["movie_id", "movie_title", "release_date", "score"]
    if pref.get("generi_desiderati"):
        cols_out += [g for g in pref["generi_desiderati"] if g in df_f.columns]
    if pref.get("prefer_award_winning"):
        cols_out.append("awards")
    if pref.get("favorite_directors"):
        cols_out.append("director")
    if pref.get("preferred_runtime") is not None:
        cols_out.append("runtime")
    if pref.get("generi_vietati"):
        cols_out += [g for g in pref["generi_vietati"] if g in df_f.columns]

    return recs, cols_out

def normalize_prefs(p: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    p = dict(p or {})
    out = {**DEFAULT_PREFS, **p}

    # numeri
    try:
        out["min_release_year"] = int(out.get("min_release_year", 0) or 0)
    except Exception:
        out["min_release_year"] = 0

    try:
        pr = out.get("preferred_runtime", None)
        out["preferred_runtime"] = None if pr in (None, "", "null") else int(pr)
    except Exception:
        out["preferred_runtime"] = None

    try:
        out["tolleranza_runtime"] = int(out.get("tolleranza_runtime", 0) or 0)
    except Exception:
        out["tolleranza_runtime"] = 0

    # liste
    for k in ("generi_desiderati", "generi_vietati", "favorite_directors"):
        v = out.get(k, [])
        if isinstance(v, (list, tuple)):
            out[k] = [str(x) for x in v]
        else:
            out[k] = []

    # booleano
    out["prefer_award_winning"] = bool(out.get("prefer_award_winning", False))
    return out


def collaborative_filtering_series(
    ratings_path: str,
    movies_path: str,
    liked_movie_title: str,
    max_components: int = 30,
) -> pd.Series:
    """Ritorna una Series 'cf_score' indicizzata per movie_id con le correlazioni
    rispetto al film 'liked_movie_title'. Se il film non è trovato, Series vuota.
    """
    ratings = pd.read_csv(ratings_path)[["user_id", "movie_id", "rating"]]
    movies  = pd.read_csv(movies_path)[["movie_id", "movie_title"]].drop_duplicates()

    like_rows = movies.loc[movies["movie_title"] == liked_movie_title, "movie_id"]
    if like_rows.empty:
        return pd.Series(dtype=float, name="cf_score")
    liked_id = int(like_rows.iloc[0])

    merged  = ratings.merge(movies, on="movie_id", how="inner")
    utility = merged.pivot_table(values="rating", index="user_id", columns="movie_id", fill_value=0)
    if utility.shape[1] < 2:
        return pd.Series(dtype=float, name="cf_score")

    X = utility.T  # (n_movies x n_users)
    n_comp = max(2, min(max_components, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Z = svd.fit_transform(X)  # (n_movies x n_comp)

    corr = np.corrcoef(Z)  # (n_movies x n_movies)
    movie_ids = list(utility.columns)
    try:
        i = movie_ids.index(liked_id)
    except ValueError:
        return pd.Series(dtype=float, name="cf_score")

    cf_vec = pd.Series(corr[i], index=movie_ids, name="cf_score")
    return cf_vec


def add_cf_and_sum(
    df_in: pd.DataFrame,
    ratings_path: str,
    movies_path: str,
    liked_movie_title: str,
    constraint_score_col: str = "score",
    alpha_cf: float = ALPHA_CF,
) -> pd.DataFrame:
    """Allinea CF per movie_id e somma al punteggio constraint in df_in.
    Ritorna df con colonne aggiunte: 'cf_score', 'hybrid_score'.
    """
    if "movie_id" not in df_in.columns:
        raise ValueError("df deve contenere la colonna 'movie_id'.")

    cf_series = collaborative_filtering_series(ratings_path, movies_path, liked_movie_title)
    df_out = df_in.copy()
    df_out = df_out.merge(cf_series, how="left", left_on="movie_id", right_index=True)
    df_out["cf_score"] = pd.to_numeric(df_out["cf_score"], errors="coerce").fillna(0.0)

    base = pd.to_numeric(df_out[constraint_score_col], errors="coerce").fillna(0.0)
    df_out["hybrid_score"] = base + alpha_cf * df_out["cf_score"]
    return df_out

def softmax_from_scores(scores: pd.Series, temperature: float = 1.0) -> pd.Series:
    """Softmax numericamente stabile: exp((s - max)/T) / sum(exp(...)).
    Gestisce NaN e casi degeneri (tutti uguali). Ritorna una Series con somma = 1.
    """
    s = pd.to_numeric(scores, errors="coerce").fillna(0.0)
    T = max(1e-8, float(temperature))
    s_shift = s - s.max()
    exps = np.exp(s_shift / T)
    # Protezione da overflow/underflow estrema
    exps = np.where(np.isfinite(exps), exps, 0.0)
    total = exps.sum()
    if total <= 0.0:
        # fallback: distribuzione uniforme
        p = np.ones_like(exps) / len(exps) if len(exps) > 0 else np.array([])
    else:
        p = exps / total
    return pd.Series(p, index=s.index, name="prob")


def sample_by_softmax(df_in: pd.DataFrame,
                      score_col: str = "hybrid_score",
                      n: int = TOP_K,
                      temperature: float = TEMPERATURE,
                      seed: int | None = RANDOM_SEED,
                      replace: bool = False) -> pd.DataFrame:
    """Calcola softmax(score_col) -> prob e campiona n righe secondo quella distribuzione.
    Ritorna il sotto-DataFrame campionato con colonna 'prob' inclusa.
    """
    rng = np.random.default_rng(seed)
    probs = softmax_from_scores(df_in[score_col], temperature=temperature)
    # Se tutte le prob sono ~0 per numerica, ricalcoliamo uniformi
    if not np.isfinite(probs.values).all() or probs.sum() <= 0:
        probs = pd.Series(np.ones(len(df_in)) / len(df_in), index=df_in.index, name="prob")

    n_eff = min(n, len(df_in))
    chosen_idx = rng.choice(df_in.index.values, size=n_eff, replace=replace, p=probs.loc[df_in.index].values)
    out = df_in.loc[chosen_idx].copy()
    out["prob"] = probs.loc[chosen_idx].values
    # Ordiniamo per probabilità decrescente solo per leggibilità
    return out.sort_values("prob", ascending=False)
