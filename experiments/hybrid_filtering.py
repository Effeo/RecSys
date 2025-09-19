import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# ========================
# 0. Config
# ========================
RATINGS_PATH   = "../data/ratings.csv"
MOVIES_PATH    = "../data/movies_enriched.csv"
TOP_K          = 10             # n film da estrarre
ALPHA_CF       = 1.0            # peso del CF nel punteggio ibrido
TEMPERATURE    = 0.3            # temperatura softmax (↓ <1 più "sharp", ↑ >1 più "morbida")
RANDOM_SEED    = None           # es. 42 per riproducibilità, None per random

# ====================================
# Pesi (bonus/malus) - constraint
# ====================================
AWARD_WEIGHT                  = 0.3   # bonus se film premiato
DIRECTOR_WEIGHT               = 1.0   # bonus se regista gradito
RUNTIME_WEIGHT                = 0.2   # bonus se durata entro la tolleranza

FORBIDDEN_GENRE_MALUS         = 1.5   # malus per ciascun genere vietato presente
RUNTIME_OUTSIDE_MALUS         = 0.3   # intensità base del malus fuori tolleranza (scalato)
YEAR_BELOW_MALUS_PER_YEAR     = 0.02  # malus per ogni anno sotto 'min_release_year'
MISSING_YEAR_MALUS            = 0.1   # malus fisso se l'anno è mancante
MISSING_RUNTIME_MALUS         = 0.0   # (facoltativo) malus lieve se runtime mancante

# ========================
# 1. Carica dataset
# ========================
df = pd.read_csv(MOVIES_PATH)

# Assicura tipi corretti
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["runtime"]      = pd.to_numeric(df["runtime"], errors="coerce")
df["awards"]       = pd.to_numeric(df.get("awards", 0), errors="coerce").fillna(0)
if "movie_id" not in df.columns:
    raise ValueError("movies_enriched.csv deve contenere la colonna 'movie_id'.")

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film_noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci_fi", "Thriller", "War", "Western"
]

# ========================
# 2. Utenti fittizi
# ========================
utenti = {
    "Francesco": {
        "min_release_year": 1990,
        "generi_desiderati": ["Action", "Thriller"],
        "generi_vietati": ["Children", "Musical"],
        "prefer_award_winning": True,
        "preferred_runtime": 130,
        "tolleranza_runtime": 20,
        "favorite_directors": ["Ridley Scott", "James Cameron"],
        "liked_movie": "Toy Story",
    },
    "Gianmarco": {
        "min_release_year": 1985,
        "generi_desiderati": ["Comedy", "Romance", "Drama"],
        "generi_vietati": ["Horror", "Sci_fi"],
        "prefer_award_winning": False,
        "preferred_runtime": 105,
        "tolleranza_runtime": 15,
        "favorite_directors": ["Nora Ephron"],
        "liked_movie": "Usual Suspects The",
    },
    "Ulderico": {
        "min_release_year": 1980,
        "generi_desiderati": ["Animation", "Children"],
        "generi_vietati": ["Crime", "Film_noir"],
        "prefer_award_winning": True,
        "preferred_runtime": None,
        "tolleranza_runtime": 0,
        "favorite_directors": ["John Lasseter", "Henry Selick"],
        "liked_movie": "Four Rooms",
    },
    "Sara": {
        "min_release_year": 1990,
        "generi_desiderati": ["Drama", "Romance"],
        "generi_vietati": ["Horror", "Sci_fi"],
        "prefer_award_winning": True,
        "preferred_runtime": 110,
        "tolleranza_runtime": 20,
        "favorite_directors": ["Nora Ephron", "James Ivory", "Wong Kar-wai"],
        "liked_movie": "Billy Madison",
    },
    "Luca": {
        "min_release_year": 1988,
        "generi_desiderati": ["Action", "Sci_fi"],
        "generi_vietati": ["Musical", "Documentary"],
        "prefer_award_winning": False,
        "preferred_runtime": 125,
        "tolleranza_runtime": 25,
        "favorite_directors": ["James Cameron", "John McTiernan", "Paul Verhoeven"],
        "liked_movie": "Clerks",
    },
}

# ========================
# 3. CF: riga i -> Series indicizzata per movie_id
# ========================
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

# ========================
# 4. Constraint recommender (no hard filter, NaN-safe)
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

# ========================
# 5. Diagnostica CF / allineamento
# ========================
def check_cf_basis(ratings_path: str, movies_path: str, liked_movie_title: str, max_components: int = 30):
    ratings = pd.read_csv(ratings_path)[["user_id", "movie_id", "rating"]]
    movies  = pd.read_csv(movies_path)[["movie_id", "movie_title"]].drop_duplicates()

    like_rows = movies.loc[movies["movie_title"] == liked_movie_title, "movie_id"]
    liked_id = int(like_rows.iloc[0]) if not like_rows.empty else None

    merged  = ratings.merge(movies, on="movie_id", how="inner")
    utility = merged.pivot_table(values="rating", index="user_id", columns="movie_id", fill_value=0)

    X = utility.T
    if X.shape[0] < 2:
        return {"ok": False, "reason": "Troppi pochi film per CF.", "n_movies": X.shape[0], "liked_id": liked_id}

    n_comp = max(2, min(max_components, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Z = svd.fit_transform(X)
    corr = np.corrcoef(Z)
    movie_ids = list(utility.columns)

    info = {
        "ok": True,
        "n_movies": len(movie_ids),
        "n_users": utility.shape[0],
        "liked_id_in_basis": liked_id in movie_ids if liked_id is not None else False,
        "liked_id": liked_id,
        "corr_shape": corr.shape,
        "basis_first5": movie_ids[:5],
    }
    return info


def debug_alignment_for_user(df_movies: pd.DataFrame,
                             user_id: str,
                             pref: dict,
                             ratings_path: str,
                             movies_path: str,
                             alpha_cf: float = 1.0,
                             top_k: int = 10) -> pd.DataFrame:
    recs, _ = recommend_movies(df_movies, user_id, pref, top_k=top_k*5)
    liked = pref.get("liked_movie")
    recs_h = add_cf_and_sum(recs, ratings_path, movies_path, liked, constraint_score_col="score", alpha_cf=alpha_cf)
    recs_h["delta"] = recs_h["hybrid_score"] - recs_h["score"]
    recs_h["in_cf"] = recs_h["cf_score"].ne(0.0)

    if liked:
        like_row = recs_h.loc[recs_h["movie_title"] == liked]
        if not like_row.empty:
            cf_like = float(like_row["cf_score"].iloc[0])
            print(f"[CHECK] cf_score per il liked '{liked}': {cf_like:.4f} (atteso ≈ 1.0)")
        else:
            print(f"[WARN] Il film liked '{liked}' non è nei candidati.")

    coverage = recs_h["in_cf"].mean() * 100
    print(f"[CHECK] Copertura CF sui candidati: {coverage:.1f}%")

    pre_cols = ["movie_id", "movie_title", "score"]
    post_cols = ["movie_id", "movie_title", "score", "cf_score", "delta", "hybrid_score"]

    print("\n--- TOP per score (PRE, constraint) ---")
    print(recs[pre_cols].head(top_k).reset_index(drop=True))

    print("\n--- TOP per hybrid_score (POST, constraint + CF) ---")
    print(recs_h.sort_values("hybrid_score", ascending=False)[post_cols].head(top_k).reset_index(drop=True))

    err = (recs_h["hybrid_score"] - (recs_h["score"] + alpha_cf * recs_h["cf_score"])).abs().max()
    print(f"\n[CHECK] max |hybrid - (score + alpha*cf)| = {err:.6g} (atteso 0)")
    return recs_h

# ========================
# 6. Softmax & Sampling
# ========================
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

# ========================
# 7. Main
# ========================
def main():
    # (Facoltativo) Controllo base del CF
    for uid, pref in utenti.items():
        liked = pref.get("liked_movie")
        if liked:
            info = check_cf_basis(RATINGS_PATH, MOVIES_PATH, liked)
            if not info["ok"]:
                print(f"[ERROR] CF basis per '{uid}' non valida:", info)
            else:
                print(f"[CF BASIS] {uid}: n_movies={info['n_movies']}, liked_in_basis={info['liked_id_in_basis']}, corr_shape={info['corr_shape']}")

    # Per ogni utente: constraint -> ibrido -> softmax -> sampling
    for uid, pref in utenti.items():
        print(f"\n================= USER: {uid} (liked: {pref.get('liked_movie')}) =================")

        # 1) Constraint
        recs, cols_out = recommend_movies(df, uid, pref, top_k=TOP_K*5)

        # 2) Add CF & somma
        liked = pref.get("liked_movie")
        if liked:
            recs_h = add_cf_and_sum(recs, RATINGS_PATH, MOVIES_PATH, liked, constraint_score_col="score", alpha_cf=ALPHA_CF)
        else:
            recs_h = recs.copy()
            recs_h["cf_score"] = 0.0
            recs_h["hybrid_score"] = recs_h["score"]

        # 3) Softmax -> probabilità
        recs_h["prob"] = softmax_from_scores(recs_h["hybrid_score"], temperature=TEMPERATURE)

        # 4) Sampling senza rimpiazzo di TOP_K film
        sampled = sample_by_softmax(recs_h, score_col="hybrid_score", n=TOP_K, temperature=TEMPERATURE, seed=RANDOM_SEED, replace=False)

        # 5) Output sintetico
        show_cols = ["movie_id", "movie_title", "score", "cf_score", "hybrid_score", "prob"]
        show_cols = [c for c in show_cols if c in sampled.columns]

        print("\n--- TOP (ibrido) deterministico ---")
        print(recs_h.sort_values("hybrid_score", ascending=False)[show_cols].head(TOP_K).reset_index(drop=True))

        print("\n--- SELEZIONE campionata da softmax ---")
        print(sampled[show_cols].reset_index(drop=True))

        # (Facoltativo) Debug coerenza: delta = hybrid - (score + alpha*cf)
        err = (recs_h["hybrid_score"] - (recs_h["score"] + ALPHA_CF * recs_h["cf_score"])).abs().max()
        print(f"\n[CHECK] max |hybrid - (score + alpha*cf)| = {err:.6g} (atteso 0)")
        # Prob sommano a 1?
        p_sum = recs_h["prob"].sum()
        print(f"[CHECK] somma(prob) = {p_sum:.6f} (atteso 1.0)")

if __name__ == "__main__":
    main()
