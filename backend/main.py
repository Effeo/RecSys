from fastapi import FastAPI
import pandas as pd
import json
import numpy as np
from typing import Dict, Any
import config
from fastapi.middleware.cors import CORSMiddleware
import random

# ========================
# Init app
# ========================
app = FastAPI(title="Movie Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8057", "http://127.0.0.1:8057", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Caricamento dataset film
# ========================
df = pd.read_csv(config.MOVIES_FILE)
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
df["awards"] = pd.to_numeric(df["awards"], errors="coerce").fillna(0)

# Carica lista film e matrice correlazioni
with open(config.MOVIES_LIST_FILE, "r", encoding="utf-8") as f:
    MOVIES_LIST = json.load(f)

MOVIES_CORR = np.load(config.MOVIES_CORR_FILE)
# pulizia NaN nella matrice di correlazione per evitare problemi in runtime
MOVIES_CORR = np.nan_to_num(MOVIES_CORR, nan=0.0)

# ========================
# Helper
# ========================
def load_users() -> Dict[str, Any]:
    if config.USERS_FILE.exists():
        with open(config.USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users: Dict[str, Any]):
    with open(config.USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

def clean_results(df_in: pd.DataFrame) -> list[dict]:
    """Converte Timestamp -> stringa YYYY-MM-DD e NaN -> None."""
    df2 = df_in.copy()
    # NaN -> None
    df2 = df2.replace({np.nan: None})
    # Timestamp -> stringa
    for col in df2.select_dtypes(include=["datetime64[ns]"]).columns:
        df2[col] = df2[col].dt.strftime("%Y-%m-%d")
    return df2.to_dict(orient="records")

# ========================
# Bandit helpers
# ========================
def is_novelty(row: pd.Series, pref: Dict[str, Any]) -> tuple[bool, str]:
    """
    Novel = fuori profilo: nessun genere desiderato matchato
            E (regista non preferito O runtime fuori tolleranza).
    Se i generi desiderati non sono indicati, non usiamo quella parte.
    """
    reasons = []
    g_des = [g for g in (pref.get("generi_desiderati") or []) if g in row.index]
    fav_dirs = set(pref.get("favorite_directors") or [])

    # Match generi
    has_genre_match = False
    if g_des:
        has_genre_match = (int(row[g_des].sum()) > 0)

    # Regista fuori preferiti?
    dir_off = (len(fav_dirs) > 0 and row.get("director") not in fav_dirs)

    # Runtime fuori tolleranza?
    rt_off = False
    if pref.get("preferred_runtime") is not None and "runtime" in row.index and pd.notna(row["runtime"]):
        tol = pref.get("tolleranza_runtime", 15)
        rt_off = abs(row["runtime"] - pref["preferred_runtime"]) > tol

    # Logica di novità
    if g_des:
        is_novel = (not has_genre_match) and (dir_off or rt_off)
        if not has_genre_match:
            reasons.append("genere fuori profilo")
    else:
        # se l'utente non ha generi desiderati, basiamoci su regista/runtime
        is_novel = (dir_off or rt_off)

    if dir_off:
        reasons.append("regista fuori preferiti")
    if rt_off:
        reasons.append("runtime fuori tolleranza")

    return is_novel, (", ".join(reasons) if reasons else "in linea con i gusti")


# ========================
# POOLS
# ========================
def _constraint_pool(df_all: pd.DataFrame, pref: Dict[str, Any], k: int) -> pd.DataFrame:
    """Top-k (ampliato) dal constraint-based, con colonna 'score' già presente."""
    return recommend_movies(df_all, pref, top_k=k).copy()

def _year_filtered(df_all: pd.DataFrame, pref: Dict[str, Any]) -> pd.DataFrame:
    """Filtro minimo per epoca/qualità base, per costruire l'explore pool."""
    df_y = df_all[df_all["release_date"].dt.year >= pref.get("min_release_year", 0)].copy()
    return df_y

def build_pools(df_all: pd.DataFrame,
                pref: Dict[str, Any],
                candidate_pool: int = 100,
                explore_extra: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ritorna (exploit_pool, explore_pool) disgiunti.
    - exploit_pool: top N dal constraint-based.
    - explore_pool: fuori profilo e non inclusi nell'exploit_pool.
      Se vuoto, riempi con: tail per punteggi bassi + random dalla stessa epoca.
    """
    exploit_pool = _constraint_pool(df_all, pref, k=max(candidate_pool, 20))
    if exploit_pool.empty:
        return exploit_pool, exploit_pool  # entrambi vuoti

    # Calcola novelty su exploit_pool (ci serve dopo, ma cloniamo)
    ex = exploit_pool.copy()

    # Costruiamo un explore candidati “larghi”
    base_wide = _year_filtered(df_all, pref)
    # rimuovi quelli già in exploit
    base_wide = base_wide[~base_wide["movie_id"].isin(ex["movie_id"])].copy()

    # contrassegna novelty su base_wide
    nov_flags, nov_reasons = zip(*[is_novelty(r, pref) for _, r in base_wide.iterrows()]) if not base_wide.empty else ([], [])
    if not base_wide.empty:
        base_wide["novel"] = list(nov_flags)
        base_wide["novelty_reason"] = list(nov_reasons)

    # explore = davvero fuori profilo
    explore_pool = base_wide[base_wide["novel"]].copy() if not base_wide.empty else base_wide

    # fallback se explore è troppo piccolo
    MIN_EXPLORE = min(50, explore_extra)
    if explore_pool.shape[0] < MIN_EXPLORE:
        # 1) aggiungi dalla coda dell'exploit (punteggi più bassi)
        tail = ex.sort_values("score", ascending=True).head(MIN_EXPLORE)
        tail = tail[~tail["movie_id"].isin(explore_pool["movie_id"] if not explore_pool.empty else [])]
        explore_pool = pd.concat([explore_pool, tail], ignore_index=True) if not explore_pool.empty else tail.copy()

        # 2) aggiungi random dal base_wide (stessa epoca)
        if not base_wide.empty:
            need = MIN_EXPLORE - explore_pool.shape[0]
            if need > 0:
                explore_pool = pd.concat(
                    [explore_pool, base_wide.sample(n=min(need, base_wide.shape[0]), random_state=None)],
                    ignore_index=True
                )

    # assicura disgiunzione
    explore_pool = explore_pool[~explore_pool["movie_id"].isin(ex["movie_id"])]
    return ex.reset_index(drop=True), explore_pool.reset_index(drop=True)


# ========================
# ε-GREEDY SU POOLS DISGIUNTI
# ========================
def epsilon_greedy_from_pools(exploit_pool: pd.DataFrame,
                              explore_pool: pd.DataFrame,
                              pref: Dict[str, Any],
                              top_k: int = 5,
                              epsilon: float = 0.2,
                              seed: int | None = None) -> pd.DataFrame:
    """
    Con prob. epsilon pesca dall'explore_pool (novità), altrimenti dall'exploit_pool.
    Ritorna df selezionato con colonne: novel, novelty_reason, pick_strategy.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if exploit_pool.empty and explore_pool.empty:
        return exploit_pool.head(0)

    # ordina lo sfruttamento per score
    ex = exploit_pool.sort_values("score", ascending=False).copy()
    ex_ids = set()
    rows = []

    # calcola novelty su entrambi i pool (se non presente)
    def _ensure_novelty(df_in: pd.DataFrame) -> pd.DataFrame:
        if "novel" in df_in.columns and "novelty_reason" in df_in.columns:
            return df_in
        flags, reasons = zip(*[is_novelty(r, pref) for _, r in df_in.iterrows()]) if not df_in.empty else ([], [])
        out = df_in.copy()
        if not df_in.empty:
            out["novel"] = list(flags)
            out["novelty_reason"] = list(reasons)
        return out

    ex = _ensure_novelty(ex)
    xp = _ensure_novelty(explore_pool)

    # indici di scorrimento per evitare ripetizioni
    ex_idx = 0

    for _ in range(top_k):
        pick_explore = (random.random() < epsilon) and not xp.empty
        chosen = None
        strategy = "explore" if pick_explore else "exploit"

        if pick_explore:
            # scegli un elemento novel NON ancora preso
            xp_available = xp[~xp["movie_id"].isin(ex_ids)]
            if not xp_available.empty:
                chosen = xp_available.sample(n=1, random_state=None).iloc[0]
            else:
                pick_explore = False  # fallback a exploit

        if not pick_explore:
            while ex_idx < len(ex) and ex.iloc[ex_idx]["movie_id"] in ex_ids:
                ex_idx += 1
            if ex_idx < len(ex):
                chosen = ex.iloc[ex_idx]
                ex_idx += 1

        if chosen is None:
            break

        ex_ids.add(int(chosen["movie_id"]))
        row = chosen.copy()
        row["pick_strategy"] = strategy
        rows.append(row)

    return pd.DataFrame(rows)

# ========================
# Raccomandazioni content-based per utente
# ========================
def recommend_movies(df, pref, top_k=5):
    df_f = df[df["release_date"].dt.year >= pref["min_release_year"]].copy()

    # rimuovi film con generi vietati (se presenti nel dataset)
    if pref.get("generi_vietati"):
        cols_vietati = [g for g in pref["generi_vietati"] if g in df_f.columns]
        if cols_vietati:
            penalita_genere = df_f[cols_vietati].sum(axis=1)
            df_f = df_f[penalita_genere == 0]

    # punteggio base sui generi desiderati
    score = 0
    if pref.get("generi_desiderati"):
        cols_desid = [g for g in pref["generi_desiderati"] if g in df_f.columns]
        if cols_desid:
            score = df_f[cols_desid].sum(axis=1)

    # bonus premi
    if pref.get("prefer_award_winning", False) and "awards" in df_f.columns:
        score = score + df_f["awards"] * config.AWARD_WEIGHT

    # bonus registi
    if pref.get("favorite_directors"):
        liked_dir = df_f["director"].isin(pref["favorite_directors"]).astype(int)
        score = score + liked_dir * config.DIRECTOR_WEIGHT

    # bonus durata
    if pref.get("preferred_runtime") is not None and "runtime" in df_f.columns:
        delta = (df_f["runtime"] - pref["preferred_runtime"]).abs()
        near = (delta <= pref.get("tolleranza_runtime", 15)).astype(int)
        score = score + near * config.RUNTIME_WEIGHT

    df_f["score"] = score
    recs = df_f.sort_values(by="score", ascending=False).head(top_k)
    return recs

# ========================
# Endpoints
# ========================
@app.get("/similar_movies/{movie_title}")
def get_similar_movies(movie_title: str, top_k: int = 5):
    # titolo esatto presente in MOVIES_LIST
    if movie_title not in MOVIES_LIST:
        return {
            "status": "no_match",
            "message": f"Movie '{movie_title}' not found in index.",
            "results": []
        }

    idx = MOVIES_LIST.index(movie_title)
    corr_vector = MOVIES_CORR[idx]

    # ordina per correlazione (decrescente), scarta self e correlazioni <= 0
    sorted_idx = np.argsort(corr_vector)[::-1]
    similar_idx = [i for i in sorted_idx if i != idx and corr_vector[i] > 0][:top_k]

    if not similar_idx:
        return {
            "status": "no_match",
            "message": f"No similar movies found for '{movie_title}'.",
            "results": []
        }

    # titoli e punteggi
    top_movies = [MOVIES_LIST[i] for i in similar_idx]
    top_scores = [float(corr_vector[i]) for i in similar_idx]
    score_map = {title: score for title, score in zip(top_movies, top_scores)}

    # prendi info dai metadata e aggiungi 'similarity'
    subset = df[df["movie_title"].isin(top_movies)].copy()
    # per mantenere l'ordine della similarità
    subset["__order"] = subset["movie_title"].map({t: k for k, t in enumerate(top_movies)})
    subset = subset.sort_values("__order")
    subset["similarity"] = subset["movie_title"].map(score_map)
    subset = subset.drop(columns="__order")

    results = clean_results(subset)
    return {
        "status": "ok",
        "input_movie": movie_title,
        "count": len(results),
        "results": results
    }

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, top_k: int = 5):
    users = load_users()
    if user_id not in users:
        return {
            "status": "no_match",
            "message": f"User '{user_id}' not found.",
            "results": []
        }

    recs_df = recommend_movies(df, users[user_id], top_k=top_k)

    if recs_df.empty:
        return {
            "status": "no_match",
            "message": f"No recommendations found for '{user_id}' with current preferences.",
            "results": []
        }

    results = clean_results(recs_df)
    return {
        "status": "ok",
        "user_id": user_id,
        "count": len(results),
        "results": results  # include 'score'
    }

@app.get("/users/{user_id}")
def get_user_preferences(user_id: str):
    users = load_users()
    if user_id not in users:
        return {
            "status": "no_match",
            "message": f"User '{user_id}' not found.",
            "results": []
        }
    return {"status": "ok", "user_id": user_id, "preferences": users[user_id]}

@app.post("/users/{user_id}")
def set_user_preferences(user_id: str, prefs: Dict[str, Any]):
    users = load_users()
    users[user_id] = prefs
    save_users(users)
    return {"status": "ok", "message": f"Preferences for {user_id} saved successfully", "preferences": prefs}

@app.get("/users")
def list_users():
    users = load_users()
    user_ids = sorted(list(users.keys()))
    return {"status": "ok", "users": user_ids}

@app.get("/recommendations_bandit/{user_id}")
def get_recommendations_bandit(user_id: str,
                               top_k: int = 5,
                               epsilon: float = 0.2,
                               candidate_pool: int = 100,
                               explore_extra: int = 200,
                               seed: int | None = None):
    users = load_users()
    if user_id not in users:
        return {"status": "no_match",
                "message": f"User '{user_id}' not found.",
                "results": []}

    # 1) costruisci pool disgiunti
    exploit_pool, explore_pool = build_pools(df, users[user_id],
                                             candidate_pool=candidate_pool,
                                             explore_extra=explore_extra)

    if exploit_pool.empty and explore_pool.empty:
        return {"status": "no_match",
                "message": f"No candidates found for '{user_id}'.",
                "results": []}

    # 2) selezione ε-greedy
    bandit_df = epsilon_greedy_from_pools(exploit_pool, explore_pool,
                                          users[user_id],
                                          top_k=top_k,
                                          epsilon=epsilon,
                                          seed=seed)

    results = clean_results(bandit_df)
    novel_titles = [r["movie_title"] for r in results if r.get("novel")]

    # diagnostica utile
    diag = {
        "exploit_pool_size": int(exploit_pool.shape[0]),
        "explore_pool_size": int(explore_pool.shape[0]),
        "explore_ratio": round(len(novel_titles) / max(1, len(results)), 3)
    }

    return {
        "status": "ok",
        "user_id": user_id,
        "epsilon": epsilon,
        "count": len(results),
        "novel_count": len(novel_titles),
        "novel_titles": novel_titles,
        "diagnostics": diag,  # per capire se stai davvero esplorando
        "results": results     # include score, novel, novelty_reason, pick_strategy
    }
