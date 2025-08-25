from fastapi import FastAPI
import pandas as pd
import json
import numpy as np
from typing import Dict, Any
import config
from fastapi.middleware.cors import CORSMiddleware

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
