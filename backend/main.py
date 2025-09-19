from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from utils import load_users, save_users, clean_results, recommend_movies, normalize_prefs, add_cf_and_sum, softmax_from_scores, sample_by_softmax
import pandas as pd
import json
from config import *

# ========================
# Init app
# ========================
app = FastAPI(title="Movie Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8057", "http://127.0.0.1:8057",
        "http://localhost:3000", "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Caricamento dataset film
# ========================
df = pd.read_csv(MOVIES_FILE)
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
df["awards"] = pd.to_numeric(df["awards"], errors="coerce").fillna(0)
with open(DATA_DIR / "users.json", "r", encoding="utf-8") as f:
    utenti = json.load(f)

# Carica lista film e matrice correlazioni
with open(MOVIES_LIST_FILE, "r", encoding="utf-8") as f:
    MOVIES_LIST = json.load(f)

# ========================
# Schemi request
# ========================
class CreateUserRequest(BaseModel):
    user_id: str
    preferences: Optional[Dict[str, Any]] = None

@app.get("/recommendations_hybrid/{user_id}")
def recommendations_hybrid(
    user_id: str,
    alpha: float = ALPHA_CF,
    temperature: float = TEMPERATURE,
    seed: int | None = RANDOM_SEED,
):
    # valida utente
    if user_id not in utenti:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found in 'utenti'.")

    # valida temperatura
    if temperature <= 0:
        raise HTTPException(status_code=400, detail="temperature deve essere > 0")

    pref = utenti[user_id]

    # 1) Constraint (no hard filter, NaN-safe) su TUTTO il dataset letto da ../data
    recs, _ = recommend_movies(df, user_id, pref, top_k=TOP_K)

    # 2) Somma CF (dalla cartella ../data) + alpha
    liked = pref.get("liked_movie")
    if liked:
        recs_h = add_cf_and_sum(
            recs,
            RATINGS_FILE,
            MOVIES_FILE,
            liked,
            constraint_score_col="score",
            alpha_cf=float(alpha),
        )
    else:
        recs_h = recs.copy()
        recs_h["cf_score"] = 0.0
        recs_h["hybrid_score"] = recs_h["score"]

    # 3) Softmax -> probabilità
    recs_h["prob"] = softmax_from_scores(recs_h["hybrid_score"], temperature=float(temperature))

    # 4) Sampling senza rimpiazzo di TOP_K film
    sampled = sample_by_softmax(
        recs_h,
        score_col="hybrid_score",
        n=TOP_K,
        temperature=float(temperature),
        seed=seed,
        replace=False,
    )

    # 5) Risposta JSON (top deterministico + selezione campionata)
    show_cols = ["movie_id", "movie_title", "score", "cf_score", "hybrid_score", "prob", "release_date", "runtime", "director"]
    show_cols = [c for c in show_cols if c in recs_h.columns]

    top_det = recs_h.sort_values("hybrid_score", ascending=False)[show_cols].head(TOP_K)
    sampled_out = sampled[show_cols]

    return {
        "status": "ok",
        "user_id": user_id,
        "params": {
            "alpha": float(alpha),
            "temperature": float(temperature),
            "seed": seed,
            "top_k": TOP_K,
        },
        "top_deterministic": clean_results(top_det),
        "sampled": clean_results(sampled_out),
    }

@app.get("/users")
def list_users():
    users = load_users()
    user_ids = sorted(list(users.keys()))
    return {"status": "ok", "users": user_ids}

@app.get("/users/{user_id}")
def get_user_preferences(user_id: str):
    users = load_users()
    if user_id not in users:
        return {"status": "no_match", "message": f"User '{user_id}' not found.", "results": []}
    return {"status": "ok", "user_id": user_id, "preferences": users[user_id]}

@app.post("/users", status_code=201)
def create_user(req: CreateUserRequest):
    user_id = (req.user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id mancante o vuoto")

    users = load_users()
    if user_id in users:
        raise HTTPException(status_code=409, detail=f"User '{user_id}' already exists")

    prefs = normalize_prefs(req.preferences)
    users[user_id] = prefs
    save_users(users)

    return {
        "status": "ok",
        "message": f"User '{user_id}' created successfully",
        "user_id": user_id,
        "preferences": prefs,
    }

@app.post("/users/{user_id}")
def set_user_preferences(user_id: str, prefs: Dict[str, Any]):
    uid = (user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id non valido")

    normalized = normalize_prefs(prefs)
    users = load_users()
    users[uid] = normalized
    save_users(users)
    return {
        "status": "ok",
        "message": f"Preferences for {uid} saved successfully",
        "user_id": uid,
        "preferences": normalized,
    }
