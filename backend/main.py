from fastapi import FastAPI
import pandas as pd
import json
import numpy as np
from typing import Dict, Any
import config
from fastapi.middleware.cors import CORSMiddleware
from utils import *

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
# Endpoints
# ========================
@app.get("/similar_movies/{movie_title}")
def get_similar_movies(movie_title: str, top_k: int = 5):
    """
    Restituisce la lista dei film più simili a quello specificato dal titolo.
    La lista dei film simili è ordinata per similarità decrescente.

    :param movie_title: Titolo del film per il quale si vuole ottenere la lista dei film simili
    :param top_k: Numero di film simili da restituire (default=5)

    :return: Dizionario contenente lo stato dell'operazione (`ok` o `no_match`), il titolo del film
             originale, il numero di film simili trovati e la lista dei film stessi.
    """
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
    """
    Restituisce la lista dei film raccomandati per l'utente specificato dall'ID.
    La lista dei film raccomandati è ordinata per punteggio decrescente.

    :param user_id: ID dell'utente per il quale si vuole ottenere la lista dei film raccomandati
    :param top_k: Numero di film raccomandati da restituire (default=5)

    :return: Dizionario contenente lo stato dell'operazione (`ok` o `no_match`), l'ID dell'utente originale,
             il numero di film raccomandati trovati e la lista dei film stessi, con punteggio.
    """
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
    """
    Restituisce le preferenze dell'utente specificato dall'ID.

    :param user_id: ID dell'utente per il quale si vuole ottenere le preferenze

    :return: Dizionario contenente lo stato dell'operazione (`ok` o `no_match`), l'ID dell'utente originale e le sue preferenze.
    """
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
    """
    Salva le preferenze dell'utente specificato dall'ID.

    :param user_id: ID dell'utente per il quale si vuole salvare le preferenze
    :param prefs: Dizionario contenente le preferenze dell'utente

    :return: Dizionario contenente lo stato dell'operazione (`ok`) e le preferenze salvate.
    """
    users = load_users()
    users[user_id] = prefs
    save_users(users)
    return {"status": "ok", "message": f"Preferences for {user_id} saved successfully", "preferences": prefs}

@app.get("/users")
def list_users():
    """
    Restituisce l'elenco degli ID degli utenti salvati.

    :return: Dizionario contenente lo stato dell'operazione (`ok`) e l'elenco degli ID degli utenti.
    """
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
    """
    Ritorna le raccomandazioni per l'utente specificato dall'ID,
    utilizzando un approccio di tipo multi-armed bandit (ε-greedy).

    La funzione restituisce un elenco di fino a top_k film raccomandati,
    selezionati con un'implementazione di ε-greedy tra due pool disgiunti:
    1. exploit_pool: top-k film raccomandati per le preferenze utente, senza alcuna novità
    2. explore_pool: candidate pool più largo, con film che rispettano la minima release year
       richiesta ma non sono in exploit_pool, contrassegnati per la novità

    :param user_id: ID dell'utente per il quale si vuole ottenere la lista dei film raccomandati
    :param top_k: Numero di film raccomandati da restituire (default=5)
    :param epsilon: Probabilità di esplorazione (default=0.2)
    :param candidate_pool: Dimensione del pool di candidati da cui estrarre le raccomandazioni (default=100)
    :param explore_extra: Numero di film aggiuntivi da esplorare, al di fuori della top_k (default=200)
    :param seed: Seed per la generazione casuale (default=None)

    :return: Dizionario contenente lo stato dell'operazione (`ok`), l'ID dell'utente originale,
             il numero di film raccomandati trovati, la lista dei film stessi, con punteggio,
             e alcune informazioni diagnostiche (exploit_pool_size, explore_pool_size, explore_ratio)
    """
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
