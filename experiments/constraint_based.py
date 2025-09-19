import pandas as pd
import numpy as np
# ========================
# 1. Carica dataset
# ========================
df = pd.read_csv("../data/movies_enriched.csv")

# Assicura tipi corretti
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")   
df["awards"] = pd.to_numeric(df["awards"], errors="coerce").fillna(0)  

# Lista dei generi (colonne 0/1)
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film_noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci_fi", "Thriller", "War", "Western"
]

# ========================
# 2. Definisci utenti fittizi (ora con extra)
# ========================
utenti = {
    "utente_1": {
        "min_release_year": 1990,
        "generi_desiderati": {"Action", "Thriller"},
        "generi_vietati": {"Children", "Musical"},
        "prefer_award_winning": True,
        "preferred_runtime": 130,          # minuti
        "tolleranza_runtime": 20,          # ± minuti
        "favorite_directors": {"Christopher Nolan", "Ridley Scott"}
    },
    "utente_2": {
        "min_release_year": 1985,
        "generi_desiderati": {"Comedy", "Romance", "Drama"},
        "generi_vietati": {"Horror", "Sci_fi"},
        "prefer_award_winning": False,
        "preferred_runtime": 105,
        "tolleranza_runtime": 15,
        "favorite_directors": {"Nora Ephron"}
    },
    "utente_3": {
        "min_release_year": 1980,
        "generi_desiderati": {"Animation", "Children"},
        "generi_vietati": {"Crime", "Film_noir"},
        "prefer_award_winning": True,
        "preferred_runtime": None,         # nessuna preferenza
        "tolleranza_runtime": 0,           # ignorato
        "favorite_directors": set()
    },
    "utente_4": {
        "min_release_year": 1970,
        "generi_desiderati": {"Documentary"},
        "generi_vietati": {"Action", "War", "Western"},
        "prefer_award_winning": False,
        "preferred_runtime": 90,
        "tolleranza_runtime": 10,
        "favorite_directors": {"Werner Herzog"}
    },
    "utente_5": {
        "min_release_year": 1995,
        "generi_desiderati": {"Drama", "Mystery", "Thriller"},
        "generi_vietati": {"Comedy", "Children"},
        "prefer_award_winning": True,
        "preferred_runtime": 110,
        "tolleranza_runtime": 15,
        "favorite_directors": {"David Fincher"}
    }
}

# ====================================
# Pesi (bonus/malus)
# ====================================
AWARD_WEIGHT                  = 0.3   # bonus se film premiato
DIRECTOR_WEIGHT               = 1.0   # bonus se regista gradito
RUNTIME_WEIGHT                = 0.2   # bonus se durata entro la tolleranza

FORBIDDEN_GENRE_MALUS         = 1.5   # malus per ciascun genere vietato presente
RUNTIME_OUTSIDE_MALUS         = 0.3   # intensità base del malus fuori tolleranza (scalato)
YEAR_BELOW_MALUS_PER_YEAR     = 0.02  # malus per ogni anno sotto 'min_release_year'
MISSING_YEAR_MALUS            = 0.1   # malus fisso se l'anno è mancante

def recommend_movies(df, utente_id, pref, top_k=5):
    # Nessun filtro duro: lavoriamo su tutto il dataset
    df_f = df.copy()

    # Punteggio iniziale
    score = pd.Series(0.0, index=df_f.index)

    # --- base: generi desiderati (bonus) -----------------
    if pref.get("generi_desiderati"):
        cols = [g for g in pref["generi_desiderati"] if g in df_f.columns]
        if cols:
            score += df_f[cols].sum(axis=1)

    # --- premi (bonus) -----------------------------------
    if pref.get("prefer_award_winning", False):
        awarded = (df_f["awards"].fillna(0) > 0).astype(int)
        score += awarded * AWARD_WEIGHT

    # --- regista gradito (bonus) -------------------------
    if pref.get("favorite_directors"):
        liked_dir = df_f["director"].isin(pref["favorite_directors"]).astype(int)
        score += liked_dir * DIRECTOR_WEIGHT

    # --- runtime: bonus dentro tolleranza, malus fuori ---
    if pref.get("preferred_runtime") is not None:
        tol = max(0, pref.get("tolleranza_runtime", 15))
        delta = (df_f["runtime"] - pref["preferred_runtime"]).abs()

        inside = (delta <= tol).astype(int)
        score += inside * RUNTIME_WEIGHT

        excess = np.maximum(0, delta - tol)
        denom = tol if tol > 0 else 1
        scale = (excess / denom).clip(0, 3)  # limita il malus
        score -= scale * RUNTIME_OUTSIDE_MALUS

    # --- generi vietati (malus, senza eliminare nulla) ---
    if pref.get("generi_vietati"):
        cols_forbidden = [g for g in pref["generi_vietati"] if g in df_f.columns]
        if cols_forbidden:
            present_forbidden = df_f[cols_forbidden].sum(axis=1)
            score -= present_forbidden * FORBIDDEN_GENRE_MALUS

    # --- ANNO: malus se sotto min_release_year (nessun drop) ---
    if pref.get("min_release_year") is not None:
        years = df_f["release_date"].dt.year
        missing_mask = years.isna()
        score -= missing_mask.astype(float) * MISSING_YEAR_MALUS

        diff = (pref["min_release_year"] - years.fillna(pref["min_release_year"])).clip(lower=0)
        score -= diff * YEAR_BELOW_MALUS_PER_YEAR

    df_f["score"] = score

    # Ordina e restituisci
    recs = df_f.sort_values(by="score", ascending=False).head(top_k)

    print(f"\nRaccomandazioni per {utente_id}:\n")
    cols_out = ["movie_title", "release_date", "score"]
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

    print(recs[cols_out])
    return recs

# ========================
# 4. Esegui per tutti gli utenti
# ========================
def main():
    for uid, pref in utenti.items():
        recommend_movies(df, uid, pref)

if __name__ == "__main__":
    main()
