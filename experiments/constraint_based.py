import pandas as pd

# ====================================
# Costanti peso per le nuove features
# ====================================
AWARD_WEIGHT     = 0.3   # bonus se film premiato
DIRECTOR_WEIGHT  = 1   # bonus se regista gradito
RUNTIME_WEIGHT   = 0.2   # bonus se durata entro la tolleranza

# ========================
# 1. Carica dataset
# ========================
df = pd.read_csv("data/movies_enriched.csv")

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

# ========================
# 3. Funzione di raccomandazione
# ========================
def recommend_movies(df, utente_id, pref, top_k=5):
    # --- hard-filter su anno -----------------------------
    df_f = df[df["release_date"].dt.year >= pref["min_release_year"]].copy()

    # --- hard-filter: rimuovi film con almeno un genere vietato
    penalita_genere = df_f[list(pref["generi_vietati"])].sum(axis=1)
    df_f = df_f[penalita_genere == 0]

    # --- punteggio di base (generi) ----------------------
    score = df_f[list(pref["generi_desiderati"])].sum(axis=1)

    # --- soft bonus: film premiato -----------------------
    if pref.get("prefer_award_winning", False):
        score += df_f["awards"] * AWARD_WEIGHT   # awards è 0/1

    # --- soft bonus: regista gradito ---------------------
    if pref.get("favorite_directors"):
        liked_dir = df_f["director"].isin(pref["favorite_directors"]).astype(int)
        score += liked_dir * DIRECTOR_WEIGHT

    # --- soft bonus: durata vicina alla preferita --------
    if pref.get("preferred_runtime") is not None:
        delta = (df_f["runtime"] - pref["preferred_runtime"]).abs()
        near = (delta <= pref.get("tolleranza_runtime", 15)).astype(int)
        score += near * RUNTIME_WEIGHT

    df_f["score"] = score

    # --- ordina e restituisci ----------------------------
    recs = df_f.sort_values(by="score", ascending=False).head(top_k)

    print(f"\nRaccomandazioni per {utente_id}:\n")
    cols_out = ["movie_title", "release_date", "score"] + list(pref["generi_desiderati"])
    if pref.get("prefer_award_winning"):
        cols_out.append("awards")
    if pref.get("favorite_directors"):
        cols_out.append("director")
    if pref.get("preferred_runtime") is not None:
        cols_out.append("runtime")
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
