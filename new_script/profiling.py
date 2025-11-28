# recsys_core/profiling.py
from typing import Any, Dict, List
import pandas as pd

def infer_user_prefs(user_train: pd.DataFrame, movies_meta: pd.DataFrame) -> Dict[str, Any]:
    """
    Analizza lo storico dell'utente per dedurre preferenze esplicite e implicite
    (Generi preferiti/vietati, registi, durata, etc.).
    """
    # Filtri base
    liked_items = user_train[user_train["rating"] >= 4.0]
    if liked_items.empty:
        liked_items = user_train # Fallback su tutto lo storico se non ci sono like forti

    liked_meta = liked_items.merge(movies_meta, on="movie_id", how="inner")
    
    disliked_items = user_train[user_train["rating"] <= 2.0]
    disliked_meta = disliked_items.merge(movies_meta, on="movie_id", how="inner")

    if liked_meta.empty:
        return {}

    # Definizione Colonne
    metadata_cols = {
        "movie_id", "movie_title", "runtime", "director", "awards", 
        "release_date", "video_release_date", "IMDb_URL", "unknown", ""
    }
    genre_cols = [c for c in movies_meta.columns if c not in metadata_cols]

    # 1. Top Genres
    top_genres = []
    if genre_cols:
        genre_sums = liked_meta[genre_cols].sum().sort_values(ascending=False)
        top_genres = genre_sums[genre_sums > 0].head(3).index.tolist()

    # 2. Forbidden Genres
    forbidden_genres = []
    if genre_cols and not disliked_meta.empty:
        dislike_sums = disliked_meta[genre_cols].sum().sort_values(ascending=False)
        candidates = dislike_sums[dislike_sums > 0].head(3).index.tolist()
        # Un genere non può essere vietato se è tra i top dell'utente
        forbidden_genres = [g for g in candidates if g not in top_genres]

    # 3. Runtime
    avg_runtime = liked_meta["runtime"].mean()

    # 4. Directors
    top_directors = _extract_top_directors(liked_meta)

    # 5. Anno Minimo (10° percentile)
    min_year = 1980
    if "release_date" in liked_meta.columns:
        years = pd.to_datetime(liked_meta["release_date"], errors="coerce").dt.year
        if not years.dropna().empty:
            min_year = int(years.quantile(0.1))

    return {
        "generi_desiderati": top_genres,
        "generi_vietati": forbidden_genres,
        "preferred_runtime": int(avg_runtime) if pd.notna(avg_runtime) else None,
        "tolleranza_runtime": 20,
        "favorite_directors": top_directors,
        "prefer_award_winning": True,
        "min_release_year": min_year,
    }

def _extract_top_directors(liked_meta: pd.DataFrame) -> List[str]:
    """Helper per estrarre registi ricorrenti."""
    if "director" not in liked_meta.columns:
        return []
    
    dir_counts = liked_meta["director"].value_counts()
    # Priorità a chi appare più di una volta
    top_dirs = dir_counts[dir_counts > 1].head(3).index.tolist()
    
    if not top_dirs and not dir_counts.empty:
        top_dirs = dir_counts.head(1).index.tolist()
        
    return top_dirs