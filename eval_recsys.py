import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
from typing import Dict, Any

# ========================
# 1. Configurazione
# ========================
class Config:
    # Pesi Ibridi
    ALPHA_CF = 0.5
    TEMPERATURE = 2.0
    TOP_K = 10
    RANDOM_SEED = 42
    
    # Debug
    VERBOSE_USERS = 3  # Quanti utenti stampare nel dettaglio

    # Pesi Constraint
    AWARD_WEIGHT = 0.5
    DIRECTOR_WEIGHT = 0.5
    RUNTIME_WEIGHT = 0.3
    RUNTIME_OUTSIDE_MALUS = 0.5
    MISSING_RUNTIME_MALUS = 0.1
    FORBIDDEN_GENRE_MALUS = 5.0
    MISSING_YEAR_MALUS = 0.1
    YEAR_BELOW_MALUS_PER_YEAR = 0.05
    
    # SVD
    SVD_COMPONENTS = 30

# ========================
# 2. Funzioni Core
# ========================

def infer_user_prefs(user_train: pd.DataFrame, movies_meta: pd.DataFrame) -> Dict[str, Any]:
    """Deduce le preferenze dai film con rating >= 4.0 nel train set."""
    liked_items = user_train[user_train["rating"] >= 4.0]
    if liked_items.empty:
        liked_items = user_train

    liked_meta = liked_items.merge(movies_meta, on="movie_id", how="inner")
    if liked_meta.empty:
        return {}

    # Generi (se colonne presenti)
    genre_cols = [c for c in movies_meta.columns if c not in ["movie_id", "movie_title", "runtime", "director", "awards", "release_date"]]
    top_genres = []
    if genre_cols:
        genre_sums = liked_meta[genre_cols].sum().sort_values(ascending=False)
        top_genres = genre_sums[genre_sums > 0].head(3).index.tolist()

    # Runtime
    avg_runtime = liked_meta["runtime"].mean()

    # Registi
    top_directors = []
    if "director" in liked_meta.columns:
        top_directors = liked_meta["director"].value_counts().head(2).index.tolist()

    return {
        "generi_desiderati": top_genres,
        "preferred_runtime": int(avg_runtime) if pd.notna(avg_runtime) else None,
        "tolleranza_runtime": 20,
        "favorite_directors": top_directors,
        "prefer_award_winning": True,
        "min_release_year": 1980
    }

def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scores = np.nan_to_num(scores, nan=0.0)
    if temperature <= 0: temperature = 1.0
    x = scores - np.max(scores)
    exp_x = np.exp(x / temperature)
    sum_exp_x = np.sum(exp_x)
    if sum_exp_x == 0:
        return np.ones_like(scores) / len(scores)
    return exp_x / sum_exp_x

# ========================
# 3. Classe Evaluator
# ========================

class HybridEvaluator:
    def __init__(self, ratings_path: Path, movies_path: Path):
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        self.movies.columns = [c.strip() for c in self.movies.columns]
        
        # Gestione colonne mancanti
        for col, val in [("awards", 0), ("director", "Unknown"), ("runtime", 90)]:
            if col not in self.movies.columns: self.movies[col] = val
            
        self.item_corr_matrix = None
        self.movie_id_to_idx = {}
        
        print(f"Loaded {len(self.ratings)} ratings and {len(self.movies)} movies.")

    def split_stratified(self, test_ratio=0.2):
        print("Eseguendo Stratified Split 80/20...")
        user_counts = self.ratings['user_id'].value_counts()
        valid_users = user_counts[user_counts >= 5].index
        
        valid_ratings = self.ratings[self.ratings['user_id'].isin(valid_users)]
        sparse_ratings = self.ratings[~self.ratings['user_id'].isin(valid_users)]
        
        train_valid, test = train_test_split(
            valid_ratings, 
            test_size=test_ratio, 
            stratify=valid_ratings['user_id'],
            random_state=Config.RANDOM_SEED
        )
        train = pd.concat([train_valid, sparse_ratings])
        print(f"Train size: {len(train)}, Test size: {len(test)}")
        return train, test

    def fit_svd(self, train_df: pd.DataFrame):
        print("Training SVD on Train Set...")
        utility_matrix = train_df.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0)
        
        self.movie_ids_in_svd = utility_matrix.columns
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids_in_svd)}
        
        X = utility_matrix.T
        n_comp = min(Config.SVD_COMPONENTS, X.shape[1]-1)
        svd = TruncatedSVD(n_components=n_comp, random_state=Config.RANDOM_SEED)
        Z = svd.fit_transform(X)
        self.item_corr_matrix = np.corrcoef(Z)
        print("SVD Training Complete.")

    def calculate_constraint_score(self, prefs: Dict, all_movies: pd.DataFrame) -> pd.Series:
        df = all_movies.set_index("movie_id").copy()
        score = pd.Series(0.0, index=df.index)
        
        if prefs.get("generi_desiderati"):
            valid_cols = [c for c in prefs["generi_desiderati"] if c in df.columns]
            if valid_cols: score += df[valid_cols].sum(axis=1)

        if prefs.get("preferred_runtime"):
            target = prefs["preferred_runtime"]
            tol = prefs.get("tolleranza_runtime", 15)
            runtimes = pd.to_numeric(df['runtime'], errors='coerce').fillna(target)
            diff = (runtimes - target).abs()
            score += (diff <= tol).astype(float) * Config.RUNTIME_WEIGHT
            score -= ((diff - tol).clip(lower=0) / tol).clip(upper=3) * Config.RUNTIME_OUTSIDE_MALUS

        if prefs.get("prefer_award_winning") and "awards" in df.columns:
             is_awarded = pd.to_numeric(df['awards'], errors='coerce').fillna(0) > 0
             score += is_awarded.astype(float) * Config.AWARD_WEIGHT

        return score

    def evaluate_user(self, user_id, train_subset, test_subset, verbose=False) -> Dict:
        # 1. Infer Prefs
        prefs = infer_user_prefs(train_subset, self.movies)
        
        # 2. Find Seed Movie (CF)
        top_rated = train_subset.sort_values("rating", ascending=False)
        if top_rated.empty: return None
        seed_movie_id = top_rated.iloc[0]["movie_id"]
        
        # Info per il verbose
        seed_title = "Unknown"
        seed_row = self.movies[self.movies.movie_id == seed_movie_id]
        if not seed_row.empty:
            seed_title = seed_row.iloc[0]['movie_title']

        # 3. Constraint Scores
        constraint_scores = self.calculate_constraint_score(prefs, self.movies)
        
        # 4. CF Scores
        cf_scores = pd.Series(0.0, index=constraint_scores.index)
        if seed_movie_id in self.movie_id_to_idx:
            idx = self.movie_id_to_idx[seed_movie_id]
            sim_vector = self.item_corr_matrix[idx]
            sim_series = pd.Series(sim_vector, index=self.movie_ids_in_svd)
            cf_scores = cf_scores.add(sim_series, fill_value=0)
        
        # 5. Hybrid & Softmax
        hybrid_scores = constraint_scores + (Config.ALPHA_CF * cf_scores)
        
        # Mask already seen
        already_seen = train_subset["movie_id"].unique()
        mask_unseen = ~hybrid_scores.index.isin(already_seen)
        valid_scores = hybrid_scores[mask_unseen]
        
        # Calcolo probabilità su tutto il set valido
        probs = softmax(valid_scores.values, temperature=Config.TEMPERATURE)
        probs_series = pd.Series(probs, index=valid_scores.index)
        
        # 6. Recommendation (Top K Deterministic)
        recs_idx = np.argsort(probs)[::-1][:Config.TOP_K]
        rec_movie_ids = valid_scores.index[recs_idx].values
        
        # 7. Ground Truth (Test Set Positives)
        # Consideriamo "Positivi" i film nel test set con rating >= 3
        positives_test = test_subset[test_subset['rating'] >= 3.0]
        positives_ids = positives_test['movie_id'].values
        
        if len(positives_ids) == 0:
            return None

        # --- Calcolo MSE (Mean Squared Error) ---
        # Confrontiamo la distribuzione predetta sui candidati (Top K + Positivi Test)
        # Pool di valutazione = I Top K raccomandati UNION i Veri Positivi
        eval_pool = list(set(rec_movie_ids) | set(positives_ids))
        
        # Vettori per MSE
        y_true = []
        y_pred = []
        
        for mid in eval_pool:
            # Pred: probabilità assegnata dal modello
            p = probs_series.get(mid, 0.0)
            y_pred.append(p)
            
            # True: 1.0 se è un positivo del test set (idealmente avremmo raccomandato questo)
            # Normalizziamo a 1/N_positives così la somma è 1.0 come la softmax? 
            # Oppure 1.0 secco? Usiamo 1/N per confrontare densità.
            t = 1.0 / len(positives_ids) if mid in positives_ids else 0.0
            y_true.append(t)
            
        mse = np.mean(np.square(np.array(y_true) - np.array(y_pred)))
        
        # --- Metriche Classiche ---
        hits = len(set(rec_movie_ids) & set(positives_ids))
        precision = hits / Config.TOP_K
        recall = hits / len(positives_ids)
        
        # --- Verbose Print ---
        if verbose:
            print(f"\n--- User {user_id} Analysis ---")
            print(f"Seed Movie (Best in Train): {seed_title} (ID: {seed_movie_id})")
            print(f"Inferred Prefs: Genres={prefs.get('generi_desiderati')}, Runtime={prefs.get('preferred_runtime')}")
            print(f"Test Set Positives ({len(positives_ids)}): {positives_ids}")
            print(f"Top 3 Recs: {rec_movie_ids[:3]} (Probs: {probs[recs_idx][:3].round(4)})")
            print(f"MSE: {mse:.6f} | Precision: {precision:.2f}")

        return {
            "precision": precision,
            "recall": recall,
            "mse": mse,
            "hits": hits
        }

    def run_evaluation(self):
        train, test = self.split_stratified()
        self.fit_svd(train)
        
        metrics = []
        test_users = test['user_id'].unique()
        
        print(f"Starting evaluation on {len(test_users)} users...")
        
        for i, user_id in enumerate(test_users):
            u_train = train[train['user_id'] == user_id]
            u_test = test[test['user_id'] == user_id]
            
            # Attiva verbose solo per i primi N utenti
            is_verbose = i < Config.VERBOSE_USERS
            
            res = self.evaluate_user(user_id, u_train, u_test, verbose=is_verbose)
            if res:
                metrics.append(res)
                
        df_m = pd.DataFrame(metrics)
        print("\n=== Hybrid Recommendation Results ===")
        print(f"Mean MSE:           {df_m['mse'].mean():.6f}")
        print(f"Mean Precision@{Config.TOP_K}: {df_m['precision'].mean():.4f}")
        print(f"Mean Recall@{Config.TOP_K}:    {df_m['recall'].mean():.4f}")
        print(f"Users Evaluated: {len(df_m)}")

# ========================
# 4. Main Execution
# ========================
if __name__ == "__main__":
    data_dir = Path("../data") 
    
    # Dummy data generation (Rimuovi se hai i file veri)
    if not (data_dir / "ratings.csv").exists():
        print("WARNING: Creating dummy data.")
        data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "user_id": np.random.randint(1, 10, 200),
            "movie_id": np.random.randint(1, 20, 200),
            "rating": np.random.randint(3, 6, 200), # High ratings for easier debug
            "timestamp": range(200)
        }).to_csv(data_dir / "ratings.csv", index=False)
        
        pd.DataFrame({
            "movie_id": range(1, 21),
            "movie_title": [f"Movie {i}" for i in range(1, 21)],
            "runtime": np.random.randint(80, 180, 20),
            "Action": np.random.randint(0, 2, 20),
            "Drama": np.random.randint(0, 2, 20)
        }).to_csv(data_dir / "movies.csv", index=False)

    evaluator = HybridEvaluator(data_dir / "ratings.csv", data_dir / "movies.csv")
    evaluator.run_evaluation()