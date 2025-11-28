# NO REFACTOR
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# ========================
# 1. Configurazione
# ========================
@dataclass
class Config:
    name: str = ""
    alpha_cf: float = 0.5
    temperature: float = 2.0
    top_k: int = 10
    random_seed: int = 42
    verbose_users: int = 0
    award_weight: float = 0.5
    director_weight: float = 0.5
    runtime_weight: float = 0.3
    runtime_outside_malus: float = 0.5
    missing_runtime_malus: float = 0.1
    forbidden_genre_malus: float = 5.0
    missing_year_malus: float = 0.1
    year_below_malus_per_year: float = 0.05
    svd_components: int = 30
    shrink_term: int = 5  # Costante C per lo shrink term

# ========================
# 2. Funzioni Core
# ========================

def infer_user_prefs(
    user_train: pd.DataFrame, movies_meta: pd.DataFrame
) -> Dict[str, Any]:
    """
    Deduce le preferenze dell'utente basandosi sui rating del training set.
    """
    liked_items = user_train[user_train["rating"] >= 4.0]
    if liked_items.empty:
        liked_items = user_train  # Fallback

    liked_meta = liked_items.merge(movies_meta, on="movie_id", how="inner")
    
    disliked_items = user_train[user_train["rating"] <= 2.0]
    disliked_meta = disliked_items.merge(movies_meta, on="movie_id", how="inner")

    if liked_meta.empty:
        return {}

    # Setup Colonne Generi (Escludiamo metadati descrittivi)
    metadata_cols = ["movie_id", "movie_title", "runtime", "director", "awards", 
                     "release_date", "video_release_date", "IMDb_URL", "unknown", ""]
    genre_cols = [c for c in movies_meta.columns if c not in metadata_cols]

    # A. Top Genres
    top_genres = []
    if genre_cols:
        genre_sums = liked_meta[genre_cols].sum().sort_values(ascending=False)
        top_genres = genre_sums[genre_sums > 0].head(3).index.tolist()

    # B. Forbidden Genres
    forbidden_genres = []
    if genre_cols and not disliked_meta.empty:
        dislike_sums = disliked_meta[genre_cols].sum().sort_values(ascending=False)
        candidates = dislike_sums[dislike_sums > 0].head(3).index.tolist()
        forbidden_genres = [g for g in candidates if g not in top_genres]

    # C. Runtime
    avg_runtime = liked_meta["runtime"].mean()

    # D. Directors
    top_directors = []
    if "director" in liked_meta.columns:
        dir_counts = liked_meta["director"].value_counts()
        top_directors = dir_counts[dir_counts > 1].head(3).index.tolist()
        if not top_directors:
            top_directors = dir_counts.head(1).index.tolist()

    # E. Anno Minimo
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

def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scores = np.nan_to_num(scores, nan=0.0)
    if temperature <= 0:
        temperature = 1.0
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
    def __init__(self, config: Config, ratings_path: Path, movies_path: Path):
        self.config = config
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        self.movies.columns = [c.strip() for c in self.movies.columns]

        # Preprocessing Dataframe Film
        for col, val in [("awards", 0), ("director", "Unknown"), ("runtime", 90)]:
            if col not in self.movies.columns:
                self.movies[col] = val

        self.item_corr_matrix = None
        self.movie_id_to_idx = {}
        
        # Variabili per i bias (Normalizzazione)
        self.global_mean = 0.0
        self.item_biases = None
        self.user_biases = None

        print(f"[{self.config.name}] Loaded {len(self.ratings)} ratings and {len(self.movies)} movies.")

    def split_stratified(self, test_ratio=0.2):
        print("Eseguendo Stratified Split 80/20...")
        user_counts = self.ratings["user_id"].value_counts()
        valid_users = user_counts[user_counts >= 5].index

        valid_ratings = self.ratings[self.ratings["user_id"].isin(valid_users)]
        sparse_ratings = self.ratings[~self.ratings["user_id"].isin(valid_users)]

        train_valid, test = train_test_split(
            valid_ratings,
            test_size=test_ratio,
            stratify=valid_ratings["user_id"],
            random_state=self.config.random_seed,
        )
        train = pd.concat([train_valid, sparse_ratings])
        print(f"Train size: {len(train)}, Test size: {len(test)}")
        return train, test

    def normalize_urm(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Implementa la normalizzazione della URM.
        Obiettivo: Rimuovere bias globali, bias utente e bias item per isolare il 'gusto reale'.
        
        Formula: Residual = r_ui - (global_avg + item_bias + user_bias)
        """
        df = train_df.copy()
        C = self.config.shrink_term

        # Step 1: Media Globale (Slide 24)
        self.global_mean = df["rating"].mean()
        
        # Step 2: Sottrazione media globale
        df["rating_norm"] = df["rating"] - self.global_mean

        # Step 3: Calcolo Item Bias con Shrinkage (Slide 25)
        # b_j = Sum(r_norm) / (N_j + C)
        item_stats = df.groupby("movie_id")["rating_norm"].agg(['sum', 'count'])
        self.item_biases = item_stats['sum'] / (item_stats['count'] + C)
        
        # Step 4: Sottrazione Item Bias
        df["rating_norm"] = df.apply(
            lambda x: x["rating_norm"] - self.item_biases.get(x["movie_id"], 0.0), axis=1
        )

        # Step 5: Calcolo User Bias con Shrinkage (Slide 25)
        # b_i = Sum(r_norm_residuo) / (N_i + C)
        user_stats = df.groupby("user_id")["rating_norm"].agg(['sum', 'count'])
        self.user_biases = user_stats['sum'] / (user_stats['count'] + C)

        # Step 6: Sottrazione User Bias per ottenere il residuo finale
        df["rating_norm"] = df.apply(
            lambda x: x["rating_norm"] - self.user_biases.get(x["user_id"], 0.0), axis=1
        )
        
        print("Normalizzazione URM completata (Global -> Item -> User bias removal).")
        return df

    def fit_svd(self, train_df: pd.DataFrame):
        print("Training SVD on Normalized Residuals...")
        
        # 1. Normalizzazione (Preprocessing)
        normalized_df = self.normalize_urm(train_df)

        # 2. Creazione matrice di utilità sui RESIDUI, non sui rating grezzi.
        # fill_value=0 ha senso perché 0 ora rappresenta "in linea con l'aspettativa (bias)"
        utility_matrix = normalized_df.pivot_table(
            index="user_id", columns="movie_id", values="rating_norm", fill_value=0
        )

        self.movie_ids_in_svd = utility_matrix.columns
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids_in_svd)}

        X = utility_matrix.T
        n_comp = min(self.config.svd_components, X.shape[1] - 1)
        
        svd = TruncatedSVD(n_components=n_comp, random_state=self.config.random_seed)
        Z = svd.fit_transform(X)
        
        # Calcolo similarità item-item nello spazio latente dei residui
        self.item_corr_matrix = np.corrcoef(Z)
        print("SVD Training Complete.")

    def calculate_constraint_score(self, prefs: Dict, all_movies: pd.DataFrame) -> pd.Series:
        df = all_movies.set_index("movie_id").copy()
        score = pd.Series(0.0, index=df.index)

        # 1. Generi Desiderati
        if prefs.get("generi_desiderati"):
            valid_cols = [c for c in prefs["generi_desiderati"] if c in df.columns]
            if valid_cols:
                score += df[valid_cols].sum(axis=1)

        # 2. Generi Vietati
        if prefs.get("generi_vietati"):
            bad_cols = [c for c in prefs["generi_vietati"] if c in df.columns]
            if bad_cols:
                has_forbidden = df[bad_cols].sum(axis=1) > 0
                score -= has_forbidden.astype(float) * self.config.forbidden_genre_malus

        # 3. Runtime
        if prefs.get("preferred_runtime"):
            target = prefs["preferred_runtime"]
            tol = prefs.get("tolleranza_runtime", 15)
            runtimes = pd.to_numeric(df["runtime"], errors="coerce")
            
            missing_mask = runtimes.isna() | (runtimes == 0)
            score -= missing_mask.astype(float) * self.config.missing_runtime_malus
            
            runtimes = runtimes.fillna(target)
            diff = (runtimes - target).abs()
            score += (diff <= tol).astype(float) * self.config.runtime_weight
            score -= ((diff - tol).clip(lower=0) / tol).clip(upper=3) * self.config.runtime_outside_malus

        # 4. Awards
        if prefs.get("prefer_award_winning") and "awards" in df.columns:
            is_awarded = pd.to_numeric(df["awards"], errors="coerce").fillna(0) > 0
            score += is_awarded.astype(float) * self.config.award_weight

        # 5. Registi
        favorite_directors = prefs.get("favorite_directors", [])
        if favorite_directors and "director" in df.columns:
            is_fav_director = df["director"].isin(favorite_directors)
            score += is_fav_director.astype(float) * self.config.director_weight

        # 6. Anno
        if "release_date" in df.columns:
            movie_years = pd.to_datetime(df["release_date"], errors="coerce").dt.year
            missing_year_mask = movie_years.isna()
            score -= missing_year_mask.astype(float) * self.config.missing_year_malus
            
            min_year = prefs.get("min_release_year", 1980)
            years_filled = movie_years.fillna(min_year)
            years_diff = (min_year - years_filled).clip(lower=0)
            score -= years_diff * self.config.year_below_malus_per_year

        return score

    def evaluate_user(self, user_id, train_subset, test_subset, verbose=False) -> Dict:
        # 1. Infer Prefs
        prefs = infer_user_prefs(train_subset, self.movies)

        # 2. Seed Movie (CF)
        top_rated = train_subset.sort_values("rating", ascending=False)
        if top_rated.empty:
            return None
        seed_movie_id = top_rated.iloc[0]["movie_id"]

        # 3. Constraint Scores
        constraint_scores = self.calculate_constraint_score(prefs, self.movies)

        # 4. CF Scores (su similarità calcolata dai residui)
        cf_scores = pd.Series(0.0, index=constraint_scores.index)
        if seed_movie_id in self.movie_id_to_idx:
            idx = self.movie_id_to_idx[seed_movie_id]
            sim_vector = self.item_corr_matrix[idx]
            sim_series = pd.Series(sim_vector, index=self.movie_ids_in_svd)
            cf_scores = cf_scores.add(sim_series, fill_value=0)

        # 5. Ibridazione
        hybrid_scores = constraint_scores + (self.config.alpha_cf * cf_scores)

        # Maschera film già visti
        already_seen = train_subset["movie_id"].unique()
        mask_unseen = ~hybrid_scores.index.isin(already_seen)
        valid_scores = hybrid_scores[mask_unseen]

        # Softmax
        probs = softmax(valid_scores.values, temperature=self.config.temperature)
        probs_series = pd.Series(probs, index=valid_scores.index)

        # 6. Raccomandazione
        recs_idx = np.argsort(probs)[::-1][: self.config.top_k]
        rec_movie_ids = valid_scores.index[recs_idx].values

        # 7. Valutazione
        positives_test = test_subset[test_subset["rating"] >= 3.0]
        positives_ids = positives_test["movie_id"].values

        if len(positives_ids) == 0:
            return None

        hits = len(set(rec_movie_ids) & set(positives_ids))
        precision = hits / self.config.top_k
        recall = hits / len(positives_ids)
        
        # MSE semplificato sulle probabilità
        eval_pool = list(set(rec_movie_ids) | set(positives_ids))
        y_true = []
        y_pred = []
        for mid in eval_pool:
            p = probs_series.get(mid, 0.0)
            y_pred.append(p)
            t = 1.0 / len(positives_ids) if mid in positives_ids else 0.0
            y_true.append(t)
        
        mse = np.mean(np.square(np.array(y_true) - np.array(y_pred)))

        if verbose:
            seed_title = "Unknown"
            seed_row = self.movies[self.movies.movie_id == seed_movie_id]
            if not seed_row.empty:
                seed_title = seed_row.iloc[0]["movie_title"]
            
            print(f"\n--- User {user_id} Analysis ---")
            print(f"Seed: {seed_title} (ID: {seed_movie_id})")
            print(f"Prefs: {prefs.get('generi_desiderati')}")
            print(f"Top 3 Recs: {rec_movie_ids[:3]}")
            print(f"Precision: {precision:.2f} | Recall: {recall:.2f}")

        return {"precision": precision, "recall": recall, "mse": mse, "hits": hits}

    def run_evaluation(self, limit_users: Optional[int] = None):
        """
        :param limit_users: Se specificato, limita il numero di utenti da valutare (per la modalità test)
        """
        train, test = self.split_stratified()
        self.fit_svd(train)

        metrics = []
        test_users = test["user_id"].unique()
        
        if limit_users:
            print(f"Limiting evaluation to first {limit_users} users (Test Mode).")
            test_users = test_users[:limit_users]

        print(f"Starting evaluation on {len(test_users)} users...")

        for i, user_id in enumerate(test_users):
            u_train = train[train["user_id"] == user_id]
            u_test = test[test["user_id"] == user_id]

            is_verbose = i < self.config.verbose_users
            res = self.evaluate_user(user_id, u_train, u_test, verbose=is_verbose)
            if res:
                metrics.append(res)

        if not metrics:
            print("No metrics collected.")
            return

        df_m = pd.DataFrame(metrics)
        print("\n=== Hybrid Recommendation Results ===")
        print(f"Mean MSE:           {df_m['mse'].mean():.6f}")
        print(f"Mean Precision@{self.config.top_k}: {df_m['precision'].mean():.4f}")
        print(f"Mean Recall@{self.config.top_k}:    {df_m['recall'].mean():.4f}")
        print(f"Users Evaluated: {len(df_m)}")

# ========================
# 4. Main Execution
# ========================
if __name__ == "__main__":
    # Gestione argomenti da riga di comando per switching run
    parser = argparse.ArgumentParser(description="Run RecSys Evaluation")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["test", "full"], 
        default="test",
        help="Mode: 'test' runs 1 config on few users, 'full' runs grid search."
    )
    args = parser.parse_args()

    data_dir = Path("../data")
    
    # Configurazioni da eseguire
    configs_to_run = []

    if args.mode == "test":
        print(">>> RUNNING IN TEST MODE (Fast) <<<")
        # Eseguiamo solo la baseline e limitiamo gli utenti
        test_config = Config(name="Test_Run_Baseline", verbose_users=5)
        configs_to_run.append(test_config)
        user_limit = 20 # Valutiamo solo 20 utenti
    else:
        print(">>> RUNNING IN FULL PRODUCTION MODE (Long) <<<")
        user_limit = None # Tutti gli utenti
        
        # --- DEFINIZIONE GRID SEARCH COMPLETA ---
        # 0. Baseline
        configs_to_run.append(Config(name="00_Baseline_Default"))
        
        # 1. Alpha Sensitivity
        configs_to_run.append(Config(name="01_Alpha_Low_0.3", alpha_cf=0.3))
        configs_to_run.append(Config(name="01_Alpha_High_0.8", alpha_cf=0.8))
        
        # 2. SVD Dimensionality
        configs_to_run.append(Config(name="02_SVD_Low_10", svd_components=10))
        configs_to_run.append(Config(name="02_SVD_High_100", svd_components=100))
        
        # 3. Strictness
        configs_to_run.append(Config(
            name="03_Strictness_Draconian", 
            forbidden_genre_malus=20.0, missing_year_malus=2.0
        ))
        
        # 4. Temperature
        configs_to_run.append(Config(name="05_Temp_Hot_5.0", temperature=5.0))

    # Loop di esecuzione
    for i, config in enumerate(configs_to_run):
        print(f"\n{'='*40}")
        print(f"Running Config {i+1}/{len(configs_to_run)}: {config.name}")
        print(f"{'='*40}")
        
        evaluator = HybridEvaluator(
            config, data_dir / "ratings.csv", data_dir / "movies_enriched.csv"
        )
        evaluator.run_evaluation(limit_users=user_limit)