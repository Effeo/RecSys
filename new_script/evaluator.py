# recsys_core/evaluator.py
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from config import Config
from profiling import infer_user_prefs
from utils import softmax

class HybridEvaluator:
    def __init__(self, config: Config, ratings_path: Path, movies_path: Path):
        self.config = config
        self.item_pop_log = {} 
        self.item_probs = {}   
        self.genre_cols = []
        
        self._load_data(ratings_path, movies_path)
        
        self.item_corr_matrix = None
        self.movie_id_to_idx = {}
        self.movie_ids_in_svd = None
        self.global_mean = 0.0
        self.item_biases = None
        self.user_biases = None

    def _load_data(self, ratings_path: Path, movies_path: Path):
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        
        self.movies.columns = [c.strip() for c in self.movies.columns]
        defaults = {"awards": 0, "director": "Unknown", "runtime": 90}
        for col, val in defaults.items():
            if col not in self.movies.columns:
                self.movies[col] = val
        
        # Casting ID
        self.movies["movie_id"] = pd.to_numeric(self.movies["movie_id"], errors='coerce').fillna(0).astype(int)
        self.ratings["movie_id"] = pd.to_numeric(self.ratings["movie_id"], errors='coerce').fillna(0).astype(int)

        metadata_cols = {
            "movie_id", "movie_title", "runtime", "director", "awards", 
            "release_date", "video_release_date", "IMDb_URL", "unknown", 
            "", "rating_norm", "Unnamed: 0"
        }
        self.genre_cols = [c for c in self.movies.columns if c not in metadata_cols]
        for col in self.genre_cols:
             self.movies[col] = pd.to_numeric(self.movies[col], errors='coerce').fillna(0)

        print(f"[{self.config.name}] Loaded {len(self.ratings)} ratings and {len(self.movies)} movies.")

    def fit(self, train_df: pd.DataFrame):
        print("Training SVD...")
        total_interactions = len(train_df)
        counts = train_df["movie_id"].value_counts()
        
        self.item_probs = (counts / total_interactions).to_dict()
        
        pop_log = np.log1p(counts)
        max_pop = pop_log.max()
        if max_pop > 0:
            pop_log = pop_log / max_pop
        self.item_pop_log = pop_log.to_dict()

        normalized_df = self._normalize_urm(train_df)
        utility_matrix = normalized_df.pivot_table(
            index="user_id", columns="movie_id", values="rating_norm", fill_value=0
        )

        self.movie_ids_in_svd = utility_matrix.columns
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids_in_svd)}

        X = utility_matrix.T
        n_comp = min(self.config.svd_components, X.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=self.config.random_seed)
        Z = svd.fit_transform(X)
        self.item_corr_matrix = np.corrcoef(Z)
        print("SVD Training Complete.")

    def _normalize_urm(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        C = self.config.shrink_term
        self.global_mean = df["rating"].mean()
        df["rating_norm"] = df["rating"] - self.global_mean
        item_stats = df.groupby("movie_id")["rating_norm"].agg(['sum', 'count'])
        self.item_biases = item_stats['sum'] / (item_stats['count'] + C)
        df["rating_norm"] -= df["movie_id"].map(self.item_biases).fillna(0)
        user_stats = df.groupby("user_id")["rating_norm"].agg(['sum', 'count'])
        self.user_biases = user_stats['sum'] / (user_stats['count'] + C)
        df["rating_norm"] -= df["user_id"].map(self.user_biases).fillna(0)
        return df

    def predict_user(self, user_train: pd.DataFrame) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
        debug_info = {}
        prefs = infer_user_prefs(user_train, self.movies)
        debug_info["prefs"] = prefs.get("generi_desiderati", [])
        
        if not prefs:
            return None, debug_info

        constraint_scores = self._calculate_constraint_score(prefs)
        cf_scores, seed_info = self._calculate_cf_score(user_train, constraint_scores.index)
        debug_info.update(seed_info)

        hybrid_scores = constraint_scores + (self.config.alpha_cf * cf_scores)

        already_seen = user_train["movie_id"].unique()
        mask_unseen = ~hybrid_scores.index.isin(already_seen)
        valid_scores = hybrid_scores[mask_unseen]

        probs = softmax(valid_scores.values, temperature=self.config.temperature)
        return pd.Series(probs, index=valid_scores.index), debug_info

    def _calculate_cf_score(self, user_train: pd.DataFrame, index_template: pd.Index) -> Tuple[pd.Series, Dict]:
        cf_scores = pd.Series(0.0, index=index_template)
        seed_info = {"seed_id": None, "seed_title": "Unknown"}
        top_rated = user_train.sort_values("rating", ascending=False)
        if top_rated.empty:
            return cf_scores, seed_info

        seed_movie_id = top_rated.iloc[0]["movie_id"]
        seed_info["seed_id"] = seed_movie_id
        seed_row = self.movies[self.movies.movie_id == seed_movie_id]
        if not seed_row.empty:
            seed_info["seed_title"] = seed_row.iloc[0]["movie_title"]

        if seed_movie_id in self.movie_id_to_idx:
            idx = self.movie_id_to_idx[seed_movie_id]
            sim_vector = self.item_corr_matrix[idx]
            sim_series = pd.Series(sim_vector, index=self.movie_ids_in_svd)
            cf_scores = cf_scores.add(sim_series, fill_value=0)
        return cf_scores, seed_info

    def _calculate_constraint_score(self, prefs: Dict) -> pd.Series:
        df = self.movies.set_index("movie_id")
        score = pd.Series(0.0, index=df.index)

        if prefs.get("generi_desiderati"):
            valid_cols = [c for c in prefs["generi_desiderati"] if c in df.columns]
            if valid_cols:
                score += df[valid_cols].sum(axis=1)

        if prefs.get("generi_vietati"):
            bad_cols = [c for c in prefs["generi_vietati"] if c in df.columns]
            if bad_cols:
                has_forbidden = df[bad_cols].sum(axis=1) > 0
                score -= has_forbidden.astype(float) * self.config.forbidden_genre_malus
        
        if prefs.get("preferred_runtime"):
            self._apply_runtime_scoring(score, df, prefs)

        if prefs.get("prefer_award_winning") and "awards" in df.columns:
            is_awarded = pd.to_numeric(df["awards"], errors="coerce").fillna(0) > 0
            score += is_awarded.astype(float) * self.config.award_weight

        favorite_directors = prefs.get("favorite_directors", [])
        if favorite_directors and "director" in df.columns:
            score += df["director"].isin(favorite_directors).astype(float) * self.config.director_weight

        if "release_date" in df.columns:
             self._apply_year_scoring(score, df, prefs)

        if self.config.use_popularity_bias:
            pop_vector = score.index.map(self.item_pop_log).fillna(0)
            score += pop_vector * self.config.popularity_weight

        return score

    def _apply_runtime_scoring(self, score_series, movies_df, prefs):
        target = prefs["preferred_runtime"]
        tol = prefs.get("tolleranza_runtime", 15)
        runtimes = pd.to_numeric(movies_df["runtime"], errors="coerce")
        missing_mask = runtimes.isna() | (runtimes == 0)
        score_series -= missing_mask.astype(float) * self.config.missing_runtime_malus
        runtimes = runtimes.fillna(target)
        diff = (runtimes - target).abs()
        score_series += (diff <= tol).astype(float) * self.config.runtime_weight
        malus = ((diff - tol).clip(lower=0) / tol).clip(upper=3) * self.config.runtime_outside_malus
        score_series -= malus

    def _apply_year_scoring(self, score_series, movies_df, prefs):
        years = pd.to_datetime(movies_df["release_date"], errors="coerce").dt.year
        score_series -= years.isna().astype(float) * self.config.missing_year_malus
        min_year = prefs.get("min_release_year", 1980)
        years_filled = years.fillna(min_year)
        years_diff = (min_year - years_filled).clip(lower=0)
        score_series -= years_diff * self.config.year_below_malus_per_year

    def evaluate(self, limit_users: Optional[int] = None) -> pd.DataFrame:
        train, test = self._split_stratified()
        self.fit(train)
        metrics = []
        test_users = test["user_id"].unique()
        
        if limit_users:
            print(f"Limiting evaluation to first {limit_users} users (Test Mode).")
            test_users = test_users[:limit_users]

        print(f"Starting evaluation on {len(test_users)} users...")
        for i, user_id in enumerate(test_users):
            u_train = train[train["user_id"] == user_id]
            u_test = test[test["user_id"] == user_id]
            probs_series, debug_info = self.predict_user(u_train)
            if probs_series is None: continue
            
            is_verbose = i < self.config.verbose_users
            res = self._compute_metrics(probs_series, u_test, user_id, debug_info, verbose=is_verbose)
            if res: metrics.append(res)

        return pd.DataFrame(metrics)

    def _compute_metrics(self, probs_series: pd.Series, u_test: pd.DataFrame, user_id: int, debug_info: Dict, verbose: bool) -> Dict:
        # --- LOGICA PARAMETRICA ---
        if self.config.use_probabilistic_sampling:
            # Recuperiamo il parametro pool size dalla config
            pool_size = self.config.sampling_top_k_pool
            
            # Se è None o enorme, usa tutto il dataset (come nel tuo run fallimentare)
            if pool_size is None or pool_size >= len(probs_series):
                pool_probs = probs_series
            else:
                # Top-K Sampling (Nucleus)
                pool_probs = probs_series.nlargest(pool_size)
            
            # Rinormalizzazione sicura
            probs_values = pool_probs.values.astype('float64')
            if probs_values.sum() == 0: 
                probs_values = np.ones_like(probs_values) / len(probs_values)
            else:
                probs_values /= probs_values.sum()
            
            k = min(self.config.top_k, len(pool_probs))
            rec_movie_ids = np.random.choice(pool_probs.index, size=k, replace=False, p=probs_values)
        else:
            # Deterministic Argmax
            recs_idx = np.argsort(probs_series.values)[::-1][: self.config.top_k]
            rec_movie_ids = probs_series.index[recs_idx].values
        # --------------------------

        positives_ids = u_test[u_test["rating"] >= 3.0]["movie_id"].values
        if len(positives_ids) == 0: return {}

        hits = len(set(rec_movie_ids) & set(positives_ids))
        precision = hits / self.config.top_k
        recall = hits / len(positives_ids)
        
        eval_pool = list(set(rec_movie_ids) | set(positives_ids))
        y_true = np.array([1.0/len(positives_ids) if mid in positives_ids else 0.0 for mid in eval_pool])
        y_pred = np.array([probs_series.get(mid, 0.0) for mid in eval_pool])
        
        mse = np.mean(np.square(y_true - y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        novelty = self._calculate_novelty(rec_movie_ids)
        diversity = self._calculate_diversity(rec_movie_ids)

        if verbose:
            print(f"\n--- User {user_id} ---")
            print(f"Seed: {debug_info.get('seed_title')}")
            print(f"Top 3: {rec_movie_ids[:3]}") 
            print(f"Prec: {precision:.2f} | Nov: {novelty:.2f} | Div: {diversity:.2f}")

        return {
            "precision": precision, "recall": recall, "mse": mse, "mae": mae, 
            "hits": hits, "novelty": novelty, "diversity": diversity
        }

    def _calculate_novelty(self, rec_movie_ids):
        if len(rec_movie_ids) == 0: return 0.0
        return np.mean([-math.log2(self.item_probs.get(mid, 1e-10)) for mid in rec_movie_ids])

    def _calculate_diversity(self, rec_movie_ids):
        if len(rec_movie_ids) == 0 or not self.genre_cols: return 0.0
        unique = set()
        recs = self.movies[self.movies["movie_id"].isin(rec_movie_ids)]
        for _, row in recs.iterrows():
            for g in self.genre_cols:
                if row[g] > 0: unique.add(g)
        return len(unique) / len(rec_movie_ids)

    def _split_stratified(self, test_ratio=0.2):
        print("Splitting Data...")
        user_counts = self.ratings["user_id"].value_counts()
        valid_users = user_counts[user_counts >= 5].index
        valid_ratings = self.ratings[self.ratings["user_id"].isin(valid_users)]
        sparse_ratings = self.ratings[~self.ratings["user_id"].isin(valid_users)]
        train_valid, test = train_test_split(valid_ratings, test_size=test_ratio, stratify=valid_ratings["user_id"], random_state=self.config.random_seed)
        return pd.concat([train_valid, sparse_ratings]), test