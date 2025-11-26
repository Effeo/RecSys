from dataclasses import dataclass
from os import name
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split


# ========================
# 1. Configurazione
# ========================
@dataclass
class Config:

    name = ""

    def __init__(
        self,
        name,
        alpha_cf: float = 0.5,
        temperature: float = 2.0,
        top_k: int = 10,
        random_seed: int = 42,
        verbose_users: int = 0,
        award_weight: float = 0.5,
        director_weight: float = 0.5,
        runtime_weight: float = 0.3,
        runtime_outside_malus: float = 0.5,
        missing_runtime_malus: float = 0.1,
        forbidden_genre_malus: float = 5.0,
        missing_year_malus: float = 0.1,
        year_below_malus_per_year: float = 0.05,
        svd_components: int = 30,
    ):
        self.name = name
        self.alpha_cf = alpha_cf
        self.temperature = temperature
        self.top_k = top_k
        self.random_seed = random_seed
        self.verbose_users = verbose_users
        self.award_weight = award_weight
        self.director_weight = director_weight
        self.runtime_weight = runtime_weight
        self.runtime_outside_malus = runtime_outside_malus
        self.missing_runtime_malus = missing_runtime_malus
        self.forbidden_genre_malus = forbidden_genre_malus
        self.missing_year_malus = missing_year_malus
        self.year_below_malus_per_year = year_below_malus_per_year
        self.svd_components = svd_components

    # # Pesi Ibridi
    # ALPHA_CF = 0.5
    # TEMPERATURE = 2.0
    # TOP_K = 10
    # RANDOM_SEED = 42

    # # Debug
    # VERBOSE_USERS = 3  # Quanti utenti stampare nel dettaglio

    # # Pesi Constraint
    # AWARD_WEIGHT = 0.5
    # DIRECTOR_WEIGHT = 0.5
    # RUNTIME_WEIGHT = 0.3
    # RUNTIME_OUTSIDE_MALUS = 0.5
    # MISSING_RUNTIME_MALUS = 0.1
    # FORBIDDEN_GENRE_MALUS = 5.0
    # MISSING_YEAR_MALUS = 0.1
    # YEAR_BELOW_MALUS_PER_YEAR = 0.05

    # # SVD
    # SVD_COMPONENTS = 30


# ========================
# 2. Funzioni Core
# ========================


def infer_user_prefs(
    user_train: pd.DataFrame, movies_meta: pd.DataFrame
) -> Dict[str, Any]:
    """
    Versione potenziata: deduce anche i generi 'vietati' e pulisce i dati.
    """
    # 1. Base Logic per i 'Mi Piace' (Rating >= 4.0)
    liked_items = user_train[user_train["rating"] >= 4.0]
    if liked_items.empty:
        liked_items = user_train  # Fallback

    liked_meta = liked_items.merge(movies_meta, on="movie_id", how="inner")

    # 2. Logica per i 'Non Mi Piace' (Rating <= 2.0)
    disliked_items = user_train[user_train["rating"] <= 2.0]
    disliked_meta = disliked_items.merge(movies_meta, on="movie_id", how="inner")

    if liked_meta.empty:
        return {}

    # Setup Colonne Generi
    metadata_cols = [
        "movie_id",
        "movie_title",
        "runtime",
        "director",
        "awards",
        "release_date",
        "video_release_date",
        "IMDb_URL",
        "unknown",
        "",
    ]
    genre_cols = [c for c in movies_meta.columns if c not in metadata_cols]

    # A. Top Genres
    top_genres = []
    if genre_cols:
        genre_sums = liked_meta[genre_cols].sum().sort_values(ascending=False)
        top_genres = genre_sums[genre_sums > 0].head(3).index.tolist()

    # B. Forbidden Genres (Generi prevalenti nei film detestati, se non sono anche nei top)
    forbidden_genres = []
    if genre_cols and not disliked_meta.empty:
        dislike_sums = disliked_meta[genre_cols].sum().sort_values(ascending=False)
        # Consideriamo forbidden se compaiono spesso nei dislike e NON sono nei top genres
        candidates = dislike_sums[dislike_sums > 0].head(3).index.tolist()
        forbidden_genres = [g for g in candidates if g not in top_genres]

    # C. Runtime
    avg_runtime = liked_meta["runtime"].mean()

    # D. Directors
    top_directors = []
    if "director" in liked_meta.columns:
        # Prendi direttori con almeno 2 occorrenze se possibile
        dir_counts = liked_meta["director"].value_counts()
        top_directors = dir_counts[dir_counts > 1].head(3).index.tolist()
        if not top_directors:  # Fallback al top 1 assoluto
            top_directors = dir_counts.head(1).index.tolist()

    # E. Anno Minimo (Euristica: prendiamo il 10° percentile degli anni guardati)
    min_year = 1980  # Default
    if "release_date" in liked_meta.columns:
        # Conversione sicura in datetime
        years = pd.to_datetime(liked_meta["release_date"], errors="coerce").dt.year
        if not years.dropna().empty:
            min_year = int(
                years.quantile(0.1)
            )  # 10% percentile per escludere outlier vecchi

    return {
        "generi_desiderati": top_genres,
        "generi_vietati": forbidden_genres,  # NUOVO
        "preferred_runtime": int(avg_runtime) if pd.notna(avg_runtime) else None,
        "tolleranza_runtime": 20,
        "favorite_directors": top_directors,
        "prefer_award_winning": True,
        "min_release_year": min_year,  # Dinamico
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

        # Gestione colonne mancanti
        for col, val in [("awards", 0), ("director", "Unknown"), ("runtime", 90)]:
            if col not in self.movies.columns:
                self.movies[col] = val

        self.item_corr_matrix = None
        self.movie_id_to_idx = {}

        print(f"Loaded {len(self.ratings)} ratings and {len(self.movies)} movies.")

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

    def fit_svd(self, train_df: pd.DataFrame):
        print("Training SVD on Train Set...")
        utility_matrix = train_df.pivot_table(
            index="user_id", columns="movie_id", values="rating", fill_value=0
        )

        self.movie_ids_in_svd = utility_matrix.columns
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids_in_svd)}

        X = utility_matrix.T
        n_comp = min(self.config.svd_components, X.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=self.config.random_seed)
        Z = svd.fit_transform(X)
        self.item_corr_matrix = np.corrcoef(Z)
        print("SVD Training Complete.")

    def calculate_constraint_score(
        self, prefs: Dict, all_movies: pd.DataFrame
    ) -> pd.Series:
        """
        Calcolo vettorializzato di TUTTI i pesi definiti nella Config.
        """
        df = all_movies.set_index("movie_id").copy()
        score = pd.Series(0.0, index=df.index)

        # 1. Generi Desiderati (Bonus)
        if prefs.get("generi_desiderati"):
            valid_cols = [c for c in prefs["generi_desiderati"] if c in df.columns]
            if valid_cols:
                # Somma 1.0 per ogni genere matchato (semplice) o pesato
                score += df[valid_cols].sum(axis=1)

        # 2. Generi Vietati (Malus Pesante) - NUOVO
        if prefs.get("generi_vietati"):
            bad_cols = [c for c in prefs["generi_vietati"] if c in df.columns]
            if bad_cols:
                # Se ha anche solo un genere vietato, applica il malus
                has_forbidden = df[bad_cols].sum(axis=1) > 0
                score -= has_forbidden.astype(float) * self.config.forbidden_genre_malus

        # 3. Runtime (Bonus e Malus)
        if prefs.get("preferred_runtime"):
            target = prefs["preferred_runtime"]
            tol = prefs.get("tolleranza_runtime", 15)

            # Gestione valori mancanti o zero
            runtimes = pd.to_numeric(df["runtime"], errors="coerce")

            # Penalità per runtime mancante (NUOVO)
            missing_runtime_mask = runtimes.isna() | (runtimes == 0)
            score -= (
                missing_runtime_mask.astype(float) * self.config.missing_runtime_malus
            )

            # Calcolo distanza per i validi
            runtimes = runtimes.fillna(target)  # Imputazione neutra per il calcolo diff
            diff = (runtimes - target).abs()

            # Bonus se dentro tolleranza
            score += (diff <= tol).astype(float) * self.config.runtime_weight
            # Malus lineare se fuori (clippato per non distruggere film troppo lunghi/corti)
            score -= ((diff - tol).clip(lower=0) / tol).clip(
                upper=3
            ) * self.config.runtime_outside_malus

        # 4. Awards (Bonus)
        if prefs.get("prefer_award_winning") and "awards" in df.columns:
            is_awarded = pd.to_numeric(df["awards"], errors="coerce").fillna(0) > 0
            score += is_awarded.astype(float) * self.config.award_weight

        # 5. Registi (Bonus) - NUOVO
        favorite_directors = prefs.get("favorite_directors", [])
        if favorite_directors and "director" in df.columns:
            # Controlla se il regista è nella lista dei preferiti
            is_fav_director = df["director"].isin(favorite_directors)
            score += is_fav_director.astype(float) * self.config.director_weight

        # 6. Anno di Uscita (Malus Temporale) - NUOVO
        if "release_date" in df.columns:
            # Estrai anno
            movie_years = pd.to_datetime(df["release_date"], errors="coerce").dt.year

            # Penalità anno mancante
            missing_year_mask = movie_years.isna()
            score -= missing_year_mask.astype(float) * self.config.missing_year_malus

            # Penalità film troppo vecchi
            min_year = prefs.get("min_release_year", 1980)
            # Riempiamo i NaNs con min_year per non applicare doppio malus
            years_filled = movie_years.fillna(min_year)

            # Calcolo anni di scarto sotto la soglia
            years_diff = (min_year - years_filled).clip(lower=0)

            # Applica malus progressivo (es: 10 anni sotto soglia * 0.05 = -0.5 punti)
            score -= years_diff * self.config.year_below_malus_per_year

        return score

    def evaluate_user(self, user_id, train_subset, test_subset, verbose=False) -> Dict:
        # 1. Infer Prefs
        prefs = infer_user_prefs(train_subset, self.movies)

        # 2. Find Seed Movie (CF)
        top_rated = train_subset.sort_values("rating", ascending=False)
        if top_rated.empty:
            return None
        seed_movie_id = top_rated.iloc[0]["movie_id"]

        # Info per il verbose
        seed_title = "Unknown"
        seed_row = self.movies[self.movies.movie_id == seed_movie_id]
        if not seed_row.empty:
            seed_title = seed_row.iloc[0]["movie_title"]

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
        hybrid_scores = constraint_scores + (self.config.alpha_cf * cf_scores)

        # Mask already seen
        already_seen = train_subset["movie_id"].unique()
        mask_unseen = ~hybrid_scores.index.isin(already_seen)
        valid_scores = hybrid_scores[mask_unseen]

        # Calcolo probabilità su tutto il set valido
        probs = softmax(valid_scores.values, temperature=self.config.temperature)
        probs_series = pd.Series(probs, index=valid_scores.index)

        # 6. Recommendation (Top K Deterministic)
        recs_idx = np.argsort(probs)[::-1][: self.config.top_k]
        rec_movie_ids = valid_scores.index[recs_idx].values

        # 7. Ground Truth (Test Set Positives)
        # Consideriamo "Positivi" i film nel test set con rating >= 3
        positives_test = test_subset[test_subset["rating"] >= 3.0]
        positives_ids = positives_test["movie_id"].values

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
        precision = hits / self.config.top_k
        recall = hits / len(positives_ids)

        # --- Verbose Print ---
        if verbose:
            print(f"\n--- User {user_id} Analysis ---")
            print(f"Seed Movie (Best in Train): {seed_title} (ID: {seed_movie_id})")
            print(
                f"Inferred Prefs: Genres={prefs.get('generi_desiderati')}, Runtime={prefs.get('preferred_runtime')}"
            )
            print(f"Test Set Positives ({len(positives_ids)}): {positives_ids}")
            print(
                f"Top 3 Recs: {rec_movie_ids[:3]} (Probs: {probs[recs_idx][:3].round(4)})"
            )
            print(f"MSE: {mse:.6f} | Precision: {precision:.2f}")

        return {"precision": precision, "recall": recall, "mse": mse, "hits": hits}

    def run_evaluation(self):
        train, test = self.split_stratified()
        self.fit_svd(train)

        metrics = []
        test_users = test["user_id"].unique()

        print(f"Starting evaluation on {len(test_users)} users...")

        for i, user_id in enumerate(test_users):
            u_train = train[train["user_id"] == user_id]
            u_test = test[test["user_id"] == user_id]

            # Attiva verbose solo per i primi N utenti
            is_verbose = i < self.config.verbose_users

            res = self.evaluate_user(user_id, u_train, u_test, verbose=is_verbose)
            if res:
                metrics.append(res)

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
    data_dir = Path("../data")

    configs = []

    # ---------------------------------------------------------
    # GRUPPO 0: BASELINE
    # ---------------------------------------------------------
    configs.append(Config(name="00_Baseline_Default"))

    # ---------------------------------------------------------
    # GRUPPO 1: ALPHA SENSITIVITY (Hybrid Balance)
    # Testiamo lo spettro da "Solo Regole" a "Solo Matematica"
    # ---------------------------------------------------------
    # Alpha basso: comanda il Content-Based (regole)
    configs.append(Config(name="01_Alpha_Low_0.2", alpha_cf=0.2))
    configs.append(Config(name="01_Alpha_Low_0.3", alpha_cf=0.3))
    
    # Alpha medio-alto: mix bilanciato
    configs.append(Config(name="01_Alpha_Mid_0.6", alpha_cf=0.6))
    
    # Alpha alto: comanda la SVD (similarità utenti)
    configs.append(Config(name="01_Alpha_High_0.8", alpha_cf=0.8))
    configs.append(Config(name="01_Alpha_High_0.9", alpha_cf=0.9))

    # ---------------------------------------------------------
    # GRUPPO 2: SVD DIMENSIONALITY (Latent Space)
    # Testiamo la complessità del modello matematico.
    # ---------------------------------------------------------
    # Pochi componenti: generalizzazione estrema (rischio underfitting)
    configs.append(Config(name="02_SVD_VeryLow_10", svd_components=10))
    
    # Componenti medi: bilanciamento
    configs.append(Config(name="02_SVD_Mid_50", svd_components=50))
    
    # Molti componenti: cattura dettagli fini (rischio overfitting su dati scarsi)
    configs.append(Config(name="02_SVD_High_100", svd_components=100))
    configs.append(Config(name="02_SVD_VeryHigh_200", svd_components=200))

    # ---------------------------------------------------------
    # GRUPPO 3: THE "STRICTNESS" SPECTRUM (Constraint Sensitivity)
    # Quanto siamo cattivi con i film che non rispettano le regole?
    # ---------------------------------------------------------
    # LENIENT: Lasciamo passare quasi tutto, i malus sono carezze.
    configs.append(Config(
        name="03_Strictness_Lenient", 
        forbidden_genre_malus=1.0, 
        missing_year_malus=0.0,
        year_below_malus_per_year=0.01,
        runtime_outside_malus=0.1
    ))

    # STRICT: Penalità medie (standard leggermente alzato)
    configs.append(Config(
        name="03_Strictness_Medium", 
        forbidden_genre_malus=5.0, 
        year_below_malus_per_year=0.1
    ))

    # DRACONIAN: Se sbagli genere o anno, il film viene annientato.
    configs.append(Config(
        name="03_Strictness_Draconian", 
        forbidden_genre_malus=20.0,       # Malus enorme
        missing_year_malus=2.0,
        year_below_malus_per_year=0.5,    # 10 anni in meno = -5 punti (mortale)
        runtime_outside_malus=2.0
    ))

    # ---------------------------------------------------------
    # GRUPPO 4: FEATURE IMPORTANCE (Ablation Study)
    # Accendiamo una feature alla volta per vedere quale "pesa" davvero.
    # Manteniamo alpha fisso a 0.5 per fairness.
    # ---------------------------------------------------------
    # Focus sul REGISTA
    configs.append(Config(
        name="04_Focus_Director", 
        director_weight=2.0, award_weight=0.1, runtime_weight=0.1
    ))

    # Focus sui PREMI (Awards)
    configs.append(Config(
        name="04_Focus_Awards", 
        director_weight=0.1, award_weight=2.0, runtime_weight=0.1
    ))

    # Focus sulla DURATA (Runtime)
    configs.append(Config(
        name="04_Focus_Runtime", 
        director_weight=0.1, award_weight=0.1, runtime_weight=2.0
    ))

    # ---------------------------------------------------------
    # GRUPPO 5: EXPLORATION (Temperature)
    # Gestione della probabilità finale
    # ---------------------------------------------------------
    # Deterministico: prende quasi sempre solo il top score
    configs.append(Config(name="05_Temp_Cold_0.5", temperature=0.5))
    
    # Esplorativo: "appiattisce" la curva, dà chance ai secondi/terzi posti
    configs.append(Config(name="05_Temp_Hot_5.0", temperature=5.0))
    
    # Molto Esplorativo + Top K ampio
    configs.append(Config(name="05_Temp_Hot_10.0_Top20", temperature=10.0, top_k=20))

    # ---------------------------------------------------------
    # GRUPPO 6: INTERACTION GRID (Alpha x Temperature)
    # Testiamo tutte le combinazioni critiche manualmente.
    # ---------------------------------------------------------

    # --- SOTTO-GRUPPO A: CONTENT DOMINATED (Alpha 0.2) ---
    # Il modello si basa quasi solo sulle regole (Generi, Registi).
    
    # 1. Il Sergente (Rigido): Segue le regole, zero fantasia.
    configs.append(Config(name="06_Grid_A0.2_T0.5_RuleEnforcer", alpha_cf=0.2, temperature=0.5))
    
    # 2. Il Curatore (Standard): Segue le regole, ma con buon senso.
    configs.append(Config(name="06_Grid_A0.2_T2.0_RuleStandard", alpha_cf=0.2, temperature=2.0))
    
    # 3. Il Caotico (High Temp): Regole forti, ma scelta finale quasi casuale tra i validi.
    configs.append(Config(name="06_Grid_A0.2_T10.0_RuleChaos",   alpha_cf=0.2, temperature=10.0))


    # --- SOTTO-GRUPPO B: BALANCED (Alpha 0.5) ---
    # Il modello è ibrido equo.
    
    # 4. Il Cecchino (Hybrid Low Temp): Prende solo il miglior punteggio ibrido assoluto.
    configs.append(Config(name="06_Grid_A0.5_T0.5_HybridSniper", alpha_cf=0.5, temperature=0.5))
    
    # 5. Hybrid Standard (Il nostro default di riferimento)
    configs.append(Config(name="06_Grid_A0.5_T2.0_HybridStd",    alpha_cf=0.5, temperature=2.0))
    
    # 6. Discovery (Hybrid High Temp): Buono per trovare gemme nascoste con score discreto.
    configs.append(Config(name="06_Grid_A0.5_T10.0_HybridExp",   alpha_cf=0.5, temperature=10.0))


    # --- SOTTO-GRUPPO C: SVD DOMINATED (Alpha 0.8) ---
    # Il modello si fida della matematica (similarità utenti).
    
    # 7. Math Rigid (Low Temp): Si fida ciecamente del numero più alto nella matrice SVD.
    configs.append(Config(name="06_Grid_A0.8_T0.5_MathRigid",    alpha_cf=0.8, temperature=0.5))
    
    # 8. Math Standard: SVD dominante ma con curva di probabilità morbida.
    configs.append(Config(name="06_Grid_A0.8_T2.0_MathStd",      alpha_cf=0.8, temperature=2.0))
    
    # 9. Math Random (High Temp): I numeri SVD sono vicini tra loro, la Temp alta li rende uguali -> Random.
    configs.append(Config(name="06_Grid_A0.8_T10.0_MathRandom",  alpha_cf=0.8, temperature=10.0))

    for i, config in enumerate(configs):
        print(f"Config {i+1}/{len(configs)}") 
        evaluator = HybridEvaluator(
            config, data_dir / "ratings.csv", data_dir / "movies_enriched.csv"
        )
        evaluator.run_evaluation()
        print("\n")




