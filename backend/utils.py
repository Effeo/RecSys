import pandas as pd
import json
import numpy as np
from typing import Dict, Any
import config
import random

# ========================
# Helper
# ========================
def load_users() -> Dict[str, Any]:
    """
    Carica il dizionario degli utenti (utente_id -> informazioni utente)
    dal file JSON specificato in config.USERS_FILE.

    Se il file non esiste, restituisce un dizionario vuoto.
    """
    if config.USERS_FILE.exists():
        with open(config.USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users: Dict[str, Any]):
    """
    Salva il dizionario degli utenti (utente_id -> informazioni utente)
    nel file JSON specificato in config.USERS_FILE.

    Se il file non esiste, viene creato.

    :param users: dizionario degli utenti
    """
    with open(config.USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

def clean_results(df_in: pd.DataFrame) -> list[dict]:
    """
    Pulisce i risultati della raccomandazione per renderli
    più facilmente serializzabili in JSON.

    :param df_in: dataframe con le colonne da pulire
    :return: lista di dizionari, ciascuno corrispondente ad una riga di df_in
    """
    df2 = df_in.copy()
    # NaN -> None
    df2 = df2.replace({np.nan: None})
    # Timestamp -> stringa
    for col in df2.select_dtypes(include=["datetime64[ns]"]).columns:
        df2[col] = df2[col].dt.strftime("%Y-%m-%d")
    return df2.to_dict(orient="records")

# ========================
# Bandit helpers
# ========================
def is_novelty(row: pd.Series, pref: Dict[str, Any]) -> tuple[bool, str]:
    """
    Verifica se un film è una novità rispetto alle preferenze utente.

    La novità viene valutata in base ai seguenti criteri:
    - il film non ha generi che interessano l'utente (se l'utente ne ha specificati);
    - il regista non è tra i preferiti dell'utente;
    - la durata del film è fuori dall'intervallo di tolleranza dell'utente
      (se l'utente ha specificato una durata preferita).

    Se il film non soddisfa alcuno di questi criteri, non è considerato una novità.
    In caso contrario, restituisce True e una stringa con le ragioni per cui
    il film è stato considerato una novità.

    :param row: riga del dataframe con le informazioni sul film
    :param pref: dizionario con le preferenze dell'utente
    :return: una tupla con un booleano che indica se il film è una novità
             e una stringa con le ragioni della novità (se presente)
    """
    reasons = []
    g_des = [g for g in (pref.get("generi_desiderati") or []) if g in row.index]
    fav_dirs = set(pref.get("favorite_directors") or [])

    # Match generi
    has_genre_match = False
    if g_des:
        has_genre_match = (int(row[g_des].sum()) > 0)

    # Regista fuori preferiti?
    dir_off = (len(fav_dirs) > 0 and row.get("director") not in fav_dirs)

    # Runtime fuori tolleranza?
    rt_off = False
    if pref.get("preferred_runtime") is not None and "runtime" in row.index and pd.notna(row["runtime"]):
        tol = pref.get("tolleranza_runtime", 15)
        rt_off = abs(row["runtime"] - pref["preferred_runtime"]) > tol

    # Logica di novità
    if g_des:
        is_novel = (not has_genre_match) and (dir_off or rt_off)
        if not has_genre_match:
            reasons.append("genere fuori profilo")
    else:
        # se l'utente non ha generi desiderati, basiamoci su regista/runtime
        is_novel = (dir_off or rt_off)

    if dir_off:
        reasons.append("regista fuori preferiti")
    if rt_off:
        reasons.append("runtime fuori tolleranza")

    return is_novel, (", ".join(reasons) if reasons else "in linea con i gusti")


# ========================
# POOLS
# ========================
def _constraint_pool(df_all: pd.DataFrame, pref: Dict[str, Any], k: int) -> pd.DataFrame:
    """
    Ritorna i top-k film raccomandati per le preferenze utente, senza alcuna novità.
    Questo è l'exploit pool per il bandit.

    :param df_all: dataframe di tutti i film, con colonne "movie_title" e "release_date"
    :param pref: dizionario con le preferenze dell'utente
    :param k: numero di film da restituire
    :return: dataframe con i top-k film, con colonna "score"
    """
    return recommend_movies(df_all, pref, top_k=k).copy()

def _year_filtered(df_all: pd.DataFrame, pref: Dict[str, Any]) -> pd.DataFrame:
    """
    Ritorna un dataframe con i film che rispettano la minima release year richiesta.

    :param df_all: dataframe di tutti i film, con colonna "release_date"
    :param pref: dizionario con le preferenze dell'utente
    :return: dataframe con i film che rispettano la minima release year richiesta
    """
    df_y = df_all[df_all["release_date"].dt.year >= pref.get("min_release_year", 0)].copy()
    return df_y

def build_pools(df_all: pd.DataFrame,
                pref: Dict[str, Any],
                candidate_pool: int = 100,
                explore_extra: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Costruisce i due pool per il bandit:
    1. exploit_pool: top-k film raccomandati per le preferenze utente, senza alcuna novità
    2. explore_pool: candidate pool più largo, con film che rispettano la minima release year
       richiesta ma non sono in exploit_pool, contrassegnati per la novità

    :param df_all: dataframe di tutti i film, con colonne "movie_title" e "release_date"
    :param pref: dizionario con le preferenze dell'utente
    :param candidate_pool: numero di film da restituire come exploit_pool
    :param explore_extra: numero di film extra da selezionare per explore_pool
    :return: due dataframe, exploit_pool e explore_pool
    """
    exploit_pool = _constraint_pool(df_all, pref, k=max(candidate_pool, 20))
    if exploit_pool.empty:
        return exploit_pool, exploit_pool  # entrambi vuoti

    # Calcola novelty su exploit_pool (ci serve dopo, ma cloniamo)
    ex = exploit_pool.copy()

    # Costruiamo un explore candidati “larghi”
    base_wide = _year_filtered(df_all, pref)
    # rimuovi quelli già in exploit
    base_wide = base_wide[~base_wide["movie_id"].isin(ex["movie_id"])].copy()

    # contrassegna novelty su base_wide
    nov_flags, nov_reasons = zip(*[is_novelty(r, pref) for _, r in base_wide.iterrows()]) if not base_wide.empty else ([], [])
    if not base_wide.empty:
        base_wide["novel"] = list(nov_flags)
        base_wide["novelty_reason"] = list(nov_reasons)

    # explore = davvero fuori profilo
    explore_pool = base_wide[base_wide["novel"]].copy() if not base_wide.empty else base_wide

    # fallback se explore è troppo piccolo
    MIN_EXPLORE = min(50, explore_extra)
    if explore_pool.shape[0] < MIN_EXPLORE:
        # 1) aggiungi dalla coda dell'exploit (punteggi più bassi)
        tail = ex.sort_values("score", ascending=True).head(MIN_EXPLORE)
        tail = tail[~tail["movie_id"].isin(explore_pool["movie_id"] if not explore_pool.empty else [])]
        explore_pool = pd.concat([explore_pool, tail], ignore_index=True) if not explore_pool.empty else tail.copy()

        # 2) aggiungi random dal base_wide (stessa epoca)
        if not base_wide.empty:
            need = MIN_EXPLORE - explore_pool.shape[0]
            if need > 0:
                explore_pool = pd.concat(
                    [explore_pool, base_wide.sample(n=min(need, base_wide.shape[0]), random_state=None)],
                    ignore_index=True
                )

    # assicura disgiunzione
    explore_pool = explore_pool[~explore_pool["movie_id"].isin(ex["movie_id"])]
    return ex.reset_index(drop=True), explore_pool.reset_index(drop=True)


# ========================
# ε-GREEDY SU POOLS DISGIUNTI
# ========================
def epsilon_greedy_from_pools(exploit_pool: pd.DataFrame,
                              explore_pool: pd.DataFrame,
                              pref: Dict[str, Any],
                              top_k: int = 5,
                              epsilon: float = 0.2,
                              seed: int | None = None) -> pd.DataFrame:
    """
    Ritorna un dataframe con le raccomandazioni per l'utente, usando ε-greedy tra due pool disgiunti:
    1. exploit_pool: top-k film raccomandati per le preferenze utente, senza alcuna novità
    2. explore_pool: candidate pool più largo, con film che rispettano la minima release year
       richiesta ma non sono in exploit_pool, contrassegnati per la novità

    :param exploit_pool: dataframe con i top-k film raccomandati per le preferenze utente
    :param explore_pool: dataframe con i film che rispettano la minima release year richiesta
    :param pref: dizionario con le preferenze dell'utente
    :param top_k: numero di film da restituire
    :param epsilon: probabilità di esplorazione
    :param seed: seed per la generazione casuale
    :return: dataframe con le raccomandazioni per l'utente, con colonne "score", "novel", "novelty_reason" e "pick_strategy"
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if exploit_pool.empty and explore_pool.empty:
        return exploit_pool.head(0)

    # ordina lo sfruttamento per score
    ex = exploit_pool.sort_values("score", ascending=False).copy()
    ex_ids = set()
    rows = []

    # calcola novelty su entrambi i pool (se non presente)
    def _ensure_novelty(df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Ritorna il dataframe con colonne "novel" e "novelty_reason" aggiunte, se non presenti.
        Le colonne sono calcolate chiamando la funzione is_novelty(row, pref) per ogni riga del dataframe.
        La funzione restituisce un dataframe copiato, con le nuove colonne aggiunte.
        """
    
        if "novel" in df_in.columns and "novelty_reason" in df_in.columns:
            return df_in
        flags, reasons = zip(*[is_novelty(r, pref) for _, r in df_in.iterrows()]) if not df_in.empty else ([], [])
        out = df_in.copy()
        if not df_in.empty:
            out["novel"] = list(flags)
            out["novelty_reason"] = list(reasons)
        return out

    ex = _ensure_novelty(ex)
    xp = _ensure_novelty(explore_pool)

    # indici di scorrimento per evitare ripetizioni
    ex_idx = 0

    for _ in range(top_k):
        pick_explore = (random.random() < epsilon) and not xp.empty
        chosen = None
        strategy = "explore" if pick_explore else "exploit"

        if pick_explore:
            # scegli un elemento novel NON ancora preso
            xp_available = xp[~xp["movie_id"].isin(ex_ids)]
            if not xp_available.empty:
                chosen = xp_available.sample(n=1, random_state=None).iloc[0]
            else:
                pick_explore = False  # fallback a exploit

        if not pick_explore:
            while ex_idx < len(ex) and ex.iloc[ex_idx]["movie_id"] in ex_ids:
                ex_idx += 1
            if ex_idx < len(ex):
                chosen = ex.iloc[ex_idx]
                ex_idx += 1

        if chosen is None:
            break

        ex_ids.add(int(chosen["movie_id"]))
        row = chosen.copy()
        row["pick_strategy"] = strategy
        rows.append(row)

    return pd.DataFrame(rows)

# ========================
# Raccomandazioni content-based per utente
# ========================
def recommend_movies(df, pref, top_k=5):
    """
    Raccomandazioni content-based per utente.

    La funzione filtra i film con anno di uscita >= min_release_year e
    rimuove quelli con generi vietati (se presenti nel dataset). Quindi,
    calcola un punteggio per ogni film come somma dei generi desiderati
    e bonus per film premiati, registi graditi e durata vicina a quella
    preferita (se specificate). I film vengono infine ordinati per punteggio
    decrescente e restituiti i primi top_k.

    :param df: dataframe con i film
    :param pref: dizionario con le preferenze dell'utente
    :param top_k: numero di film da restituire
    :return: dataframe con i top_k film raccomandati, con colonna "score"
    """
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
