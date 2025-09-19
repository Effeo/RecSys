from pathlib import Path
from typing import Any, Dict
import numpy as np
# directory del file corrente (config.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# cartella data (fuori da backend/, ma montata dentro docker-compose)
DATA_DIR = BASE_DIR / "data"


# ====== Path file ======
MOVIES_FILE = DATA_DIR / "movies_enriched.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_CORR_FILE = DATA_DIR / "movies_corr.npy"
MOVIES_LIST_FILE = DATA_DIR / "movies_list.json"
USERS_FILE = DATA_DIR / "users.json"

MOVIES_CORR = np.load(MOVIES_CORR_FILE)
# pulizia NaN nella matrice di correlazione per evitare problemi in runtime
MOVIES_CORR = np.nan_to_num(MOVIES_CORR, nan=0.0)
ALPHA_CF       = 1.0            # peso del CF nel punteggio ibrido
TEMPERATURE    = 0.3            # temperatura softmax (↓ <1 più "sharp", ↑ >1 più "morbida")
RANDOM_SEED    = None           # es. 42 per riproducibilità, None per random
AWARD_WEIGHT                  = 0.3   # bonus se film premiato
DIRECTOR_WEIGHT               = 1.0   # bonus se regista gradito
RUNTIME_WEIGHT                = 0.2   # bonus se durata entro la tolleranza

FORBIDDEN_GENRE_MALUS         = 1.5   # malus per ciascun genere vietato presente
RUNTIME_OUTSIDE_MALUS         = 0.3   # intensità base del malus fuori tolleranza (scalato)
YEAR_BELOW_MALUS_PER_YEAR     = 0.02  # malus per ogni anno sotto 'min_release_year'
MISSING_YEAR_MALUS            = 0.1   # malus fisso se l'anno è mancante
MISSING_RUNTIME_MALUS         = 0.0   # (facoltativo) malus lieve se runtime mancante
TOP_K          = 10             # n film da estrarre

DEFAULT_PREFS: Dict[str, Any] = {
    "min_release_year": 0,
    "generi_desiderati": [],
    "generi_vietati": [],
    "prefer_award_winning": False,
    "preferred_runtime": None,
    "tolleranza_runtime": 0,
    "favorite_directors": [],
}