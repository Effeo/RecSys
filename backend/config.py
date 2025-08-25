from pathlib import Path

# directory del file corrente (config.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# cartella data (fuori da backend/, ma montata dentro docker-compose)
DATA_DIR = BASE_DIR / "data"

# ====== Pesi ======
AWARD_WEIGHT = 0.3
DIRECTOR_WEIGHT = 1.0
RUNTIME_WEIGHT = 0.2

# ====== Path file ======
MOVIES_FILE = DATA_DIR / "movies_enriched.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"
MOVIES_CORR_FILE = DATA_DIR / "movies_corr.npy"
MOVIES_LIST_FILE = DATA_DIR / "movies_list.json"
USERS_FILE = DATA_DIR / "users.json"
