from pathlib import Path

# ====== Pesi ======
AWARD_WEIGHT = 0.3
DIRECTOR_WEIGHT = 1.0
RUNTIME_WEIGHT = 0.2

# ====== Path file ======
DATA_DIR = Path("../data")
MOVIES_FILE = DATA_DIR / "movies_enriched.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"

MOVIES_CORR_FILE = Path("../data/movies_corr.npy")
MOVIES_LIST_FILE = Path("../data/movies_list.json")
USERS_FILE = Path("../data/users.json")
