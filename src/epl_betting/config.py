from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

N_SIMULATIONS = 20000

# Betting parameters
MODEL_WEIGHT = 0.3      # how much we trust our model vs market
MIN_EDGE = 0.07         # minimum edge (3%) to place a bet
KELLY_FRACTION = 0.25   # fraction of full Kelly stake to actually use
