import pandas as pd
from ..config import RAW_DIR, PROCESSED_DIR


def load_raw_matches() -> pd.DataFrame:
    """
    Load raw Premier League match data from CSV.

    Expected columns (example, can evolve):
    - match_id
    - date
    - season
    - home_team
    - away_team
    - home_goals
    - away_goals
    - home_xg
    - away_xg
    """
    path = RAW_DIR / "matches_premier_league.csv"
    return pd.read_csv(path, parse_dates=["date"])


def load_processed_matches() -> pd.DataFrame:
    """
    Load feature-engineered match data for modelling/backtesting.
    """
    path = PROCESSED_DIR / "matches_features.csv"
    return pd.read_csv(path, parse_dates=["date"])
