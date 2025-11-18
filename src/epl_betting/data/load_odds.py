import pandas as pd
from ..config import RAW_DIR


def load_odds() -> pd.DataFrame:
    """
    Load historical odds for Premier League matches.

    Expected columns (example):
    - match_id
    - date
    - home_team
    - away_team
    - odds_home
    - odds_draw
    - odds_away
    - bookmaker
    """
    path = RAW_DIR / "odds_premier_league.csv"
    return pd.read_csv(path, parse_dates=["date"])
