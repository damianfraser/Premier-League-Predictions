import pandas as pd
from ..config import RAW_DIR

def load_players_matchstats() -> pd.DataFrame:
    path = RAW_DIR / "players_this_season.csv"
    df = pd.read_csv(path)
    return df
