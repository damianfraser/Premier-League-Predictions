import pandas as pd
from ..config import RAW_DIR

def load_raw_matches() -> pd.DataFrame:
    path = RAW_DIR / "matches_this_season.csv"
    df = pd.read_csv(path)

    # Standardise date and team/goal/xG columns using the schema from README
    if "kickoff_time" in df.columns:
        df["date"] = pd.to_datetime(df["kickoff_time"])

    rename_map = {
        "home_score": "home_goals",
        "away_score": "away_goals",
        "home_expected_goals_xg": "home_xg",
        "away_expected_goals_xg": "away_xg",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    return df
