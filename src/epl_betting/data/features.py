import pandas as pd
from ..config import PROCESSED_DIR


def save_features(df: pd.DataFrame, name: str = "matches_features.csv") -> None:
    """
    Save feature-engineered data to the processed directory.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / name, index=False)