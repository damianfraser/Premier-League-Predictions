from typing import Tuple

import pandas as pd


def compute_roi(results: pd.DataFrame) -> float:
    """
    Compute ROI given a results DataFrame.

    Expected columns:
    - stake
    - odds
    - outcome (1 if bet won, 0 if lost)
    """
    total_staked = results["stake"].sum()
    total_return = (results["stake"] * results["odds"] * results["outcome"]).sum()
    profit = total_return - total_staked
    return profit / total_staked if total_staked > 0 else 0.0


def equity_curve(results: pd.DataFrame) -> pd.Series:
    """
    Compute cumulative profit over time, ordered by date.
    Expected columns: 'date', 'stake', 'odds', 'outcome'
    """
    df = results.sort_values("date").copy()
    df["profit"] = df["stake"] * (df["odds"] * df["outcome"] - 1.0)
    return df["profit"].cumsum()
