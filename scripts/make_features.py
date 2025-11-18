"""
Offline script to create feature-engineered datasets.
Run this after updating raw match/odds data.
"""
import pandas as pd

from src.epl_betting.data.load_matches import load_raw_matches
from src.epl_betting.data.load_odds import load_odds
from src.epl_betting.data.features import save_features


def main():
    matches = load_raw_matches()
    odds = load_odds()

    # TODO: join matches + odds and create rolling/xG features.
    features = matches  # temporary: just save matches as-is

    save_features(features)


if __name__ == "__main__":
    main()
