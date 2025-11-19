import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SEASON_DIR = RAW_DIR / "FPL-Elo-Insights" / "data" / "2025-2026"


def load_matches_with_names() -> pd.DataFrame:
    """
    Load matches_this_season.csv and attach odds-style team names directly
    from team IDs, so they match the names used in odds_this_season.csv.
    """
    matches_path = RAW_DIR / "matches_this_season.csv"
    matches = pd.read_csv(matches_path)

    # Standardise date (useful to keep around, even though we don't join on it)
    if "kickoff_time" in matches.columns:
        matches["date"] = pd.to_datetime(matches["kickoff_time"]).dt.date
    elif "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"]).dt.date
    else:
        raise ValueError("No date or kickoff_time column in matches_this_season.csv")

    # Rename key stats if present
    rename_map = {
        "home_score": "home_goals",
        "away_score": "away_goals",
        "home_expected_goals_xg": "home_xg",
        "away_expected_goals_xg": "away_xg",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in matches.columns}
    matches = matches.rename(columns=rename_map)

    # Explicit mapping: FotMob / FPL-Elo team IDs → odds-style team names
    # These IDs come from matches_this_season.csv (examples you pasted).
    TEAM_ID_TO_NAME = {
        1:  "Man United",
        2:  "Leeds",
        3:  "Arsenal",
        4:  "Newcastle",
        6:  "Tottenham",
        7:  "Aston Villa",
        8:  "Chelsea",
        11: "Everton",
        14: "Liverpool",
        17: "Nott'm Forest",
        21: "West Ham",
        31: "Crystal Palace",
        36: "Brighton",
        39: "Wolves",
        43: "Man City",
        54: "Fulham",
        56: "Sunderland",
        90: "Burnley",
        91: "Bournemouth",
        94: "Brentford",
    }

    # IDs are floats in the CSV (e.g. 14.0), so cast to int before mapping
    matches["home_team_id"] = matches["home_team"].astype(int)
    matches["away_team_id"] = matches["away_team"].astype(int)

    matches["home_team_name"] = matches["home_team_id"].map(TEAM_ID_TO_NAME)
    matches["away_team_name"] = matches["away_team_id"].map(TEAM_ID_TO_NAME)

    # These are the join keys we will use with the odds file
    matches["home_name_for_join"] = matches["home_team_name"]
    matches["away_name_for_join"] = matches["away_team_name"]

    return matches


def load_odds() -> pd.DataFrame:
    """
    Load odds_this_season.csv and expose team names in the same format
    as load_matches_with_names, so we can join directly on names.
    """
    odds_path = RAW_DIR / "odds_this_season.csv"
    odds = pd.read_csv(odds_path, parse_dates=["date"])
    odds["date"] = odds["date"].dt.date

    # odds_this_season.csv already uses the desired names, e.g.
    # Liverpool, Bournemouth, Aston Villa, Newcastle, Brighton, Fulham, ...
    odds["home_name_for_join"] = odds["home_team"]
    odds["away_name_for_join"] = odds["away_team"]

    return odds


def make_match_features() -> pd.DataFrame:
    matches = load_matches_with_names()
    odds = load_odds()

    # In case the odds file has multiple rows per fixture (e.g. different
    # bookmakers / timestamps), reduce to a single row per (home, away).
    odds_for_merge = (
        odds.sort_values("date")  # keep the latest by date if duplicates exist
        .drop_duplicates(
            subset=["home_name_for_join", "away_name_for_join"],
            keep="last",
        )[
            [
                "home_name_for_join",
                "away_name_for_join",
                "odds_home",
                "odds_draw",
                "odds_away",
                "bookmaker",
                "match_id",  # odds match_id (e.g. 2025-08-15_Liverpool-Bournemouth)
                "date",
            ]
        ]
    )

    # 🔑 Join only on (home team name, away team name).
    # We assume each home team plays each away team once per season.
    merged = matches.merge(
        odds_for_merge,
        on=["home_name_for_join", "away_name_for_join"],
        how="inner",  # only matches that currently have odds
        suffixes=("", "_odds"),
    )

    print(f"Merged {len(merged)} matches with odds out of {len(matches)} total matches.")

    # ---- Debugging: which odds fixtures didn't match any FPL-Elo match? ----
    merged_keys = merged[["home_name_for_join", "away_name_for_join"]].drop_duplicates()
    odds_keys = odds_for_merge[["home_name_for_join", "away_name_for_join"]].drop_duplicates()

    unmatched = odds_keys.merge(
        merged_keys,
        on=["home_name_for_join", "away_name_for_join"],
        how="left",
        indicator=True,
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"]

    if not unmatched.empty:
        print("⚠ Some odds rows did not match any FPL-Elo matches (by team names).")
        debug = unmatched.merge(
            odds,
            on=["home_name_for_join", "away_name_for_join"],
            how="left",
        )[["date", "home_team", "away_team"]].drop_duplicates()
        print(debug.head(20))

    return merged


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features = make_match_features()
    out_path = PROCESSED_DIR / "data" / "processed" / "matches_features.csv"
    # Slight correction: we’re already in PROJECT_ROOT, so:
    out_path = PROCESSED_DIR / "matches_features.csv"
    features.to_csv(out_path, index=False)
    print(f"✅ Saved match features to {out_path}")


if __name__ == "__main__":
    main()
