import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SEASON_DIR = RAW_DIR / "FPL-Elo-Insights" / "data" / "2025-2026"
BY_TOURNAMENT_PL_DIR = SEASON_DIR / "By Tournament" / "Premier League"


def collect_matches() -> pd.DataFrame:
    """Combine Premier League matches from all GW folders into one DataFrame."""
    all_matches = []

    # Each subfolder is like 'GW1', 'GW2', ...
    for gw_dir in sorted(BY_TOURNAMENT_PL_DIR.iterdir()):
        if not gw_dir.is_dir():
            continue
        matches_path = gw_dir / "matches.csv"
        if matches_path.exists():
            df = pd.read_csv(matches_path)
            df["gw_folder"] = gw_dir.name  # optional, GW trace
            all_matches.append(df)

    if not all_matches:
        raise RuntimeError("No matches.csv files found under By Tournament/Premier League")

    matches = pd.concat(all_matches, ignore_index=True).drop_duplicates(subset=["match_id"])
    return matches


def collect_player_matchstats() -> pd.DataFrame:
    """Combine Premier League player match stats from all GW folders."""
    all_pms = []

    for gw_dir in sorted(BY_TOURNAMENT_PL_DIR.iterdir()):
        if not gw_dir.is_dir():
            continue
        pms_path = gw_dir / "playermatchstats.csv"
        if pms_path.exists():
            df = pd.read_csv(pms_path)
            df["gw_folder"] = gw_dir.name  # optional
            all_pms.append(df)

    if not all_pms:
        raise RuntimeError("No playermatchstats.csv files found under By Tournament/Premier League")

    pms = pd.concat(all_pms, ignore_index=True).drop_duplicates(subset=["match_id", "player_id"])
    return pms


def main():
    # 1. Collect & save matches
    matches = collect_matches()
    matches_out = RAW_DIR / "matches_this_season.csv"
    matches.to_csv(matches_out, index=False)
    print(f"Saved Premier League matches to {matches_out}")

    # 2. Collect & save player match stats
    pms = collect_player_matchstats()
    pms_out = RAW_DIR / "players_this_season.csv"
    pms.to_csv(pms_out, index=False)
    print(f"Saved Premier League player match stats to {pms_out}")


if __name__ == "__main__":
    main()
