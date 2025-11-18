import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")

def main():
    # 1. Load the E0.csv file
    df = pd.read_csv(RAW_DIR / "E0.csv", parse_dates=["Date"])

    # 2. Remove rows without Pinnacle odds (PSH/PSD/PSA)
    df = df.dropna(subset=["PSH", "PSD", "PSA"])

    # 3. Build match_id
    df["match_id"] = (
        df["Date"].dt.strftime("%Y-%m-%d") + "_" +
        df["HomeTeam"].str.replace(" ", "") + "-" +
        df["AwayTeam"].str.replace(" ", "")
    )

    # 4. Build final clean dataset
    cleaned = df[[
        "match_id", "Date", "HomeTeam", "AwayTeam",
        "PSH", "PSD", "PSA"
    ]].rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "PSH": "odds_home",
        "PSD": "odds_draw",
        "PSA": "odds_away"
    })

    # 5. Add bookmaker column
    cleaned["bookmaker"] = "Pinnacle"

    # 6. Save to raw folder as odds_this_season.csv
    cleaned.to_csv(RAW_DIR / "odds_this_season.csv", index=False)

    print("Saved cleaned odds to data/raw/odds_this_season.csv")

if __name__ == "__main__":
    main()
