import pandas as pd
from pathlib import Path

from epl_betting.models.team_strength import fit_team_strength_model
from epl_betting.models.probability import outcome_probs
from epl_betting.betting.odds_utils import implied_probs_from_odds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "matches_features.csv")

    # Fit team strength model on ALL finished matches
    strength = fit_team_strength_model(df, use_xg=True)

    records = []

    for _, row in df.iterrows():
        home = row["home_team_name"]
        away = row["away_team_name"]

        # 1) model probs
        model = outcome_probs(strength, home, away)

        # 2) market probs from odds
        market = implied_probs_from_odds(
            odds_home=row["odds_home"],
            odds_draw=row["odds_draw"],
            odds_away=row["odds_away"],
        )

        # 3) edges (simple version = model prob - market prob)
        edge_home = model["p_home_model"] - market["p_home_market"]
        edge_draw = model["p_draw_model"] - market["p_draw_market"]
        edge_away = model["p_away_model"] - market["p_away_market"]

        records.append({
            "date": row.get("date", None),
            "home_team": home,
            "away_team": away,
            "odds_home": row["odds_home"],
            "odds_draw": row["odds_draw"],
            "odds_away": row["odds_away"],
            "p_home_model": model["p_home_model"],
            "p_draw_model": model["p_draw_model"],
            "p_away_model": model["p_away_model"],
            "p_home_market": market["p_home_market"],
            "p_draw_market": market["p_draw_market"],
            "p_away_market": market["p_away_market"],
            "edge_home": edge_home,
            "edge_draw": edge_draw,
            "edge_away": edge_away,
        })

    out_df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "historical_model_vs_market.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved evaluation to {out_path}")

    # Optional: quick summary of average edge
    print("Average edge (model - market):")
    print(out_df[["edge_home", "edge_draw", "edge_away"]].mean())


if __name__ == "__main__":
    main()