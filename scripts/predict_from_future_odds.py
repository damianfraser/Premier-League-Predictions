import pandas as pd
from pathlib import Path

from epl_betting.models.team_strength import fit_team_strength_model
from epl_betting.models.probability import outcome_probs
from epl_betting.betting.odds_utils import implied_probs_from_odds

# Betting parameters
MODEL_WEIGHT = 0.30      # how much we trust our model vs market
MIN_EDGE = 0.03          # minimum edge (3%) to recommend a bet
KELLY_FRACTION = 0.25    # fraction of full Kelly stake to actually use

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"


def blended_prob(p_model: float, p_market: float) -> float:
    """
    Blend model and market probabilities.
    """
    return MODEL_WEIGHT * p_model + (1.0 - MODEL_WEIGHT) * p_market


def load_training_matches() -> pd.DataFrame:
    """
    Load matches_features.csv and keep only matches that have actually been played
    (i.e. have goals/xG).
    """
    df = pd.read_csv(PROCESSED_DIR / "matches_features.csv")
    # adjust these column names if you ended up renaming
    if "home_goals" in df.columns:
        df = df[df["home_goals"].notna()]
    elif "home_xg" in df.columns:
        df = df[df["home_xg"].notna()]
    else:
        raise ValueError("matches_features.csv must have home_goals or home_xg for training.")

    return df


def load_future_odds() -> pd.DataFrame:
    """
    Load manually-entered future odds from data/raw/future_odds.csv.

    Date column is optional. Required columns:
      - home_team
      - away_team
      - odds_home
      - odds_draw
      - odds_away
    """
    path = RAW_DIR / "future_odds.csv"

    # Simple read: no date parsing, since we don't require a date column
    df = pd.read_csv(path)

    # If there's a date column, normalise it; otherwise it's fine
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    else:
        df["date"] = None  # or just drop this line if you don't care at all

    required_cols = ["home_team", "away_team", "odds_home", "odds_draw", "odds_away"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"future_odds.csv is missing required columns: {missing}")

    return df


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Train / refit team strength model on historic matches
    train_df = load_training_matches()
    print(f"Using {len(train_df)} past matches to fit team strengths...")
    strength = fit_team_strength_model(train_df, use_xg=True)

    # 2) Load FUTURE odds that you entered manually
    future_odds = load_future_odds()
    print(f"Loaded {len(future_odds)} future fixtures with odds.")

    records = []

    for _, row in future_odds.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # --- model probabilities
        model_probs = outcome_probs(strength, home, away)

        # --- market probabilities from odds (remove overround)
        market_probs = implied_probs_from_odds(
            odds_home=row["odds_home"],
            odds_draw=row["odds_draw"],
            odds_away=row["odds_away"],
        )

        outcomes = {
            "Home": ("p_home_model", "p_home_market", row["odds_home"]),
            "Draw": ("p_draw_model", "p_draw_market", row["odds_draw"]),
            "Away": ("p_away_model", "p_away_market", row["odds_away"]),
        }

        for outcome, (m_key, mk_key, odd) in outcomes.items():
            p_model = model_probs[m_key]
            p_market = market_probs[mk_key]

            # blended probability
            p_final = blended_prob(p_model, p_market)

            # edge vs market probability
            edge = p_final - p_market

            # Kelly stake fraction (full Kelly)
            kelly_full = max((p_final * odd - 1) / (odd - 1), 0.0)
            stake_fraction = KELLY_FRACTION * kelly_full

            records.append({
                "date": row["date"],
                "home_team": home,
                "away_team": away,
                "bet_side": outcome,          # Home / Draw / Away
                "odds": odd,
                "p_model": p_model,
                "p_market": p_market,
                "p_final": p_final,
                "edge": edge,
                "edge_pct": edge * 100,
                "kelly_full": kelly_full,
                "stake_fraction": stake_fraction,
            })

    results = pd.DataFrame(records)

    # Sort by edge descending
    results = results.sort_values("edge", ascending=False).reset_index(drop=True)

    # Save all and also filtered recommendations
    all_path = RESULTS_DIR / "future_odds_all_edges.csv"
    rec_path = RESULTS_DIR / "future_odds_recommended_bets.csv"

    results.to_csv(all_path, index=False)
    recs = results[results["edge"] >= MIN_EDGE].copy()
    recs.to_csv(rec_path, index=False)

    print(f"\n💾 Saved all edges to: {all_path}")
    print(f"💾 Saved recommended bets (edge ≥ {MIN_EDGE*100:.1f}%) to: {rec_path}\n")

    # Print a quick summary to the console
    if recs.empty:
        print("⚠ No recommended bets for these odds (no edge ≥ threshold).")
    else:
        print("🎯 Recommended bets based on current future odds:\n")
        for _, r in recs.iterrows():
            print(
                f"{r['date']} - {r['home_team']} vs {r['away_team']} | "
                f"Bet: {r['bet_side']} @ {r['odds']} | "
                f"Edge: {r['edge_pct']:.1f}% | Stake: {r['stake_fraction']:.3f} bankroll"
            )


if __name__ == "__main__":
    main()
