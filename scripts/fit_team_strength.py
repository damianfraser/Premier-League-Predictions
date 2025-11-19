import pickle
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = MODELS_DIR / "team_strength.pkl"


class TeamStrengthModel:
    """
    Simple Poisson attack/defense model:
    log(expected_goals) = home_advantage + attack_home - defence_away
    """

    def __init__(self, teams):
        self.teams = teams
        self.attack = {t: 0.0 for t in teams}
        self.defence = {t: 0.0 for t in teams}
        self.home_advantage = 0.0


def fit_poisson_strength_model(df: pd.DataFrame) -> TeamStrengthModel:
    """
    Fits a Poisson regression model for attack/defence.

    df must contain:
      home_team_name, away_team_name, home_goals, away_goals
    """

    teams = sorted(set(df["home_team_name"]) | set(df["away_team_name"]))
    model = TeamStrengthModel(teams)

    # Initial guess
    attack = {t: 0.0 for t in teams}
    defence = {t: 0.0 for t in teams}
    home_adv = 0.0

    # Convert to arrays for optimization
    team_index = {t: i for i, t in enumerate(teams)}

    def pack_params(a, d, h):
        return np.concatenate([
            np.array([a[t] for t in teams]),
            np.array([d[t] for t in teams]),
            np.array([h])
        ])

    def unpack_params(params):
        a = params[:len(teams)]
        d = params[len(teams):len(teams)*2]
        h = params[-1]

        attack_dict = {t: a[i] for i, t in enumerate(teams)}
        defence_dict = {t: d[i] for i, t in enumerate(teams)}
        return attack_dict, defence_dict, h

    # Negative log-likelihood
    def nll(params):
        attack, defence, h = unpack_params(params)
        nll_sum = 0.0

        for _, row in df.iterrows():
            home = row["home_team_name"]
            away = row["away_team_name"]
            hg = row["home_goals"]
            ag = row["away_goals"]

            lam_home = np.exp(h + attack[home] - defence[away])
            lam_away = np.exp(attack[away] - defence[home])

            # Poisson log likelihood
            nll_sum += -(
                hg * np.log(lam_home) - lam_home +
                ag * np.log(lam_away) - lam_away
            )

        return nll_sum

    # Run optimization
    from scipy.optimize import minimize

    params0 = pack_params(attack, defence, home_adv)

    result = minimize(nll, params0, method="L-BFGS-B")

    attack, defence, home_adv = unpack_params(result.x)

    model.attack = attack
    model.defence = defence
    model.home_advantage = home_adv

    return model


def main():
    df = pd.read_csv(PROCESSED_DIR / "matches_features.csv")

    required_cols = [
        "home_team_name", "away_team_name",
        "home_goals", "away_goals"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print("Fitting team strength model on", len(df), "matches...")
    model = fit_poisson_strength_model(df)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Saved model →", OUTPUT_PATH)
    print("\n--- Home Advantage ---")
    print(model.home_advantage)

    print("\n--- Attack Strength (top 5) ---")
    for t, v in sorted(model.attack.items(), key=lambda x: -x[1])[:5]:
        print(f"{t}: {v:.3f}")

    print("\n--- Best Defences (highest defence rating) ---")
    best_def = sorted(model.defence.items(), key=lambda x: x[1], reverse=True)[:5]
    for t, v in best_def:
        print(f"{t}: {v:.3f}")

    print("\n--- Worst Defences (lowest defence rating) ---")
    worst_def = sorted(model.defence.items(), key=lambda x: x[1])[:5]
    for t, v in worst_def:
        print(f"{t}: {v:.3f}")


if __name__ == "__main__":
    main()
