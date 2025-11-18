from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class TeamStrength:
    attack: Dict[str, float]
    defence: Dict[str, float]
    home_advantage: float
    intercept: float


def fit_team_strength_model(df: pd.DataFrame) -> TeamStrength:
    """
    Fit a simple log-linear model for goals/xG (Poisson-style).

    df should have, at minimum:
    - home_team, away_team
    - home_xg, away_xg (or goals)

    TODO: implement actual Poisson or GLM fit using statsmodels.
    For now, this returns neutral strengths for all teams.
    """
    teams = sorted(set(df["home_team"]).union(df["away_team"]))
    attack = {t: 0.0 for t in teams}
    defence = {t: 0.0 for t in teams}
    home_adv = 0.0
    intercept = np.log(df[["home_xg", "away_xg"]].values.mean() + 1e-6)

    return TeamStrength(attack=attack, defence=defence,
                        home_advantage=home_adv, intercept=intercept)


def expected_goals(strength: TeamStrength, home_team: str, away_team: str) -> Tuple[float, float]:
    """
    Compute expected goals (lambda_home, lambda_away) given team strengths.
    """
    a = strength.attack
    d = strength.defence
    lam_home = np.exp(strength.intercept + a[home_team] - d[away_team] + strength.home_advantage)
    lam_away = np.exp(strength.intercept + a[away_team] - d[home_team])
    return lam_home, lam_away
