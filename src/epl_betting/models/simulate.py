from typing import Dict

import numpy as np

from .team_strength import TeamStrength, expected_goals
from ..config import N_SIMULATIONS


def simulate_match(strength: TeamStrength,
                   home_team: str,
                   away_team: str,
                   n_simulations: int = N_SIMULATIONS) -> Dict[str, float]:
    """
    Simulate a match using a Poisson model for home/away goals.
    Returns model probabilities for home/draw/away and lambdas.
    """
    lam_home, lam_away = expected_goals(strength, home_team, away_team)

    home_goals = np.random.poisson(lam_home, size=n_simulations)
    away_goals = np.random.poisson(lam_away, size=n_simulations)

    home_wins = (home_goals > away_goals).mean()
    draws = (home_goals == away_goals).mean()
    away_wins = (home_goals < away_goals).mean()

    return {
        "p_home_model": home_wins,
        "p_draw_model": draws,
        "p_away_model": away_wins,
        "lambda_home": lam_home,
        "lambda_away": lam_away,
    }
