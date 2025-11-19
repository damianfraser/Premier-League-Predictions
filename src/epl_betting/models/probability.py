import numpy as np
from typing import Dict
from .team_strength import TeamStrength, expected_goals


def _poisson_pmf(lam: float, max_goals: int = 10):
    return [np.exp(-lam) * lam**k / np.math.factorial(k) for k in range(max_goals + 1)]


def outcome_probs(strength: TeamStrength,
                  home_team: str,
                  away_team: str,
                  max_goals: int = 10) -> Dict[str, float]:
    """
    Return model probabilities for home win / draw / away win
    using independent Poisson goals.
    """
    lam_home, lam_away = expected_goals(strength, home_team, away_team)

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    p_h = _poisson_pmf(lam_home, max_goals)
    p_a = _poisson_pmf(lam_away, max_goals)

    for i, p_hi in enumerate(p_h):
        for j, p_aj in enumerate(p_a):
            p = p_hi * p_aj
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

    # For sanity, normalise (tiny numerical error fix)
    s = p_home + p_draw + p_away
    return {
        "p_home_model": p_home / s,
        "p_draw_model": p_draw / s,
        "p_away_model": p_away / s,
    }