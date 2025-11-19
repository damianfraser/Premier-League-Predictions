from typing import Dict


def implied_probs_from_odds(odds_home: float,
                            odds_draw: float,
                            odds_away: float) -> Dict[str, float]:
    """
    Convert decimal odds for 1X2 into implied probabilities
    (removing the bookmaker overround).
    """
    inv_h = 1.0 / odds_home
    inv_d = 1.0 / odds_draw
    inv_a = 1.0 / odds_away

    overround = inv_h + inv_d + inv_a

    return {
        "p_home_market": inv_h / overround,
        "p_draw_market": inv_d / overround,
        "p_away_market": inv_a / overround,
    }