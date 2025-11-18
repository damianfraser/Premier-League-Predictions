from typing import Dict


def implied_probs_from_odds(odds_home: float, odds_draw: float, odds_away: float) -> Dict[str, float]:
    """
    Convert 1X2 decimal odds into implied probabilities, removing overround.
    """
    ph = 1.0 / odds_home
    pd = 1.0 / odds_draw
    pa = 1.0 / odds_away
    overround = ph + pd + pa

    return {
        "p_home_market": ph / overround,
        "p_draw_market": pd / overround,
        "p_away_market": pa / overround,
    }
