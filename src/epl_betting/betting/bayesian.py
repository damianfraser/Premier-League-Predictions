from typing import Dict
from ..config import MODEL_WEIGHT


def combine_probs(model_probs: Dict[str, float],
                  market_probs: Dict[str, float],
                  w: float = MODEL_WEIGHT) -> Dict[str, float]:
    """
    Simple Bayesian-style blend of model and market probabilities:
    posterior = w * model + (1 - w) * market
    """
    p_home = w * model_probs["p_home_model"] + (1 - w) * market_probs["p_home_market"]
    p_draw = w * model_probs["p_draw_model"] + (1 - w) * market_probs["p_draw_market"]
    p_away = w * model_probs["p_away_model"] + (1 - w) * market_probs["p_away_market"]

    s = p_home + p_draw + p_away
    return {
        "p_home_posterior": p_home / s,
        "p_draw_posterior": p_draw / s,
        "p_away_posterior": p_away / s,
    }
