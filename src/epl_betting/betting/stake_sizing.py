def kelly_fraction(p: float, odds: float) -> float:
    """
    Full Kelly fraction for a single outcome.

    p: posterior probability of winning
    odds: decimal odds
    """
    b = odds - 1.0
    f_star = (p * (b + 1) - 1) / b  # standard Kelly formula
    return max(0.0, f_star)
