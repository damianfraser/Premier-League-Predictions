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


def fit_team_strength_model(df: pd.DataFrame, use_xg: bool = True) -> TeamStrength:
    # Pick xG if available, fallback to actual goals
    if use_xg and "home_xg" in df.columns and "away_xg" in df.columns:
        home_for = df["home_xg"]
        home_ag = df["away_xg"]
        away_for = df["away_xg"]
        away_ag = df["home_xg"]
    else:
        home_for = df["home_goals"]
        home_ag = df["away_goals"]
        away_for = df["away_goals"]
        away_ag = df["home_goals"]

    teams = sorted(set(df["home_team_name"]).union(df["away_team_name"]))

    league_avg = (home_for.sum() + away_for.sum()) / (len(df) * 2)

    avg_home_goals = home_for.mean()
    avg_away_goals = away_for.mean()
    home_advantage = np.log((avg_home_goals + 1e-8) / (avg_away_goals + 1e-8))

    attack = {}
    defence = {}

    for team in teams:
        home_mask = df["home_team_name"] == team
        away_mask = df["away_team_name"] == team

        gf = home_for[home_mask].sum() + away_for[away_mask].sum()
        ga = home_ag[away_mask].sum() + away_ag[home_mask].sum()
        n_games = home_mask.sum() + away_mask.sum()

        if n_games == 0:
            attack[team] = 0
            defence[team] = 0
            continue

        gf_pg = gf / n_games
        ga_pg = ga / n_games

        attack[team] = np.log((gf_pg + 1e-8) / (league_avg + 1e-8))
        defence[team] = np.log((league_avg + 1e-8) / (ga_pg + 1e-8))

    return TeamStrength(
        attack=attack,
        defence=defence,
        home_advantage=home_advantage,
        intercept=np.log(league_avg + 1e-8),
    )


def expected_goals(strength: TeamStrength, home: str, away: str) -> Tuple[float, float]:
    lam_home = np.exp(
        strength.intercept +
        strength.attack[home] -
        strength.defence[away] +
        strength.home_advantage
    )
    lam_away = np.exp(
        strength.intercept +
        strength.attack[away] -
        strength.defence[home]
    )
    return lam_home, lam_away
