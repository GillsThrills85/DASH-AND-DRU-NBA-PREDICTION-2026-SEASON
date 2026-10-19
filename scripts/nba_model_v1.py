"""
NBA Prediction Model v1 — transparent per-factor model.

Pipeline:
  1. Load games + team box scores (from scripts/pull_data.py).
  2. For each game, build each team's trailing-10-game profile
     (min 5 games; fewer blends toward league average). STRICTLY no lookahead.
  3. Ratings per 100 possessions, light SOS adjustment, pace, rest, home edge.
  4. Expected margin -> win probability via normal CDF (sigma = 13.5).

Usage:
  python3 scripts/nba_model_v1.py --backtest
  python3 scripts/nba_model_v1.py --predict LAL BOS        # away, home
  python3 scripts/nba_model_v1.py --predict LAL BOS --injuries data/injuries.csv

injuries.csv format: date,team,player,pts_impact   (one row per inactive)
"""
import argparse
import csv
import math
import os
from collections import defaultdict
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

WINDOW = 10          # trailing games in each team's profile
MIN_GAMES = 5        # below this, blend toward league average
HOME_EDGE = 2.0      # points
MARGIN_SIGMA = 13.5  # std dev of NBA final margins
B2B_PENALTY = -1.5   # played yesterday
REST_BONUS = 1.0     # 3+ days rest vs opponent on <=1


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def possessions(fga, fta, orb, tov, ofga, ofta, oorb, otov):
    p1 = fga + 0.44 * fta - orb + tov
    p2 = ofga + 0.44 * ofta - oorb + otov
    return 0.5 * (p1 + p2)


def load():
    games = []
    with open(os.path.join(DATA, "nba_games_2023-2025.csv")) as f:
        for r in csv.DictReader(f):
            if r["away_score"] == "" or r["home_score"] == "":
                continue
            games.append({
                "id": int(r["game_id"]), "date": r["date"],
                "away": r["away_team"], "home": r["home_team"],
                "away_pts": int(r["away_score"]), "home_pts": int(r["home_score"]),
            })
    box = defaultdict(dict)  # game_id -> team -> stats
    with open(os.path.join(DATA, "nba_boxscores_2023-2025.csv")) as f:
        for r in csv.DictReader(f):
            box[int(r["game_id"])][r["team"]] = {k: float(r[k]) for k in
                ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
                 "reb", "ast", "stl", "blk", "turnover", "pf", "pts"]}
    games.sort(key=lambda g: (g["date"], g["id"]))
    return games, box


def load_injuries(path):
    imp = defaultdict(float)  # (date, team) -> total pts docked
    if path and os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                imp[(r["date"], r["team"])] += float(r["pts_impact"])
    return imp


class Profiler:
    """Trailing-window team profiles with no lookahead."""

    def __init__(self, games, box):
        self.games, self.box = games, box
        self.history = defaultdict(list)  # team -> list of game dicts (chronological)
        self.last_date = {}               # team -> date of most recent game
        for g in games:
            b = box.get(g["id"], {})
            if g["away"] not in b or g["home"] not in b:
                continue
            for team, opp, pts, opp_pts, is_home in [
                (g["away"], g["home"], g["away_pts"], g["home_pts"], 0),
                (g["home"], g["away"], g["home_pts"], g["away_pts"], 1),
            ]:
                s, o = b[team], b[opp]
                poss = possessions(s["fga"], s["fta"], s["oreb"], s["turnover"],
                                   o["fga"], o["fta"], o["oreb"], o["turnover"])
                if poss <= 0:
                    continue
                self.history[team].append({
                    "date": g["date"], "opp": opp,
                    "ortg": 100 * pts / poss, "drtg": 100 * opp_pts / poss,
                    "pace": poss,
                    "net": 100 * (pts - opp_pts) / poss,
                })
            self.last_date[g["away"]] = g["date"]
            self.last_date[g["home"]] = g["date"]
        # league averages for blending early-season
        all_nets = [h["net"] for hs in self.history.values() for h in hs]
        self.league_net = sum(all_nets) / len(all_nets) if all_nets else 0.0
        all_o = [h["ortg"] for hs in self.history.values() for h in hs]
        all_d = [h["drtg"] for hs in self.history.values() for h in hs]
        all_p = [h["pace"] for hs in self.history.values() for h in hs]
        self.lg_ortg = sum(all_o) / len(all_o)
        self.lg_drtg = sum(all_d) / len(all_d)
        self.lg_pace = sum(all_p) / len(all_p)

    def profile(self, team, before_date):
        """Trailing WINDOW games strictly before before_date."""
        hs = [h for h in self.history[team] if h["date"] < before_date][-WINDOW:]
        n = len(hs)
        if n == 0:
            return None
        w = min(n, MIN_GAMES) / MIN_GAMES  # 1.0 once MIN_GAMES reached
        ortg = sum(h["ortg"] for h in hs) / n
        drtg = sum(h["drtg"] for h in hs) / n
        pace = sum(h["pace"] for h in hs) / n
        net = sum(h["net"] for h in hs) / n
        # light SOS: average opponent net faced (from full history = mild leak,
        # acceptable v1 approximation; v2 does iterative SRS)
        sos = sum(self._team_net(h["opp"]) for h in hs) / n
        adj_net = net + 0.25 * sos
        return {
            "n": n,
            "ortg": w * ortg + (1 - w) * self.lg_ortg,
            "drtg": w * drtg + (1 - w) * self.lg_drtg,
            "pace": w * pace + (1 - w) * self.lg_pace,
            "net": w * net + (1 - w) * self.league_net,
            "adj_net": w * adj_net + (1 - w) * self.league_net,
        }

    def _team_net(self, team):
        hs = self.history[team][-WINDOW:]
        return sum(h["net"] for h in hs) / len(hs) if hs else self.league_net

    def rest_days(self, team, date):
        d0 = self.last_date.get(team)
        if not d0:
            return 99
        return (datetime.strptime(date, "%Y-%m-%d") -
                datetime.strptime(d0, "%Y-%m-%d")).days - 1


def predict(prof, away, home, date, injuries):
    pa = prof.profile(away, date)
    ph = prof.profile(home, date)
    if not pa or not ph or pa["n"] < MIN_GAMES or ph["n"] < MIN_GAMES:
        return None
    pace = (pa["pace"] + ph["pace"]) / 2
    home_pts = ((ph["ortg"] + pa["drtg"]) / 2) * pace / 100
    away_pts = ((pa["ortg"] + ph["drtg"]) / 2) * pace / 100

    factors = [("base (efficiency matchup)", home_pts - away_pts)]
    margin = home_pts - away_pts

    margin += HOME_EDGE
    factors.append(("home court", HOME_EDGE))

    ra, rh = prof.rest_days(away, date), prof.rest_days(home, date)
    rest_edge = 0.0
    if ra == 0 and rh > 0:
        rest_edge += -B2B_PENALTY  # away on back-to-back helps home (+1.5)
        factors.append(("away back-to-back", round(-B2B_PENALTY, 1)))
    if rh == 0 and ra > 0:
        rest_edge += B2B_PENALTY  # home on back-to-back hurts home (-1.5)
        factors.append(("home back-to-back", round(B2B_PENALTY, 1)))
    if rh >= 3 and ra <= 1:
        rest_edge += REST_BONUS
        factors.append(("home rest advantage", REST_BONUS))
    if ra >= 3 and rh <= 1:
        rest_edge -= REST_BONUS
        factors.append(("away rest advantage", -REST_BONUS))
    margin += rest_edge

    inj = injuries.get((date, home), 0.0) - injuries.get((date, away), 0.0)
    if inj:
        margin -= inj
        factors.append(("injuries (home minus away dock)", -inj))

    prob = phi(margin / MARGIN_SIGMA)
    return {
        "away": away, "home": home, "date": date,
        "proj_away": round(away_pts, 1), "proj_home": round(home_pts, 1),
        "margin": round(margin, 1), "home_win_prob": round(prob, 3),
        "factors": [(k, round(v, 1)) for k, v in factors],
        "n_away": pa["n"], "n_home": ph["n"],
    }


def backtest():
    games, box = load()
    prof = Profiler(games, box)
    injuries = load_injuries(None)
    res = []
    for g in games:
        p = predict(prof, g["away"], g["home"], g["date"], injuries)
        if not p:
            continue
        actual_margin = g["home_pts"] - g["away_pts"]
        res.append((p["home_win_prob"], 1 if actual_margin > 0 else 0,
                    p["margin"], actual_margin))
    n = len(res)
    acc = sum(1 for pr, y, _, _ in res if (pr >= 0.5) == (y == 1)) / n
    brier = sum((pr - y) ** 2 for pr, y, _, _ in res) / n
    logloss = -sum(y * math.log(max(pr, 1e-9)) + (1 - y) * math.log(max(1 - pr, 1e-9))
                   for pr, y, _, _ in res) / n
    mae = sum(abs(pm - am) for _, _, pm, am in res) / n
    print(f"games scored: {n}")
    print(f"accuracy:     {acc:.1%}")
    print(f"brier score:  {brier:.4f}  (lower is better; 0.25 = coin flip)")
    print(f"log loss:     {logloss:.4f}")
    print(f"margin MAE:   {mae:.1f} pts")
    print("\ncalibration by confidence band:")
    for lo, hi, label in [(0.8, 1.01, ">=80%"), (0.6, 0.8, "60-80%"), (0.5, 0.6, "50-60%")]:
        band = [(pr, y) for pr, y, _, _ in res if lo <= max(pr, 1 - pr) < hi]
        if band:
            hit = sum(y if pr >= 0.5 else 1 - y for pr, y in band) / len(band)
            print(f"  {label:>7}: {len(band):5d} games, won {hit:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--predict", nargs=2, metavar=("AWAY", "HOME"))
    ap.add_argument("--date", default=None)
    ap.add_argument("--injuries", default=None)
    a = ap.parse_args()

    if a.backtest:
        backtest()
        return

    if a.predict:
        away, home = (s.upper() for s in a.predict)
        games, box = load()
        prof = Profiler(games, box)
        date = a.date or max(g["date"] for g in games)
        injuries = load_injuries(a.injuries)
        p = predict(prof, away, home, date, injuries)
        if not p:
            print("Not enough trailing data for one of these teams.")
            return
        print(f"{away} @ {home}  ({p['date']})")
        print(f"projected: {away} {p['proj_away']} - {home} {p['proj_home']}")
        print(f"home win prob: {p['home_win_prob']:.1%}  (margin {p['margin']:+.1f})")
        print("how this number is built:")
        for k, v in p["factors"]:
            print(f"  {v:+.1f}  {k}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
