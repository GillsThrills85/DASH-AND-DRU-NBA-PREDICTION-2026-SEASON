"""
NBA data pull for the prediction model (balldontlie.io).

Pulls:
  1. All regular-season games for 2023-24 and 2024-25 (backtest/training data)
  2. Team box scores for every one of those games (ratings + Four Factors)

Usage:
  export BALLDONTLIE_API_KEY="your-key-here"
  python3 scripts/pull_data.py

Output (data/ folder):
  nba_games_2023-2025.csv       - one row per game: date, teams, final score
  nba_boxscores_2023-2025.csv   - one row per team-game: all box score stats

The key is read from the environment only -- never put it in this file.
"""
import csv
import os
import time
import urllib.request
import urllib.parse
import json

API_KEY = os.environ.get("BALLDONTLIE_API_KEY")
if not API_KEY:
    raise SystemExit("Set BALLDONTLIE_API_KEY first: export BALLDONTLIE_API_KEY=\"...\"")

BASE = "https://api.balldontlie.io/nba/v1"
PAUSE = 1.0  # be polite; raise if your tier allows more
SEASONS = [2023, 2024]  # balldontlie labels by starting year: 2023 = 2023-24
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(OUT, exist_ok=True)


def api(path, params):
    q = urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(
        f"{BASE}{path}?{q}",
        headers={"Authorization": API_KEY, "User-Agent": "nba-model-pull/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def paged(path, params):
    page, per_page = 1, 100
    while True:
        d = api(path, {**params, "page": page, "per_page": per_page})
        items = d.get("data", [])
        if not items:
            break
        yield from items
        meta = d.get("meta", {})
        if not meta.get("next_page"):
            break
        page = meta["next_page"]
        time.sleep(PAUSE)


print("Pulling games...")
games = []
for g in paged("/games", [("seasons[]", s) for s in SEASONS] + [("postseason", "false")]):
    games.append({
        "game_id": g["id"],
        "date": g["date"][:10],
        "season": g["season"],
        "away_team": g["visitor_team"]["abbreviation"],
        "home_team": g["home_team"]["abbreviation"],
        "away_score": g["visitor_team_score"],
        "home_score": g["home_team_score"],
    })
    if len(games) % 500 == 0:
        print(f"  ...{len(games)} games")
print(f"Total games: {len(games)}")

with open(os.path.join(OUT, "nba_games_2023-2025.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(games[0].keys()))
    w.writeheader()
    w.writerows(games)

print("Pulling box scores (this takes a while)...")
FIELDS = ["game_id", "team", "home", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
          "oreb", "dreb", "reb", "ast", "stl", "blk", "turnover", "pf", "pts"]
rows = []
ids = [g["game_id"] for g in games]
for i in range(0, len(ids), 25):
    chunk = ids[i:i + 25]
    params = [("game_ids[]", gid) for gid in chunk]
    for b in paged("/box_scores", params):
        t = b["team"]["abbreviation"]
        rows.append({
            "game_id": b["game"]["id"],
            "team": t,
            "home": 1 if t == b["game"]["home_team"]["abbreviation"] else 0,
            "fgm": b["fgm"], "fga": b["fga"], "fg3m": b["fg3m"], "fg3a": b["fg3a"],
            "ftm": b["ftm"], "fta": b["fta"], "oreb": b["oreb"], "dreb": b["dreb"],
            "reb": b["reb"], "ast": b["ast"], "stl": b["stl"], "blk": b["blk"],
            "turnover": b["turnover"], "pf": b["pf"], "pts": b["pts"],
        })
    done = min(i + 25, len(ids))
    if (i // 25) % 10 == 0:
        print(f"  ...{done}/{len(ids)} games")
    time.sleep(PAUSE)

with open(os.path.join(OUT, "nba_boxscores_2023-2025.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

print(f"Done. {len(rows)} team-games written to {OUT}/")
