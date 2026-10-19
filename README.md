# Dash's Work on NBA Prediction Model

Transparent, per-factor NBA game predictor. Every number shows its work —
no black boxes. (Same philosophy as the Yardline CFB app.)

## How it works (v1)

For each game, the model builds each team's **trailing-10-game** profile
(minimum 5 games; fewer than that blends with league average):

- **Offensive / defensive rating** — points scored/allowed per 100 possessions
- **Pace** — possessions per game
- **Four Factors** (both sides): eFG%, turnover rate, offensive rebound rate, FT rate
- **Strength of schedule** — light adjustment from average opponent net rating

Then:

```
expected pace        = avg(home pace, away pace)
home expected pts    = avg(home ORtg, away DRtg) × pace / 100
away expected pts    = avg(away ORtg, home DRtg) × pace / 100
expected margin      = home pts − away pts
                     + home court edge (~+2.0)
                     + rest adjustments (back-to-back, rest advantage)
                     − injury adjustments (manual input, see below)
win probability      = Φ(margin / 13.5)      # normal CDF, NBA margin σ ≈ 13.5
```

Inspired by the best public models: FiveThirtyEight's RAPTOR+Elo blend
(~66–67% accuracy) and ESPN's BPI (efficiency + SOS + rest + location).
v1 is team-level and fully transparent; player-level detail is a v2 option.

## Workflow

1. **Pull the data** — `python3 scripts/pull_data.py` (needs `BALLDONTLIE_API_KEY`
   in your environment). Writes two seasons of games + box scores to `data/`.
2. **Backtest** — `python3 scripts/nba_model_v1.py --backtest` walks forward
   through both seasons with no lookahead and prints the honest scorecard:
   accuracy, Brier score, log loss, margin error, calibration by confidence band.
3. **Predict** — `python3 scripts/nba_model_v1.py --predict "LAL" "BOS"`
   (away, home) prints projected score, win %, and every factor's point
   contribution.

## Injuries

Box scores don't carry inactives, and they're the biggest single-game swing
factor in the NBA. v1 takes them as a manual input: add rows to
`data/injuries.csv` (`date,team,player,pts_impact`) on game day and the
model docks that team's expected points accordingly.

## Data

CSVs live in `data/` (git-ignored if large). Schema matches the pull script:
`nba_games_2023-2025.csv`, `nba_boxscores_2023-2025.csv`.

## Roadmap

- v1: team-level transparent model + walk-forward backtest (this repo now)
- v2: full SRS-style SOS iteration, Four-Factor-driven matchup edges
- v3: player-level adjustments (BPM/on-off) for injury impact automation
