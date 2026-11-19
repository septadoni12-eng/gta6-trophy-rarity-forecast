# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/septadoni12-eng/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/septadoni12-eng/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts the earn rate of all 55 GTA VI PS5 trophies on Exophase at 2026-11-26T00:00:00Z — five quantiles each, 275 numbers — by forecasting earners and tracked players separately and dividing at the end. A printed rate is earners / players_tracked at Exophase's last refresh, and the two move differently in launch week: the tracked base explodes (diluting every rate), while earners accumulate per trophy at an archetype's pace — the tutorial mission stays pinned near 100%, story chapters decay with depth, quick feats adopt fast, grind trophies barely move, the platinum stays at a handful of earners. The program collapses each case's read history to its distinct refresh states, fits a saturating growth curve to the player count and a Weibull adoption curve to each trophy's earners (speed and ceiling shrunk toward the trophy's archetype), shrinks the resulting median toward the last-read rate on the logit scale, and reads its spread off the case's own realised logit moves between refreshes — recency-weighted, pooled by rate level — plus parameter jitter, a point mass on "no move", and a jump mixture. A seeded Monte Carlo turns that distribution into the five quantiles, so two runs write identical bytes. Python 3.13, standard library only, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=tvfbjxfd&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
