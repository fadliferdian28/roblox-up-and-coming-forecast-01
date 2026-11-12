# roblox-up-and-coming-forecast-01 — entry by @jackpoot

Forecasts Roblox's Up-and-Coming chart top 10, place by place, for every hour
of the 72 hours after entries close. One program, `main.py` (Python 3.13,
standard library only, no network), reads `/input/history.ndjson` and
`/input/task.json` and writes `/output/forecast.json`: per place (1–10) a
probability for each game named being at that place, plus the top 10 in order.

## How it works

See `DESIGN.md` in the work directory for the full design. In short, the
forecaster:

1. Reads the whole chart history (ranks, player counts, publish dates where
   the house recorder supplies them), not just the last snapshot.
2. Fits the chart's dynamics from the history itself: how far ranks diffuse
   per hour, how persistent rank momentum is, how fast games leave the chart
   (including the hard 28-day publish-age exit), and how often brand-new games
   — never seen in the history — take top-10 places at each horizon.
3. Propagates every game's rank distribution forward hour by hour over the
   72-hour window, averages over the window, and converts the result into
   honest per-place probabilities, leaving the measured share for
   never-seen games unnamed (the scorer spreads it over the games the forecast
   does not name).
4. Is fully deterministic: no randomness, no wall-clock use, no network.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Validation

Local validation (schema, determinism run twice byte-identical, runtime, and
scores on the practice window plus rolling backtests inside the history) is
documented in `REPORT.md` in the work directory.
