#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01 entry (v1: empirical transition kernel).

Reads the whole chart history and forecasts each game's probability of
holding each of the top 10 places, averaged over the 72-hour scored window.

Model (see DESIGN.md for the evidence):
  1. Fit P(destination | rank-bin, horizon) from every pair of reads:
     destination in {places 1..10, 11+, OFF}, rank bins 1..10 singletons
     then 11-15, 16-20, 21-30, 31-45, 46-70, 71-120, OFF-recent, OFF-old.
     Horizon buckets [0,3,6,12,24,36,48,72,120] h, log-horizon
     interpolation, tau=0 anchor = identity. Dirichlet smoothing toward each
     horizon bucket's pooled destination distribution.
  2. Fit, per horizon bucket, q[k](tau): the share of place-k slots held by
     games never seen in the history, and E[# never-seen games on the chart].
  3. Per game: rank at the last read, plus a momentum displacement
     beta * d_prev * (1 - exp(-tau/tau_m)) where d_prev is the recent
     6-hour-equivalent climb; look up the kernel at the effective rank and
     average over tau = 0..71 h. Games with a known publish date
     (details.created) are truncated to OFF past created + 28 days.
  4. Per place: renormalise known-game probabilities to sum to 1 - q_k,
     then name games greedily where naming beats leaving the game in the
     unnamed pool under the model's own expected log score. The unnamed
     mass covers brand-new games.
  5. top10: the 10 games with the highest place-weighted score.

Fallback: with too little history, or if the model fails for any reason,
emit the persistence baseline (v0) rather than no forecast.

Python 3.13, standard library only, no network, fully deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- hyperparameters (tuned on rolling backtests; see REPORT.md) ---
# Each is overridable by environment variable for tuning runs; the grader
# leaves them unset, so the defaults below are the entry's behaviour.
ALPHA = float(os.environ.get("ROBLOX_ALPHA", "1000.0"))     # Dirichlet pseudocounts
BETA = float(os.environ.get("ROBLOX_BETA", "0.5"))          # momentum multiplier
TAU_M = float(os.environ.get("ROBLOX_TAU_M", "12.0"))       # momentum saturation (h)
DPREV_CAP = 40.0      # cap on |d_prev| (ranks per 6 h)
SCALE_CAP = (0.5, 2.0)  # clamp on the per-place renormalisation scale
# Tuning hooks (env-gated; unset in the grader, so defaults rule there):
#   ROBLOX_NAME_MODE: greedy (default) | top<N> (name the N likeliest) | all
#   ROBLOX_PRIOR: global (default) | local (neighbouring source bins)
#   ROBLOX_Q_SHRINK: 0 (default) .. 1, shrink per-place q toward the mean
#   ROBLOX_RECENCY_DAYS: 0 (default, off) or e-folding days for pair weights
#   ROBLOX_DEBUG=1: print per-place diagnostics to stderr
NAME_MODE = os.environ.get("ROBLOX_NAME_MODE", "greedy")
PRIOR_MODE = os.environ.get("ROBLOX_PRIOR", "local")
Q_SHRINK = float(os.environ.get("ROBLOX_Q_SHRINK", "0.5"))
RECENCY_DAYS = float(os.environ.get("ROBLOX_RECENCY_DAYS", "0.0"))
DEBUG = os.environ.get("ROBLOX_DEBUG", "") == "1"
AGE_LIMIT_DAYS = 28   # chart only lists games published in the last 28 days
OFF_RECENT_HOURS = 24.0

TAU_EDGES = [0.0, 3.0, 6.0, 12.0, 24.0, 36.0, 48.0, 72.0, 120.0]
N_DEST = 12           # places 1..10, 11+, OFF
DEST_OFF = 11
DEST_REST = 10
N_SRC = 18            # ranks 1..10, 11-15, 16-20, 21-30, 31-45, 46-70, 71-120,
                      # OFF-recent, OFF-old
SRC_OFF_RECENT = 16
SRC_OFF_OLD = 17
MAX_ROWS_PER_PLACE = 100


def parse_time(s):
    if s is None:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def rank_bin(rank):
    """Source bin for a rank (1-based)."""
    if rank <= 10:
        return rank - 1
    if rank <= 15:
        return 10
    if rank <= 20:
        return 11
    if rank <= 30:
        return 12
    if rank <= 45:
        return 13
    if rank <= 70:
        return 14
    return 15


def dest_bin(rank):
    """Destination bin for a rank (1-based), or DEST_OFF when off-chart."""
    if rank is None:
        return DEST_OFF
    if rank <= 10:
        return rank - 1
    return DEST_REST


def tau_bucket(dt_hours):
    for b in range(len(TAU_EDGES) - 1):
        if TAU_EDGES[b] <= dt_hours < TAU_EDGES[b + 1]:
            return b
    return len(TAU_EDGES) - 2


def bucket_center(b):
    lo, hi = TAU_EDGES[b], TAU_EDGES[b + 1]
    if lo <= 0:
        return 1.5
    return math.sqrt(lo * hi)


class Kernel:
    """Empirical P(dest | source bin, tau) with log-tau interpolation."""

    def __init__(self, counts, totals):
        # counts[src][bucket][dest], totals[src][bucket]
        self.anchors = []  # (tau, dists[src] -> list of 12 probs)
        n_buckets = len(TAU_EDGES) - 1
        # pooled destination distribution per bucket (the Dirichlet prior)
        pooled = []
        for b in range(n_buckets):
            agg = [0.0] * N_DEST
            tot = 0.0
            for s in range(N_SRC):
                for d in range(N_DEST):
                    agg[d] += counts[s][b][d]
                tot += totals[s][b]
            if tot > 0:
                pooled.append([a / tot for a in agg])
            else:
                pooled.append(None)
        # global pooled distribution (fallback prior for empty buckets)
        gtot = [0.0] * N_DEST
        gsum = 0.0
        for b in range(n_buckets):
            if pooled[b] is not None:
                for d in range(N_DEST):
                    gtot[d] += pooled[b][d]
                gsum += 1.0
        global_prior = (
            [g / gsum for g in gtot] if gsum > 0 else [1.0 / N_DEST] * N_DEST
        )
        # local priors: pooled destination distribution of the neighbouring
        # source bins (s-1, s, s+1) in the same bucket; falls back to the
        # bucket pooled distribution when the neighbourhood is too thin.
        local_prior = {}
        if PRIOR_MODE == "local":
            for b in range(n_buckets):
                for s in range(N_SRC):
                    agg = [0.0] * N_DEST
                    tot = 0.0
                    for s2 in (s - 1, s, s + 1):
                        if 0 <= s2 < N_SRC:
                            for d in range(N_DEST):
                                agg[d] += counts[s2][b][d]
                            tot += totals[s2][b]
                    if tot >= 20:
                        local_prior[(s, b)] = [a / tot for a in agg]
        # identity anchor at tau = 0
        ident = []
        for s in range(N_SRC):
            row = [0.0] * N_DEST
            if s < 10:
                row[s] = 1.0
            elif s < SRC_OFF_RECENT:
                row[DEST_REST] = 1.0
            else:
                row[DEST_OFF] = 1.0
            ident.append(row)
        self.anchors.append((0.0, ident))
        for b in range(n_buckets):
            tot_b = sum(totals[s][b] for s in range(N_SRC))
            if tot_b <= 0:
                continue  # no observations in this bucket: not an anchor
            dists = []
            for s in range(N_SRC):
                n = totals[s][b]
                prior_s = local_prior.get(
                    (s, b), pooled[b] if pooled[b] is not None else global_prior
                )
                row = [
                    (counts[s][b][d] + ALPHA * prior_s[d]) / (n + ALPHA)
                    for d in range(N_DEST)
                ]
                dists.append(row)
            self.anchors.append((bucket_center(b), dists))
        self.anchors.sort(key=lambda a: a[0])

    def dist(self, src, tau):
        """Interpolated destination distribution for source bin src at tau."""
        anchors = self.anchors
        if tau <= anchors[0][0]:
            return anchors[0][1][src]
        if tau >= anchors[-1][0]:
            return anchors[-1][1][src]
        for i in range(len(anchors) - 1):
            t0, d0 = anchors[i]
            t1, d1 = anchors[i + 1]
            if t0 <= tau <= t1:
                if t0 <= 0:
                    w = tau / t1 if t1 > 0 else 0.0
                else:
                    w = (math.log(tau) - math.log(t0)) / (
                        math.log(t1) - math.log(t0)
                    )
                r0, r1 = d0[src], d1[src]
                return [r0[d] + w * (r1[d] - r0[d]) for d in range(N_DEST)]
        return anchors[-1][1][src]


class Curve:
    """A scalar curve over tau with log-tau interpolation (anchor at 0)."""

    def __init__(self, points):
        # points: list of (tau, value), sorted; (0, v0) should be first
        self.points = sorted(points)

    def value(self, tau):
        pts = self.points
        if tau <= pts[0][0]:
            return pts[0][1]
        if tau >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            t0, v0 = pts[i]
            t1, v1 = pts[i + 1]
            if t0 <= tau <= t1:
                if t0 <= 0:
                    w = tau / t1 if t1 > 0 else 0.0
                else:
                    w = (math.log(tau) - math.log(t0)) / (
                        math.log(t1) - math.log(t0)
                    )
                return v0 + w * (v1 - v0)
        return pts[-1][1]


def load_history(path):
    reads = []
    created = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("ok") is False:
                continue
            chart = rec.get("up_and_coming")
            if not chart:
                continue
            t = parse_time(rec.get("at"))
            if t is None:
                continue
            ranks = {}
            for row in chart:
                try:
                    uid = int(row["universe_id"])
                    ranks[uid] = int(row["rank"])
                except (KeyError, TypeError, ValueError):
                    continue
            if ranks:
                reads.append((t, ranks))
            details = rec.get("details")
            if isinstance(details, dict):
                for key, det in details.items():
                    if not isinstance(det, dict):
                        continue
                    try:
                        uid = int(key)
                    except (TypeError, ValueError):
                        continue
                    if uid in created:
                        continue
                    c = parse_time(det.get("created"))
                    if c is not None:
                        created[uid] = c
    reads.sort(key=lambda r: r[0])
    return reads, created


def fit(reads):
    """Fit the transition kernel, never-seen curves and new-games curve."""
    n = len(reads)
    n_buckets = len(TAU_EDGES) - 1
    counts = [
        [[0.0] * N_DEST for _ in range(n_buckets)] for _ in range(N_SRC)
    ]
    totals = [[0.0] * n_buckets for _ in range(N_SRC)]
    q_counts = [[0.0] * 10 for _ in range(n_buckets)]
    q_tot = [[0.0] * 10 for _ in range(n_buckets)]
    new_counts = [0.0] * n_buckets
    new_tot = [0.0] * n_buckets

    seen = set()
    last_seen = {}
    t_end = reads[-1][0]
    for i in range(n):
        t_i, chart_i = reads[i]
        seen |= set(chart_i)
        for uid in chart_i:
            last_seen[uid] = t_i
        if RECENCY_DAYS > 0:
            age_days = (t_end - t_i).total_seconds() / 86400.0
            w_pair = math.exp(-age_days / RECENCY_DAYS)
        else:
            w_pair = 1.0
        for j in range(i + 1, n):
            t_j, chart_j = reads[j]
            dt = (t_j - t_i).total_seconds() / 3600.0
            if dt <= 0:
                continue
            b = tau_bucket(dt)
            # never-seen share per place + new games on chart
            new_on = 0
            for uid, rk in chart_j.items():
                if uid not in seen:
                    new_on += 1
                    if rk <= 10:
                        q_counts[b][rk - 1] += w_pair
            new_counts[b] += new_on * w_pair
            new_tot[b] += w_pair
            for rk in range(1, 11):
                q_tot[b][rk - 1] += w_pair
            # transitions for every game seen up to i
            for uid in seen:
                rk_i = chart_i.get(uid)
                if rk_i is not None:
                    src = rank_bin(rk_i)
                else:
                    age = (t_i - last_seen[uid]).total_seconds() / 3600.0
                    src = (
                        SRC_OFF_RECENT
                        if age <= OFF_RECENT_HOURS
                        else SRC_OFF_OLD
                    )
                d = dest_bin(chart_j.get(uid))
                counts[src][b][d] += w_pair
                totals[src][b] += w_pair

    kernel = Kernel(counts, totals)
    q_curves = []
    for k in range(10):
        pts = [(0.0, 0.0)]
        for b in range(n_buckets):
            if q_tot[b][k] > 0:
                pts.append(
                    (bucket_center(b), q_counts[b][k] / q_tot[b][k])
                )
        q_curves.append(Curve(pts))
    new_pts = [(0.0, 0.0)]
    for b in range(n_buckets):
        if new_tot[b] > 0:
            new_pts.append(
                (bucket_center(b), new_counts[b] / new_tot[b])
            )
    new_curve = Curve(new_pts)
    return kernel, q_curves, new_curve


def forecast_model(reads, created, places, window_hours, t0):
    kernel, q_curves, new_curve = fit(reads)
    t_last, last_chart = reads[-1]
    n_hours = max(1, int(round(window_hours)))

    # per-game recent climb (6h-equivalent) and last-seen info
    appearances = {}
    for t, chart in reads:
        for uid, rk in chart.items():
            appearances.setdefault(uid, []).append((t, rk))
    last_seen = {uid: app[-1][0] for uid, app in appearances.items()}

    def d_prev(uid):
        app = appearances[uid]
        if len(app) < 2:
            return 0.0
        t_now, rk_now = app[-1]
        best = None
        for t, rk in app[:-1]:
            dt = (t_now - t).total_seconds() / 3600.0
            if 2.0 <= dt <= 18.0:
                err = abs(dt - 6.0)
                if best is None or err < best[0]:
                    best = (err, dt, rk)
        if best is None:
            return 0.0
        _, dt, rk_then = best
        scale = 6.0 / max(3.0, min(12.0, dt))
        d = (rk_then - rk_now) * scale
        return max(-DPREV_CAP, min(DPREV_CAP, d))

    # average never-seen mass and new-game count over the window
    q_k = [0.0] * places
    e_new = 0.0
    for tau in range(n_hours):
        for k in range(places):
            q_k[k] += q_curves[k].value(float(tau))
        e_new += new_curve.value(float(tau))
    q_k = [q / n_hours for q in q_k]
    if Q_SHRINK > 0:
        q_mean = sum(q_k) / len(q_k)
        q_k = [(1.0 - Q_SHRINK) * q + Q_SHRINK * q_mean for q in q_k]
    e_new /= n_hours

    # per-game place probabilities, averaged over the window
    prob = {}   # uid -> list of places probabilities
    p_on = {}   # uid -> mean P(on chart)
    for uid in sorted(appearances):
        rk0 = last_chart.get(uid)
        if rk0 is not None:
            src0 = rank_bin(rk0)
            mom = d_prev(uid)
        else:
            age = (t_last - last_seen[uid]).total_seconds() / 3600.0
            src0 = SRC_OFF_RECENT if age <= OFF_RECENT_HOURS else SRC_OFF_OLD
            mom = 0.0
        exit_hour = None
        c = created.get(uid)
        if c is not None:
            exit_hour = (c + timedelta(days=AGE_LIMIT_DAYS) - t0).total_seconds() / 3600.0
            if exit_hour < 0:
                # on the chart past the supposed limit: trust the chart
                exit_hour = None
        acc = [0.0] * places
        on_acc = 0.0
        for tau in range(n_hours):
            if exit_hour is not None and tau > exit_hour:
                continue  # aged out: all mass OFF
            if rk0 is not None:
                disp = BETA * mom * (1.0 - math.exp(-tau / TAU_M)) if tau > 0 else 0.0
                r_eff = rk0 - disp
                src = rank_bin(max(1, int(round(r_eff)))) if r_eff <= 120 else 15
            else:
                src = src0
            dist = kernel.dist(src, float(tau))
            for k in range(places):
                acc[k] += dist[k]
            on_acc += 1.0 - dist[DEST_OFF]
        prob[uid] = [a / n_hours for a in acc]
        p_on[uid] = on_acc / n_hours

    # per-place assembly: renormalise, then greedy naming
    rows = []
    for k in range(places):
        qk = min(max(q_k[k], 0.0), 0.9)
        total = sum(prob[uid][k] for uid in prob)
        if total <= 0:
            continue
        scale = (1.0 - qk) / total
        scale_clamped = max(SCALE_CAP[0], min(SCALE_CAP[1], scale))
        cand = sorted(
            ((prob[uid][k] * scale_clamped, uid) for uid in prob if prob[uid][k] > 0),
            key=lambda x: (-x[0], x[1]),
        )
        named_mass = sum(p for p, _ in cand)  # == (1-qk) up to the scale clamp
        unnamed_mass = named_mass  # mass still in the pool (known games)
        pool_games = sum(p_on.values()) + e_new  # expected unnamed games
        picked = []
        if NAME_MODE == "all":
            picked = [(uid, p) for p, uid in cand[:MAX_ROWS_PER_PLACE]]
        elif NAME_MODE.startswith("top"):
            try:
                n_top = int(NAME_MODE[3:])
            except ValueError:
                n_top = MAX_ROWS_PER_PLACE
            picked = [(uid, p) for p, uid in cand[:n_top]]
        else:  # greedy: name a game iff naming beats leaving it in the pool
            for p, uid in cand:
                if len(picked) >= MAX_ROWS_PER_PLACE:
                    break
                pool_mass = unnamed_mass + qk
                if pool_mass <= 0 or pool_games <= 0:
                    break
                u = pool_mass / pool_games
                if u <= 0:
                    break
                u_mass_after = unnamed_mass - p + qk
                games_after = pool_games - p_on[uid]
                u_after = u_mass_after / games_after if games_after > 0 else u
                if u_after <= 0:
                    gain = p * math.log(p / u) if p > 0 else 0.0
                else:
                    gain = p * math.log(p / u) + u_mass_after * math.log(
                        u_after / u
                    )
                if gain > 0:
                    picked.append((uid, p))
                    unnamed_mass -= p
                    pool_games -= p_on[uid]
        if DEBUG:
            print(
                f"place {k+1}: q={qk:.3f} total_known={total:.3f} "
                f"scale={scale:.3f} named={len(picked)} "
                f"max_p={cand[0][0]:.4f} pool_games={pool_games:.1f}",
                file=sys.stderr,
            )
        for uid, p in picked:
            pf = math.floor(p * 1e6) / 1e6
            if pf > 0:
                rows.append({"universe_id": uid, "place": k + 1, "p": pf})

    # top10: select the 10 games most likely to be in the top 10 at all
    # (maximises expected hits), ordered by place-weighted score (best
    # expected rank first, for the agreement tie-break).
    p_top10 = {uid: sum(prob[uid]) for uid in prob}
    place_w = {}
    for uid in prob:
        s = 0.0
        for k in range(places):
            s += prob[uid][k] * (places + 1 - (k + 1))
        place_w[uid] = s
    selected = sorted(prob, key=lambda u: (-p_top10[u], u))[:places]
    top10 = sorted(selected, key=lambda u: (-place_w[u], u))
    # pad (degenerate histories only) with dummy ids that match no game
    pad = 9_000_000_000
    while len(top10) < places:
        top10.append(pad)
        pad += 1
    return {"probabilities": rows, "top10": top10}


def forecast_persistence(reads, places):
    """v0 fallback: the chart does not move."""
    _, last_chart = reads[-1]
    ids = [uid for uid, _ in sorted(last_chart.items(), key=lambda kv: kv[1])]
    rows = []
    for place in range(1, places + 1):
        if place > len(ids):
            break
        rows.append({"universe_id": ids[place - 1], "place": place, "p": 0.5})
        if place == 1:
            if len(ids) > 1:
                rows.append(
                    {"universe_id": ids[1], "place": place, "p": 0.3}
                )
            continue
        rows.append(
            {"universe_id": ids[place - 2], "place": place, "p": 0.15}
        )
        if place < len(ids):
            rows.append(
                {"universe_id": ids[place], "place": place, "p": 0.15}
            )
    top10 = ids[:places]
    pad = 9_000_000_000
    while len(top10) < places:
        top10.append(pad)
        pad += 1
    return {"probabilities": rows, "top10": top10}


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    output = task.get("output", "forecast.json")
    window = task.get("window") or {}
    slots = window.get("slots", 72)
    slot_minutes = window.get("slot_minutes", 60)
    window_hours = slots * slot_minutes / 60.0
    t0 = parse_time(task.get("history_until"))

    reads, created = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    if not reads:
        raise SystemExit("history.ndjson holds no usable read")
    if t0 is None:
        t0 = reads[-1][0]

    forecast = None
    if len(reads) >= 3:
        try:
            forecast = forecast_model(reads, created, places, window_hours, t0)
        except Exception as exc:  # never fail the run: fall back
            print(f"model failed ({exc!r}); using persistence", file=sys.stderr)
    if forecast is None:
        forecast = forecast_persistence(reads, places)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output), "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
