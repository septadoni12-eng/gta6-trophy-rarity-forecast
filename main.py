"""Forecast GTA VI's PS5 trophy earn rates on Exophase, five quantiles a trophy.

Written for make.fun's gta6-trophy-rarity-forecast-01 (Python 3.13, standard
library only, no network). The grader runs this file once per case: it reads
the single case file in the input directory and writes one forecast file per
case to the output directory, in the exact shape the competition specifies.

How the forecast is built (the short version; the repo README tells it too):

  A printed earn rate is earners / players_tracked at Exophase's last
  refresh. The two move differently, so they are forecast separately and
  divided at the end:

  * players_tracked follows a saturating launch-week growth curve, fitted
    to the case's own refresh states (the history repeats verbatim between
    Exophase refreshes, so it is first collapsed to the distinct states);
    a history that shows no growth yet falls back to a launch prior.
  * each trophy's earners follow a saturating adoption curve (a Weibull
    CDF in hours since release) whose speed and ceiling are fitted per
    trophy and shrunk toward the trophy's archetype: the tutorial mission
    is pinned near 100%, story chapters decay with depth, one-off feats
    adopt fast, grind trophies barely move inside week one, the platinum
    stays at a handful of earners.
  * the median is that structural forecast shrunk toward the last-read
    rate on the logit scale (persistence is the anchor every winning
    forecast on this platform starts from; the structure moves it only
    as far as the evidence carries).
  * the spread is an empirical fan, not a formula's sigma: the case's own
    realised logit moves between refreshes, recency-weighted and pooled
    by rate level (rare trophies swing harder in logit terms), scaled to
    the horizon, plus parameter jitter, a small point mass on "no move",
    and a jump mixture for regime surprises. Quantiles are read off a
    seeded Monte Carlo of that distribution, so two runs of the same case
    write the same bytes.
"""

import json
import math
import os
import random
import zlib
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
TAU_GRID = [0.7 * (3000.0 / 0.7) ** (i / 27.0) for i in range(28)]
SIMS = 400

# Archetype table: (Weibull shape k, prior time constant tau0 in hours,
# ceiling as a share of the tracked base).
ARCH = {
    "platinum": (1.00, 2000.0, 0.030),
    "mission": (1.00, 60.0, 0.60),
    "strand": (1.00, 90.0, 0.45),
    "feat": (0.90, 20.0, 0.50),
    "grind": (0.80, 300.0, 0.35),
}
# Prior-mode earners growth over unobserved public play time: earners
# multiply by (1 + play_hours / tau_g) ** gamma_g. Calibrated on the
# documented launch: in the first hours the tracked base multiplied ~6x
# while early adopters earned fast (story/feat earners grew several-fold)
# and grind earners barely moved at all.
GROWTH = {
    "platinum": (48.0, 0.30),
    "story": (2.5, 0.65),
    "mission": (4.0, 0.50),
    "strand": (5.0, 0.50),
    "feat": (1.0, 0.90),
    "grind": (24.0, 0.35),
}
STORY_CAPS = (0.92, 0.82, 0.72, 0.63, 0.55)
GRIND_IDS = {14, 15, 17, 18, 19, 25, 28, 30, 31, 32, 35, 40, 42, 43, 52, 53, 54}
STRAND_IDS = {22, 23, 24, 26, 27}


def archetype(trophy):
    tid = trophy["id"]
    if tid == 1:
        return "platinum", 0
    if tid == 2:
        return "pinned", 0
    if 3 <= tid <= 7:
        return "story", tid - 3
    if 8 <= tid <= 13:
        return "mission", 0
    if tid in STRAND_IDS:
        return "strand", 0
    if tid in GRIND_IDS:
        return "grind", 0
    return "feat", 0


def arch_params(name, depth):
    if name == "story":
        return 1.15, 30.0, STORY_CAPS[min(depth, 4)]
    return ARCH[name]


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(z):
    if z >= 0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def round4(x):
    v = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(v) if float(v).is_integer() else v


def weibull(h, tau, k):
    if h <= 0:
        return 0.0
    return 1.0 - math.exp(-((h / tau) ** k))


class State:
    __slots__ = ("h", "players", "rates", "span")

    def __init__(self, h, players, rates):
        self.h = h
        self.players = players
        self.rates = rates
        self.span = 0.25


def build_states(case, release):
    """Collapse the read history to the distinct page states, oldest first."""
    states = []
    for read in case["history"]:
        h = (parse_time(read["slot"]) - release).total_seconds() / 3600.0
        key = (read["players_tracked"], tuple(read["rates"]))
        if states and states[-1][0] == key:
            continue
        states.append((key, State(h, read["players_tracked"], list(read["rates"]))))
    out = [s for _, s in states]
    for j, st in enumerate(out):
        nxt = out[j + 1].h if j + 1 < len(out) else (parse_time(case["at"]) - release).total_seconds() / 3600.0
        st.span = max(0.17, nxt - st.h)
    return out


def players_prior(p0, h):
    return p0 * (1.0 + max(h, 0.0) / 1.5) ** 1.35


def forecast_players(states, h_last, h_target):
    """Return (players at target, prior_mode flag)."""
    p0 = states[0].players
    p_last = states[-1].players
    distinct = sorted({s.players for s in states})
    if len(distinct) < 2 or distinct[-1] / max(distinct[0], 1) < 1.05:
        return players_prior(p0, h_target), True
    xs, ys, ws = [], [], []
    for s in states:
        xs.append(math.log1p(max(s.h, 0.0)))
        ys.append(math.log(max(s.players, 1)))
        ws.append(s.span)
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx if sxx > 1e-9 else 0.0
    intercept = my - slope * mx
    fit = math.exp(intercept + slope * math.log1p(max(h_target, 0.0)))
    span_growth = max(s.h for s in states) - min(s.h for s in states)
    w_fit = min(1.0, span_growth / 48.0)
    blended = math.exp(w_fit * math.log(fit) + (1.0 - w_fit) * math.log(players_prior(p0, h_target)))
    blended = min(max(blended, p_last * 1.05), p_last * 80.0)
    return blended, False


def earners_series(states, idx):
    pts = []
    for s in states:
        r = s.rates[idx]
        if r is not None:
            pts.append((max(s.h, 0.02), r * s.players / 100.0, s.span))
    return pts


def fit_earners(pts, name, depth, p_target, h_last, h_target, prior_mode):
    """Median earners at the target hour for one trophy."""
    k, tau0, cap = arch_params(name, depth)
    e_last = pts[-1][1] if pts else 0.0
    cap_abs = p_target * cap
    if len(pts) >= 2 and pts[-1][1] > pts[0][1] * 1.02 and pts[-1][1] > 0:
        soft = max(e_last, 1.0) * 0.02
        best = None
        for tau in TAU_GRID:
            num = den = 0.0
            for h, e, w in pts:
                wv = weibull(h, tau, k)
                if wv > 1e-6:
                    num += w * e / wv
                    den += w
            if den <= 0:
                continue
            a = num / den
            sse = 0.0
            for h, e, w in pts:
                pred = a * weibull(h, tau, k)
                sse += w * (math.log(e + soft) - math.log(pred + soft)) ** 2
            if best is None or sse < best[0]:
                best = (sse, tau, a)
        _, tau_fit, a_fit = best
        shrink = len(pts) / (len(pts) + 6.0)
        tau = math.exp(shrink * math.log(tau_fit) + (1.0 - shrink) * math.log(tau0))
        a = min(a_fit, cap_abs / max(weibull(h_target, tau, k), 1e-6) if a_fit > cap_abs else a_fit)
        e_target = a * weibull(h_target, tau, k)
        resid = math.sqrt(best[0] / max(sum(w for _, _, w in pts), 1e-9))
        return max(e_target, e_last * 0.99), min(0.85, 0.18 + resid)
    # Flat or empty series. In prior mode the observed earners predate public
    # play (a pre-launch page), so grow them over the public play time ahead
    # at the archetype's rate. In fit mode a flat series is real evidence:
    # earners are simply not moving, so project them (nearly) flat.
    if prior_mode:
        tau_g, gamma_g = GROWTH[name]
        play_h = h_target - max(h_last, 0.0)
        e_target = e_last * ((1.0 + max(play_h, 0.0) / tau_g) ** gamma_g)
        sd_e = 0.42
    else:
        e_target = e_last * 1.03
        sd_e = 0.30
    e_target = min(e_target, cap_abs)
    floor = 1.0 if cap > 0.05 else 0.3
    return max(e_target, min(floor, cap_abs), e_last * 0.99), sd_e


def band_sigmas(states, h_at):
    """Recency-weighted robust scale of realised logit moves, per sqrt hour,
    pooled by the rate level the move started from."""
    moves = {"rare": [], "mid": [], "common": []}
    for j in range(len(states) - 1):
        a, b = states[j], states[j + 1]
        dh = b.h - a.h
        if dh <= 0:
            continue
        age = max(0.0, h_at - b.h)
        w = math.exp(-age / 36.0)
        for i in range(len(a.rates)):
            r1, r2 = a.rates[i], b.rates[i]
            if r1 is None or r2 is None:
                continue
            u = abs(logit(r2) - logit(r1)) / math.sqrt(dh)
            band = "rare" if r1 < 0.5 else ("mid" if r1 <= 5.0 else "common")
            moves[band].append((u, w))
    pooled = [uw for band in moves.values() for uw in band]

    def scale(pairs):
        if len(pairs) < 6:
            return None
        pairs = sorted(pairs)
        total = sum(w for _, w in pairs)
        acc = 0.0
        med = pairs[-1][0]
        for u, w in pairs:
            acc += w
            if acc >= total / 2.0:
                med = u
                break
        return min(0.85, 1.4826 * med)

    pooled_scale = scale(pooled)
    out = {}
    for band, pairs in moves.items():
        s = scale(pairs)
        out[band] = s if s is not None else (pooled_scale if pooled_scale is not None else 0.34)
    return out


def quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def forecast_case(case, raw):
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    at = parse_time(case["at"])
    h_at = (at - release).total_seconds() / 3600.0
    h_target = h_at + case["horizon_minutes"] / 60.0
    levels = list(case.get("levels") or LEVELS)
    states = build_states(case, release)
    if not states:
        states = [State(0.0, 1, [None] * len(case["trophies"]))]
    last = states[-1]
    p_target, prior_mode = forecast_players(states, last.h, h_target)
    sigmas = band_sigmas(states, h_at) if not prior_mode else {"rare": 0.34, "mid": 0.30, "common": 0.22}
    horizon_h = max(case["horizon_minutes"] / 60.0, 0.05)
    h_eff = horizon_h / (1.0 + horizon_h / 72.0)
    rng = random.Random(zlib.crc32(raw))
    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        name, depth = archetype(trophy)
        series = [s.rates[idx] for s in states if s.rates[idx] is not None]
        r_last = series[-1] if series else None
        if name == "pinned":
            base = r_last if r_last is not None else 99.9
            med = min(99.99, max(96.0, base - 0.10 * (horizon_h / 24.0)))
            draws = []
            for _ in range(SIMS):
                z = logit(med) + rng.gauss(0.0, 0.45)
                if rng.random() < 0.08:
                    z -= abs(rng.gauss(0.0, 1.6))
                draws.append(min(99.99, inv_logit(z)))
            draws.sort()
            rows.append({"id": trophy["id"], "q": [round4(quantile(draws, t)) for t in levels]})
            continue
        pts = earners_series(states, idx)
        e_target, sd_e = fit_earners(pts, name, depth, p_target, last.h, h_target, prior_mode)
        struct_rate = min(99.9, 100.0 * e_target / max(p_target, 1.0))
        w_struct = 0.85 if prior_mode else 0.80
        anchor_logit = logit(r_last) if r_last is not None else logit(struct_rate)
        med_logit = w_struct * logit(struct_rate) + (1.0 - w_struct) * anchor_logit
        band = "rare" if (r_last or struct_rate) < 0.5 else ("mid" if (r_last or struct_rate) <= 5.0 else "common")
        sd_move = sigmas[band] * math.sqrt(h_eff)
        if prior_mode:
            sd_move *= 1.25
        sd_move *= 1.0 + 0.5 * math.exp(-max(e_target, 0.0) / 50.0)
        sd_move = min(1.50, max(0.38 if prior_mode else 0.30, sd_move))
        sd_p = 0.20 if prior_mode else 0.12
        p_stick = 0.22 if prior_mode else 0.03
        draws = []
        for _ in range(SIMS):
            u = rng.random()
            if u < p_stick:
                draws.append(r_last if (prior_mode and r_last is not None) else inv_logit(med_logit))
                continue
            e_draw = e_target * math.exp(rng.gauss(0.0, sd_e))
            p_draw = p_target * math.exp(rng.gauss(0.0, sd_p))
            r0 = min(99.9, max(0.001, 100.0 * e_draw / max(p_draw, 1.0)))
            z = w_struct * logit(r0) + (1.0 - w_struct) * anchor_logit
            z += rng.gauss(0.0, sd_move)
            if rng.random() < 0.10:
                z += rng.gauss(0.0, sd_move * 0.8)
            draws.append(inv_logit(z))
        draws.sort()
        rows.append({"id": trophy["id"], "q": [round4(quantile(draws, t)) for t in levels]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        raw = path.read_bytes()
        case = json.loads(raw.decode("utf-8"))
        out = forecast_case(case, raw)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
