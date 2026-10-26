"""Matchweek projections in this league's scoring.

For each player and fixture:
  rates     per-90 rate of every stat the league scores, from the current season (recent
            matches weighted more) blended with the previous season; goals/assists pulled
            toward xG/xA where available
  minutes   recency-weighted minutes in his team's recent matches (0 when unused/injured),
            times injury availability for live projections
  matchup   attacking stats scaled by opponent strength (elasticities from matchup_study.py,
            bigger swings for low/middling players) and home/away; clean sheets, goals
            conceded and keeper saves from a Poisson model of own defence x opponent attack
All tunable settings live in PARAMS (tuned on 2025-26 by tune.py -> data/tuned_params.json).
"""

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

import core

PARAMS = {
    "decay": 0.70,        # weight multiplier per older match (current season)
    "prior_min": 600,     # previous season counts as this many minutes of evidence
    "min_decay": 0.70,    # minutes model: per older team match
    "min_window": 6,      # minutes model: team matches looked at
    "team_prior": 8,      # previous season counts as this many matches for team strength
    "elast_scale": 1.0,   # multiplier on the measured opponent-strength effects
    "home_att": 0.0,      # attacking boost at home (minus the same away)
    "home_def": 0.0,      # goals-conceded cut at home (plus the same away)
    "xg_share": 0.5,      # how far goals/assists are pulled toward xG/xA
}
ATTACK = {"g", "at", "sot", "kp", "acnc", "sm", "pkd", "pks", "cos"}
TEAM_STATS = {"cs", "ga"}
DEF_ACTIONS = {"tkw", "int", "clr", "aer", "bs", "sv"}
INJURY_AVAIL = {"Out": 0.0, "Doubtful": 0.25, "Questionable": 0.6, "GTD": 0.6, "Suspended": 0.0}
# Share of OUR projection in the final number (rest = Sleeper's, rescaled), by position.
MODEL_SHARE = {"F": 1.0, "M": 0.75, "D": 0.25, "GK": 0.0}
# Multiplier on the final projection so its average matches actual scoring (model runs ~1-1.5 pts low).
CALIBRATION = {"F": 1.0, "M": 1.0, "D": 1.0, "GK": 1.0}

_tuned = core.ROOT / "data" / "tuned_params.json"
if _tuned.exists():
    _t = json.loads(_tuned.read_text())
    PARAMS.update(_t.get("params", {}))
    MODEL_SHARE.update(_t.get("model_share", {}))
    CALIBRATION.update(_t.get("calibration", {}))


@lru_cache(maxsize=1)
def elasticity():
    return json.loads((core.ROOT / "data" / "matchup_elasticity.json").read_text())


def team_strength(tm, before_t, season, P):
    """Per team: recency-weighted strength goals/shots for and against (+ previous-season prior)."""
    lg = tm[tm.t < before_t]
    cur = lg[lg.season == season].sort_values("t")
    prev = tm[tm.season == season - 1].groupby("team")[["sg", "sga", "sot", "sota"]].mean()
    league_avg = float(cur.sg.mean()) if len(cur) else float(prev.sg.mean())
    weak = {"sg": prev.sg.quantile(0.2), "sga": prev.sga.quantile(0.8),       # promoted teams: below average
            "sot": prev.sot.quantile(0.2), "sota": prev.sota.quantile(0.8)}
    rows = {}
    for team, g in cur.groupby("team"):
        w = P["decay"] ** np.arange(len(g))[::-1]
        p = prev.loc[team] if team in prev.index else pd.Series(weak)
        pw = P["team_prior"] * w.mean()
        rows[team] = {c: (np.dot(w, g[c]) + pw * p[c]) / (w.sum() + pw) for c in ("sg", "sga", "sot", "sota")}
    out = pd.DataFrame(rows).T
    out["xgd"] = out.sg - out.sga
    out.attrs["avg"] = league_avg
    out.attrs["avg_sot"] = float(out.sot.mean())
    return out


def player_rates(pm, before_t, season, P):
    """Per-90 rates for every raw stat, recency-weighted, with previous-season prior."""
    hist = pm[(pm.t < before_t) & pm.season.isin([season - 1, season])].copy()
    has = (hist.xg + hist.xa) > 0
    xs = P["xg_share"]
    hist.loc[has, "g"] = (1 - xs) * hist.loc[has, "g"] + xs * hist.loc[has, "xg"]
    hist.loc[has, "at"] = (1 - xs) * hist.loc[has, "at"] + xs * hist.loc[has, "xa"]
    stats = sorted(set().union(*[core.weights(p).keys() for p in core.POSITIONS]) - {"min", "cs90"})
    stats = [s for s in stats if s in hist]
    cur = hist[hist.season == season].sort_values("t")
    cur = cur.assign(k=cur.groupby("pid").cumcount(ascending=False))
    cur["w"] = P["decay"] ** cur.k
    prev = hist[hist.season == season - 1]
    agg_c = cur[stats].mul(cur.w, axis=0).groupby(cur.pid).sum()
    min_c = (cur["min"] * cur.w).groupby(cur.pid).sum()
    agg_p = prev.groupby("pid")[stats].sum()
    min_p = prev.groupby("pid")["min"].sum()
    pids = agg_c.index.union(agg_p.index)
    agg_c, min_c = agg_c.reindex(pids, fill_value=0), min_c.reindex(pids, fill_value=0)
    agg_p, min_p = agg_p.reindex(pids, fill_value=0), min_p.reindex(pids, fill_value=0)
    pw = np.minimum(min_p, P["prior_min"])
    prior_rate = agg_p.div(min_p.replace(0, np.nan), axis=0).fillna(0)
    num = agg_c.add(prior_rate.mul(pw, axis=0))
    den = (min_c + pw).replace(0, np.nan)
    rates = num.div(den, axis=0).fillna(0) * 90
    rates["evidence_min"] = min_c + pw
    return rates


def expected_minutes(pm, tm, before_t, season, P):
    cur_tm = tm[(tm.t < before_t) & (tm.season == season)]
    cur_pm = pm[(pm.t < before_t) & (pm.season == season)]
    last_team = cur_pm.sort_values("t").groupby("pid").team.last()
    mins = cur_pm.set_index(["pid", "game"])["min"]
    out = {}
    for team, games in cur_tm.sort_values("t").groupby("team").game:
        g = list(games)[-int(P["min_window"]):]
        w = P["min_decay"] ** np.arange(len(g))[::-1]
        for pid in last_team[last_team == team].index:
            m = np.array([mins.get((pid, x), 0.0) for x in g])
            out[pid] = float(np.dot(w, m) / w.sum())
    return pd.Series(out)


def project(target_t, fixtures, live=False, season=None, home=None, params=None):
    """fixtures: team id -> opponent id; home: team id -> 1 home / 0 away (optional)."""
    season = season or core.SEASON
    P = {**PARAMS, **(params or {})}
    home = home or {}
    pm, tm = core.player_matches(), core.team_matches()
    el = elasticity()
    ts = team_strength(tm, target_t, season, P)
    avg = ts.attrs["avg"]
    rates = player_rates(pm, target_t, season, P)
    emin = expected_minutes(pm, tm, target_t, season, P)
    hist = pm[(pm.t < target_t) & pm.season.isin([season - 1, season])].sort_values("t")
    pos_of = hist.groupby("pid").pos.last()
    team_of = hist[hist.season == season].groupby("pid").team.last()
    base_p90 = {}
    for pid in rates.index.intersection(pos_of.index):
        w = core.weights(pos_of[pid])
        r = rates.loc[pid]
        base_p90[pid] = sum(w.get(s, 0) * r.get(s, 0) for s in w if s in r.index and s not in TEAM_STATS)
    bp = pd.Series(base_p90, dtype=float)
    tiers = bp.groupby(pos_of.reindex(bp.index)).transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "middling", "elite"]))
    tier_mult = {k: v / np.mean(list(el["by_tier"].values())) for k, v in el["by_tier"].items()}

    rows = []
    db = core.players()
    for pid in emin.index:
        pos, team = pos_of.get(pid), team_of.get(pid)
        if pos not in core.POSITIONS or team not in fixtures or team not in ts.index:
            continue
        if (db.get(pid) or {}).get("position") not in core.POSITIONS:   # skip team/club entities
            continue
        opp = fixtures[team]
        if opp not in ts.index:
            continue
        m = emin[pid]
        if live:
            m *= INJURY_AVAIL.get(((db.get(pid) or {}).get("metadata") or {}).get("soccer_injury_status"), 1.0)
        r = rates.loc[pid] if pid in rates.index else pd.Series(dtype=float)
        w = core.weights(pos)
        z = (ts.loc[opp, "xgd"] - el["opp_xgd_mean"]) / el["opp_xgd_sd"]
        tier = tiers.get(pid, "middling")
        h = home.get(team, np.nan)
        hsign = 0.0 if h != h else (1.0 if h == 1 else -1.0)
        att_mult = 1 + P["elast_scale"] * el["by_pos"][pos]["att"] * z * tier_mult.get(tier, 1)
        att_mult = float(np.clip(att_mult * (1 + P["home_att"] * hsign), 0.5, 1.8))
        # Poisson team defence: expected goals against, shots on target faced.
        opp_att = (ts.loc[opp, "sg"] / avg) * (1 - P["home_att"] * hsign)
        lam = ts.loc[team, "sga"] * opp_att * (1 - P["home_def"] * hsign)
        sota = ts.loc[team, "sota"] * (ts.loc[opp, "sot"] / ts.attrs["avg_sot"]) * (1 - P["home_def"] * hsign)
        p_cs = math.exp(-lam)
        play = min(m, 90) / 90
        pts = 0.0
        comp = {"att": 0.0, "dfn": 0.0, "team": 0.0, "other": 0.0}
        for s, wt in w.items():
            if not wt or s in ("min", "cs90"):
                continue
            if s == "cs":
                v, k = wt * p_cs * (1.0 if m >= 60 else m / 60 * 0.5), "team"
            elif s == "ga":
                v, k = wt * lam * play, "team"
            elif s == "sv" and pos == "GK":
                v, k = wt * max(sota - lam * 0.9, 0) * play, "dfn"
            else:
                v = wt * r.get(s, 0.0) * play * (att_mult if s in ATTACK else 1.0)
                k = "att" if s in ATTACK else ("dfn" if s in DEF_ACTIONS else "other")
            comp[k] += v
            pts += v
        rows.append({"pid": pid, "name": core.pname(pid), "pos": pos, "team": team, "opp": opp, "home": h,
                     "e_min": m, "proj": pts, "att": comp["att"], "dfn": comp["dfn"], "teampts": comp["team"],
                     "other": comp["other"], "p_cs": p_cs, "opp_z": z, "tier": tier,
                     "evidence_min": float(r.get("evidence_min", 0))})
    return pd.DataFrame(rows).set_index("pid") if rows else pd.DataFrame()


def live_projection(week=None):
    """Final projection for a future matchweek: our model blended with Sleeper by position."""
    week = week or core.DONE + 1
    p = project(target_t=core.DONE + 1, fixtures=core.fixtures(week), live=True, home=core.fixture_home(week))
    path = core.RAW / f"sleeper_future_{core.SEASON}.json"
    fut = json.loads(path.read_text()) if path.exists() else {}
    sl = fut.get(str(week), {})
    # Sleeper fills future weeks with flat placeholders (identical week to week) until shortly
    # before the matchweek; those carry no fixture information, so only blend real ones.
    nxt = fut.get(str(week + 1), {})
    common = [k for k in sl if k in nxt]
    if common and sum(sl[k] == nxt[k] for k in common) / len(common) > 0.8:
        sl = {}
    p.attrs["sleeper_live"] = bool(sl)
    p["model"] = p.proj
    p["sleeper"] = [sl.get(pid, np.nan) for pid in p.index]
    # Sleeper's levels are off (e.g. GKs ~4 pts vs ~7-12 actual), but its ordering tested well:
    # rescale each position to our model's average before blending.
    p["sleeper_raw"] = p.sleeper
    for pos in core.POSITIONS:
        m = (p.pos == pos) & p.sleeper.notna() & (p.e_min >= 45)
        if m.sum() >= 5 and p.loc[m, "sleeper"].mean() > 0:
            p.loc[p.pos == pos, "sleeper"] *= p.loc[m, "model"].mean() / p.loc[m, "sleeper"].mean()
    share = p.pos.map(MODEL_SHARE).fillna(1.0)
    has = p.sleeper.notna()
    p.loc[has, "proj"] = share[has] * p.model[has] + (1 - share[has]) * p.sleeper[has]
    p["proj"] *= p.pos.map(CALIBRATION).fillna(1.0)
    p.loc[p.e_min < 1, "proj"] = 0.0     # injured/unused: no points, whatever Sleeper projects
    p["week"] = week
    return p
