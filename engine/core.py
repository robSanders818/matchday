"""Shared data loading: league scoring, player-match table, team-match table.

Player-match rows (both seasons) carry raw stats, minutes, team/opponent and fantasy points
in this league's position-specific scoring (Sleeper tags each stat line with pos_<p>_<stat>,
so a player is scored at the position he's listed at).
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
ME = "robSanders818"
POSITIONS = ("F", "M", "D", "GK")


def load(name):
    return json.loads((RAW / name).read_text())


LEAGUE = load("league.json")
SCORING = LEAGUE["scoring_settings"]
META = load("meta.json")
SEASON, DONE = META["season"], META["done"]


def weights(pos: str) -> dict:
    """Points per unit of each raw stat for a position, e.g. {'g': 9, 'kp': 2, ...}."""
    pre = f"pos_{pos.lower()}_"
    return {k[len(pre):]: v for k, v in SCORING.items() if k.startswith(pre)}


def score(stats: dict) -> tuple[float, str | None]:
    pts, pos = 0.0, None
    for k, v in stats.items():
        if k.startswith("pos_"):
            pos = pos or k.split("_")[1].upper()
            pts += SCORING.get(k, 0) * v
    return pts, pos


@lru_cache(maxsize=1)
def players() -> dict:
    return load("players.json")


@lru_cache(maxsize=1)
def team_names() -> dict:
    """Sleeper team id -> abbreviation."""
    out = {}
    for p in players().values():
        if p.get("team") and p.get("team_abbr"):
            out[str(p["team"])] = p["team_abbr"]
    return out


ESPN_TO_SLEEPER = {"MNC": "MCI", "MAN": "MUN"}


@lru_cache(maxsize=1)
def espn_home() -> dict:
    """(date, frozenset{abbr, abbr}) -> home abbr, from ESPN (dates +-1 day for time zones)."""
    p = RAW / "espn_matches.json"
    if not p.exists():
        return {}
    out = {}
    for m in json.loads(p.read_text()):
        h, a = ESPN_TO_SLEEPER.get(m["home"], m["home"]), ESPN_TO_SLEEPER.get(m["away"], m["away"])
        d = pd.Timestamp(m["date"][:10])
        for off in (-1, 0, 1):
            out[((d + pd.Timedelta(days=off)).strftime("%Y-%m-%d"), frozenset((h, a)))] = h
    return out


def is_home(date, team_id, opp_id):
    tn = team_names()
    t, o = tn.get(str(team_id)), tn.get(str(opp_id))
    h = espn_home().get((str(date)[:10], frozenset((t, o)))) if t and o else None
    return np.nan if h is None else float(h == t)


@lru_cache(maxsize=1)
def player_matches() -> pd.DataFrame:
    rows = []
    seasons = [y for y in (SEASON - 2, SEASON - 1, SEASON) if (RAW / f"stats_{y}.json").exists()]
    for season in seasons:
        for w, recs in load(f"stats_{season}.json").items():
            for r in recs:
                st = r["stats"]
                if not st.get("gp") and not st.get("min"):
                    continue
                pts, pos = score(st)
                raw = {k: v for k, v in st.items() if not k.startswith("pos_")}
                rows.append({"season": season, "week": int(w), "pid": r["player_id"], "team": str(r.get("team")),
                             "opp": str(r.get("opponent")), "game": r.get("game_id"), "date": r.get("date"),
                             "pos": pos, "pts": pts, **raw})
    df = pd.DataFrame(rows)
    num = df.columns.difference(["season", "week", "pid", "team", "opp", "game", "date", "pos", "pts"])
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").fillna(0)
    df["t"] = (df.season - SEASON) * 38 + df.week          # time index: 2026 wk1 = 1, 2025 wk38 = 0
    df = df[df["min"] > 0].reset_index(drop=True)
    games = df.drop_duplicates(["game", "team"])[["game", "team", "opp", "date"]]
    games["home"] = [is_home(d, t, o) for d, t, o in zip(games.date, games.team, games.opp)]
    return df.merge(games[["game", "team", "home"]], on=["game", "team"], how="left")


@lru_cache(maxsize=1)
def team_matches() -> pd.DataFrame:
    """One row per team per match: goals/xG/shots on target for and against."""
    pm = player_matches()
    g = pm.groupby(["season", "week", "game", "team", "opp", "t", "home"], dropna=False).agg(
        gf=("g", "sum"), xg=("xg", "sum"), sot=("sot", "sum"), og_by=("og", "sum")).reset_index()
    opp = g.rename(columns={"team": "opp", "opp": "team", "gf": "ga_raw", "xg": "xga", "sot": "sota", "og_by": "og_for"})
    tm = g.merge(opp[["season", "week", "game", "team", "ga_raw", "xga", "sota", "og_for"]], on=["season", "week", "game", "team"], how="left")
    tm["gf"] = tm.gf + tm.og_for.fillna(0)          # own goals by the opponent count for us
    tm["ga"] = tm.ga_raw.fillna(0) + tm.og_by
    # xG only exists from 2026-27, so team strength uses a proxy available in both seasons:
    # half actual goals, half shots-on-target x league conversion rate (~0.32).
    conv = tm.gf.sum() / max(tm.sot.sum(), 1)
    tm["sg"] = 0.5 * tm.gf + 0.5 * conv * tm.sot          # "strength goals" for
    tm["sga"] = 0.5 * tm.ga + 0.5 * conv * tm.sota        # ... and against
    return tm.drop(columns=["ga_raw"])


def rosters() -> dict:
    users = {u["user_id"]: u for u in load("users.json")}
    out = {}
    for r in load("rosters.json"):
        u = users.get(r["owner_id"], {})
        st = r["settings"]
        out[r["roster_id"]] = {
            "owner": u.get("display_name", "?"),
            "team": (u.get("metadata") or {}).get("team_name") or u.get("display_name", "?"),
            "players": [str(p) for p in (r.get("players") or [])],
            "reserve": [str(p) for p in (r.get("reserve") or [])],
            "record": f"{st.get('wins', 0)}-{st.get('losses', 0)}-{st.get('ties', 0)}",
            "wins": st.get("wins", 0), "ties": st.get("ties", 0),
            "pf": st.get("fpts", 0) + st.get("fpts_decimal", 0) / 100,
        }
    return out


def my_roster_id() -> int:
    return next(r for r, v in rosters().items() if v["owner"].lower() == ME.lower())


def fixtures(week: int) -> dict:
    """team id -> opponent id for a future matchweek."""
    f = load(f"fixtures_{SEASON}.json").get(str(week), [])
    return {str(r["team"]): str(r["opponent"]) for r in f if r.get("team")}


def fixture_home(week: int) -> dict:
    """team id -> 1.0 home / 0.0 away (nan if unknown) for a future matchweek."""
    f = load(f"fixtures_{SEASON}.json").get(str(week), [])
    return {str(r["team"]): is_home(r.get("date"), r["team"], r["opponent"]) for r in f if r.get("team")}


def pname(pid) -> str:
    p = players().get(str(pid), {})
    return p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or str(pid)


def ppos(pid) -> str:
    p = players().get(str(pid), {})
    return p.get("position") or ((p.get("fantasy_positions") or ["?"])[0])


def scheduled_opp(week: int) -> int | None:
    """Your opponent's roster id for a matchweek, from data/schedule.json (Sleeper's API doesn't expose soccer matchups)."""
    p = ROOT / "data" / "schedule.json"
    return json.loads(p.read_text()).get(str(week)) if p.exists() else None
