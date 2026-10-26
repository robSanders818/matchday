"""Phone page for a matchweek: projections for every rostered player, kickoffs, injuries and
confirmed ESPN lineups (published about an hour before each kickoff).

    uv run matchday.py [--week W] [--opp N] [--no-fetch]

Writes out/matchday.html (web/matchday.template.html + embedded data). Run it again after
lineups come out; players in games that already have lineups get marked starting/bench.
"""

import argparse
import json
import os
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone

import pandas as pd
import requests

import core
import model
import soccer

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1"
es = requests.Session()
es.headers["User-Agent"] = "Mozilla/5.0"


def norm(s: str) -> str:
    s = (s or "").replace("ß", "ss").replace("ø", "o").replace("ł", "l")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower().replace("-", " ").strip()


def espn_week(dates: list[str]) -> list[dict]:
    """ESPN matches on these dates (YYYY-MM-DD) with kickoff and, once announced, lineups."""
    events = {}
    for d in dates:
        for e in es.get(f"{ESPN}/scoreboard", params={"dates": d.replace("-", "")}, timeout=30).json().get("events", []):
            events[e["id"]] = e
    events = list(events.values())
    out = []
    for e in events:
        comp = e["competitions"][0]
        teams = {c["homeAway"]: core.ESPN_TO_SLEEPER.get(c["team"]["abbreviation"], c["team"]["abbreviation"]) for c in comp["competitors"]}
        m = {"id": e["id"], "kickoff": e["date"], "home": teams["home"], "away": teams["away"],
             "status": comp["status"]["type"]["name"], "lineups": {}}
        try:
            s = es.get(f"{ESPN}/summary", params={"event": e["id"]}, timeout=30).json()
            for r in s.get("rosters", []):
                ab = core.ESPN_TO_SLEEPER.get(r["team"]["abbreviation"], r["team"]["abbreviation"])
                if r.get("roster"):
                    m["lineups"][ab] = [{"name": norm(x["athlete"].get("displayName")), "last": norm(x["athlete"].get("lastName")),
                                         "starter": bool(x.get("starter"))} for x in r["roster"]]
        except (requests.RequestException, ValueError):
            pass
        out.append(m)
    return out


def lineup_status(name: str, squad: list[dict] | None) -> str | None:
    """'start' / 'bench' (in the matchday squad but not starting) / 'out' (not in the squad, or name not found), None if no lineup yet."""
    if not squad:
        return None
    n = norm(name)
    hit = [x for x in squad if x["name"] == n]
    if not hit:
        last = n.split()[-1]
        hit = [x for x in squad if x["last"] == last or x["name"].split()[-1] == last]
        hit = hit if len(hit) == 1 else []
    if not hit:
        toks = set(n.split())
        hit = [x for x in squad if toks & set(x["name"].split())]
        hit = hit if len(hit) == 1 else []
    if not hit:
        # Not in the 20-man squad: out, unless the name just didn't match. A whole team's names
        # failing would mean a format change, so only trust it when most of the squad is known.
        return "out" if len(squad) >= 18 else None
    return "start" if hit[0]["starter"] else "bench"


def build(week: int, opp: int | None) -> dict:
    p = soccer.proj(week)
    tn = core.team_names()
    db = core.players()
    R = core.rosters()
    dates = sorted({r["date"] for r in core.load(f"fixtures_{core.SEASON}.json").get(str(week), []) if r.get("date")})
    matches = espn_week(dates) if dates else []
    by_team = {}
    for m in matches:
        for side in ("home", "away"):
            by_team[m[side]] = m
    # Points per 90 when fit, for players the model has at ~0 minutes (injured) who turn out to start.
    pm = core.player_matches()
    pm = pm[pm.season >= core.SEASON - 1]
    g = pm.groupby("pid").agg(pts=("pts", "sum"), mins=("min", "sum"))
    p90 = (g.pts / g.mins.clip(lower=1) * 90).where(g.mins >= 270)
    players = []
    for rid, v in R.items():
        for pid in dict.fromkeys(v["players"] + v["reserve"]):
            if core.ppos(pid) not in core.POSITIONS:
                continue
            info = db.get(pid, {})
            inj = ((info.get("metadata") or {}).get("soccer_injury_status")) or info.get("injury_status")
            r = p.loc[pid] if pid in p.index else None
            team = tn.get(str(r.team), "") if r is not None else info.get("team_abbr", "")
            m = by_team.get(team)
            players.append({
                "pid": pid, "rid": rid, "name": core.pname(pid), "pos": core.ppos(pid), "elig": soccer.eligible(pid), "team": team,
                "opp": tn.get(str(r.opp), "") if r is not None else "",
                "home": None if r is None or pd.isna(r.home) else bool(r.home),
                "proj": round(float(r.proj), 2) if r is not None else 0.0,
                "emin": round(float(r.e_min), 1) if r is not None else 0.0,
                "pcs": round(float(r.p_cs), 3) if r is not None and core.ppos(pid) in ("D", "GK") else None,
                "p90": round(float(p90.get(pid)) * model.CALIBRATION.get(core.ppos(pid), 1.0), 2) if pd.notna(p90.get(pid)) else None,
                "inj": inj, "ir": pid in v["reserve"],
                "kickoff": m["kickoff"] if m else None,
                "lineup": lineup_status(core.pname(pid), m["lineups"].get(team)) if m else None,
            })
    return {
        "week": week, "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "sleeperLive": bool(p.attrs.get("sleeper_live")), "me": core.my_roster_id(), "opp": opp,
        "teams": [{"rid": rid, "name": v["team"], "owner": v["owner"], "record": v["record"]} for rid, v in R.items()],
        "formations": soccer.FORMATIONS, "playerSd": soccer.PLAYER_SD,
        # Your lineup as saved in Sleeper right now ("0" = empty slot).
        "myStarters": next((r.get("starters") or [] for r in core.load("rosters.json") if r["roster_id"] == core.my_roster_id()), []),
        "matches": [{k: m[k] for k in ("kickoff", "home", "away", "status")} | {"lineups": sorted(m["lineups"])} for m in matches],
        "players": players,
    }


def adjusted(p: dict) -> float:
    """Projection given confirmed lineup / injury, same rules as the phone page."""
    s = p["lineup"] or ("out" if p["ir"] or p["inj"] in ("Out", "Suspended") else None)
    r90 = p["proj"] / p["emin"] * 90 if p["emin"] >= 5 else (p["p90"] or 0)
    return {"out": 0.0, "start": r90 * (90 if p["pos"] == "GK" else 78) / 90, "bench": r90 * 10 / 90}.get(s, p["proj"])


def report(data: dict, out) -> None:
    """Print my recommended XI and what changed since the last run (for alerts)."""
    me = [p for p in data["players"] if p["rid"] == data["me"] and not p["ir"]]
    df = pd.DataFrame([{"proj": adjusted(p)} for p in me], index=[p["pid"] for p in me])
    by = {p["pid"]: p for p in me}
    total, lineup = soccer.best_lineup(list(df.index), df)
    xi = [pid for _, pid in lineup]
    print(f"Recommended XI ({soccer.formation(lineup)}, {total:.1f} pts):")
    for slot, pid in lineup:
        p = by[pid]
        print(f"  {slot:<4}{p['name']:<24}{p['team']:>4} v {p['opp']:<4} {df.at[pid, 'proj']:5.1f}  {p['lineup'] or 'lineup not out'}")
    flagged = [p for p in me if p["pid"] in xi and p["lineup"] in ("bench", "out")]
    for p in flagged:
        print(f"  !! {p['name']} is {'on the bench' if p['lineup'] == 'bench' else 'not in the squad'} but still the best option")
    saved = data["myStarters"]
    empty = sum(1 for x in saved if x in ("0", "", None))
    dead = [by[x]["name"] for x in saved if x in by and adjusted(by[x]) == 0]
    if empty or dead:
        print(f"  !! Your saved Sleeper lineup has {empty} empty slot(s)" + (f" and starts {', '.join(dead)} (not playing)" if dead else ""))
    last = out / "last_xi.json"
    prev = json.loads(last.read_text()) if last.exists() else {}
    if prev.get("week") == data["week"]:
        ins = [by[x]["name"] for x in xi if x not in prev["xi"]]
        outs = [by[x]["name"] if x in by else core.pname(x) for x in prev["xi"] if x not in xi]
        print("CHANGES: " + (f"start {', '.join(ins)} instead of {', '.join(outs)}" if ins or outs else "none"))
    last.write_text(json.dumps({"week": data["week"], "xi": xi}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int)
    ap.add_argument("--opp", type=int)
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    if not a.no_fetch:
        # Fresh stats/rosters first, then restart so core picks up the new meta (last scored week).
        subprocess.run([sys.executable, str(core.ROOT / "fetch_sleeper.py")], check=True)
        os.execv(sys.executable, [sys.executable, __file__, "--no-fetch"] + sys.argv[1:])
    week = a.week or core.DONE + 1
    data = build(week, a.opp or core.scheduled_opp(week))
    out = core.ROOT / "out"
    out.mkdir(exist_ok=True)
    (out / "matchday.json").write_text(json.dumps(data, indent=1))
    tpl = (core.ROOT / "web" / "matchday.template.html").read_text()
    (out / "matchday.html").write_text(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    report(data, out)
    known = sum(1 for m in data["matches"] if m["lineups"])
    print(f"MW{week}: {len(data['players'])} players, {len(data['matches'])} matches ({known} with lineups) -> out/matchday.html")


if __name__ == "__main__":
    main()
