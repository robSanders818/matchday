"""Sleeper EPL fantasy assistant.

    uv run soccer.py week [--opp N] [--week W]   # best lineup for the next matchweek (+ win odds vs team N)
    uv run soccer.py waivers [--pos D]           # best free agents next week / next 5 weeks, who to drop
    uv run soccer.py values [--team N]           # rest-of-season value of every rostered player
    uv run soccer.py trade --give "A, B" --get "C"   # lineup impact for you and the other team
    uv run soccer.py league                      # standings, luck, positional strength by team
    uv run soccer.py teams                       # roster ids, to use with --opp / --team

Run `uv run fetch_sleeper.py` first each week for fresh stats.
"""

import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import ndtr

import core
import model

# Sleeper soccer lineups: a goalkeeper plus 10 outfielders in one of these formations (D-M-F).
# A player can fill any position he's eligible for (e.g. Okafor F or M); he's scored the same either way.
FORMATIONS = [(3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1), (5, 3, 2), (5, 4, 1)]
OUTFIELD = ("D", "M", "F")
ROS_WEEKS = 5
PLAYER_SD = 6.0        # typical miss on one player's matchweek score (backtest MAE ~5.3)


@lru_cache(maxsize=None)
def proj(week: int) -> pd.DataFrame:
    return model.live_projection(week)


def weeks_ahead(n=ROS_WEEKS):
    return list(range(core.DONE + 1, min(core.DONE + 1 + n, 39)))


def eligible(pid) -> list[str]:
    fp = [x for x in (core.players().get(str(pid), {}).get("fantasy_positions") or []) if x in core.POSITIONS]
    return fp or [core.ppos(pid)]


def best_lineup(pids: list[str], p: pd.DataFrame) -> tuple[float, list[tuple[str, str]]]:
    """Optimal starters [(position, pid)] for one matchweek: best goalkeeper, then the best
    formation and assignment of outfielders by eligibility (exact, small DP over position counts)."""
    val = {pid: float(p.at[pid, "proj"]) if pid in p.index else 0.0 for pid in dict.fromkeys(pids)}
    gks = sorted((x for x in val if "GK" in eligible(x)), key=lambda x: -val[x])
    cap = tuple(max(f[i] for f in FORMATIONS) for i in range(3))
    dp = {(0, 0, 0): (0.0, ())}
    for pid in val:
        el = [OUTFIELD.index(e) for e in eligible(pid) if e in OUTFIELD]
        if not el:
            continue
        nxt = dict(dp)
        for state, (tot, lu) in dp.items():
            for i in el:
                if state[i] < cap[i]:
                    st = tuple(c + (j == i) for j, c in enumerate(state))
                    cand = (tot + val[pid], lu + ((OUTFIELD[i], pid),))
                    if st not in nxt or cand[0] > nxt[st][0]:
                        nxt[st] = cand
        dp = nxt
    best = (-1e9, ())
    for f in FORMATIONS:
        for st, cand in dp.items():
            if all(st[i] <= f[i] for i in range(3)) and cand[0] > best[0]:
                best = cand
    total, lineup = best
    lineup = sorted(lineup, key=lambda x: (OUTFIELD.index(x[0]), -val[x[1]]))
    if gks:
        total += val[gks[0]]
        lineup = [("GK", gks[0])] + lineup
    return total, lineup


def formation(lineup) -> str:
    return "-".join(str(sum(1 for s, _ in lineup if s == pos)) for pos in OUTFIELD)


def label(pid, p):
    tn = core.team_names()
    if pid not in p.index:
        return f"{core.pname(pid):<24}{core.ppos(pid):<3} no fixture/minutes data"
    r = p.loc[pid]
    cs = f" CS {r.p_cs:.0%}" if r.pos in ("D", "GK") else ""
    return (f"{r['name']:<24}{r.pos:<3}{tn.get(r.team, r.team):>4} v {tn.get(r.opp, r.opp):<4} "
            f"{r.proj:5.1f} pts  ({r.e_min:3.0f} min{cs})")


def cmd_teams(_):
    for rid, v in core.rosters().items():
        print(f"  {rid}: {v['team']} ({v['owner']}) {v['record']}  PF {v['pf']:.0f}" + ("   <- you" if rid == core.my_roster_id() else ""))


def cmd_week(a):
    week = a.week or core.DONE + 1
    p = proj(week)
    R = core.rosters()
    me = core.my_roster_id()
    mine = R[me]["players"]
    total, lineup = best_lineup(mine, p)
    print(f"Matchweek {week}: {R[me]['team']} — projected {total:.1f} pts in a {formation(lineup)}"
          + ("" if p.attrs.get("sleeper_live") else "   (Sleeper hasn't posted real projections for this week yet: model only)"))
    for slot, pid in lineup:
        print(f"  {slot:<4}{label(pid, p)}")
    starters = {pid for _, pid in lineup}
    print("  Bench:")
    for pid in sorted((x for x in mine if x not in starters), key=lambda x: -(p.at[x, "proj"] if x in p.index else 0)):
        print(f"      {label(pid, p)}")
    a.opp = a.opp or core.scheduled_opp(week)
    if a.opp:
        ot, ol = best_lineup(R[a.opp]["players"], p)
        sd = PLAYER_SD * np.sqrt(len(lineup) + len(ol))
        pw = float(ndtr((total - ot) / sd))
        print(f"\nVs {R[a.opp]['team']}: their best lineup ({formation(ol)}) {ot:.1f} pts -> your win chance {pw:.0%} (margin {total - ot:+.1f})")
        for slot, pid in ol:
            print(f"  {slot:<4}{label(pid, p)}")


def rostered() -> set:
    return {pid for v in core.rosters().values() for pid in v["players"] + v["reserve"]}


def ros_table() -> pd.DataFrame:
    ws = weeks_ahead()
    frames = [proj(w)[["name", "pos", "team", "proj", "e_min"]].assign(week=w) for w in ws]
    allp = pd.concat(frames)
    t = allp.groupby(level=0).agg(name=("name", "first"), pos=("pos", "first"), team=("team", "first"),
                                  ros=("proj", "mean"), e_min=("e_min", "mean"))
    t["next"] = proj(ws[0]).proj.reindex(t.index)
    return t


def cmd_waivers(a):
    t = ros_table()
    owned = rostered()
    fa = t[~t.index.isin(owned)]
    if a.pos:
        fa = fa[fa.pos == a.pos.upper()]
    tn = core.team_names()
    p = proj(core.DONE + 1)
    me = core.my_roster_id()
    mine = core.rosters()[me]["players"]
    base, lineup = best_lineup(mine, p)
    print(f"Best free agents (next week = MW{core.DONE + 1}, ROS = avg next {ROS_WEEKS} matchweeks):")
    rows = []
    for pid, r in fa.sort_values("ros", ascending=False).head(40).iterrows():
        # Lineup gain this week if added in place of your worst player at any position.
        drop = min(mine, key=lambda x: t.at[x, "ros"] if x in t.index else -9)
        gain = best_lineup([x for x in mine if x != drop] + [pid], p)[0] - base
        rows.append((r.ros, pid, r, gain, drop))
    for ros, pid, r, gain, drop in rows[:20]:
        print(f"  {r['name']:<24}{r.pos:<3}{tn.get(r.team, r.team):>4}  next {r.next:5.1f}  ROS {ros:5.1f}  ({r.e_min:3.0f} min)  this-week lineup {gain:+.1f}")
    worst = sorted(mine, key=lambda x: t.at[x, "ros"] if x in t.index else -9)[:4]
    print("\nYour lowest rest-of-season players (drop candidates): " + ", ".join(
        f"{core.pname(x)} {core.ppos(x)} ({t.at[x, 'ros']:.1f})" if x in t.index else f"{core.pname(x)} (no data)" for x in worst))


def replacement(t: pd.DataFrame) -> dict:
    owned = rostered()
    fa = t[~t.index.isin(owned)]
    return {pos: float(fa[fa.pos == pos].ros.nlargest(3).mean()) for pos in core.POSITIONS}


def cmd_values(a):
    t = ros_table()
    rep = replacement(t)
    R = core.rosters()
    rid = a.team or core.my_roster_id()
    tn = core.team_names()
    print(f"{R[rid]['team']} — rest-of-season value (avg next {ROS_WEEKS} MWs, vs best free agents at the position)")
    print(f"  replacement level: " + "  ".join(f"{k} {v:.1f}" for k, v in rep.items()))
    print("  value = ROS minus the best free agent he could be swapped for; dual-position players")
    print("  count against the weaker position (and are scored at their listed position wherever they play).")
    rows = []
    for pid in dict.fromkeys(R[rid]["players"] + R[rid]["reserve"]):  # reserve players are also listed in players
        if pid in t.index:
            r = t.loc[pid]
            el = [e for e in eligible(pid) if e in rep]
            rows.append((r.ros - min(rep[e] for e in el), r, "/".join(el)))
    for v, r, el in sorted(rows, key=lambda x: -x[0]):
        print(f"  {r['name']:<24}{el:<5}{tn.get(r.team, r.team):>4}  ROS {r.ros:5.1f}  next {r.next:5.1f}  value {v:+5.1f}  ({r.e_min:3.0f} min)")


def team_ros(pids) -> float:
    return float(np.mean([best_lineup(pids, proj(w))[0] for w in weeks_ahead()]))


def cmd_trade(a):
    R = core.rosters()
    by_name = {}
    for rid, v in R.items():
        for pid in v["players"]:
            by_name[core.pname(pid).lower()] = (pid, rid)
    find = lambda n: by_name.get(n.strip().lower()) or next(((p, r) for k, (p, r) in by_name.items() if n.strip().lower() in k), None)
    give = [find(n) for n in a.give.split(",")]
    get = [find(n) for n in a.get.split(",")]
    if None in give + get:
        raise SystemExit("couldn't find one of those players on a roster")
    me, other = core.my_roster_id(), get[0][1]
    mine, theirs = R[me]["players"], R[other]["players"]
    g_ids, r_ids = [x[0] for x in give], [x[0] for x in get]
    new_mine = [x for x in mine if x not in g_ids] + r_ids
    new_theirs = [x for x in theirs if x not in r_ids] + g_ids
    b_me, a_me, b_th, a_th = team_ros(mine), team_ros(new_mine), team_ros(theirs), team_ros(new_theirs)
    print(f"Trade with {R[other]['team']}: give {', '.join(core.pname(x) for x in g_ids)} | get {', '.join(core.pname(x) for x in r_ids)}")
    print(f"  You:  best lineup {b_me:.1f} -> {a_me:.1f} pts/MW ({a_me - b_me:+.1f}) over the next {ROS_WEEKS} matchweeks")
    print(f"  Them: best lineup {b_th:.1f} -> {a_th:.1f} pts/MW ({a_th - b_th:+.1f})")


def cmd_league(_):
    R = core.rosters()
    ws = weeks_ahead()
    tn = core.team_names()
    rows = []
    for rid, v in R.items():
        lineups = [best_lineup(v["players"], proj(w)) for w in ws]
        pos_pts = {pos: 0.0 for pos in core.POSITIONS}
        for total, lineup in lineups:
            for slot, pid in lineup:
                pos_pts[core.ppos(pid)] = pos_pts.get(core.ppos(pid), 0) + proj(ws[0]).proj.get(pid, 0) / len(lineups)
        rows.append({"team": v["team"][:24], "rid": rid, "record": v["record"], "wins": v["wins"] + 0.5 * v["ties"],
                     "PF": v["pf"], "proj/MW": np.mean([x[0] for x in lineups]), **{k: round(x, 1) for k, x in pos_pts.items()}})
    t = pd.DataFrame(rows)
    t["PF rank"] = t.PF.rank(ascending=False).astype(int)
    t["win rank"] = t.wins.rank(ascending=False, method="min").astype(int)
    t["luck"] = t["PF rank"] - t["win rank"]
    pd.set_option("display.width", 200)
    print(t.sort_values("proj/MW", ascending=False).round(1).drop(columns=["wins"]).to_string(index=False))
    print("\nluck > 0: record better than points scored; < 0: unlucky. Position columns = projected points from starters at that position next MW.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("week"); s.add_argument("--opp", type=int); s.add_argument("--week", type=int); s.set_defaults(fn=cmd_week)
    s = sub.add_parser("waivers"); s.add_argument("--pos"); s.set_defaults(fn=cmd_waivers)
    s = sub.add_parser("values"); s.add_argument("--team", type=int); s.set_defaults(fn=cmd_values)
    s = sub.add_parser("trade"); s.add_argument("--give", required=True); s.add_argument("--get", required=True); s.set_defaults(fn=cmd_trade)
    s = sub.add_parser("league"); s.set_defaults(fn=cmd_league)
    s = sub.add_parser("teams"); s.set_defaults(fn=cmd_teams)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
