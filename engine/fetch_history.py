"""One-off/occasional history pull used for tuning and home/away.

    uv run fetch_history.py

  data/raw/stats_2024.json          Sleeper 2024-25 stats (prior season for backtesting 2025-26)
  data/raw/sleeper_proj_2025.json   Sleeper's 2025-26 weekly projections, in league scoring
  data/raw/espn_matches.json        every EPL match (2025-26, 2026-27 incl. future): date, home, away
"""

import json
import time
from pathlib import Path

import requests

import core
from fetch_sleeper import RAW, get, weekly

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
es = requests.Session()
es.headers["User-Agent"] = "Mozilla/5.0"


def espn_season(first_day: str) -> list:
    """All matches of the EPL season whose calendar contains first_day (YYYYMMDD)."""
    cal = es.get(ESPN, params={"dates": first_day}, timeout=30).json()["leagues"][0]["calendar"]
    out = []
    for d in cal:
        day = d[:10].replace("-", "")
        for e in es.get(ESPN, params={"dates": day}, timeout=30).json().get("events", []):
            comp = e["competitions"][0]
            teams = {c["homeAway"]: c["team"] for c in comp["competitors"]}
            out.append({"id": e["id"], "date": e["date"], "home": teams["home"]["abbreviation"],
                        "away": teams["away"]["abbreviation"], "home_name": teams["home"]["displayName"],
                        "away_name": teams["away"]["displayName"], "status": comp["status"]["type"]["name"]})
        time.sleep(0.2)
    return out


def main():
    p = RAW / "stats_2024.json"
    if not p.exists():
        print("Sleeper 2024-25 stats...")
        p.write_text(json.dumps({w: weekly("stats", 2024, w) for w in range(1, 39)}))
    q = RAW / "sleeper_proj_2025.json"
    if not q.exists():
        print("Sleeper 2025-26 projections...")
        out = {}
        for w in range(1, 39):
            rows = get(f"https://api.sleeper.com/projections/clubsoccer:epl/2025/{w}", {"season_type": "regular"}) or []
            out[w] = {r["player_id"]: core.score(r["stats"])[0] for r in rows}
            time.sleep(0.25)
        q.write_text(json.dumps(out))
    print("ESPN fixtures (home/away)...")
    matches = espn_season("20250816") + espn_season("20260822")
    (RAW / "espn_matches.json").write_text(json.dumps(matches))
    print(f"saved {len(matches)} ESPN matches")


if __name__ == "__main__":
    main()
