"""Pull the Sleeper EPL league + player stats into data/raw/.

    uv run fetch_sleeper.py              # league, rosters, 2026 stats to date, upcoming fixtures
    uv run fetch_sleeper.py --history    # also (re)download last season (2025-26) stats
    uv run fetch_sleeper.py --players    # refresh the player database (Sleeper: max once a day)

The public API works even though the website doesn't show soccer leagues. Weekly stat rows
include each player's team and opponent (game_id), which is what the matchup model needs.
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests

LEAGUE_ID = os.environ.get("SLEEPER_LEAGUE_ID") or ""
SPORT = "clubsoccer:epl"
ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
WEEKS_PER_SEASON = 38

s = requests.Session()
s.headers["User-Agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 Chrome/130 Safari/537.36"


def get(url, params=None):
    for attempt in range(4):
        r = s.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(1.5 * (attempt + 1))
    r.raise_for_status()


def save(name, obj):
    (RAW / name).write_text(json.dumps(obj))


def weekly(kind: str, season: int, week: int) -> list:
    rows = get(f"https://api.sleeper.com/{kind}/{SPORT}/{season}/{week}", {"season_type": "regular"}) or []
    time.sleep(0.25)
    return [{k: v for k, v in r.items() if k != "player"} for r in rows]  # player blob is huge; db has it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--players", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    league = get(f"https://api.sleeper.app/v1/league/{LEAGUE_ID}")
    season = int(league["season"])
    done = int(league["settings"].get("last_scored_leg") or league["settings"]["leg"])
    save("league.json", league)
    save("users.json", get(f"https://api.sleeper.app/v1/league/{LEAGUE_ID}/users"))
    save("rosters.json", get(f"https://api.sleeper.app/v1/league/{LEAGUE_ID}/rosters"))
    save("transactions.json", {w: get(f"https://api.sleeper.app/v1/league/{LEAGUE_ID}/transactions/{w}") or []
                               for w in range(1, done + 2)})
    save("meta.json", {"season": season, "done": done})
    print(f"{league['name']} {season}: matchweeks scored {done}")

    if args.players or not (RAW / "players.json").exists():
        print("fetching player database...")
        save("players.json", get(f"https://api.sleeper.app/v1/players/{SPORT}"))

    prev = RAW / f"stats_{season - 1}.json"
    if args.history or not prev.exists():
        print(f"fetching {season - 1} stats (38 matchweeks)...")
        save(prev.name, {w: weekly("stats", season - 1, w) for w in range(1, WEEKS_PER_SEASON + 1)})
    save(f"stats_{season}.json", {w: weekly("stats", season, w) for w in range(1, done + 1)})
    # Future fixtures come from Sleeper's projection rows (team + opponent per week). Sleeper's
    # projected stats are also kept: the model blends them in for D/GK, where the backtest
    # showed they help.
    import core  # noqa: E402  (scoring helper; safe after league.json is saved)
    fixtures, sleeper = {}, {}
    for w in range(done + 1, WEEKS_PER_SEASON + 1):
        rows = weekly("projections", season, w)
        fixtures[w] = [{k: r.get(k) for k in ("team", "opponent", "game_id", "date")} for r in rows]
        sleeper[w] = {r["player_id"]: core.score(r["stats"])[0] for r in rows}
    save(f"fixtures_{season}.json", fixtures)
    save(f"sleeper_future_{season}.json", sleeper)
    print(f"saved stats {season - 1} + {season} weeks 1-{done}, fixtures weeks {done + 1}-{WEEKS_PER_SEASON}")


if __name__ == "__main__":
    main()
