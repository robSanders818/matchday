"""Build the matchday page and publish it to GitHub Pages (repo robSanders818/matchday).

    uv run publish_pages.py --push    # one-time setup from the Mac: creates the repo, secrets, Pages, schedule

After that GitHub updates the projections by itself (Fri/Sat/Sun ~5am Eastern, see
web/update.yml), and the page reads confirmed lineups from ESPN live. Running this again with
--push from the Mac also syncs any model changes to the repo.

The repo is public, so league data is only ever committed encrypted:
    data/matchday.drs = "DRS1" | salt(16) | iv(12) | AES-256-GCM(gzip(json)), key = PBKDF2-SHA256(passphrase)
Passphrase: macOS Keychain (service "matchday"), set on the first run, copied into the repo's
Actions secrets by gh; the phone asks for it once.
"""

import argparse
import getpass
import gzip
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).parent.resolve()
IN_REPO = ROOT.name == "engine"                     # running inside the Pages repo (GitHub Actions)
SITE = Path(os.environ["MATCHDAY_SITE"]) if os.environ.get("MATCHDAY_SITE") else ROOT.parent if IN_REPO else ROOT.parent / "Matchday"
REPO = "robSanders818/matchday"
MAGIC = b"DRS1"
PBKDF2_ITERATIONS = 310_000  # must match ITERATIONS in the page
KEYCHAIN = ("matchday", "matchday-passphrase")
ENGINE_FILES = ["core.py", "model.py", "soccer.py", "matchday.py", "fetch_sleeper.py", "fetch_history.py",
                "publish_pages.py", "web/matchday.template.html", "data/tuned_params.json",
                "data/matchup_elasticity.json", "data/schedule.json"]
REQUIREMENTS = "numpy>=2.5.4\npandas>=3.1.0\nrequests>=2.34.2\nscipy>=1.18.1\ncryptography\n"
GITIGNORE = "engine/data/raw/\nengine/out/\nengine/.venv/\n__pycache__/\n"


def passphrase(new: bool = False) -> str:
    pw = None if new else os.environ.get("MATCHDAY_PASSPHRASE")
    if pw:
        return pw
    import keyring  # Mac only; GitHub Actions passes the secret in the environment
    pw = None if new else keyring.get_password(*KEYCHAIN)
    if pw:
        return pw
    while True:
        p1 = getpass.getpass("New matchday passphrase (you'll type this on your phone once): ")
        if len(p1) >= 8 and p1 == getpass.getpass("Repeat it: "):
            keyring.set_password(*KEYCHAIN, p1)
            return p1
        print("Use at least 8 characters, typed the same twice.")


def encrypt(plain: bytes, pw: str) -> bytes:
    salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)
    key = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, PBKDF2_ITERATIONS, 32)
    return MAGIC + salt + iv + AESGCM(key).encrypt(iv, gzip.compress(plain, 9), None)


def write_index():
    (SITE / "index.html").write_text("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
                                     "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                                     "<meta name=\"robots\" content=\"noindex\">\n</head><body>\n"
                                     + (ROOT / "web" / "matchday.template.html").read_text() + "\n</body></html>\n")


def git(*a, check=True):
    return subprocess.run(["git", *a], cwd=SITE, check=check, capture_output=True, text=True)


def league_id() -> str:
    return json.loads((ROOT / "data" / "raw" / "league.json").read_text())["league_id"]


def sync_engine():
    """Copy the model into the repo (minus the league id, which lives in an Actions secret)."""
    LEAGUE_ID = league_id()
    eng = SITE / "engine"
    for f in ENGINE_FILES:
        (eng / f).parent.mkdir(parents=True, exist_ok=True)
        text = (ROOT / f).read_text()
        if f == "fetch_sleeper.py":
            text = text.replace(f'or "{LEAGUE_ID}"', 'or ""')
        if LEAGUE_ID in text:
            raise SystemExit(f"{f} contains the league id; not syncing it to a public repo")
        (eng / f).write_text(text)
    (eng / "requirements.txt").write_text(REQUIREMENTS)
    (SITE / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "web" / "update.yml", SITE / ".github" / "workflows" / "update.yml")
    (SITE / ".gitignore").write_text(GITIGNORE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", type=int, help="opponent roster id; defaults to data/schedule.json")
    ap.add_argument("--week", type=int)
    ap.add_argument("--no-fetch", action="store_true", help="data/raw is already fresh")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--new-passphrase", action="store_true", help="replace the saved passphrase")
    a = ap.parse_args()

    cmd = [sys.executable, str(ROOT / "matchday.py")] + (["--opp", str(a.opp)] if a.opp else []) \
        + (["--week", str(a.week)] if a.week else []) + (["--no-fetch"] if a.no_fetch else [])
    subprocess.run(cmd, check=True, cwd=ROOT)
    data = json.loads((ROOT / "out" / "matchday.json").read_text())
    pw = passphrase(a.new_passphrase)
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    write_index()
    (SITE / "data" / "matchday.drs").write_bytes(encrypt(json.dumps(data, separators=(",", ":")).encode(), pw))
    (SITE / ".nojekyll").write_text("")
    print(f"wrote {SITE}/index.html and data/matchday.drs")
    if IN_REPO:
        return  # the workflow commits
    sync_engine()
    if not a.push:
        return

    first = not (SITE / ".git").exists()
    if first:
        git("init", "-b", "main")
    git("add", "index.html", "data/matchday.drs", ".nojekyll", ".gitignore", "engine", ".github")
    staged = git("diff", "--cached", "--name-only").stdout.split()
    bad = [f for f in staged if f.startswith(("engine/data/raw", "engine/out")) or f.endswith(".json") and f not in
           {"engine/" + x for x in ENGINE_FILES}]
    if bad:
        raise SystemExit(f"refusing to commit plaintext data: {bad}")
    if staged:
        git("commit", "-m", "Matchday " + data["generated"])
    if not git("remote", check=False).stdout.strip():
        subprocess.run(["gh", "repo", "create", REPO, "--public", "--source", str(SITE), "--push",
                        "--description", "Matchday start/sit page (data encrypted)"], check=True)
        subprocess.run(["gh", "api", "-X", "POST", f"repos/{REPO}/pages", "-f", "source[branch]=main", "-f", "source[path]=/"],
                       check=True, capture_output=True)
    else:
        git("pull", "--rebase", "-X", "theirs", "origin", "main", check=False)  # pick up the bot's data commits
        git("push", "origin", "main")
    # Secrets for the scheduled updates (values go straight from here to GitHub).
    subprocess.run(["gh", "secret", "set", "MATCHDAY_PASSPHRASE", "-R", REPO], input=pw, text=True, check=True, capture_output=True)
    subprocess.run(["gh", "secret", "set", "SLEEPER_LEAGUE_ID", "-R", REPO], input=league_id(), text=True, check=True, capture_output=True)
    print(f"https://{REPO.split('/')[0].lower()}.github.io/{REPO.split('/')[1]}/  (updates itself Fri/Sat/Sun ~5am Eastern)")


if __name__ == "__main__":
    main()
