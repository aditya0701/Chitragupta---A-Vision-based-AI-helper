"""Run every harness written for the 2026-07-29 fixes.

    python tests/run_all.py          # from the repo root

These are ad-hoc harnesses, not a test suite — the project has never had one
(DECISIONS.md "Testing notes"). They cover the logic that can be checked
without a live session; everything about how the *model* behaves still needs a
real run and a read of the export.

The two JS harnesses need node on PATH. They drive the real app.js/debug.js in
a stubbed DOM.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

PY = [
    ("detail budget      ", "t_detail.py"),
    ("retraction         ", "t_retract.py"),
    ("tool integration   ", "t_integration.py"),
    ("search exclusion   ", "t_exclude.py"),
    ("silence detection  ", "t_silence.py"),
    ("apology loop       ", "t_apology.py"),
    ("v2 live prompts    ", "t_live_prompts.py"),
    ("v2 tick/chat overlap", "t_live_parallel.py"),
]

results = []
for label, script in PY:
    p = subprocess.run([sys.executable, "-X", "utf8", str(HERE / script)],
                       cwd=ROOT, capture_output=True, text=True)
    tail = [l for l in (p.stdout or "").strip().splitlines() if l.strip()]
    results.append((label, p.returncode == 0, tail[-1] if tail else (p.stderr or "").strip()[-120:]))

for label, script, arg in [
    ("input race app.js  ", "t_race2.js", str(ROOT / "server/static/app.js")),
    ("input race debug.js", "t_race2.js", str(ROOT / "server/static/debug.js")),
]:
    try:
        p = subprocess.run(["node", str(HERE / script), arg],
                           cwd=ROOT, capture_output=True, text=True)
        tail = [l for l in (p.stdout or "").strip().splitlines() if l.strip()]
        results.append((label, p.returncode == 0, tail[-1] if tail else "no output"))
    except FileNotFoundError:
        results.append((label, False, "node not found on PATH"))

print()
for label, passed, note in results:
    print(f"  {'PASS' if passed else 'FAIL'}  {label}  {note}")

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} harnesses passed")
sys.exit(1 if failed else 0)
