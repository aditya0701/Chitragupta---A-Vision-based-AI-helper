"""v2 must not send frames to Groq — and must fail at startup if it would.

The failure this guards against cost a live session. On 2026-08-10 a ~18-minute
cooking session died mid-answer on Groq's daily token cap, while server/.env,
render.yaml and every doc in the repo said `deepinfra`. Nothing looked wrong
anywhere: the running process simply predated the setting, and no code path
ever compared what was configured against what was actually happening.

Two independent defects, both covered here:

  1. LIVE_BACKEND_MODE defaulted to "hybrid" — the Groq path. The default
     nobody sets was the one configuration that cannot work.
  2. Nothing checked. A Groq-vision backend was accepted silently and the
     consequence arrived as a 429 eighteen minutes later.

Run:  python tests/t_live_backend.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAIL = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if not cond and detail else ""))
    if not cond:
        FAIL.append(label)


# ── [1] The declared provider ────────────────────────────────────────────────
# VISION_PROVIDER exists because "which provider gets the pixels" was not
# otherwise knowable: DeepInfraHybridBackend extends DeepSeekBackend and
# replaces a vision client built in the parent's __init__, so neither the class
# name nor the mode string tells you where an image ends up.
print("\n[1] every backend declares where its pixels go")

from server.backends import VisionBackend
from server.backends.deepseek_backend import DeepSeekBackend
from server.backends.deepinfra_backend import DeepInfraHybridBackend
from server.backends.groq_backend import GroqBackend

check("the base class declares the attribute", hasattr(VisionBackend, "VISION_PROVIDER"))
check("DeepSeek hybrid reports groq", DeepSeekBackend.VISION_PROVIDER == "groq",
      DeepSeekBackend.VISION_PROVIDER)
check("GroqBackend reports groq", GroqBackend.VISION_PROVIDER == "groq",
      GroqBackend.VISION_PROVIDER)
# The override is the entire point of the subclass. Without it the check below
# passes vacuously and v2 refuses to start on a perfectly good backend.
check("DeepInfra OVERRIDES its parent's groq", DeepInfraHybridBackend.VISION_PROVIDER == "deepinfra",
      DeepInfraHybridBackend.VISION_PROVIDER)


# ── [2] The default ──────────────────────────────────────────────────────────
print("\n[2] an unset LIVE_BACKEND_MODE lands on deepinfra, not groq")

os.environ.pop("LIVE_BACKEND_MODE", None)
for mod in [m for m in sys.modules if m.startswith("server.live")]:
    del sys.modules[mod]
from server.live import config as live_config

check("default is deepinfra", live_config.LIVE_BACKEND_MODE == "deepinfra",
      live_config.LIVE_BACKEND_MODE)
check("the groq escape hatch is OFF by default", live_config.ALLOW_GROQ_VISION is False,
      str(live_config.ALLOW_GROQ_VISION))


# ── [3] The startup check ────────────────────────────────────────────────────
# Stub backends rather than real ones: constructing DeepInfraHybridBackend
# needs a live API key, and this is a test about a policy decision, not about
# whether a vendor's SDK initializes.
print("\n[3] a groq-vision backend is refused, a deepinfra one is accepted")

from server.live.routes import _check_vision_provider


class _Stub:
    def __init__(self, provider):
        self.VISION_PROVIDER = provider


def refused(backend, mode="hybrid"):
    try:
        _check_vision_provider(backend, mode)
        return None
    except RuntimeError as e:
        return str(e)


err = refused(_Stub("groq"))
check("groq vision is refused", err is not None)
check("the error says how to fix it", err and "LIVE_BACKEND_MODE=deepinfra" in err, str(err)[:120])
# Restarting is the actual fix — the config was already correct on disk last
# time, and a reader who doesn't know that will change a file that needs no
# changing and conclude the check is broken.
check("the error says to restart", err and "RESTART" in err.upper(), str(err)[:120])
check("the error names the override", err and "LIVE_ALLOW_GROQ_VISION" in err, str(err)[:120])

check("deepinfra vision is accepted", refused(_Stub("deepinfra"), "deepinfra") is None)
# "unknown" is the base-class default, so any backend that forgot to declare
# would land here. Refusing those too would break v1's other backends for no
# reason; only groq is known-impossible.
check("an undeclared backend is not refused", refused(_Stub("unknown"), "colab") is None)


# ── [4] The escape hatch ─────────────────────────────────────────────────────
# Choosing Groq deliberately is fine. Arriving there by accident is the bug.
print("\n[4] the escape hatch works, and only when set explicitly")

live_config.ALLOW_GROQ_VISION = True
check("groq is allowed when explicitly opted in", refused(_Stub("groq")) is None)
live_config.ALLOW_GROQ_VISION = False
check("and refused again once revoked", refused(_Stub("groq")) is not None)


print("\n" + ("FAILURES: " + ", ".join(FAIL) if FAIL else "ALL BACKEND-GUARD CHECKS PASSED"))
sys.exit(1 if FAIL else 0)
