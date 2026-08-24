"""
Footage-server credentials for repo.octopus-intelligence.org.

Single source of truth: reads OCTOPUS_USER / OCTOPUS_PASS from the environment,
falling back to the repo-root .env file (which is gitignored). No third-party
deps — we parse .env ourselves so this works without python-dotenv installed.

Usage:
    from server_creds import USER, PASS          # phase2/* and ui/* scripts
    # (both dirs are one level under the repo root, so add the root to sys.path:)
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""
import os
from pathlib import Path


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (without overriding
    anything already set in the real environment)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env_file(Path(__file__).resolve().parent / ".env")

USER = os.environ.get("OCTOPUS_USER", "")
PASS = os.environ.get("OCTOPUS_PASS", "")

if not USER or not PASS:
    raise RuntimeError(
        "Footage-server credentials missing. Set OCTOPUS_USER and OCTOPUS_PASS "
        "in the environment or in the repo-root .env file."
    )
