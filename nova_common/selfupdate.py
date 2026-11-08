"""Safety net for NOVA's self-updates, shared by Core and the launcher.

When NOVA applies a change to her own code she writes data/autonomy/update.json and restarts
(exit code RESTART_CODE). If Core then crashes within TRIAL_SECONDS, the launcher reverts the
change with git and restarts the previous version. Core clears the marker once it has run fine.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time

from .config import ROOT

RESTART_CODE = 75          # "restart me to load an update" (not a crash)
TRIAL_SECONDS = 180
MARKER = ROOT / "data" / "autonomy" / "update.json"


def write_marker(merge_id: str, proposal_id: str, title: str) -> None:
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(json.dumps({"merge_id": merge_id, "proposal": proposal_id, "title": title,
                                  "applied_at": time.time(), "status": "trial"}), encoding="utf-8")


def read_marker() -> dict | None:
    try:
        return json.loads(MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def set_status(status: str, **extra) -> None:
    m = read_marker()
    if m:
        m.update(status=status, **extra)
        MARKER.write_text(json.dumps(m), encoding="utf-8")


def clear() -> None:
    MARKER.unlink(missing_ok=True)


def git(*args: str, cwd=ROOT, timeout: int = 120) -> subprocess.CompletedProcess:
    exe = shutil.which("git")
    if not exe:
        raise FileNotFoundError("git is not installed")
    return subprocess.run([exe, *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def rollback_trial() -> str:
    """Revert the update on trial (called by the launcher when Core crashes right after one)."""
    m = read_marker()
    if not m or m.get("status") != "trial":
        return ""
    r = git("revert", "-m", "1", "--no-edit", m["merge_id"])
    if r.returncode != 0:
        git("revert", "--abort")
        set_status("rollback_failed", error=(r.stderr or r.stdout)[-500:])
        return f"Could not undo NOVA's update '{m.get('title')}': {(r.stderr or r.stdout)[-300:]}"
    set_status("rolled_back", rolled_back_at=time.time())
    return f"NOVA's update '{m.get('title')}' made Core crash, so it was undone automatically."
