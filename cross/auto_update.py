"""Auto-update — periodically checks PyPI and installs newer versions."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys

import httpx

logger = logging.getLogger("cross.auto_update")

_PYPI_PACKAGE = "cross-ai"
_PYPI_URL = f"https://pypi.org/pypi/{_PYPI_PACKAGE}/json"


def _get_installed_version() -> str | None:
    import importlib.metadata

    for name in ("cross-ai", "cross"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


async def _check_pypi() -> str | None:
    """Fetch the latest version from PyPI. Returns version string or None."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_PYPI_URL)
            if resp.status_code == 200:
                return resp.json()["info"]["version"]
    except Exception:
        logger.debug("Failed to check PyPI for updates")
    return None


def _install_update() -> bool:
    """Install the latest version from PyPI. Returns True on success."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", _PYPI_PACKAGE],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


async def run_update_loop(interval_hours: int, notify_fn=None):
    """Background loop that checks for updates and installs them.

    Args:
        interval_hours: Hours between update checks.
        notify_fn: Optional callback(title, body) for notifications.
    """
    interval_s = interval_hours * 3600
    while True:
        await asyncio.sleep(interval_s)
        try:
            current = _get_installed_version()
            latest = await _check_pypi()
            if not current or not latest or latest == current:
                continue

            # Simple version comparison (works for semver)
            if latest <= current:
                continue

            logger.info(f"Update available: {current} -> {latest}")
            if _install_update():
                logger.info(f"Updated cross to {latest}")
                msg = f"cross updated: {current} → {latest}. Restart to apply."
                if notify_fn:
                    notify_fn("cross — updated", msg)
            else:
                logger.warning("Auto-update failed")
        except Exception:
            logger.debug("Auto-update check failed", exc_info=True)
