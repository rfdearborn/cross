"""Tests for the auto-update module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cross.auto_update import _check_pypi, _install_update, run_update_loop


class TestCheckPypi:
    @pytest.mark.anyio
    async def test_returns_latest_version(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"info": {"version": "1.2.3"}}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("cross.auto_update.httpx.AsyncClient", return_value=mock_client):
            version = await _check_pypi()
        assert version == "1.2.3"

    @pytest.mark.anyio
    async def test_returns_none_on_error(self):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("network error"))

        with patch("cross.auto_update.httpx.AsyncClient", return_value=mock_client):
            version = await _check_pypi()
        assert version is None

    @pytest.mark.anyio
    async def test_returns_none_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("cross.auto_update.httpx.AsyncClient", return_value=mock_client):
            version = await _check_pypi()
        assert version is None


class TestInstallUpdate:
    @patch("subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert _install_update() is True

    @patch("subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        assert _install_update() is False


class TestRunUpdateLoop:
    @pytest.mark.anyio
    async def test_installs_when_newer(self):
        with (
            patch("cross.auto_update._get_installed_version", return_value="0.1.0"),
            patch("cross.auto_update._check_pypi", return_value="0.2.0"),
            patch("cross.auto_update._install_update", return_value=True) as mock_install,
            patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]),
        ):
            notify = MagicMock()
            with pytest.raises(asyncio.CancelledError):
                await run_update_loop(1, notify_fn=notify)
            mock_install.assert_called_once()
            notify.assert_called_once()

    @pytest.mark.anyio
    async def test_skips_when_same_version(self):
        with (
            patch("cross.auto_update._get_installed_version", return_value="0.1.0"),
            patch("cross.auto_update._check_pypi", return_value="0.1.0"),
            patch("cross.auto_update._install_update") as mock_install,
            patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]),
        ):
            with pytest.raises(asyncio.CancelledError):
                await run_update_loop(1)
            mock_install.assert_not_called()

    @pytest.mark.anyio
    async def test_skips_when_older(self):
        with (
            patch("cross.auto_update._get_installed_version", return_value="0.2.0"),
            patch("cross.auto_update._check_pypi", return_value="0.1.0"),
            patch("cross.auto_update._install_update") as mock_install,
            patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]),
        ):
            with pytest.raises(asyncio.CancelledError):
                await run_update_loop(1)
            mock_install.assert_not_called()

    @pytest.mark.anyio
    async def test_no_notify_when_no_callback(self):
        with (
            patch("cross.auto_update._get_installed_version", return_value="0.1.0"),
            patch("cross.auto_update._check_pypi", return_value="0.2.0"),
            patch("cross.auto_update._install_update", return_value=True),
            patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]),
        ):
            # Should not raise even without notify_fn
            with pytest.raises(asyncio.CancelledError):
                await run_update_loop(1)

    @pytest.mark.anyio
    async def test_handles_check_failure(self):
        with (
            patch("cross.auto_update._get_installed_version", return_value="0.1.0"),
            patch("cross.auto_update._check_pypi", return_value=None),
            patch("cross.auto_update._install_update") as mock_install,
            patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]),
        ):
            with pytest.raises(asyncio.CancelledError):
                await run_update_loop(1)
            mock_install.assert_not_called()

    @pytest.mark.anyio
    async def test_handles_install_failure(self):
        with (
            patch("cross.auto_update._get_installed_version", return_value="0.1.0"),
            patch("cross.auto_update._check_pypi", return_value="0.2.0"),
            patch("cross.auto_update._install_update", return_value=False),
            patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError]),
        ):
            notify = MagicMock()
            with pytest.raises(asyncio.CancelledError):
                await run_update_loop(1, notify_fn=notify)
            notify.assert_not_called()
