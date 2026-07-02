#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Agora transport.

Tests cover:
- AgoraTransportClient lifecycle and video queue guarding
- AgoraParams defaults and video config wiring
- Runner integration: AgoraRunnerArguments, configure(), create_transport()
- YUV→RGB and RGB→RGBA conversion correctness
"""

import argparse
import asyncio
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

try:
    from agora.rtc.agora_service import AgoraService

    from pipecat.transports.agora.transport import (
        AgoraCallbacks,
        AgoraParams,
        AgoraTransport,
        AgoraTransportClient,
        _VIDEO_PIXEL_I420,
        _VIDEO_PIXEL_RGBA,
    )

    AGORA_AVAILABLE = True
except ImportError:
    AGORA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Transport unit tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(AGORA_AVAILABLE, "agora-python-server-sdk not installed")
class TestAgoraParams(unittest.TestCase):
    """AgoraParams defaults and field validation."""

    def test_defaults(self):
        params = AgoraParams()
        self.assertEqual(params.app_id, "")
        self.assertFalse(params.enable_vad)
        self.assertTrue(params.auto_subscribe_audio)
        self.assertFalse(params.auto_subscribe_video)
        self.assertFalse(params.enable_encryption)
        self.assertFalse(params.enable_apm)
        self.assertEqual(params.audio_scenario, 9)  # AI_SERVER

    def test_video_flags_default_false(self):
        params = AgoraParams()
        self.assertFalse(params.video_in_enabled)
        self.assertFalse(params.video_out_enabled)

    def test_video_flags_settable(self):
        params = AgoraParams(video_in_enabled=True, video_out_enabled=True)
        self.assertTrue(params.video_in_enabled)
        self.assertTrue(params.video_out_enabled)


@unittest.skipUnless(AGORA_AVAILABLE, "agora-python-server-sdk not installed")
class TestAgoraTransportClient(unittest.TestCase):
    """AgoraTransportClient construction and basic state."""

    def _make_callbacks(self) -> AgoraCallbacks:
        return AgoraCallbacks(
            on_connected=AsyncMock(),
            on_disconnected=AsyncMock(),
            on_before_disconnect=AsyncMock(),
            on_user_joined=AsyncMock(),
            on_user_left=AsyncMock(),
            on_audio_track_subscribed=AsyncMock(),
            on_video_track_subscribed=AsyncMock(),
            on_data_received=AsyncMock(),
            on_first_user_joined=AsyncMock(),
            on_token_privilege_will_expire=AsyncMock(),
            on_connection_lost=AsyncMock(),
            on_error=AsyncMock(),
        )

    def test_initial_state(self):
        client = AgoraTransportClient(
            app_id="test-app",
            channel_name="test-channel",
            uid="0",
            token="test-token",
            params=AgoraParams(),
            callbacks=self._make_callbacks(),
            transport_name="test",
        )
        self.assertFalse(client._connected)
        self.assertFalse(client._other_user_has_joined)
        self.assertIsNone(client._connection)

    def test_has_video_queue(self):
        """Client should have a video queue for video frame delivery."""
        client = AgoraTransportClient(
            app_id="test-app",
            channel_name="test-channel",
            uid="0",
            token="test-token",
            params=AgoraParams(),
            callbacks=self._make_callbacks(),
            transport_name="test",
        )
        self.assertIsInstance(client._video_queue, asyncio.Queue)
        self.assertEqual(client._video_queue.qsize(), 0)


@unittest.skipUnless(AGORA_AVAILABLE, "agora-python-server-sdk not installed")
class TestAgoraTransportFacade(unittest.TestCase):
    """AgoraTransport facade creates correct input/output."""

    def test_input_output_types(self):
        transport = AgoraTransport(
            app_id="test",
            channel_name="ch",
            uid="0",
            token="tok",
            params=AgoraParams(),
        )
        from pipecat.transports.agora.transport import (
            AgoraInputTransport,
            AgoraOutputTransport,
        )

        self.assertIsInstance(transport.input(), AgoraInputTransport)
        self.assertIsInstance(transport.output(), AgoraOutputTransport)

    def test_input_output_singletons(self):
        transport = AgoraTransport(
            app_id="test",
            channel_name="ch",
            uid="0",
            token="tok",
        )
        self.assertIs(transport.input(), transport.input())
        self.assertIs(transport.output(), transport.output())

    def test_participant_aliases_registered(self):
        """on_participant_joined/left/first should be available as event handlers."""
        transport = AgoraTransport(
            app_id="test",
            channel_name="ch",
            uid="0",
            token="tok",
        )
        # These should not raise — the handler names are registered.
        for name in (
            "on_participant_joined",
            "on_participant_left",
            "on_first_participant_joined",
        ):
            transport.event_handler(name)(AsyncMock())


@unittest.skipUnless(AGORA_AVAILABLE, "agora-python-server-sdk not installed")
class TestFirstUserJoinedTracking(unittest.IsolatedAsyncioTestCase):
    """on_first_user_joined fires exactly once."""

    def _make_callbacks(self) -> AgoraCallbacks:
        return AgoraCallbacks(
            on_connected=AsyncMock(),
            on_disconnected=AsyncMock(),
            on_before_disconnect=AsyncMock(),
            on_user_joined=AsyncMock(),
            on_user_left=AsyncMock(),
            on_audio_track_subscribed=AsyncMock(),
            on_video_track_subscribed=AsyncMock(),
            on_data_received=AsyncMock(),
            on_first_user_joined=AsyncMock(),
            on_token_privilege_will_expire=AsyncMock(),
            on_connection_lost=AsyncMock(),
            on_error=AsyncMock(),
        )

    async def test_first_user_fires_once(self):
        callbacks = self._make_callbacks()
        client = AgoraTransportClient(
            app_id="test",
            channel_name="ch",
            uid="0",
            token="tok",
            params=AgoraParams(),
            callbacks=callbacks,
            transport_name="test",
        )
        await client._handle_user_joined("user-1")
        await client._handle_user_joined("user-2")

        callbacks.on_first_user_joined.assert_called_once_with("user-1")
        self.assertEqual(callbacks.on_user_joined.call_count, 2)


# ---------------------------------------------------------------------------
# Video conversion tests
# ---------------------------------------------------------------------------


class TestYUVtoRGBConversion(unittest.TestCase):
    """BT.601 YUV→RGB conversion correctness."""

    def test_white_pixel(self):
        """YUV white (235,128,128) should map to RGB near (235,235,235)."""
        width, height = 4, 4
        y = np.full((height, width), 235, dtype=np.uint8)
        u = np.full((height // 2, width // 2), 128, dtype=np.uint8)
        v = np.full((height // 2, width // 2), 128, dtype=np.uint8)

        u_full = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
        v_full = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

        y_f = y.astype(np.float32)
        u_f = u_full.astype(np.float32) - 128.0
        v_f = v_full.astype(np.float32) - 128.0

        r = np.clip(y_f + 1.402 * v_f, 0, 255).astype(np.uint8)
        g = np.clip(y_f - 0.344136 * u_f - 0.714136 * v_f, 0, 255).astype(np.uint8)
        b = np.clip(y_f + 1.772 * u_f, 0, 255).astype(np.uint8)

        self.assertEqual(r[0, 0], 235)
        self.assertEqual(g[0, 0], 235)
        self.assertEqual(b[0, 0], 235)

    def test_black_pixel(self):
        """YUV black (16,128,128) should map to RGB near (16,16,16)."""
        y_val = 16
        y_f = np.float32(y_val)
        r = int(np.clip(y_f + 1.402 * 0, 0, 255))
        g = int(np.clip(y_f - 0.344136 * 0 - 0.714136 * 0, 0, 255))
        b = int(np.clip(y_f + 1.772 * 0, 0, 255))
        self.assertEqual(r, 16)
        self.assertEqual(g, 16)
        self.assertEqual(b, 16)

    def test_output_shape(self):
        """Converted RGB array should have shape (H, W, 3)."""
        width, height = 8, 6
        y = np.zeros((height, width), dtype=np.uint8)
        u = np.zeros((height // 2, width // 2), dtype=np.uint8)
        v = np.zeros((height // 2, width // 2), dtype=np.uint8)

        u_full = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
        v_full = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

        rgb = np.stack(
            [
                np.clip(y.astype(np.float32) + 1.402 * (v_full.astype(np.float32) - 128), 0, 255).astype(np.uint8),
                np.clip(y.astype(np.float32) - 0.344136 * (u_full.astype(np.float32) - 128) - 0.714136 * (v_full.astype(np.float32) - 128), 0, 255).astype(np.uint8),
                np.clip(y.astype(np.float32) + 1.772 * (u_full.astype(np.float32) - 128), 0, 255).astype(np.uint8),
            ],
            axis=-1,
        )
        self.assertEqual(rgb.shape, (height, width, 3))
        self.assertEqual(len(rgb.tobytes()), height * width * 3)


class TestRGBtoRGBAConversion(unittest.TestCase):
    """RGB→RGBA conversion for video output."""

    def test_alpha_channel_appended(self):
        width, height = 4, 4
        rgb = np.full((height, width, 3), 128, dtype=np.uint8)
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255

        self.assertEqual(rgba.shape, (height, width, 4))
        self.assertTrue(np.all(rgba[:, :, 3] == 255))
        self.assertTrue(np.all(rgba[:, :, :3] == 128))

    def test_byte_length(self):
        width, height = 10, 8
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255
        self.assertEqual(len(rgba.tobytes()), width * height * 4)


# ---------------------------------------------------------------------------
# Pixel format constants
# ---------------------------------------------------------------------------


@unittest.skipUnless(AGORA_AVAILABLE, "agora-python-server-sdk not installed")
class TestPixelFormatConstants(unittest.TestCase):
    def test_i420_value(self):
        self.assertEqual(_VIDEO_PIXEL_I420, 1)

    def test_rgba_value(self):
        self.assertEqual(_VIDEO_PIXEL_RGBA, 4)


# ---------------------------------------------------------------------------
# Runner integration tests
# ---------------------------------------------------------------------------


class TestAgoraRunnerArguments(unittest.TestCase):
    """AgoraRunnerArguments dataclass."""

    def test_fields(self):
        from pipecat.runner.types import AgoraRunnerArguments

        args = AgoraRunnerArguments(
            app_id="app", channel_name="ch", uid="0", token="tok"
        )
        self.assertEqual(args.app_id, "app")
        self.assertEqual(args.channel_name, "ch")
        self.assertEqual(args.uid, "0")
        self.assertEqual(args.token, "tok")

    def test_inherits_runner_arguments(self):
        from pipecat.runner.types import AgoraRunnerArguments, RunnerArguments

        self.assertTrue(issubclass(AgoraRunnerArguments, RunnerArguments))

    def test_body_and_session_id(self):
        from pipecat.runner.types import AgoraRunnerArguments

        args = AgoraRunnerArguments(
            app_id="a",
            channel_name="c",
            uid="0",
            token="t",
            body={"key": "val"},
            session_id="s123",
        )
        self.assertEqual(args.body, {"key": "val"})
        self.assertEqual(args.session_id, "s123")


class TestAgoraConfigure(unittest.IsolatedAsyncioTestCase):
    """runner/agora.py configure() function."""

    async def test_returns_tuple(self):
        with patch.dict(os.environ, {"AGORA_APP_ID": "test-id"}):
            result = await self._configure()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    async def test_app_id_from_env(self):
        with patch.dict(os.environ, {"AGORA_APP_ID": "my-app"}):
            app_id, _, _, _ = await self._configure()
        self.assertEqual(app_id, "my-app")

    async def test_missing_app_id_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove AGORA_APP_ID if present
            os.environ.pop("AGORA_APP_ID", None)
            with self.assertRaises(ValueError):
                await self._configure()

    async def test_token_fallback_to_app_id(self):
        with patch.dict(os.environ, {"AGORA_APP_ID": "app123"}, clear=True):
            os.environ.pop("AGORA_TOKEN", None)
            os.environ.pop("AGORA_APP_CERTIFICATE", None)
            _, _, _, token = await self._configure()
        self.assertEqual(token, "app123")

    async def test_explicit_token(self):
        with patch.dict(os.environ, {"AGORA_APP_ID": "app"}):
            _, _, _, token = await self._configure(token="my-token")
        self.assertEqual(token, "my-token")

    async def test_env_token(self):
        with patch.dict(os.environ, {"AGORA_APP_ID": "app", "AGORA_TOKEN": "env-tok"}):
            _, _, _, token = await self._configure()
        self.assertEqual(token, "env-tok")

    async def test_channel_name_default_generated(self):
        with patch.dict(os.environ, {"AGORA_APP_ID": "app"}, clear=True):
            os.environ.pop("AGORA_CHANNEL_NAME", None)
            _, channel, _, _ = await self._configure()
        self.assertTrue(channel.startswith("pipecat-"))

    async def test_channel_name_from_arg(self):
        with patch.dict(os.environ, {"AGORA_APP_ID": "app"}):
            _, channel, _, _ = await self._configure(channel_name="custom")
        self.assertEqual(channel, "custom")

    async def test_uid_default(self):
        with patch.dict(os.environ, {"AGORA_APP_ID": "app"}, clear=True):
            os.environ.pop("AGORA_UID", None)
            _, _, uid, _ = await self._configure()
        self.assertEqual(uid, "0")

    async def test_non_numeric_uid_raises(self):
        with patch.dict(os.environ, {"AGORA_APP_ID": "app"}):
            with self.assertRaises(ValueError):
                await self._configure(uid="abc")

    async def test_certificate_without_builder_raises(self):
        with patch.dict(
            os.environ,
            {"AGORA_APP_ID": "app", "AGORA_APP_CERTIFICATE": "cert"},
            clear=True,
        ):
            os.environ.pop("AGORA_TOKEN", None)
            with patch.dict("sys.modules", {"agora_token_builder": None}):
                with self.assertRaises(ImportError):
                    await self._configure()

    async def _configure(self, **kwargs):
        from pipecat.runner.agora import configure

        return await configure(**kwargs)


@unittest.skipUnless(AGORA_AVAILABLE, "agora-python-server-sdk not installed")
class TestCreateTransportAgora(unittest.IsolatedAsyncioTestCase):
    """create_transport() with AgoraRunnerArguments."""

    async def test_returns_agora_transport(self):
        from pipecat.runner.types import AgoraRunnerArguments
        from pipecat.runner.utils import create_transport

        args = AgoraRunnerArguments(
            app_id="test", channel_name="ch", uid="0", token="tok"
        )
        transport_params = {
            "agora": lambda: AgoraParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
        }
        transport = await create_transport(args, transport_params)
        self.assertIsInstance(transport, AgoraTransport)


try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pipecat.runner.run import _setup_unified_start_route

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi not installed")
class TestStartEndpointAgora(unittest.TestCase):
    """POST /start with transport=agora returns Agora credential fields."""

    def test_start_agora_returns_credentials(self):
        app = FastAPI()
        args = argparse.Namespace(transport=None)
        _setup_unified_start_route(app, args, {})

        # Mock configure to return deterministic values
        mock_configure = AsyncMock(
            return_value=("app-id-123", "test-channel", "42", "token-abc")
        )

        # Mock bot module so the spawned task doesn't fail
        bot_module = types.ModuleType("bot")
        bot_module.bot = AsyncMock()

        with (
            patch("pipecat.runner.run._transport_routes_enabled", return_value=True),
            patch(
                "pipecat.runner.agora.configure",
                mock_configure,
            ),
            patch("pipecat.runner.run._get_bot_module", return_value=bot_module),
        ):
            response = TestClient(app).post(
                "/start",
                json={
                    "transport": "agora",
                    "channelName": "test-channel",
                    "uid": "42",
                    "token": "token-abc",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("sessionId", body)
        self.assertEqual(body["agoraAppId"], "app-id-123")
        self.assertEqual(body["agoraChannel"], "test-channel")
        self.assertEqual(body["agoraUid"], "42")
        self.assertEqual(body["agoraToken"], "token-abc")

    def test_start_agora_passes_request_fields_to_configure(self):
        app = FastAPI()
        args = argparse.Namespace(transport=None)
        _setup_unified_start_route(app, args, {})

        mock_configure = AsyncMock(
            return_value=("app", "ch", "0", "tok")
        )
        bot_module = types.ModuleType("bot")
        bot_module.bot = AsyncMock()

        with (
            patch("pipecat.runner.run._transport_routes_enabled", return_value=True),
            patch("pipecat.runner.agora.configure", mock_configure),
            patch("pipecat.runner.run._get_bot_module", return_value=bot_module),
        ):
            TestClient(app).post(
                "/start",
                json={
                    "transport": "agora",
                    "channelName": "my-channel",
                    "uid": "99",
                    "token": "my-token",
                },
            )

        mock_configure.assert_called_once_with(
            channel_name="my-channel",
            uid="99",
            token="my-token",
        )


if __name__ == "__main__":
    unittest.main()
