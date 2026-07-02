#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Integration tests for Agora transport.

These tests connect to a real Agora channel using credentials from
environment variables. They are skipped when AGORA_APP_ID is not set.

Required env vars:
    AGORA_APP_ID         - Agora App ID

Optional env vars:
    AGORA_TOKEN          - Pre-minted token (defaults to app_id for testing-mode)
    AGORA_CHANNEL_NAME   - Channel name (defaults to a generated name)
    AGORA_UID            - User ID (defaults to "0")
"""

import asyncio
import os
import struct
import unittest
from unittest.mock import AsyncMock

import numpy as np

AGORA_APP_ID = os.getenv("AGORA_APP_ID")

try:
    from pipecat.transports.agora.transport import (
        AgoraCallbacks,
        AgoraParams,
        AgoraTransportClient,
    )

    AGORA_AVAILABLE = True
except ImportError:
    AGORA_AVAILABLE = False

SKIP_REASON = (
    "agora-python-server-sdk not installed"
    if not AGORA_AVAILABLE
    else "AGORA_APP_ID not set"
)
SHOULD_RUN = AGORA_AVAILABLE and AGORA_APP_ID


def _make_callbacks() -> "AgoraCallbacks":
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


@unittest.skipUnless(SHOULD_RUN, SKIP_REASON)
class TestAgoraConnectDisconnect(unittest.IsolatedAsyncioTestCase):
    """Verify the bot can join and leave an Agora channel."""

    async def test_connect_sets_connected_flag(self):
        """Client connects to Agora and _connected becomes True."""
        app_id = AGORA_APP_ID
        token = os.getenv("AGORA_TOKEN", app_id)
        channel = os.getenv("AGORA_CHANNEL_NAME", "pipecat-test")
        uid = os.getenv("AGORA_UID", "0")

        client = AgoraTransportClient(
            app_id=app_id,
            channel_name=channel,
            uid=uid,
            token=token,
            params=AgoraParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
            callbacks=_make_callbacks(),
            transport_name="integration-test",
        )

        self.assertFalse(client._connected)
        await client.connect()
        self.assertTrue(client._connected)
        self.assertIsNotNone(client._connection)

        await client.disconnect()
        self.assertFalse(client._connected)


@unittest.skipUnless(SHOULD_RUN, SKIP_REASON)
class TestAgoraAudioPush(unittest.IsolatedAsyncioTestCase):
    """Verify the bot can push audio frames into a channel."""

    async def test_push_audio_frame(self):
        """Push a short 16-bit PCM frame and verify no error."""
        app_id = AGORA_APP_ID
        token = os.getenv("AGORA_TOKEN", app_id)
        channel = os.getenv("AGORA_CHANNEL_NAME", "pipecat-test")
        uid = os.getenv("AGORA_UID", "0")

        client = AgoraTransportClient(
            app_id=app_id,
            channel_name=channel,
            uid=uid,
            token=token,
            params=AgoraParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=16000,
            ),
            callbacks=_make_callbacks(),
            transport_name="integration-audio",
        )

        await client.connect()
        try:
            # Generate 20ms of silence (16kHz, mono, 16-bit PCM)
            num_samples = 320  # 16000 * 0.02
            silence = b"\x00\x00" * num_samples

            result = await client.write_audio(silence, 16000, 1)
            self.assertTrue(result)

            # Generate 20ms of a 440Hz tone
            t = np.arange(num_samples) / 16000.0
            tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
            tone_bytes = tone.tobytes()

            result = await client.write_audio(tone_bytes, 16000, 1)
            self.assertTrue(result)
        finally:
            await client.disconnect()


@unittest.skipUnless(SHOULD_RUN, SKIP_REASON)
class TestAgoraVideoPush(unittest.IsolatedAsyncioTestCase):
    """Verify the bot can push video frames into a channel."""

    async def test_push_rgb_video_frame(self):
        """Push a small RGB test frame and verify no error."""
        app_id = AGORA_APP_ID
        token = os.getenv("AGORA_TOKEN", app_id)
        channel = os.getenv("AGORA_CHANNEL_NAME", "pipecat-test")
        uid = os.getenv("AGORA_UID", "0")

        client = AgoraTransportClient(
            app_id=app_id,
            channel_name=channel,
            uid=uid,
            token=token,
            params=AgoraParams(
                audio_in_enabled=False,
                audio_out_enabled=False,
                video_in_enabled=False,
                video_out_enabled=True,
            ),
            callbacks=_make_callbacks(),
            transport_name="integration-video",
        )

        await client.connect()
        try:
            # Create a 64x64 red RGB frame
            width, height = 64, 64
            red_frame = np.zeros((height, width, 3), dtype=np.uint8)
            red_frame[:, :, 0] = 255  # R=255, G=0, B=0
            rgb_bytes = red_frame.tobytes()

            result = await client.write_video(rgb_bytes, width, height, "RGB")
            self.assertTrue(result)

            # Create a 64x64 green RGBA frame
            green_frame = np.zeros((height, width, 4), dtype=np.uint8)
            green_frame[:, :, 1] = 255  # G=255
            green_frame[:, :, 3] = 255  # A=255
            rgba_bytes = green_frame.tobytes()

            result = await client.write_video(rgba_bytes, width, height, "RGBA")
            self.assertTrue(result)
        finally:
            await client.disconnect()

    async def test_unsupported_video_format_returns_false(self):
        """Pushing an unsupported format returns False without crashing."""
        app_id = AGORA_APP_ID
        token = os.getenv("AGORA_TOKEN", app_id)
        channel = os.getenv("AGORA_CHANNEL_NAME", "pipecat-test")
        uid = os.getenv("AGORA_UID", "0")

        client = AgoraTransportClient(
            app_id=app_id,
            channel_name=channel,
            uid=uid,
            token=token,
            params=AgoraParams(
                video_out_enabled=True,
            ),
            callbacks=_make_callbacks(),
            transport_name="integration-video-bad-fmt",
        )

        await client.connect()
        try:
            result = await client.write_video(b"\x00" * 100, 10, 10, "BGR")
            self.assertFalse(result)
        finally:
            await client.disconnect()


if __name__ == "__main__":
    unittest.main()
