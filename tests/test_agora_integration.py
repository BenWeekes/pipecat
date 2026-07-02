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

End-to-end voice agent test additionally requires:
    DEEPGRAM_API_KEY     - Deepgram STT API key
    OPENAI_API_KEY       - OpenAI LLM API key
    ELEVENLABS_API_KEY   - ElevenLabs TTS API key
"""

import asyncio
import os
import struct
import unittest
from unittest.mock import AsyncMock

import numpy as np

AGORA_APP_ID = os.getenv("AGORA_APP_ID")

try:
    from agora.rtc.agora_base import (
        AgoraServiceConfig,
        AudioParams,
        AudioPublishType,
        AudioScenarioType,
        AudioSubscriptionOptions,
        ChannelProfileType,
        ClientRoleType,
        RTCConnConfig,
        RtcConnectionPublishConfig,
        VideoPublishType,
    )
    from agora.rtc.agora_service import AgoraService
    from agora.rtc.audio_frame_observer import IAudioFrameObserver
    from agora.rtc.rtc_connection_observer import IRTCConnectionObserver

    from pipecat.transports.agora.transport import (
        AgoraCallbacks,
        AgoraParams,
        AgoraTransportClient,
        get_agora_service,
        release_agora_service,
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

# Check whether all service keys are available for the full voice agent test.
_HAS_SERVICE_KEYS = all(
    os.getenv(k) for k in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY")
)
SHOULD_RUN_E2E = SHOULD_RUN and _HAS_SERVICE_KEYS
E2E_SKIP_REASON = (
    SKIP_REASON
    if not SHOULD_RUN
    else "DEEPGRAM_API_KEY, OPENAI_API_KEY, and ELEVENLABS_API_KEY required"
)


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


# ---------------------------------------------------------------------------
# Helper: lightweight Agora viewer client (subscribe-only)
# ---------------------------------------------------------------------------


class _ViewerConnectionObserver(IRTCConnectionObserver):
    """Tracks connection state for the viewer client.

    Callbacks fire on native Agora threads.  State is recorded in
    plain booleans (thread-safe for single-writer / single-reader on
    CPython) and polled from the asyncio side with ``await poll()``.
    """

    def __init__(self):
        self.connected = False
        self.user_joined_uid: str | None = None

    def on_connected(self, conn, conn_info, reason):
        self.connected = True

    def on_disconnected(self, conn, conn_info, reason):
        self.connected = False

    def on_user_joined(self, conn, user_id):
        self.user_joined_uid = user_id

    def on_user_left(self, conn, user_id, reason):
        pass

    def on_connection_lost(self, conn, conn_info):
        self.connected = False

    def on_error(self, conn, error_code, error_msg):
        pass

    def on_token_privilege_will_expire(self, conn, token):
        pass


class _ViewerAudioObserver(IAudioFrameObserver):
    """Collects PCM audio frames from remote users.

    Callbacks fire on native Agora threads.  We append to a plain
    list and read ``frame_count`` from the asyncio side via polling.
    """

    def __init__(self, sample_rate: int = 16000):
        self._sample_rate = sample_rate
        self.frames: list[bytes] = []
        self.frame_count = 0

    def on_playback_audio_frame_before_mixing(
        self, local_user, channel_id, uid, frame, vad_result_state, vad_result_bytearray
    ):
        pcm = bytes(frame.buffer)
        self.frames.append(pcm)
        self.frame_count += 1
        return 1

    def on_get_playback_audio_frame_param(self, local_user):
        return AudioParams(
            sample_rate=self._sample_rate,
            channels=1,
            mode=0,
            samples_per_call=self._sample_rate // 100,
        )


class _AgoraViewer:
    """A minimal Agora client that joins a channel and subscribes to audio.

    Used by integration tests to verify that a publisher's audio arrives
    through Agora's servers.
    """

    def __init__(self, app_id: str, channel: str, uid: str, token: str):
        self._app_id = app_id
        self._channel = channel
        self._uid = uid
        self._token = token
        self._connection = None
        self.conn_observer = _ViewerConnectionObserver()
        self.audio_observer = _ViewerAudioObserver()

    async def connect(self):
        config = AgoraServiceConfig(
            appid=self._app_id,
            enable_audio_processor=1,
            enable_audio_device=0,
            enable_video=0,
            audio_scenario=AudioScenarioType.AUDIO_SCENARIO_AI_SERVER,
        )
        self._agora_service = get_agora_service(config)

        audio_sub = AudioSubscriptionOptions(
            pcm_data_only=1,
            bytes_per_sample=2,
            number_of_channels=1,
            sample_rate_hz=16000,
        )
        conn_config = RTCConnConfig(
            auto_subscribe_audio=1,
            auto_subscribe_video=0,
            client_role_type=ClientRoleType.CLIENT_ROLE_BROADCASTER,
            channel_profile=ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING,
            audio_subs_options=audio_sub,
        )
        publish_config = RtcConnectionPublishConfig(
            is_publish_audio=False,
            is_publish_video=False,
            audio_publish_type=AudioPublishType.AUDIO_PUBLISH_TYPE_NONE,
            video_publish_type=VideoPublishType.VIDEO_PUBLISH_TYPE_NONE,
        )
        self._connection = self._agora_service.create_rtc_connection(
            conn_config, publish_config
        )
        self._connection.register_observer(self.conn_observer)

        local_user = self._connection.get_local_user()
        local_user.set_playback_audio_frame_before_mixing_parameters(1, 16000)
        self._connection.register_audio_frame_observer(
            self.audio_observer, 0, None
        )
        local_user.subscribe_all_audio()

        ret = self._connection.connect(self._token, self._channel, self._uid)
        if ret != 0:
            raise RuntimeError(f"Viewer connect failed: {ret}")

    async def wait_connected(self, timeout: float = 15):
        """Poll until the viewer is connected to the channel."""
        elapsed = 0.0
        while not self.conn_observer.connected and elapsed < timeout:
            await asyncio.sleep(0.5)
            elapsed += 0.5
        if not self.conn_observer.connected:
            raise TimeoutError("Viewer did not connect within timeout")

    async def wait_audio(self, min_frames: int = 1, timeout: float = 15):
        """Poll until at least *min_frames* audio frames have arrived."""
        elapsed = 0.0
        while self.audio_observer.frame_count < min_frames and elapsed < timeout:
            await asyncio.sleep(0.2)
            elapsed += 0.2
        if self.audio_observer.frame_count < min_frames:
            raise TimeoutError(
                f"Expected >= {min_frames} audio frames, "
                f"got {self.audio_observer.frame_count}"
            )

    async def disconnect(self):
        if self._connection:
            self._connection.disconnect()
            self._connection.release()
            self._connection = None
        release_agora_service()


# ---------------------------------------------------------------------------
# Basic transport tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# End-to-end: publisher pushes audio, viewer receives it via Agora
# ---------------------------------------------------------------------------


@unittest.skipUnless(SHOULD_RUN, SKIP_REASON)
class TestAgoraAudioRelay(unittest.IsolatedAsyncioTestCase):
    """Two clients on the same channel: one pushes audio, the other receives it."""

    async def test_viewer_receives_audio_from_publisher(self):
        """Publisher pushes a 440Hz tone; viewer confirms non-silent audio arrives."""
        app_id = AGORA_APP_ID
        token = os.getenv("AGORA_TOKEN", app_id)
        channel = os.getenv("AGORA_CHANNEL_NAME", "pipecat-test")

        # Publisher (uid "1") — pushes audio
        publisher = AgoraTransportClient(
            app_id=app_id,
            channel_name=channel,
            uid="1",
            token=token,
            params=AgoraParams(
                audio_in_enabled=False,
                audio_out_enabled=True,
                audio_out_sample_rate=16000,
            ),
            callbacks=_make_callbacks(),
            transport_name="e2e-publisher",
        )

        # Viewer (uid "2") — subscribes to audio
        viewer = _AgoraViewer(app_id, channel, "2", token)

        try:
            # Both join the channel
            await publisher.connect()
            await viewer.connect()
            await viewer.wait_connected(timeout=15)

            # Give Agora a moment to set up the media path
            await asyncio.sleep(2)

            # Publisher pushes 2 seconds of 440Hz tone in 20ms chunks
            sample_rate = 16000
            chunk_samples = 320  # 20ms at 16kHz
            total_chunks = 100  # 2 seconds
            t_full = np.arange(chunk_samples * total_chunks) / sample_rate
            tone_full = (np.sin(2 * np.pi * 440 * t_full) * 16000).astype(np.int16)

            for i in range(total_chunks):
                chunk = tone_full[i * chunk_samples : (i + 1) * chunk_samples]
                result = await publisher.write_audio(chunk.tobytes(), sample_rate, 1)
                self.assertTrue(result)

            # Wait for audio to arrive at the viewer
            await viewer.wait_audio(min_frames=1, timeout=15)

            # Verify we received audio frames
            self.assertGreater(
                len(viewer.audio_observer.frames), 0,
                "Viewer should have received at least one audio frame",
            )

            # Verify at least some frames are non-silent
            all_audio = b"".join(viewer.audio_observer.frames)
            samples = np.frombuffer(all_audio, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
            self.assertGreater(
                rms, 100,
                f"Audio should be non-silent (RMS={rms:.1f}), "
                f"received {len(viewer.audio_observer.frames)} frames, "
                f"{len(all_audio)} bytes",
            )
        finally:
            await viewer.disconnect()
            await publisher.disconnect()


# ---------------------------------------------------------------------------
# End-to-end: full voice agent pipeline with viewer
# ---------------------------------------------------------------------------


@unittest.skipUnless(SHOULD_RUN_E2E, E2E_SKIP_REASON)
class TestAgoraVoiceAgent(unittest.IsolatedAsyncioTestCase):
    """Full voice agent pipeline: STT + LLM + TTS over Agora, verified by a viewer."""

    async def test_voice_agent_produces_audio(self):
        """A voice agent joins the channel, speaks a greeting, and
        a viewer client confirms it receives real TTS audio.
        """
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.frames.frames import EndFrame, TTSSpeakFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineParams, PipelineWorker
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
            LLMUserAggregatorParams,
        )
        from pipecat.services.deepgram.stt import DeepgramSTTService
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.transports.agora.transport import AgoraTransport
        from pipecat.workers.runner import WorkerRunner

        app_id = AGORA_APP_ID
        token = os.getenv("AGORA_TOKEN", app_id)
        channel = os.getenv("AGORA_CHANNEL_NAME", "pipecat-test")

        transport = AgoraTransport(
            app_id=app_id,
            channel_name=channel,
            uid="10",
            token=token,
            params=AgoraParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=24000,
            ),
        )

        stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(
                model="gpt-4o-mini",
                system_instruction="You are a test bot. Respond briefly.",
            ),
        )
        tts = ElevenLabsTTSService(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            settings=ElevenLabsTTSService.Settings(
                voice="21m00Tcm4TlvDq8ikWAM",
            ),
        )

        context = LLMContext()
        user_agg, assistant_agg = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                user_agg,
                llm,
                tts,
                transport.output(),
                assistant_agg,
            ]
        )

        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(enable_metrics=True),
            idle_timeout_secs=None,
        )

        # When the viewer joins, the bot speaks a greeting.
        # Uses on_first_participant_joined (the framework-convention alias)
        # to exercise that it works identically to on_first_user_joined.
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, user_id):
            await asyncio.sleep(1)
            await worker.queue_frame(
                TTSSpeakFrame("Hello, this is a test of the Agora voice pipeline.")
            )

        # Viewer (uid "20") — subscribes to audio
        viewer = _AgoraViewer(app_id, channel, "20", token)

        runner = WorkerRunner()
        await runner.add_workers(worker)

        # Run the pipeline in a background task
        runner_task = asyncio.create_task(runner.run())

        try:
            # Wait for bot transport to connect
            await asyncio.sleep(5)

            # Viewer joins — triggers on_first_participant_joined on the bot
            await viewer.connect()
            await viewer.wait_connected(timeout=15)

            # Wait for TTS audio to arrive (greeting takes several seconds:
            # the bot sleeps 1s, then calls the LLM, then streams TTS)
            await viewer.wait_audio(min_frames=1, timeout=30)

            # Let more audio accumulate
            await asyncio.sleep(3)

            # Verify we received real audio
            self.assertGreater(
                len(viewer.audio_observer.frames), 0,
                "Viewer should have received audio from the voice agent",
            )

            all_audio = b"".join(viewer.audio_observer.frames)
            samples = np.frombuffer(all_audio, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
            self.assertGreater(
                rms, 50,
                f"Voice agent audio should be non-silent (RMS={rms:.1f}), "
                f"received {len(viewer.audio_observer.frames)} frames, "
                f"{len(all_audio)} bytes",
            )
        finally:
            # Shut down the pipeline
            await viewer.disconnect()
            await worker.queue_frame(EndFrame())
            try:
                await asyncio.wait_for(runner_task, timeout=15)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                runner_task.cancel()


if __name__ == "__main__":
    unittest.main()
