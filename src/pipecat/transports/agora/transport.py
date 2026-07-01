#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agora transport implementation for Pipecat.

This module provides Agora real-time communication integration
including audio and video streaming, data messaging, user management,
and channel event handling for conversational AI applications.
"""

import asyncio
import json
import time
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    BotConnectedFrame,
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    Frame,
    InputTransportMessageFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    UserAudioRawFrame,
    UserImageRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.asyncio.task_manager import BaseTaskManager

try:
    from agora.rtc.agora_base import (
        AgoraServiceConfig,
        AudioParams,
        AudioPublishType,
        AudioScenarioType,
        AudioSubscriptionOptions,
        ChannelProfileType,
        ClientRoleType,
        ExternalVideoFrame,
        RTCConnConfig,
        RtcConnectionPublishConfig,
        VideoPublishType,
        VideoSubscriptionOptions,
    )
    from agora.rtc.agora_service import AgoraService
    from agora.rtc.audio_frame_observer import IAudioFrameObserver
    from agora.rtc.local_user_observer import IRTCLocalUserObserver
    from agora.rtc.rtc_connection_observer import IRTCConnectionObserver
    from agora.rtc.video_frame_observer import IVideoFrameObserver
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Agora, you need to install the Agora Python Server SDK. "
        'See: https://github.com/AgoraIO-Extensions/Agora-python-Server-SDK'
    )
    raise ImportError(f"Missing module: {e}") from e

# Agora ExternalVideoFrame pixel format constants (from C SDK headers)
_VIDEO_PIXEL_I420 = 1
_VIDEO_PIXEL_RGBA = 4


# ---------------------------------------------------------------------------
# Custom frame types
# ---------------------------------------------------------------------------


@dataclass
class AgoraOutputTransportMessageFrame(OutputTransportMessageFrame):
    """Frame for transport messages in Agora channels.

    Parameters:
        participant_id: Optional ID of the target participant.
    """

    participant_id: str | None = None


@dataclass
class AgoraOutputTransportMessageUrgentFrame(OutputTransportMessageUrgentFrame):
    """Frame for urgent transport messages in Agora channels.

    Parameters:
        participant_id: Optional ID of the target participant.
    """

    participant_id: str | None = None


@dataclass
class AgoraInputTransportMessageFrame(InputTransportMessageFrame):
    """Frame for inbound transport messages in Agora channels.

    Parameters:
        participant_id: Optional ID of the sending participant.
    """

    participant_id: str | None = None


# ---------------------------------------------------------------------------
# AgoraParams
# ---------------------------------------------------------------------------


class AgoraParams(TransportParams):
    """Configuration parameters for Agora transport.

    Parameters:
        app_id: Agora App ID from the Agora Console.
        audio_scenario: Audio scenario for AI server optimization.
        enable_apm: Enable Audio Processing Module.
        apm_config: APM configuration when enable_apm is True.
        enable_vad: Enable Agora's built-in Voice Activity Detection.
        vad_configure: VAD configuration parameters.
        auto_subscribe_audio: Auto-subscribe to all remote audio tracks.
        auto_subscribe_video: Auto-subscribe to all remote video tracks.
        enable_encryption: Enable media encryption.
        encryption_config: Encryption configuration when enable_encryption is True.
    """

    app_id: str = ""
    audio_scenario: int = 9  # AUDIO_SCENARIO_AI_SERVER
    enable_apm: bool = False
    apm_config: Any = None
    enable_vad: bool = False
    vad_configure: Any = None
    auto_subscribe_audio: bool = True
    auto_subscribe_video: bool = False
    enable_encryption: bool = False
    encryption_config: Any = None


# ---------------------------------------------------------------------------
# AgoraCallbacks
# ---------------------------------------------------------------------------


class AgoraCallbacks(BaseModel):
    """Callback handlers for Agora events.

    Parameters:
        on_connected: Called when connected to the Agora channel.
        on_disconnected: Called when disconnected from the channel.
        on_before_disconnect: Called just before disconnecting (sync).
        on_user_joined: Called when a remote user joins the channel.
        on_user_left: Called when a remote user leaves the channel.
        on_audio_track_subscribed: Called when a remote audio track is subscribed.
        on_video_track_subscribed: Called when a remote video track is subscribed.
        on_data_received: Called when a stream message is received.
        on_first_user_joined: Called when the first remote user joins.
        on_token_privilege_will_expire: Called when the token is about to expire.
        on_connection_lost: Called when the connection is lost.
        on_error: Called when an error occurs.
    """

    on_connected: Callable[[], Awaitable[None]]
    on_disconnected: Callable[[], Awaitable[None]]
    on_before_disconnect: Callable[[], Awaitable[None]]
    on_user_joined: Callable[[str], Awaitable[None]]
    on_user_left: Callable[[str, int], Awaitable[None]]
    on_audio_track_subscribed: Callable[[str], Awaitable[None]]
    on_video_track_subscribed: Callable[[str], Awaitable[None]]
    on_data_received: Callable[[bytes, str], Awaitable[None]]
    on_first_user_joined: Callable[[str], Awaitable[None]]
    on_token_privilege_will_expire: Callable[[str], Awaitable[None]]
    on_connection_lost: Callable[[], Awaitable[None]]
    on_error: Callable[[int, str], Awaitable[None]]


# ---------------------------------------------------------------------------
# AgoraService singleton management
# ---------------------------------------------------------------------------

_agora_service: AgoraService | None = None
_agora_service_config: AgoraServiceConfig | None = None
_agora_service_ref_count = 0
_agora_service_lock = threading.Lock()


def _configs_compatible(
    a: AgoraServiceConfig, b: AgoraServiceConfig
) -> tuple[bool, str]:
    """Check whether two AgoraServiceConfigs are materially compatible.

    Returns (True, "") if compatible, or (False, reason) if not.
    """
    checks = [
        ("appid", a.appid, b.appid),
        ("enable_audio_processor", a.enable_audio_processor, b.enable_audio_processor),
        ("enable_audio_device", a.enable_audio_device, b.enable_audio_device),
        ("enable_video", a.enable_video, b.enable_video),
        ("audio_scenario", a.audio_scenario, b.audio_scenario),
        ("channel_profile", a.channel_profile, b.channel_profile),
        ("area_code", a.area_code, b.area_code),
        ("use_string_uid", a.use_string_uid, b.use_string_uid),
        ("enable_apm", a.enable_apm, b.enable_apm),
        ("apm_config", a.apm_config, b.apm_config),
    ]
    for field_name, val_a, val_b in checks:
        if val_a != val_b:
            return False, f"{field_name}: existing={val_a}, requested={val_b}"
    return True, ""


def get_agora_service(config: AgoraServiceConfig) -> AgoraService:
    """Get or create the global AgoraService singleton.

    Thread-safe. The first caller's config is used for initialization.
    Subsequent callers reuse the existing service. Raises ValueError on
    incompatible config.
    """
    global _agora_service, _agora_service_config, _agora_service_ref_count
    with _agora_service_lock:
        if _agora_service is None:
            _agora_service = AgoraService()
            _agora_service.initialize(config)
            _agora_service_config = config
        else:
            compatible, reason = _configs_compatible(_agora_service_config, config)
            if not compatible:
                raise ValueError(
                    f"AgoraService singleton already initialized with "
                    f"incompatible config. Conflict: {reason}. "
                    f"The Agora SDK supports only one AgoraService per process. "
                    f"Use separate processes for different configurations."
                )
        _agora_service_ref_count += 1
        return _agora_service


def release_agora_service():
    """Release a reference to the global AgoraService.

    When the last reference is released, the service is destroyed.
    """
    global _agora_service, _agora_service_ref_count
    with _agora_service_lock:
        _agora_service_ref_count -= 1
        if _agora_service_ref_count <= 0 and _agora_service:
            _agora_service.release()
            _agora_service = None
            _agora_service_ref_count = 0


# ---------------------------------------------------------------------------
# AgoraTransportClient
# ---------------------------------------------------------------------------


class AgoraTransportClient:
    """Core client managing the Agora SDK connection.

    Wraps AgoraService, RTCConnection, and all observers. Handles the
    native-thread to asyncio-event-loop bridge for all SDK callbacks.
    """

    # ------ Internal observer classes ------

    class _ConnectionObserver(IRTCConnectionObserver):
        """Routes Agora connection events to the async event queue."""

        def __init__(self, client: "AgoraTransportClient"):
            self._client = client

        def on_connected(self, conn, conn_info, reason):
            self._client._queue_event(self._client._callbacks.on_connected)

        def on_disconnected(self, conn, conn_info, reason):
            self._client._sdk_disconnected = True
            self._client._connected = False

        def on_user_joined(self, conn, user_id):
            self._client._queue_event(self._client._handle_user_joined, user_id)

        def on_user_left(self, conn, user_id, reason):
            self._client._queue_event(
                self._client._callbacks.on_user_left, user_id, reason
            )

        def on_token_privilege_will_expire(self, conn, token):
            self._client._queue_event(
                self._client._callbacks.on_token_privilege_will_expire, token
            )

        def on_connection_lost(self, conn, conn_info):
            if not self._client._connection_lost_delivered:
                self._client._connection_lost_delivered = True
                self._client._connected = False
                self._client._queue_event(self._client._callbacks.on_connection_lost)

        def on_error(self, conn, error_code, error_msg):
            self._client._queue_event(
                self._client._callbacks.on_error, error_code, error_msg
            )

    class _AudioObserver(IAudioFrameObserver):
        """Routes received audio frames to the async audio queue."""

        def __init__(self, client: "AgoraTransportClient"):
            self._client = client

        def on_playback_audio_frame_before_mixing(
            self, local_user, channel_id, uid, frame, vad_result_state, vad_result_bytearray
        ):
            pcm_bytes = bytes(frame.buffer)
            sample_rate = frame.samples_per_sec
            channels = frame.channels
            loop = self._client._task_manager.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                self._client._audio_queue.put(
                    (pcm_bytes, uid, sample_rate, channels)
                ),
                loop,
            )
            return 1

        def on_get_playback_audio_frame_param(self, local_user):
            return AudioParams(
                sample_rate=self._client._in_sample_rate,
                channels=self._client._params.audio_in_channels,
                mode=0,
                samples_per_call=self._client._in_sample_rate // 100,
            )

    class _LocalUserObserver(IRTCLocalUserObserver):
        """Routes local user events to the async event queue."""

        def __init__(self, client: "AgoraTransportClient"):
            self._client = client

        def on_user_audio_track_subscribed(self, local_user, user_id, remote_track):
            self._client._queue_event(
                self._client._callbacks.on_audio_track_subscribed, user_id
            )

        def on_user_video_track_subscribed(
            self, local_user, user_id, info, remote_track
        ):
            self._client._queue_event(
                self._client._callbacks.on_video_track_subscribed, user_id
            )

        def on_stream_message(self, local_user, user_id, stream_id, data, length):
            raw = data.encode() if isinstance(data, str) else data
            self._client._queue_event(
                self._client._callbacks.on_data_received, raw, user_id
            )

    class _VideoObserver(IVideoFrameObserver):
        """Routes received video frames to the async video queue.

        The Agora SDK delivers video in I420 (YUV420P) planar format.
        We convert to RGB here using numpy for efficient BT.601 conversion,
        then queue the RGB bytes along with the user_id and dimensions.
        """

        def __init__(self, client: "AgoraTransportClient"):
            self._client = client

        def on_frame(self, channel_id, remote_uid, frame):
            try:
                width = frame.width
                height = frame.height

                # Extract I420 planes, copying data before the native
                # buffer is reclaimed.
                y = np.frombuffer(bytes(frame.y_buffer), dtype=np.uint8).reshape(
                    (height, frame.y_stride)
                )[:, :width]
                u = np.frombuffer(bytes(frame.u_buffer), dtype=np.uint8).reshape(
                    (height // 2, frame.u_stride)
                )[:, : width // 2]
                v = np.frombuffer(bytes(frame.v_buffer), dtype=np.uint8).reshape(
                    (height // 2, frame.v_stride)
                )[:, : width // 2]

                # Upsample U and V to full resolution
                u_full = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
                v_full = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)

                # BT.601 YUV→RGB conversion
                y_f = y.astype(np.float32)
                u_f = u_full.astype(np.float32) - 128.0
                v_f = v_full.astype(np.float32) - 128.0

                r = np.clip(y_f + 1.402 * v_f, 0, 255).astype(np.uint8)
                g = np.clip(y_f - 0.344136 * u_f - 0.714136 * v_f, 0, 255).astype(
                    np.uint8
                )
                b = np.clip(y_f + 1.772 * u_f, 0, 255).astype(np.uint8)

                rgb = np.stack([r, g, b], axis=-1)
                rgb_bytes = rgb.tobytes()

                loop = self._client._task_manager.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    self._client._video_queue.put(
                        (rgb_bytes, remote_uid, width, height)
                    ),
                    loop,
                )
            except Exception as e:
                logger.error(f"Error converting video frame: {e}")

    # ------ Initialization ------

    def __init__(
        self,
        app_id: str,
        channel_name: str,
        uid: str,
        token: str,
        params: AgoraParams,
        callbacks: AgoraCallbacks,
        transport_name: str,
    ):
        self._app_id = app_id
        self._channel_name = channel_name
        self._uid = uid
        self._token = token
        self._params = params
        self._callbacks = callbacks
        self._transport_name = transport_name

        self._agora_service: AgoraService | None = None
        self._connection = None
        self._connected = False
        self._disconnect_counter = 0
        self._other_user_has_joined = False
        self._sdk_disconnected = False
        self._connection_lost_delivered = False

        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._video_queue: asyncio.Queue = asyncio.Queue()
        self._event_queue: asyncio.Queue = asyncio.Queue()

        self._task_manager: BaseTaskManager | None = None
        self._event_task: asyncio.Task | None = None
        self._async_lock = asyncio.Lock()

        self._in_sample_rate = 16000
        self._out_sample_rate = 24000

    # ------ Thread-safe event queuing ------

    def _queue_event(self, callback, *args):
        """Thread-safe: called from Agora native callback threads."""
        loop = self._task_manager.get_event_loop()
        asyncio.run_coroutine_threadsafe(
            self._event_queue.put((callback, *args)), loop
        )

    async def _event_task_handler(self):
        """Runs on the asyncio event loop, dispatches queued events."""
        while True:
            item = await self._event_queue.get()
            callback = item[0]
            args = item[1:]
            try:
                await callback(*args)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
            self._event_queue.task_done()

    # ------ Lifecycle ------

    async def setup(self, setup: FrameProcessorSetup):
        """Initialize task manager and start event handler."""
        if self._task_manager:
            return
        self._task_manager = setup.task_manager
        self._event_task = self._task_manager.create_task(
            self._event_task_handler(), f"{self}::_event_task_handler"
        )

    async def start(self, frame: StartFrame):
        """Initialize sample rates from StartFrame."""
        self._in_sample_rate = (
            self._params.audio_in_sample_rate or frame.audio_in_sample_rate
        )
        self._out_sample_rate = (
            self._params.audio_out_sample_rate or frame.audio_out_sample_rate
        )

    async def connect(self):
        """Connect to the Agora channel."""
        async with self._async_lock:
            if self._connected:
                self._disconnect_counter += 1
                return

            logger.info(f"Connecting to Agora channel {self._channel_name}")

            video_enabled = self._params.video_in_enabled or self._params.video_out_enabled

            config = AgoraServiceConfig(
                appid=self._app_id,
                enable_audio_processor=1,
                enable_audio_device=0,
                enable_video=1 if video_enabled else 0,
                audio_scenario=AudioScenarioType(self._params.audio_scenario),
                enable_apm=self._params.enable_apm,
                apm_config=self._params.apm_config,
            )
            self._agora_service = get_agora_service(config)

            try:
                audio_sub_options = AudioSubscriptionOptions(
                    pcm_data_only=1,
                    bytes_per_sample=2,
                    number_of_channels=self._params.audio_in_channels,
                    sample_rate_hz=self._in_sample_rate,
                )
                conn_config = RTCConnConfig(
                    auto_subscribe_audio=1 if self._params.auto_subscribe_audio else 0,
                    auto_subscribe_video=1 if self._params.auto_subscribe_video else 0,
                    client_role_type=ClientRoleType.CLIENT_ROLE_BROADCASTER,
                    channel_profile=ChannelProfileType.CHANNEL_PROFILE_LIVE_BROADCASTING,
                    audio_subs_options=audio_sub_options,
                )
                publish_config = RtcConnectionPublishConfig(
                    is_publish_audio=self._params.audio_out_enabled,
                    is_publish_video=self._params.video_out_enabled,
                    audio_publish_type=AudioPublishType.AUDIO_PUBLISH_TYPE_PCM,
                    video_publish_type=(
                        VideoPublishType.VIDEO_PUBLISH_TYPE_YUV
                        if self._params.video_out_enabled
                        else VideoPublishType.VIDEO_PUBLISH_TYPE_NONE
                    ),
                    audio_scenario=AudioScenarioType(self._params.audio_scenario),
                )
                self._connection = self._agora_service.create_rtc_connection(
                    conn_config, publish_config
                )

                # Register observers
                self._conn_observer = self._ConnectionObserver(self)
                self._connection.register_observer(self._conn_observer)

                self._local_user_observer = self._LocalUserObserver(self)
                self._connection.register_local_user_observer(self._local_user_observer)

                if self._params.audio_in_enabled:
                    local_user = self._connection.get_local_user()
                    local_user.set_playback_audio_frame_before_mixing_parameters(
                        self._params.audio_in_channels, self._in_sample_rate
                    )
                    self._audio_observer = self._AudioObserver(self)
                    vad_config = self._params.vad_configure
                    enable_vad = 1 if self._params.enable_vad else 0
                    self._connection.register_audio_frame_observer(
                        self._audio_observer, enable_vad, vad_config
                    )
                    local_user.subscribe_all_audio()

                if self._params.video_in_enabled:
                    self._video_observer = self._VideoObserver(self)
                    self._connection.register_video_frame_observer(
                        self._video_observer
                    )
                    local_user = self._connection.get_local_user()
                    local_user.subscribe_all_video(VideoSubscriptionOptions())

                if self._params.enable_encryption and self._params.encryption_config:
                    self._connection.enable_encryption(
                        1, self._params.encryption_config
                    )

                ret = self._connection.connect(
                    self._token, self._channel_name, self._uid
                )
                if ret != 0:
                    raise RuntimeError(
                        f"Agora connect failed with error code {ret}"
                    )

                if self._params.audio_out_enabled:
                    self._connection.publish_audio()

                if self._params.video_out_enabled:
                    self._connection.publish_video()

            except Exception:
                logger.error(
                    f"Failed to connect to Agora channel {self._channel_name}"
                )
                if self._connection:
                    self._connection.release()
                    self._connection = None
                release_agora_service()
                self._agora_service = None
                raise

            self._connected = True
            self._disconnect_counter += 1
            logger.info(f"Connected to Agora channel {self._channel_name}")

    async def disconnect(self):
        """Disconnect from the Agora channel."""
        async with self._async_lock:
            self._disconnect_counter -= 1
            if not self._connected or self._disconnect_counter > 0:
                return

            logger.info(f"Disconnecting from Agora channel {self._channel_name}")
            await self._callbacks.on_before_disconnect()

            if self._connection:
                if not self._sdk_disconnected:
                    self._connection.disconnect()
                self._connection.release()
                self._connection = None

            release_agora_service()
            self._agora_service = None
            self._connected = False
            self._sdk_disconnected = False
            self._connection_lost_delivered = False

            logger.info(f"Disconnected from Agora channel {self._channel_name}")
            await self._callbacks.on_disconnected()

    async def cleanup(self):
        """Final cleanup."""
        await self.disconnect()
        if self._event_task:
            self._event_task.cancel()

    # ------ Media I/O ------

    async def write_audio(
        self, audio_data: bytes, sample_rate: int, channels: int
    ) -> bool:
        """Write PCM audio to the Agora channel.

        Includes real-time pacing: sleeps for the duration of the audio
        chunk after pushing it. The Agora SDK's push_audio_pcm_data is
        non-blocking and buffers internally, so without pacing we would
        overwhelm the buffer and audio would cut off.
        """
        if not self._connected or not self._connection:
            return False
        try:
            ret = self._connection.push_audio_pcm_data(
                bytearray(audio_data), sample_rate, channels
            )
            if ret != 0:
                logger.warning(
                    f"push_audio_pcm_data returned {ret}, "
                    f"len={len(audio_data)}, sr={sample_rate}, ch={channels}"
                )
                return False
            # Pace output: sleep for the duration of this audio chunk so
            # we don't push faster than real-time.
            bytes_per_sample = 2
            num_samples = len(audio_data) // (bytes_per_sample * channels)
            duration_secs = num_samples / sample_rate
            await asyncio.sleep(duration_secs)
            return True
        except Exception as e:
            logger.error(f"Error writing audio to Agora: {e}")
            return False

    async def send_data(self, data: bytes) -> bool:
        """Send a data stream message."""
        if not self._connected or not self._connection:
            return False
        try:
            ret = self._connection.send_stream_message(bytearray(data))
            return ret == 0
        except Exception as e:
            logger.error(f"Error sending data: {e}")
            return False

    async def write_video(
        self, image_data: bytes, width: int, height: int, fmt: str
    ) -> bool:
        """Write a video frame to the Agora channel.

        Accepts RGB or RGBA image data. RGB is converted to RGBA by
        appending 0xFF alpha bytes. The ExternalVideoFrame uses
        format=4 (VIDEO_PIXEL_RGBA).
        """
        if not self._connected or not self._connection:
            return False
        try:
            if fmt == "RGB":
                # Convert RGB to RGBA by adding alpha channel
                rgb = np.frombuffer(image_data, dtype=np.uint8).reshape(
                    (height, width, 3)
                )
                rgba = np.empty((height, width, 4), dtype=np.uint8)
                rgba[:, :, :3] = rgb
                rgba[:, :, 3] = 255
                buffer = bytearray(rgba.tobytes())
            elif fmt in ("RGBA", "RGBX"):
                buffer = bytearray(image_data)
            else:
                logger.warning(
                    f"Unsupported video format '{fmt}', expected RGB or RGBA"
                )
                return False

            frame = ExternalVideoFrame(
                type=1,  # raw pixels
                format=_VIDEO_PIXEL_RGBA,
                buffer=buffer,
                stride=width,
                height=height,
                timestamp=int(time.time() * 1000),
            )
            ret = self._connection.push_video_frame(frame)
            if ret != 0:
                logger.warning(f"push_video_frame returned {ret}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error writing video to Agora: {e}")
            return False

    async def interrupt_audio(self):
        """Clear the audio buffer on interruption."""
        if self._connection:
            self._connection.interrupt_audio()

    async def _handle_user_joined(self, user_id: str):
        """Internal handler for user-joined with first-user tracking."""
        await self._callbacks.on_user_joined(user_id)
        if not self._other_user_has_joined:
            self._other_user_has_joined = True
            await self._callbacks.on_first_user_joined(user_id)

    def __str__(self):
        return f"{self._transport_name}::AgoraTransportClient"


# ---------------------------------------------------------------------------
# AgoraInputTransport
# ---------------------------------------------------------------------------


class AgoraInputTransport(BaseInputTransport):
    """Handles incoming media streams and events from Agora channels."""

    def __init__(
        self,
        transport: BaseTransport,
        client: AgoraTransportClient,
        params: AgoraParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        self._audio_in_task: asyncio.Task | None = None
        self._video_in_task: asyncio.Task | None = None
        self._resampler = create_stream_resampler()
        self._initialized = False

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        await super().cleanup()
        await self._transport.cleanup()

    async def start(self, frame: StartFrame):
        await super().start(frame)

        if self._initialized:
            return
        self._initialized = True

        await self._client.start(frame)
        await self._client.connect()

        if self._params.audio_in_enabled and not self._audio_in_task:
            self._audio_in_task = self.create_task(self._audio_in_task_handler())

        if self._params.video_in_enabled and not self._video_in_task:
            self._video_in_task = self.create_task(self._video_in_task_handler())

        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.disconnect()
        if self._audio_in_task:
            await self.cancel_task(self._audio_in_task)
        if self._video_in_task:
            await self.cancel_task(self._video_in_task)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.disconnect()
        if self._audio_in_task:
            await self.cancel_task(self._audio_in_task)
        if self._video_in_task:
            await self.cancel_task(self._video_in_task)

    async def push_app_message(self, message: Any, sender: str):
        """Push an application message received from Agora as a transport frame."""
        frame = AgoraInputTransportMessageFrame(
            message=message, participant_id=sender
        )
        await self.push_frame(frame)

    async def _audio_in_task_handler(self):
        """Consume audio frames from the Agora SDK audio queue."""
        while True:
            pcm_bytes, uid, sample_rate, channels = (
                await self._client._audio_queue.get()
            )

            audio_data = pcm_bytes

            if sample_rate != self.sample_rate:
                audio_data = await self._resampler.resample(
                    audio_data, sample_rate, self.sample_rate
                )

            if len(audio_data) == 0:
                continue

            input_frame = UserAudioRawFrame(
                user_id=uid,
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=channels,
            )
            await self.push_audio_frame(input_frame)

    async def _video_in_task_handler(self):
        """Consume video frames from the Agora SDK video queue."""
        while True:
            rgb_bytes, uid, width, height = await self._client._video_queue.get()
            frame = UserImageRawFrame(
                user_id=uid,
                image=rgb_bytes,
                size=(width, height),
                format="RGB",
            )
            await self.push_video_frame(frame)


# ---------------------------------------------------------------------------
# AgoraOutputTransport
# ---------------------------------------------------------------------------


class AgoraOutputTransport(BaseOutputTransport):
    """Handles outgoing media streams and events to Agora channels."""

    def __init__(
        self,
        transport: BaseTransport,
        client: AgoraTransportClient,
        params: AgoraParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client
        self._initialized = False

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        await super().cleanup()
        await self._transport.cleanup()

    async def start(self, frame: StartFrame):
        await super().start(frame)

        if self._initialized:
            return
        self._initialized = True

        await self._client.start(frame)
        await self._client.connect()
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            await self._client.interrupt_audio()

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write a Pipecat audio frame to the Agora channel."""
        return await self._client.write_audio(
            frame.audio, self.sample_rate, self._params.audio_out_channels
        )

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a Pipecat video frame to the Agora channel."""
        width, height = frame.size
        return await self._client.write_video(
            frame.image, width, height, frame.format or "RGB"
        )

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message via Agora data stream.

        Agora data streams are channel-wide broadcast; per-participant
        targeting is not supported by the SDK. If a participant_id is
        set on the frame it is logged and ignored.
        """
        if isinstance(frame, (AgoraOutputTransportMessageFrame, AgoraOutputTransportMessageUrgentFrame)):
            if frame.participant_id:
                logger.warning(
                    f"Agora data streams do not support per-participant targeting. "
                    f"Message will be broadcast to all users in the channel "
                    f"(requested participant_id={frame.participant_id})."
                )
        message = frame.message
        if isinstance(message, dict):
            message = json.dumps(message, ensure_ascii=False)
        if isinstance(message, str):
            message = message.encode("utf-8")
        await self._client.send_data(message)


# ---------------------------------------------------------------------------
# AgoraTransport
# ---------------------------------------------------------------------------


class AgoraTransport(BaseTransport):
    """Transport implementation for Agora real-time communication.

    Event handlers available:

    - on_connected: Called when the bot connects to the channel.
    - on_disconnected: Called when the bot disconnects from the channel.
    - on_before_disconnect: [sync] Called just before disconnecting.
    - on_user_joined: Called when a remote user joins. Args: (user_id: str)
    - on_user_left: Called when a remote user leaves.
      Args: (user_id: str, reason: int)
    - on_audio_track_subscribed: Called when a remote audio track is subscribed.
      Args: (user_id: str)
    - on_video_track_subscribed: Called when a remote video track is subscribed.
      Args: (user_id: str)
    - on_data_received: Called when data is received. Args: (data: bytes, user_id: str)
    - on_first_user_joined: Called when the first remote user joins.
      Args: (user_id: str)
    - on_token_privilege_will_expire: Called when the token is about to expire.
      Args: (token: str)
    - on_connection_lost: Called when the connection is lost.
    - on_error: Called when an error occurs. Args: (error_code: int, error_msg: str)

    Example::

        @transport.event_handler("on_first_user_joined")
        async def on_first_user_joined(transport, user_id):
            await worker.queue_frame(TTSSpeakFrame("Hello!"))

        @transport.event_handler("on_user_left")
        async def on_user_left(transport, user_id, reason):
            await worker.queue_frame(EndFrame())
    """

    def __init__(
        self,
        app_id: str,
        channel_name: str,
        uid: str,
        token: str,
        params: AgoraParams | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the Agora transport.

        Args:
            app_id: Agora App ID.
            channel_name: Name of the Agora channel to join.
            uid: Numeric user ID string.
            token: Authentication token (use app_id for testing-mode projects).
            params: Configuration parameters for the transport.
            input_name: Optional name for the input transport.
            output_name: Optional name for the output transport.
        """
        super().__init__(input_name=input_name, output_name=output_name)

        self._params = params or AgoraParams()

        callbacks = AgoraCallbacks(
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_before_disconnect=self._on_before_disconnect,
            on_user_joined=self._on_user_joined,
            on_user_left=self._on_user_left,
            on_audio_track_subscribed=self._on_audio_track_subscribed,
            on_video_track_subscribed=self._on_video_track_subscribed,
            on_data_received=self._on_data_received,
            on_first_user_joined=self._on_first_user_joined,
            on_token_privilege_will_expire=self._on_token_privilege_will_expire,
            on_connection_lost=self._on_connection_lost,
            on_error=self._on_error,
        )

        self._client = AgoraTransportClient(
            app_id=app_id or self._params.app_id,
            channel_name=channel_name,
            uid=uid,
            token=token,
            params=self._params,
            callbacks=callbacks,
            transport_name=self.name,
        )
        self._input: AgoraInputTransport | None = None
        self._output: AgoraOutputTransport | None = None

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_user_joined")
        self._register_event_handler("on_user_left")
        self._register_event_handler("on_audio_track_subscribed")
        self._register_event_handler("on_video_track_subscribed")
        self._register_event_handler("on_data_received")
        self._register_event_handler("on_first_user_joined")
        self._register_event_handler("on_token_privilege_will_expire")
        self._register_event_handler("on_connection_lost")
        self._register_event_handler("on_error")
        self._register_event_handler("on_before_disconnect", sync=True)

    def input(self) -> AgoraInputTransport:
        """Get the input transport for receiving media and events."""
        if not self._input:
            self._input = AgoraInputTransport(
                self, self._client, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> AgoraOutputTransport:
        """Get the output transport for sending media and events."""
        if not self._output:
            self._output = AgoraOutputTransport(
                self, self._client, self._params, name=self._output_name
            )
        return self._output

    # --- Event handler routing ---

    async def _on_connected(self):
        await self._call_event_handler("on_connected")
        if self._input:
            await self._input.push_frame(BotConnectedFrame())

    async def _on_disconnected(self):
        await self._call_event_handler("on_disconnected")

    async def _on_before_disconnect(self):
        await self._call_event_handler("on_before_disconnect")

    async def _on_user_joined(self, user_id: str):
        await self._call_event_handler("on_user_joined", user_id)
        if self._input:
            await self._input.push_frame(ClientConnectedFrame())

    async def _on_user_left(self, user_id: str, reason: int):
        await self._call_event_handler("on_user_left", user_id, reason)

    async def _on_audio_track_subscribed(self, user_id: str):
        await self._call_event_handler("on_audio_track_subscribed", user_id)

    async def _on_video_track_subscribed(self, user_id: str):
        await self._call_event_handler("on_video_track_subscribed", user_id)

    async def _on_data_received(self, data: bytes, user_id: str):
        await self._call_event_handler("on_data_received", data, user_id)
        if self._input:
            try:
                message = data.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning(
                    f"Non-UTF-8 data message from user {user_id} "
                    f"({len(data)} bytes). Raw bytes delivered via "
                    f"on_data_received; skipping transport frame."
                )
                return
            await self._input.push_app_message(message, user_id)

    async def _on_first_user_joined(self, user_id: str):
        await self._call_event_handler("on_first_user_joined", user_id)

    async def _on_token_privilege_will_expire(self, token: str):
        await self._call_event_handler("on_token_privilege_will_expire", token)

    async def _on_connection_lost(self):
        await self._call_event_handler("on_connection_lost")

    async def _on_error(self, error_code: int, error_msg: str):
        await self._call_event_handler("on_error", error_code, error_msg)

    # --- Convenience methods ---

    async def send_message(self, message: str):
        """Send a message to the Agora channel."""
        if self._output:
            frame = OutputTransportMessageFrame(message=message)
            await self._output.send_message(frame)

    async def renew_token(self, token: str):
        """Renew the Agora token.

        Persists the new token so reconnections use the updated value.
        """
        self._client._token = token
        if self._client._connection:
            self._client._connection.renew_token(token)
