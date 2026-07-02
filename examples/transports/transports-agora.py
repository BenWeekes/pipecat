#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agora transport example for Pipecat.

This example shows how to build a voice agent using the Agora transport
with Deepgram STT, OpenAI LLM, and ElevenLabs TTS.

Install::

    pip install "pipecat-ai[agora]"

Usage::

    set -a && source .env && set +a
    python transports-agora.py

This connects directly to an Agora channel (no HTTP server). When
``AGORA_APP_CERTIFICATE`` is set, the script auto-mints tokens, prints
a browser URL for the Agora web demo, and opens it so you can speak to
the bot immediately.

Without a certificate, a separate Agora client (web or mobile) must
join the same channel to interact with the bot.

Required environment variables:

    AGORA_APP_ID       - Agora App ID from the Agora Console
    DEEPGRAM_API_KEY   - Deepgram API key for STT
    OPENAI_API_KEY     - OpenAI API key for LLM
    ELEVENLABS_API_KEY - ElevenLabs API key for TTS

Optional environment variables:

    AGORA_APP_CERTIFICATE - Agora App Certificate (enables auto token
                            minting and browser viewer URL)
    AGORA_TOKEN           - Pre-minted token (default: app_id for testing-mode)
    AGORA_CHANNEL_NAME    - Channel name (default: auto-generated)
    AGORA_UID             - User ID (default: "0")

See ``src/pipecat/transports/agora/README.md`` for the full quickstart.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.agora import viewer_uid, build_viewer_url, configure, mint_token
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.agora.transport import AgoraParams, AgoraTransport
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    (app_id, channel_name, uid, token) = await configure()

    app_certificate = os.getenv("AGORA_APP_CERTIFICATE")
    if app_certificate:
        import webbrowser

        vuid = viewer_uid(uid)
        viewer_token = mint_token(app_id, app_certificate, channel_name, vuid)
        viewer_url = build_viewer_url(app_id, channel_name, viewer_token, vuid)
        print(f"   → Viewer URL: {viewer_url}")
        webbrowser.open(viewer_url)

    transport = AgoraTransport(
        app_id=app_id,
        channel_name=channel_name,
        uid=uid,
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
            system_instruction=(
                "You are a helpful assistant in a voice conversation. "
                "Your responses will be spoken aloud, so avoid emojis, "
                "bullet points, or other formatting that can't be spoken. "
                "Respond to what the user said in a creative, helpful, and brief way."
            ),
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(
            voice="21m00Tcm4TlvDq8ikWAM",  # Rachel
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=None,
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, user_id):
        await asyncio.sleep(1)
        await worker.queue_frame(
            TTSSpeakFrame("Hello! I'm your Agora-powered voice assistant. How can I help you?")
        )

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, user_id, reason):
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
