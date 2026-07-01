#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agora transport example for Pipecat.

This example shows how to build a voice agent using the Agora transport
with Deepgram STT, OpenAI LLM, and ElevenLabs TTS.

Usage (runner mode -- recommended):

    python transports-agora.py -t agora

Usage (direct mode):

    python transports-agora.py

Required environment variables:

    AGORA_APP_ID       - Agora App ID from the Agora Console
    DEEPGRAM_API_KEY   - Deepgram API key for STT
    OPENAI_API_KEY     - OpenAI API key for LLM
    ELEVENLABS_API_KEY - ElevenLabs API key for TTS

Optional environment variables:

    AGORA_CHANNEL_NAME - Channel name (default: "pipecat")
    AGORA_UID          - User ID (default: "0")
    AGORA_TOKEN        - Pre-minted token (default: app_id for testing-mode)
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
from pipecat.runner.types import AgoraRunnerArguments, RunnerArguments
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.agora.transport import AgoraParams, AgoraTransport
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def bot(runner_args: RunnerArguments):
    """Bot entry point -- works with both runner and direct invocation."""

    # When launched via the runner, runner_args is AgoraRunnerArguments with
    # pre-configured credentials. For direct invocation we build the
    # transport from environment variables.
    if isinstance(runner_args, AgoraRunnerArguments):
        transport = AgoraTransport(
            app_id=runner_args.app_id,
            channel_name=runner_args.channel_name,
            uid=runner_args.uid,
            token=runner_args.token,
            params=AgoraParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=24000,
            ),
        )
    else:
        app_id = os.environ["AGORA_APP_ID"]
        transport = AgoraTransport(
            app_id=app_id,
            channel_name=os.getenv("AGORA_CHANNEL_NAME", "pipecat"),
            uid=os.getenv("AGORA_UID", "0"),
            token=os.getenv("AGORA_TOKEN", app_id),
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

    @transport.event_handler("on_first_user_joined")
    async def on_first_user_joined(transport, user_id):
        await asyncio.sleep(1)
        await worker.queue_frame(
            TTSSpeakFrame("Hello! I'm your Agora-powered voice assistant. How can I help you?")
        )

    @transport.event_handler("on_user_left")
    async def on_user_left(transport, user_id, reason):
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    # Support both runner mode (python bot.py -t agora) and direct mode
    if len(sys.argv) > 1 and ("-t" in sys.argv or "--transport" in sys.argv):
        from pipecat.runner.run import main

        main()
    else:
        # Direct mode: run without the development runner
        app_id = os.environ["AGORA_APP_ID"]
        args = AgoraRunnerArguments(
            app_id=app_id,
            channel_name=os.getenv("AGORA_CHANNEL_NAME", "pipecat"),
            uid=os.getenv("AGORA_UID", "0"),
            token=os.getenv("AGORA_TOKEN", app_id),
        )
        args.handle_sigint = True
        asyncio.run(bot(args))
