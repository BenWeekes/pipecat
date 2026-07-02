# Agora Transport for Pipecat

Real-time voice and video transport using [Agora's](https://www.agora.io/) server-side SDK. Connects your Pipecat bot to an Agora channel where browser or mobile clients can talk to it.

## Quickstart

### 1. Install

```bash
pip install "pipecat-ai[agora]"
```

This installs `agora-python-server-sdk` (the Linux/macOS server SDK) and `agora-token-builder` (for minting tokens).

### 2. Get credentials

1. Create a project at [console.agora.io](https://console.agora.io/)
2. Copy your **App ID**
3. Enable **App Certificate** under the project's security settings and copy the certificate

### 3. Set environment variables

```bash
# Required
AGORA_APP_ID=your_app_id

# Optional - set the certificate to enable auto-minting of tokens
AGORA_APP_CERTIFICATE=your_app_certificate

# Optional - override defaults
AGORA_CHANNEL_NAME=pipecat        # default: auto-generated
AGORA_UID=0                       # default: "0"
```

**Token resolution order:**

1. Pre-minted token passed as argument or `AGORA_TOKEN` env var
2. If `AGORA_APP_CERTIFICATE` is set: auto-mints via `agora-token-builder`
3. If neither: uses `app_id` as token (works when App Certificate is disabled in the Agora Console)

### 4. Run the example

```bash
# Set your .env, then:
set -a && source .env && set +a
python examples/transports/transports-agora.py
```

If `AGORA_APP_CERTIFICATE` is set, the bot mints a viewer token, prints a browser URL, and opens it automatically. You can speak to the bot directly in the browser.

### 5. Minimal bot code

```python
import asyncio
from pipecat.runner.agora import configure
from pipecat.transports.agora.transport import AgoraTransport, AgoraParams

async def main():
    app_id, channel_name, uid, token = await configure()

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

    # Build your pipeline with transport.input() and transport.output()
    # See examples/transports/transports-agora.py for a full voice agent

asyncio.run(main())
```

## Using with the development runner

The Pipecat development runner supports Agora as a direct-connect transport (no HTTP server):

```bash
python your_bot.py -t agora
```

This calls `_run_agora()` in the runner, which configures credentials, starts the bot, and (if a certificate is set) opens a browser viewer URL.

Agora is also available via `POST /start` with `{"transport": "agora"}` when the runner is in multi-transport mode.

## Helper functions

### `configure()`

Async function that resolves credentials from arguments and environment variables. Returns `(app_id, channel_name, uid, token)`.

```python
from pipecat.runner.agora import configure

app_id, channel_name, uid, token = await configure(
    channel_name="my-channel",  # optional override
    uid="0",                    # optional override
    token="pre-minted",         # optional override
    token_ttl=3600,             # seconds, default 3600
)
```

### `mint_token()`

Mint a token directly (requires `AGORA_APP_CERTIFICATE`):

```python
from pipecat.runner.agora import mint_token

token = mint_token(app_id, app_certificate, channel_name, uid=12345, ttl=3600)
```

### `build_viewer_url()`

Build an Agora web demo URL for browser testing:

```python
from pipecat.runner.agora import build_viewer_url

url = build_viewer_url(app_id, channel_name, token, viewer_uid=12345)
# Returns: https://webdemo.agora.io/basicVoiceCall/index.html?appid=...&channel=...&token=...&uid=...
```

## Transport parameters

`AgoraParams` extends `TransportParams` with Agora-specific settings:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_in_enabled` | bool | False | Enable audio input from remote users |
| `audio_out_enabled` | bool | False | Enable audio output to the channel |
| `video_in_enabled` | bool | False | Enable video input from remote users |
| `video_out_enabled` | bool | False | Enable video output to the channel |
| `audio_in_sample_rate` | int | 16000 | Input audio sample rate (Hz) |
| `audio_out_sample_rate` | int | 24000 | Output audio sample rate (Hz) |
| `audio_scenario` | int | 9 | Audio scenario (9 = AI server optimized) |
| `auto_subscribe_audio` | bool | True | Auto-subscribe to remote audio tracks |
| `auto_subscribe_video` | bool | False | Auto-subscribe to remote video tracks |
| `enable_apm` | bool | False | Enable Audio Processing Module |
| `enable_vad` | bool | False | Enable Agora's built-in VAD |
| `enable_encryption` | bool | False | Enable media encryption |

## Event handlers

Register handlers on `AgoraTransport` with the `@transport.event_handler()` decorator:

```python
@transport.event_handler("on_first_participant_joined")
async def on_first_participant_joined(transport, user_id):
    await worker.queue_frame(TTSSpeakFrame("Hello!"))

@transport.event_handler("on_participant_left")
async def on_participant_left(transport, user_id, reason):
    await worker.queue_frame(EndFrame())
```

| Event | Args | Description |
|-------|------|-------------|
| `on_connected` | — | Bot connected to the channel |
| `on_disconnected` | — | Bot disconnected from the channel |
| `on_before_disconnect` | — | Just before disconnecting (sync) |
| `on_participant_joined` | `user_id` | Remote user joined |
| `on_participant_left` | `user_id, reason` | Remote user left |
| `on_first_participant_joined` | `user_id` | First remote user joined (fires once) |
| `on_audio_track_subscribed` | `user_id` | Remote audio track subscribed |
| `on_video_track_subscribed` | `user_id` | Remote video track subscribed |
| `on_data_received` | `data, user_id` | Stream message received |
| `on_token_privilege_will_expire` | `token` | Token is about to expire |
| `on_connection_lost` | — | Connection lost |
| `on_error` | `error_code, error_msg` | Error occurred |

Aliases `on_user_joined`, `on_user_left`, and `on_first_user_joined` are also available for consistency with Daily/LiveKit transports.

## File layout

```
src/pipecat/transports/agora/
    transport.py          # AgoraTransport, AgoraTransportClient, AgoraParams
src/pipecat/runner/
    agora.py              # configure(), mint_token(), build_viewer_url()
    run.py                # _run_agora() for -t agora mode
    types.py              # AgoraRunnerArguments dataclass
    utils.py              # create_transport() Agora factory
examples/transports/
    transports-agora.py   # Full voice agent example
tests/
    test_agora_transport.py                # Unit tests
    integration/test_agora_integration.py  # Integration tests
```

## Troubleshooting

**`ImportError: agora-python-server-sdk not installed`** — Run `pip install "pipecat-ai[agora]"`. The Agora server SDK supports Linux and macOS.

**`ImportError: agora-token-builder is not installed`** — Set `AGORA_APP_CERTIFICATE` but the token builder package is missing. Run `pip install agora-token-builder`.

**Viewer gets error code 110** — Token authentication failed. Each UID needs its own token when App Certificate is enabled. Make sure the viewer token is minted for the viewer's UID, not the bot's UID.

**No audio from bot** — Check that `audio_out_enabled=True` is set in `AgoraParams` and that your TTS service is configured.

**Bot doesn't respond to speech** — Check that `audio_in_enabled=True` is set and your STT service API key is valid.
