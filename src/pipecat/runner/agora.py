#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agora channel and token configuration utilities for the development runner.

Required environment variables:

- AGORA_APP_ID - Agora App ID from the Agora Console

Optional environment variables:

- AGORA_APP_CERTIFICATE - Agora App Certificate (secret, never commit).
  When set, the runner mints tokens automatically (requires agora-token-builder).
- AGORA_TOKEN - Pre-minted token. Used when AGORA_APP_CERTIFICATE is not set.
- AGORA_CHANNEL_NAME - Default channel name (otherwise auto-generated).
- AGORA_UID - Default user ID (otherwise "0"). Must be numeric.

Token resolution order (deterministic -- always produces a string token):

1. Pre-minted token from /start body, then AGORA_TOKEN env var.
2. If neither, and AGORA_APP_CERTIFICATE is set: mint via agora-token-builder
   (ImportError is fatal -- the user intended minting).
3. If no token and no certificate: token = app_id (Agora SDK convention
   for projects with App Certificate disabled).

Install::

    pip install "pipecat-ai[agora]"

Example::

    from pipecat.runner.agora import configure

    app_id, channel_name, uid, token = await configure()
"""

import os
import secrets
import time
import urllib.parse

from loguru import logger


async def configure(
    channel_name: str | None = None,
    uid: str | None = None,
    token: str | None = None,
    token_ttl: int = 3600,
) -> tuple[str, str, str, str]:
    """Configure Agora channel credentials from arguments or environment.

    Always returns a non-None token string. The SDK's connect() expects a
    string token -- this function ensures one is always available.

    Args:
        channel_name: Channel name. Defaults to env AGORA_CHANNEL_NAME or
            a generated name.
        uid: User ID (numeric string). Defaults to env AGORA_UID or "0".
        token: Pre-minted token. Defaults to env AGORA_TOKEN.
        token_ttl: Token validity in seconds when minting (default 3600).

    Returns:
        Tuple of (app_id, channel_name, uid, token). Token is always a
        non-empty string.

    Raises:
        ValueError: If AGORA_APP_ID is not set, or if uid is not numeric.
        ImportError: If AGORA_APP_CERTIFICATE is set but agora-token-builder
            is not installed.
    """
    app_id = os.getenv("AGORA_APP_ID")
    if not app_id:
        raise ValueError(
            "AGORA_APP_ID must be set in environment variables. "
            "Get it from the Agora Console."
        )

    channel_name = channel_name or os.getenv("AGORA_CHANNEL_NAME") or f"pipecat-{secrets.token_hex(4)}"
    uid = uid or os.getenv("AGORA_UID") or "0"

    # Validate UID is numeric (v1 policy)
    if not uid.isdigit():
        raise ValueError(
            f"Agora UID must be numeric in v1 (got '{uid}'). "
            "String UIDs require separate SDK and token-builder handling."
        )

    token = token or os.getenv("AGORA_TOKEN")

    # Token resolution: always produce a non-None string
    if not token:
        app_certificate = os.getenv("AGORA_APP_CERTIFICATE")
        if app_certificate:
            try:
                from agora_token_builder import RtcTokenBuilder
            except ImportError:
                raise ImportError(
                    "AGORA_APP_CERTIFICATE is set but agora-token-builder is not installed. "
                    "Install it with: pip install agora-token-builder"
                )
            token = RtcTokenBuilder.buildTokenWithUid(
                app_id, app_certificate, channel_name, int(uid), 1, int(time.time()) + token_ttl
            )
            logger.info(f"Generated Agora token for channel {channel_name}")
        else:
            # No certificate and no pre-minted token. Use app_id as token,
            # which is the Agora SDK convention for projects with App
            # Certificate disabled.
            token = app_id
            logger.info(
                "No AGORA_TOKEN or AGORA_APP_CERTIFICATE set. "
                "Using app_id as token (Agora testing-mode convention)."
            )

    return (app_id, channel_name, uid, token)


def viewer_uid(bot_uid: str) -> int:
    """Derive a viewer UID that won't collide with the bot UID.

    When the bot UID is 0 (Agora assigns one at connect time), we generate
    a random UID. Otherwise we pick bot_uid + 1, wrapping within Agora's
    32-bit unsigned range.
    """
    bot = int(bot_uid)
    if bot == 0:
        import random

        return random.randint(10000, 2**31 - 1)
    return (bot + 1) % (2**32)


def mint_token(
    app_id: str,
    app_certificate: str,
    channel_name: str,
    uid: int,
    ttl: int = 3600,
) -> str:
    """Mint an Agora RTC token.

    Args:
        app_id: Agora App ID.
        app_certificate: Agora App Certificate.
        channel_name: Channel name the token grants access to.
        uid: Numeric user ID.
        ttl: Token validity in seconds (default 3600).

    Returns:
        The token string.

    Raises:
        ImportError: If agora-token-builder is not installed.
    """
    from agora_token_builder import RtcTokenBuilder

    return RtcTokenBuilder.buildTokenWithUid(
        app_id, app_certificate, channel_name, uid, 1, int(time.time()) + ttl
    )


def build_viewer_url(
    app_id: str,
    channel_name: str,
    token: str,
    viewer_uid: int,
) -> str:
    """Build an Agora web demo URL for testing.

    Args:
        app_id: Agora App ID.
        channel_name: Channel name.
        token: Pre-minted token for the viewer.
        viewer_uid: Numeric UID encoded in the token.

    Returns:
        URL string for the Agora web demo basic voice call page.
    """
    encoded_token = urllib.parse.quote(token, safe="")
    return (
        f"https://webdemo.agora.io/basicVoiceCall/index.html"
        f"?appid={app_id}&channel={channel_name}&token={encoded_token}&uid={viewer_uid}"
    )
