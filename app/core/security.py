"""Internal service authentication.

The NestJS backend proxies requests to this service and attaches a shared
secret in the ``X-Internal-Token`` header. Every route except ``/health``
depends on :func:`verify_internal_token`.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.core.config import get_settings


def verify_internal_token(
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> None:
    """Validate the shared internal token.

    Raises ``401`` when the header is missing or does not match the configured
    secret. Compares UTF-8 bytes in constant time so malformed headers cannot
    raise.
    """

    expected = get_settings().internal_service_token.encode("utf-8")
    provided = (x_internal_token or "").encode("utf-8")

    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Internal-Token header.",
        )
