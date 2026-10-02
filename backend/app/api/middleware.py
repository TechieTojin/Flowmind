from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from app.config import get_settings

# Allowance for multipart boundaries and form fields around the file itself.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024


async def reject_oversized_uploads(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject requests whose declared Content-Length is clearly above the upload limit,
    before the multipart body is received. The route also enforces the exact limit."""
    max_bytes = get_settings().max_upload_bytes
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit():
        if int(content_length) > max_bytes + _MULTIPART_OVERHEAD_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"File exceeds the maximum upload size of {max_bytes // (1024 * 1024)} MB."
                },
            )
    return await call_next(request)
