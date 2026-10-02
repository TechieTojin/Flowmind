"""Minimal in-process ASGI client so endpoint tests need no extra dependency (e.g. httpx)."""

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class ASGIResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


def encode_multipart(
    fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]
) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    for name, (filename, content, content_type) in files.items():
        header = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        parts.append(header + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class ASGIClient:
    def __init__(self, app: Any) -> None:
        self.app = app

    def request(
        self, method: str, path: str, body: bytes = b"", headers: dict[str, str] | None = None
    ) -> ASGIResponse:
        all_headers = {"host": "testserver", "content-length": str(len(body)), **(headers or {})}
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(k.lower().encode(), v.encode()) for k, v in all_headers.items()],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
        }
        status = 500
        response_headers: dict[str, str] = {}
        chunks: list[bytes] = []

        async def run() -> None:
            nonlocal status
            body_sent = False

            async def receive() -> dict[str, Any]:
                nonlocal body_sent
                if not body_sent:
                    body_sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await asyncio.sleep(3600)  # nothing else to receive
                return {"type": "http.disconnect"}

            async def send(message: dict[str, Any]) -> None:
                nonlocal status
                if message["type"] == "http.response.start":
                    status = message["status"]
                    response_headers.update(
                        {k.decode(): v.decode() for k, v in message.get("headers", [])}
                    )
                elif message["type"] == "http.response.body":
                    chunks.append(message.get("body", b""))

            await self.app(scope, receive, send)

        asyncio.run(run())
        return ASGIResponse(status, response_headers, b"".join(chunks))

    def get(self, path: str) -> ASGIResponse:
        return self.request("GET", path)

    def post_json(self, path: str, payload: Any) -> ASGIResponse:
        return self.request(
            "POST", path, json.dumps(payload).encode(), {"content-type": "application/json"}
        )

    def post_file(
        self,
        path: str,
        filename: str,
        content: bytes,
        fields: dict[str, str] | None = None,
        content_type: str = "application/octet-stream",
    ) -> ASGIResponse:
        body, multipart_type = encode_multipart(
            fields or {}, {"file": (filename, content, content_type)}
        )
        return self.request("POST", path, body, {"content-type": multipart_type})
