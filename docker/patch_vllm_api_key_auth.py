#!/usr/bin/env python3
"""Apply the Python production changes from vLLM PR #58028 before wheel builds.

Keep API-key authentication on by default outside the liveness allowlist.
Match the middleware code and docstring notes separately to support the regular
and B12X layouts on fresh source. Unexpected layouts fail before either
production file is written. No vLLM imports needed.
"""

import argparse
from pathlib import Path


AUTH_REL = Path("vllm/entrypoints/serve/middleware/authenticate.py")
CLI_REL = Path("vllm/entrypoints/launchers/cli_args.py")

# Production blocks from PR #58028 at ab14efdfba01.
OLD_GUARD = '''GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")
'''

NEW_GUARD = '''UNGUARDED_PATHS = frozenset({"/health", "/ping", "/load", "/version"})


def _is_cors_preflight(scope: Scope, headers: Headers) -> bool:
    """Return True for a CORS preflight, which browsers send without a token.

    This is the same test that Starlette's CORSMiddleware uses before it
    answers a preflight itself. Both headers are necessary. Without "origin",
    CORSMiddleware sends the request through to the app, and a mount such as
    /metrics answers any method. A bare OPTIONS request must therefore pass
    the token check like every other request.
    """
    return (
        scope.get("method") == "OPTIONS"
        and "origin" in headers
        and "access-control-request-method" in headers
    )
'''

OLD_CALL = '''    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if (
            scope["type"] not in ("http", "websocket")
            or scope.get("method") == "OPTIONS"
        ):
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        root_path = scope.get("root_path", "")
        url_path = scope["path"].removeprefix(root_path)
        headers = Headers(scope=scope)
        # Type narrow to satisfy mypy.
        if url_path.startswith(GUARDED_PREFIX) and not self.verify_token(headers):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)
'''

NEW_CALL = '''    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] not in ("http", "websocket"):
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        root_path = scope.get("root_path", "")
        url_path = scope["path"].removeprefix(root_path)
        # This middleware runs ahead of the router, so a path that differs from
        # an allowlisted one only by a trailing slash never reaches FastAPI's
        # redirect_slashes: match on the normalized path, or a liveness probe
        # configured as /health/ is answered with 401.
        probe_path = url_path.rstrip("/") or "/"
        headers = Headers(scope=scope)
        if _is_cors_preflight(scope, headers):
            return self.app(scope, receive, send)
        # Type narrow to satisfy mypy.
        if probe_path not in UNGUARDED_PATHS and not self.verify_token(headers):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)
'''

OLD_NOTES = '''        1. The HTTP method is OPTIONS.
        2. The request path doesn't start with GUARDED_PREFIX (e.g. /health).
'''

NEW_NOTES = '''        1. The request is a CORS preflight: OPTIONS with an Origin header
           and an Access-Control-Request-Method header.
        2. The request path, ignoring a trailing slash, is one of
           UNGUARDED_PATHS (e.g. /health).
'''

OLD_HELP = '''    """If provided, the server will require one of these keys to be presented in
    the header.

    Warning: this only authenticates endpoints under the `/v1`, `/v2`, and
    `/inference` path prefixes. Other endpoints on the same server, including
    `/invocations` (which exposes the same inference capabilities as `/v1`),
    remain unauthenticated. Do not rely on `--api-key` alone to secure vLLM;
    see
    https://docs.vllm.ai/en/latest/usage/security.html#api-key-authentication-limitations
    for what it does and does not protect."""'''

NEW_HELP = '''    """If provided, the server will require one of these keys to be presented in
    the header.

    This authenticates every endpoint on the server, except a small liveness
    allowlist (`/health`, `/ping`, `/load`, `/version`). All other endpoints
    need the key. This includes `/invocations`, `/tokenize` and `/metrics`.
    A scraper that must read `/metrics` without a key needs a separate
    listener. See
    https://docs.vllm.ai/en/latest/usage/security.html#api-key-authentication-limitations
    for what `--api-key` does and does not protect."""'''


def replace_blocks(source: str, replacements: tuple[tuple[str, str], ...]) -> str:
    if not all(source.count(old) == 1 for old, _ in replacements):
        raise ValueError("Unexpected API-key authentication source layout")
    for old, new in replacements:
        source = source.replace(old, new, 1)
    return source


def patch_authentication(source: str) -> str:
    patched = replace_blocks(source, (
        (OLD_GUARD, NEW_GUARD),
        (OLD_CALL, NEW_CALL),
        (OLD_NOTES, NEW_NOTES),
    ))
    compile(patched, str(AUTH_REL), "exec")
    return patched


def patch_cli_help(source: str) -> str:
    patched = replace_blocks(source, ((OLD_HELP, NEW_HELP),))
    compile(patched, str(CLI_REL), "exec")
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    auth_path = args.source_root / AUTH_REL
    if not auth_path.exists():
        print("Separate authentication middleware is absent; API-key patch is not applicable")
        return
    updates = []
    try:
        for relative, patcher in ((AUTH_REL, patch_authentication), (CLI_REL, patch_cli_help)):
            path = args.source_root / relative
            source = path.read_text()
            patched = patcher(source)
            updates.append((path, patched))
    except (OSError, ValueError, SyntaxError) as exc:
        raise SystemExit(f"Unable to apply API-key authentication patch: {exc}") from exc
    for path, patched in updates:
        path.write_text(patched)
    print("Applied vLLM PR #58028 Python API-key authentication patch")


if __name__ == "__main__":
    main()
