#!/usr/bin/env python3
"""CPU-only source and route-policy checks for the API-key authentication patch."""

import ast
import hashlib
import importlib.util
import secrets
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Awaitable
from pathlib import Path
from unittest.mock import Mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCH_PATH = PROJECT_DIR / "docker/patch_vllm_api_key_auth.py"
SPEC = importlib.util.spec_from_file_location("api_key_auth_patch", PATCH_PATH)
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

# Upstream middleware before PR #58028, with an unrelated helper to preserve.
AUTH_SOURCE = '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import secrets
from collections.abc import Awaitable

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

GUARDED_PREFIX = ("/v1", "/v2", "/inference", "/cohere")


class AuthenticationMiddleware:
    """Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    Notes
    -----
    There are two cases in which authentication is skipped:
        1. The HTTP method is OPTIONS.
        2. The request path doesn't start with GUARDED_PREFIX (e.g. /health).

    """

    def __init__(self, app: ASGIApp, tokens: list[str]) -> None:
        self.app = app
        self.api_tokens = [hashlib.sha256(t.encode("utf-8")).digest() for t in tokens]

    def verify_token(self, headers: Headers) -> bool:
        authorization_header_value = headers.get("Authorization")
        if not authorization_header_value:
            return False

        scheme, _, param = authorization_header_value.partition(" ")
        if scheme.lower() != "bearer":
            return False

        param_hash = hashlib.sha256(param.encode("utf-8")).digest()

        token_match = False
        for token_hash in self.api_tokens:
            token_match |= secrets.compare_digest(param_hash, token_hash)

        return token_match

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
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


def unrelated_route_helper():
    return "preserve me"
'''

CLI_SOURCE = '''class FrontendArgs:
    api_key: list[str] | None = None
    """If provided, the server will require one of these keys to be presented in
    the header.

    Warning: this only authenticates endpoints under the `/v1`, `/v2`, and
    `/inference` path prefixes. Other endpoints on the same server, including
    `/invocations` (which exposes the same inference capabilities as `/v1`),
    remain unauthenticated. Do not rely on `--api-key` alone to secure vLLM;
    see
    https://docs.vllm.ai/en/latest/usage/security.html#api-key-authentication-limitations
    for what it does and does not protect."""
    ssl_keyfile: str | None = None
'''

B12X_SOURCE = AUTH_SOURCE.replace(
    '    """Pure ASGI', '    """\n    Pure ASGI',
).replace('(e.g. /health).\n\n    """', '(e.g. /health).\n    """')


class Headers(dict):
    """The case-insensitive lookup contract used from Starlette Headers."""

    def __init__(self, *, scope):
        super().__init__((key.lower(), value) for key, value in scope["headers"].items())

    def get(self, key, default=None):
        return super().get(key.lower(), default)


class ApiKeyAuthPatchTests(unittest.TestCase):
    def middleware(self, source):
        tree = ast.parse(PATCHER.patch_authentication(source))
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        self.app = Mock(return_value="allowed")
        self.response = Mock(return_value="denied")
        self.response_factory = Mock(return_value=self.response)
        namespace = {
            "hashlib": hashlib, "secrets": secrets, "Awaitable": Awaitable,
            "ASGIApp": object, "Receive": object, "Scope": dict, "Send": object,
            "Headers": Headers, "JSONResponse": self.response_factory,
        }
        exec(compile(tree, "patched_authenticate.py", "exec"), namespace)
        return namespace["AuthenticationMiddleware"](self.app, ["first-key", "second-key"])

    def request(self, middleware, path, *, method="GET", headers=None, **extra):
        self.app.reset_mock()
        self.response_factory.reset_mock()
        scope = {"type": "http", "path": path, "method": method, "headers": headers or {}}
        scope.update(extra)
        return middleware(scope, None, None)

    def test_all_non_allowlisted_routes_require_valid_tokens(self):
        for source in (AUTH_SOURCE, B12X_SOURCE):
            middleware = self.middleware(source)
            for path in (
                "/v1/models", "/v1/models/", "/v2/test", "/inference", "/cohere",
                "/invocations", "/tokenize", "/detokenize", "/metrics", "/metrics/",
                "/docs", "/openapi.json", "/server_info", "/new-route",
                "/health/details", "/healthz", "/",
            ):
                for token in (None, "Bearer wrong", "Basic first-key"):
                    with self.subTest(path=path, token=token, b12x=source == B12X_SOURCE):
                        headers = {"Authorization": token} if token else {}
                        self.assertEqual(self.request(middleware, path, headers=headers), "denied")
                        self.app.assert_not_called()
                        self.response_factory.assert_called_once_with(
                            content={"error": "Unauthorized"}, status_code=401,
                        )
                for token in ("Bearer first-key", "bearer second-key"):
                    self.assertEqual(self.request(middleware, path, headers={
                        "Authorization": token,
                    }), "allowed")

    def test_liveness_routes_and_trailing_slashes_remain_public(self):
        for source in (AUTH_SOURCE, B12X_SOURCE):
            middleware = self.middleware(source)
            for path in ("/health", "/ping", "/load", "/version"):
                for suffix in ("", "/"):
                    with self.subTest(path=path + suffix):
                        self.assertEqual(self.request(middleware, path + suffix), "allowed")
                        self.response_factory.assert_not_called()

    def test_only_true_cors_preflights_bypass_authentication(self):
        for source in (AUTH_SOURCE, B12X_SOURCE):
            middleware = self.middleware(source)
            for headers, expected in (
                ({}, "denied"),
                ({"Origin": "https://dashboard.example"}, "denied"),
                ({"Access-Control-Request-Method": "GET"}, "denied"),
                ({"Origin": "https://dashboard.example", "Access-Control-Request-Method": "GET"}, "allowed"),
                ({"Authorization": "Bearer first-key"}, "allowed"),
            ):
                with self.subTest(headers=headers):
                    self.assertEqual(self.request(middleware, "/metrics", method="OPTIONS", headers=headers), expected)
            self.assertEqual(self.request(middleware, "/metrics", headers={
                "Origin": "https://dashboard.example", "Access-Control-Request-Method": "GET",
            }), "denied")

    def test_root_path_and_non_request_scopes_are_preserved(self):
        middleware = self.middleware(B12X_SOURCE)
        self.assertEqual(self.request(middleware, "/prefix/health/", root_path="/prefix"), "allowed")
        self.assertEqual(self.request(middleware, "/prefix/metrics", root_path="/prefix"), "denied")
        self.assertEqual(middleware({"type": "lifespan"}, None, None), "allowed")

    def test_both_layouts_preserve_other_code(self):
        for source in (AUTH_SOURCE, B12X_SOURCE):
            patched = PATCHER.patch_authentication(source)
            old_tree, new_tree = ast.parse(source), ast.parse(patched)
            def preserved(tree):
                cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
                return [ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name != "__call__"]
            self.assertEqual(preserved(old_tree), preserved(new_tree))
            self.assertEqual(ast.dump(old_tree.body[-1]), ast.dump(new_tree.body[-1]))
        patched_help = PATCHER.patch_cli_help(CLI_SOURCE)
        self.assertIn("This authenticates every endpoint", patched_help)

    def test_unexpected_source_is_rejected(self):
        for source in (
            AUTH_SOURCE.replace('scope.get("method")', 'scope.get("verb")'),
            AUTH_SOURCE.replace('GUARDED_PREFIX =', 'UNGUARDED_PATHS ='),
            AUTH_SOURCE.replace('"/cohere"', '"/extra"'),
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                PATCHER.patch_authentication(source)

    def test_entry_point_validates_both_files_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auth_path, cli_path = root / PATCHER.AUTH_REL, root / PATCHER.CLI_REL
            auth_path.parent.mkdir(parents=True)
            cli_path.parent.mkdir(parents=True)
            auth_path.write_text(B12X_SOURCE)
            cli_path.write_text("unexpected = True\n")
            result = subprocess.run([sys.executable, str(PATCH_PATH), str(root)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(auth_path.read_text(), B12X_SOURCE)
            self.assertEqual(cli_path.read_text(), "unexpected = True\n")
            for args in ([str(root)], []):
                auth_path.write_text(B12X_SOURCE)
                cli_path.write_text(CLI_SOURCE)
                result = subprocess.run([sys.executable, str(PATCH_PATH), *args], cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(auth_path.read_text(), PATCHER.patch_authentication(B12X_SOURCE))
                self.assertEqual(cli_path.read_text(), PATCHER.patch_cli_help(CLI_SOURCE))

    def test_absent_middleware_is_not_applicable(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(PATCH_PATH), directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("not applicable", result.stdout)

    def test_dockerfile_patches_before_wheel_compilation(self):
        build, runner = (PROJECT_DIR / "Dockerfile").read_text().split("FROM ${CUDA_IMAGE} AS runner\n", 1)
        command = f"RUN python3 /tmp/vllm-patches/{PATCH_PATH.name} .\n"
        self.assertIn(command, build)
        self.assertLess(build.index(command), build.index("uv build --no-build-isolation --wheel ."))
        self.assertNotIn(PATCH_PATH.name, runner)


if __name__ == "__main__":
    unittest.main()
