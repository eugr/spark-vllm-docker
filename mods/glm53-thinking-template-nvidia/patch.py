#!/usr/bin/env python3
"""Generate an external GLM-5.3 template with a thinking-off prefix.

This variant targets the 257-line template in nvidia/GLM-5.3-Flash-NVFP4,
rather than the 261-line zai-org template. Both variants make the same
replacement at a single exact anchor. Their input/output hashes and output
paths differ so they can coexist.

The checkpoint is never modified. Default/on rendering is preserved. Explicit
off uses the GLM-4.7 closing-only prefix; GLM-5.3 does not document this mode.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

SOURCE_SHA256 = "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891"
OUTPUT_SHA256 = "d55b07445fea7fcde691dcf6b1b4e006234695dc4a319fc0206201bb1c3a077e"
DEFAULT_OUTPUT = Path("/tmp/glm53-nvidia-chat-template.jinja")
OLD = """{%- if add_generation_prompt -%}
    <|assistant|>{{- '<think>' -}}
{%- endif -%}"""
NEW = """{%- if add_generation_prompt -%}
    {%- set requested_thinking = thinking if thinking is defined else none -%}
    {%- set requested_enable_thinking = enable_thinking if enable_thinking is defined else none -%}
    {%- set effective_thinking = true if requested_thinking is none and requested_enable_thinking is none else requested_thinking or requested_enable_thinking -%}
    <|assistant|>{{- '<think>' if effective_thinking else '</think>' -}}
{%- endif -%}"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_template(source: bytes) -> bytes:
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Unknown source template hash")
    text = source.decode()
    if text.count(OLD) != 1:
        raise ValueError("Expected exactly one generation-prefix anchor")
    result = text.replace(OLD, NEW, 1).encode()
    if digest(result) != OUTPUT_SHA256:
        raise ValueError("Unexpected generated template hash")
    return result


def generate(source: Path, output: Path, check: bool = False) -> dict:
    source, output = Path(source), Path(output)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Source must be a regular non-symlink file")
    original = source.read_bytes()
    generated = build_template(original)
    if source.resolve() == output.resolve():
        raise ValueError("Output must not overwrite the source template")
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise ValueError("Output must be a regular non-symlink file")
    exists = output.exists()
    if exists and digest(output.read_bytes()) != OUTPUT_SHA256:
        raise ValueError("Output already exists with an unknown hash")
    report = {"status": "already_generated" if exists else "ready",
              "source": str(source), "output": str(output),
              "source_sha256": SOURCE_SHA256, "output_sha256": OUTPUT_SHA256,
              "experimental_glm53_thinking_off": True}
    if check or exists:
        return report
    fd, temporary = tempfile.mkstemp(prefix=".glm53-template-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(generated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        if source.is_symlink() or source.read_bytes() != original:
            raise ValueError("Source changed while generating template")
        if output.exists() or output.is_symlink():
            raise ValueError("Output appeared while generating template")
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    report["status"] = "generated"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).with_name("chat_template.base.jinja"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        report = generate(args.source, args.output, args.check)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
