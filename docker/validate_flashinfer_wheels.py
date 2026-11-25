#!/usr/bin/env python3
"""Validate local providers required by a FlashInfer JIT-cache shim.

Uses only the standard library so host-side validation needs no package installs.
Older monolithic wheels declare no provider dependencies and remain supported.
"""

import argparse
from email.parser import BytesParser
from pathlib import Path
import re
import sys
from zipfile import BadZipFile, ZipFile


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_metadata(wheel: Path):
    try:
        with ZipFile(wheel) as archive:
            entries = [
                name for name in archive.namelist()
                if re.fullmatch(r"[^/]+\.dist-info/METADATA", name)
            ]
            if len(entries) != 1:
                raise ValueError("expected exactly one .dist-info/METADATA entry")
            metadata = BytesParser().parsebytes(archive.read(entries[0]))
    except (OSError, BadZipFile, ValueError) as exc:
        raise ValueError(f"Cannot read wheel metadata from {wheel.name}: {exc}") from exc
    if not metadata.get("Name") or not metadata.get("Version"):
        raise ValueError(f"Missing Name or Version in {wheel.name}")
    return metadata


def provider_tag(architecture: str) -> str:
    # Match FlashInfer's shim naming, retaining the architecture feature suffix.
    normalized = re.sub(r"^(compute_|sm_|sm)", "", architecture.lower())
    normalized = normalized.replace(".", "").replace("_", "")
    if not re.fullmatch(r"\d{2,3}[af]?", normalized):
        raise ValueError(f"Invalid GPU architecture: {architecture!r}")
    return f"sm{normalized}"


def validate_providers(jit_wheel: Path, architectures: str) -> list[Path]:
    metadata = read_metadata(jit_wheel)
    if normalize_name(metadata["Name"]) != "flashinfer-jit-cache":
        raise ValueError(f"Expected flashinfer-jit-cache metadata in {jit_wheel.name}")

    requirements = {}
    for requirement in metadata.get_all("Requires-Dist", []):
        name_match = re.match(r"[\w.-]+", requirement)
        if not name_match:
            continue
        name = normalize_name(name_match[0])
        if not name.startswith("flashinfer-jit-cache-sm"):
            continue
        # Upstream generates unconditional, exact pins for every provider.
        # Fail explicitly if that format changes instead of skipping a dependency.
        pin = re.fullmatch(
            r"[\w.-]+\s*(?:==\s*([\w.+!-]+)|\(\s*==\s*([\w.+!-]+)\s*\))\s*",
            requirement,
        )
        if pin is None:
            raise ValueError(f"Unsupported FlashInfer provider requirement: {requirement}")
        version = pin[1] or pin[2]
        if name in requirements and requirements[name] != version:
            raise ValueError(f"Conflicting FlashInfer provider requirements for {name}")
        requirements[name] = version

    if not requirements:
        return []

    for architecture in architectures.split():
        name = f"flashinfer-jit-cache-{provider_tag(architecture)}"
        if name not in requirements:
            raise ValueError(
                f"FlashInfer JIT-cache shim {jit_wheel.name} does not declare "
                f"the provider for GPU arch {architecture} ({name})"
            )

    # Check every declared dependency, including additional architectures in a
    # multi-architecture shim: the installer requires all of them, not just one.
    providers = []
    for name, version in requirements.items():
        pattern = f"{name.replace('-', '_')}-*.whl"
        wheels = sorted(jit_wheel.parent.glob(pattern))
        arch = name.removeprefix("flashinfer-jit-cache-")
        if not wheels:
            raise ValueError(
                f"FlashInfer JIT-cache provider wheel for arch {arch} "
                f"({pattern}) is missing; required by {jit_wheel.name}: {name}=={version}"
            )
        if len(wheels) != 1:
            raise ValueError(f"Expected exactly one FlashInfer provider wheel for {name}: {pattern}")
        provider = read_metadata(wheels[0])
        actual_version = provider["Version"]
        # Literal upstream pins share the provider's version. A public-version
        # pin also permits a local build label (e.g. ==0.7.0 accepts 0.7.0+cu134).
        matches_version = actual_version == version or (
            "+" not in version and actual_version.split("+", 1)[0] == version
        )
        if normalize_name(provider["Name"]) != name or not matches_version:
            raise ValueError(
                f"FlashInfer provider {wheels[0].name} does not satisfy {name}=={version}; "
                f"wheel metadata declares {provider['Name']}=={actual_version}"
            )
        providers.append(wheels[0])
    return providers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jit_wheel", type=Path)
    parser.add_argument("architectures", nargs="?", default="12.1a")
    parser.add_argument("--list-provider-wheels", action="store_true",
                        help="Print validated provider paths for release publication")
    args = parser.parse_args()
    try:
        providers = validate_providers(args.jit_wheel, args.architectures)
    except ValueError as exc:
        print(f"Error: {exc}.", file=sys.stderr)
        print(
            "       Re-run with --rebuild-flashinfer to regenerate a complete "
            f"FlashInfer wheel set for GPU arch {args.architectures}.",
            file=sys.stderr,
        )
        return 1
    if args.list_provider_wheels:
        for provider in providers:
            print(provider)
    return 0


if __name__ == "__main__":
    sys.exit(main())
