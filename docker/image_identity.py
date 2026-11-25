#!/usr/bin/env python3
"""Fingerprint runnable image content from one `docker image inspect` result.

Classic Docker IDs hash the image config; containerd IDs can instead identify a
manifest or index. RepoDigests can also change or disappear during save/load.
Compare the platform, uncompressed layer digests, and runtime config instead.
This is an equivalence check, not a registry digest or an image signature.
"""

import hashlib
import json
import re
import sys


def omit_empty_fields(config):
    # Docker API versions differ in whether unset config fields are serialized.
    # Only normalize the field level: empty values inside Labels, Volumes, and
    # ExposedPorts carry meaning, and array order (e.g. Env or Cmd) matters.
    return {
        key: value for key, value in config.items()
        if value not in (None, "", False, 0, [], {})
    }


def fingerprint(result):
    if not isinstance(result, list) or len(result) != 1:
        raise ValueError("expected one inspected image")
    image = result[0]
    if not isinstance(image, dict):
        raise ValueError("invalid image metadata")
    for field in ("Os", "Architecture"):
        if not isinstance(image.get(field), str) or not image[field]:
            raise ValueError("missing image platform")
    config = image.get("Config")
    rootfs = image.get("RootFS")
    if not isinstance(config, dict) or not isinstance(rootfs, dict):
        raise ValueError("missing image config or filesystem")
    layers = rootfs.get("Layers")
    if layers is None:
        layers = []
    if rootfs.get("Type") != "layers" or not isinstance(layers, list):
        raise ValueError("invalid image filesystem")
    if any(not isinstance(layer, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", layer)
           for layer in layers):
        raise ValueError("invalid image layer digest")

    config = omit_empty_fields(config)
    if isinstance(config.get("Healthcheck"), dict):
        config["Healthcheck"] = omit_empty_fields(config["Healthcheck"])
        config = omit_empty_fields(config)
    content = {
        "Os": image["Os"],
        "Architecture": image["Architecture"],
        "Variant": image.get("Variant") or "",
        "OsVersion": image.get("OsVersion") or "",
        "RootFS": {"Type": "layers", "Layers": layers},
        "Config": config,
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main():
    try:
        print(fingerprint(json.load(sys.stdin)))
    except (ValueError, TypeError):
        # Never echo inspect data: image configs can contain sensitive values.
        print("Could not fingerprint Docker image metadata.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
