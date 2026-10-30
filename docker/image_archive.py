#!/usr/bin/env python3
"""Check that a Docker save archive contains its referenced config and layers.

The containerd image store can export metadata without the referenced blobs and
still report success. Check tar headers without extracting files or reading the
large layer payloads. Missing foreign-platform OCI descriptors are intentional;
manifest.json lists the runnable images actually included in the export.
"""

import json
import os
import sys
import tarfile


def validate(path):
    archive_size = os.path.getsize(path)
    with tarfile.open(path, mode="r:") as archive:
        members = {entry.name: entry for entry in archive.getmembers()}

        def require_file(name):
            seen = set()
            while isinstance(name, str) and name in members and name not in seen:
                seen.add(name)
                entry = members[name]
                if entry.islnk():
                    name = entry.linkname
                    continue
                if (entry.isfile() and entry.size > 0
                        and entry.offset_data + entry.size <= archive_size):
                    return entry
                break
            raise ValueError("a referenced config or layer file is missing or truncated")

        manifest = json.load(archive.extractfile(require_file("manifest.json")))
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("no runnable images in manifest.json")
        for image in manifest:
            if not isinstance(image, dict) or not isinstance(image.get("Layers"), list):
                raise ValueError("invalid image manifest")
            config = json.load(archive.extractfile(require_file(image.get("Config"))))
            if (not isinstance(config, dict) or not config.get("os")
                    or not config.get("architecture")):
                raise ValueError("missing image platform in saved config")
            rootfs = config.get("rootfs")
            if (not isinstance(rootfs, dict) or rootfs.get("type") != "layers"
                    or not isinstance(rootfs.get("diff_ids", []), (list, type(None)))):
                raise ValueError("invalid saved filesystem config")
            if len(rootfs.get("diff_ids") or []) != len(image["Layers"]):
                raise ValueError("saved config and layer list disagree")
            for layer in image["Layers"]:
                require_file(layer)


def main():
    try:
        validate(sys.argv[1])
    except (OSError, tarfile.TarError, ValueError, TypeError, KeyError):
        # Image configs can contain credentials; never echo archive contents.
        print("Error: Saved image archive is incomplete or invalid; "
              "a runnable image config and all referenced layers are required.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
