"""Fetch only the pinned public preview's vLLM installation layer.

This does not import the donor OS, Torch, CUDA, NCCL or private local artifacts.
All network bytes are checked against the immutable OCI layer digest before
extraction. Install RECORD hashes are checked before the output can be used.
"""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

REPOSITORY = 'vllm/vllm-openai'
MANIFEST = 'sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e'
LAYER = 'sha256:bfecc5abad27302cf3d93f03f9fdc5b57a90eb6f2da5f42c46fb1b070a323ef5'
VERSION = '0.1.dev20073+g8e685d198'
SITE = PurePosixPath('usr/local/lib/python3.12/dist-packages')
DIST = 'vllm-' + VERSION + '.dist-info'


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120)


def allowed(path):
    return (path == PurePosixPath('usr/local/bin/vllm') or
            path.is_relative_to(SITE / 'vllm') or path.is_relative_to(SITE / DIST))


def fetch(output):
    if output.exists():
        raise FileExistsError(output)
    with request('https://auth.docker.io/token?service=registry.docker.io&scope=repository:' + REPOSITORY + ':pull') as response:
        token = json.load(response)['token']
    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json'}
    root = 'https://registry-1.docker.io/v2/' + REPOSITORY
    with request(root + '/manifests/' + MANIFEST, headers) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != MANIFEST.split(':')[1]:
        raise ValueError('donor manifest checksum mismatch')
    manifest = json.loads(raw)
    entry = next(x for x in manifest['layers'] if x['digest'] == LAYER)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='qwen38-preview-', dir=output.parent) as tmp:
        tmp = Path(tmp)
        archive = tmp / 'layer.tar.gz'
        digest, count = hashlib.sha256(), 0
        with request(root + '/blobs/' + LAYER, headers) as response, archive.open('wb') as dest:
            while data := response.read(8 * 1024 * 1024):
                digest.update(data)
                count += len(data)
                dest.write(data)
        if count != entry['size'] or digest.hexdigest() != LAYER.split(':')[1]:
            raise ValueError('donor layer size/checksum mismatch')
        stage = tmp / 'root'
        stage.mkdir()
        extracted = 0
        with tarfile.open(archive, 'r:gz') as archive_file:
            for member in archive_file:
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('unsafe archive path')
                if not allowed(path):
                    continue
                destination = stage / path
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive_file.extractfile(member) as source, destination.open('wb') as target:
                        shutil.copyfileobj(source, target)
                    destination.chmod(member.mode & 0o777)
                    extracted += 1
                else:
                    raise ValueError('unexpected link/device in vLLM artifact: ' + str(path))
        verify_record(stage)
        provenance = {'source_manifest': MANIFEST, 'source_layer': LAYER,
                      'vllm_version': VERSION, 'compressed_bytes': count,
                      'extracted_files': extracted,
                      'scope': 'vllm distribution only; not donor OS/dependencies'}
        (stage / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        stage.rename(output)
    print(json.dumps(provenance), flush=True)


def verify_record(root):
    site = root / SITE
    record = site / DIST / 'RECORD'
    if not record.is_file() or not (site / 'vllm/models/qwen3_8_flash_next/nvidia/model.py').is_file():
        raise ValueError('missing expected preview model or install RECORD')
    with record.open(newline='') as handle:
        for name, hash_value, size in csv.reader(handle):
            target = (site / name).resolve()
            if not target.is_relative_to(root.resolve()) or not allowed(PurePosixPath(target.relative_to(root.resolve()))):
                raise ValueError('unaccounted vLLM-owned file: ' + name)
            if not target.is_file():
                # Interpreter bytecode is generated during install, not needed
                # for a clean reinstallation. All hashed files are mandatory.
                if not hash_value and name.endswith('.pyc'):
                    continue
                raise ValueError('missing installed file: ' + name)
            if size and target.stat().st_size != int(size):
                raise ValueError('RECORD size mismatch: ' + name)
            if hash_value:
                algorithm, expected = hash_value.split('=', 1)
                h = hashlib.new(algorithm)
                with target.open('rb') as stream:
                    while block := stream.read(8 * 1024 * 1024):
                        h.update(block)
                actual = base64.urlsafe_b64encode(h.digest()).rstrip(b'=').decode()
                if actual != expected:
                    raise ValueError('RECORD checksum mismatch: ' + name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    fetch(parser.parse_args().output.resolve())
