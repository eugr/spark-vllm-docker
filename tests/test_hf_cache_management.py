#!/usr/bin/env python3
"""Cache management on tiny real files, with mocked hf and SSH endpoints."""

import csv
import io
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MOCK = r'''
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
node = os.environ.get("TEST_NODE", "head")
state = Path(os.environ["TEST_STATE"])
mapping = json.loads(os.environ["TEST_CACHES"])
if tool == "python3":
    if node != "head" and "--cache-dir" in args:
        args[args.index("--cache-dir") + 1] = mapping[node]
    os.execv(sys.executable, [sys.executable, *args])
with (state / "calls.jsonl").open("a") as log:
    log.write(json.dumps({"tool": tool, "args": args, "node": node}) + "\n")
if tool == "ssh":
    target, command = args[-2:]
    peer = target.split("@", 1)[1]
    if peer == os.environ.get("TEST_FAIL_SSH"):
        sys.exit(1)
    if peer == os.environ.get("TEST_NEW_FILE_AFTER_REPAIR") and "repair_cache_ownership" in command:
        snapshot = Path(mapping[peer]) / "models--org--model/snapshots" / ("a" * 40)
        (snapshot / "newly-readable.safetensors").write_bytes(b"new shard")
    remote_env = dict(os.environ, TEST_NODE=peer)
    remote_env.update(json.loads(os.environ.get("TEST_REMOTE_ENVS", "{}")).get(peer, {}))
    sys.exit(subprocess.run(["bash", "-c", command], env=remote_env).returncode)
if tool in ("curl", "wget"):
    assert "https://astral.sh/uv/install.sh" in args, args
    destination = (args[args.index("--output") + 1] if tool == "curl" else
                   next(arg.split("=", 1)[1] for arg in args if arg.startswith("--output-document=")))
    Path(destination).write_text("""#!/bin/sh
set -e
[ "$UV_NO_MODIFY_PATH" = 1 ]
[ "$UV_INSTALL_DIR" = "$HOME/.local/bin" ]
printf 'Mock uv installer output\\n'
[ "${TEST_INSTALL_FAIL:-}" != 1 ]
mkdir -p "$UV_INSTALL_DIR"
cp "$TEST_UVX_SOURCE" "$UV_INSTALL_DIR/uvx"
chmod +x "$UV_INSTALL_DIR/uvx"
""")
    sys.exit(22 if os.environ.get("TEST_INSTALL_DOWNLOAD_FAIL") else 0)
if tool == "rsync":
    if os.environ.get("TEST_FAIL_RSYNC"):
        sys.exit(23)
    if ":" in args[-1] and "@" in args[-1]:
        peer, destination = args[-1].split(":", 1)
        peer = peer.split("@", 1)[1]
        if peer == os.environ.get("TEST_FAIL_COPY"):
            sys.exit(23)
        args[-1] = str(Path(mapping[peer]) / Path(destination.rstrip("/")).name) + "/"
    portable_root = os.environ.get("TEST_NO_SYMLINK_ROOT")
    if portable_root and Path(args[-1]).is_relative_to(portable_root):
        # Simulate a drive that rejects symlinks and Unix ownership/mode options,
        # while still using real rsync for the resulting file transfer.
        if "-a" in args or any(arg in args for arg in ("--perms", "--owner", "--group")):
            sys.exit("Destination does not support Unix file attributes")
        result = subprocess.run([os.environ["TEST_RSYNC"], *args])
        if any(path.is_symlink() for path in Path(args[-1]).rglob("*")):
            sys.exit("Destination does not support symlinks")
        sys.exit(result.returncode)
    os.execv(os.environ["TEST_RSYNC"], ["rsync", *args])
assert tool == "uvx", (tool, args)
if args[:2] == ["hf", "download"]:
    assert len(args) == 3, args
    (Path(mapping[node]) / ("models--" + args[2].replace("/", "--"))).mkdir(exist_ok=True)
    sys.exit(0)
assert args[:2] == ["hf", "cache"], args
cache = Path(args[args.index("--cache-dir") + 1])
if args[2] == "ls":
    assert args[args.index("--format") + 1] == "json", args
    assert "--revisions" in args and "--sort" in args, args
    entries = []
    for repo in sorted(cache.glob("*--*")):
        if node == os.environ.get("TEST_OMIT_REPO_NODE"):
            continue
        kind, *parts = repo.name.split("--")
        refs = {}
        for ref in (repo / "refs").rglob("*"):
            if ref.is_file():
                refs.setdefault(ref.read_text().strip(), []).append(ref.relative_to(repo / "refs").as_posix())
        for snapshot in sorted((repo / "snapshots").glob("*")):
            entries.append({"repo_type": kind[:-1], "repo_id": "/".join(parts),
                            "revision": snapshot.name, "refs": refs.get(snapshot.name, []),
                            "size": "1.0G", "snapshot_path": str(snapshot)})
    print(json.dumps(entries))
elif args[2] == "rm":
    assert "--yes" in args
    if node == os.environ.get("TEST_FAIL_DELETE"):
        sys.exit(1)
    targets = args[3:args.index("--cache-dir")]
    for target in targets:
        if target.startswith("model/"):
            repo = cache / ("models--" + target[6:].replace("/", "--"))
            if repo.exists():
                shutil.rmtree(repo)
        else:
            for snapshot in cache.glob("*/snapshots/" + target):
                repo = snapshot.parent.parent
                shutil.rmtree(snapshot)
                for ref in (repo / "refs").rglob("*"):
                    if ref.is_file() and ref.read_text().strip() == target:
                        ref.unlink()
                if not any((repo / "snapshots").iterdir()):
                    shutil.rmtree(repo)
else:
    sys.exit("Unexpected hf command")
'''


class CacheManagementTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="hf-management-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.cache = self.root / "home/cache with 'quotes' $(touch INJECTED)"
        self.hub = self.cache / "hub"
        self.peer1 = self.root / "peer1/hub"
        self.peer2 = self.root / "peer2/hub"
        self.backup = self.root / "SSD backup"
        for path in (self.hub, self.peer1, self.peer2, self.backup):
            path.mkdir(parents=True)
        self.config = self.root / "cluster.env"
        self.script = ROOT / "hf-download.sh"
        self.config.write_text("COPY_HOSTS=peer1,peer2\n")
        self.log = self.root / "calls.jsonl"
        self.log.touch()
        self.env = {"PATH": f"{self.bin}:{os.environ['PATH']}", "USER": "tester",
                    "HOME": str(self.root / "home"), "HF_HOME": str(self.cache),
                    "TEST_STATE": str(self.root), "TEST_RSYNC": shutil.which("rsync"),
                    "TEST_CACHES": json.dumps({"head": str(self.hub),
                                               "peer1": str(self.peer1), "peer2": str(self.peer2)})}
        for tool in ("uvx", "ssh", "rsync", "python3"):
            executable = self.bin / tool
            executable.write_text(f"#!{sys.executable}\n" + MOCK)
            executable.chmod(0o755)

    def model(self, cache=None, model="org/model", revision="a" * 40,
              data=b"weights", referenced=True, shared=False, kind="models"):
        cache = cache or self.hub
        repo = cache / (kind + "--" + model.replace("/", "--"))
        blob = repo / "blobs" / revision
        blob.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists():
            if shared:
                payload = cache / "blobs" / revision[:2] / revision
                payload.parent.mkdir(parents=True, exist_ok=True)
                payload.write_bytes(data)
                blob.symlink_to(os.path.relpath(payload, blob.parent))
            else:
                blob.write_bytes(data)
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        link = snapshot / "model.safetensors"
        if not link.exists():
            link.symlink_to("../../blobs/" + revision)
        if referenced:
            (repo / "refs").mkdir(exist_ok=True)
            (repo / "refs/main").write_text(revision)
        return repo

    def run_script(self, *args, success=True, input=""):
        result = subprocess.run(["bash", str(self.script), "--config",
                                 str(self.config), *args], cwd=self.root, env=self.env,
                                input=input, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        self.assertFalse((self.root / "INJECTED").exists())
        return result

    def calls(self, tool="uvx", action=None):
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [c for c in calls if c["tool"] == tool and (action is None or c["args"][2] == action)]

    def backup_without_symlinks(self, error="EPERM"):
        # Inject only the filesystem limitation; the real helper must choose
        # its fallback. Every cache and install remains in this test's tempdir.
        hook = self.root / "python-hooks"
        hook.mkdir()
        (hook / "sitecustomize.py").write_text('''
import errno
import os
from pathlib import Path

original_symlink = os.symlink
def limited_symlink(src, dst, *args, **kwargs):
    if Path(dst).is_relative_to(os.environ["TEST_NO_SYMLINK_ROOT"]):
        code = getattr(errno, os.environ["TEST_SYMLINK_ERRNO"])
        raise OSError(code, os.strerror(code), str(dst))
    return original_symlink(src, dst, *args, **kwargs)
os.symlink = limited_symlink
''')
        self.env["PYTHONPATH"] = str(hook)
        self.env["TEST_NO_SYMLINK_ROOT"] = str(self.backup)
        self.env["TEST_SYMLINK_ERRNO"] = error

    def test_portable_backup_delete_list_restore_and_distribute(self):
        self.backup_without_symlinks()
        first, second = "a" * 40, "b" * 40
        repo = self.model(shared=True)
        self.model(revision=second, referenced=False)
        self.model(self.peer1)
        # Both revisions share one payload, and the second has its own payload.
        (repo / "snapshots" / second / "common.safetensors").symlink_to("../../blobs/" + first)
        result = self.run_script("--backup", "org/model", "--backup-dir", str(self.backup),
                                 "--delete", "-c", "--force")
        self.assertIn("backing up regular snapshot files", result.stderr)
        self.assertFalse(repo.exists())
        saved = self.backup / repo.name
        self.assertFalse(any(path.is_symlink() for path in self.backup.rglob("*")))
        self.assertFalse((saved / "blobs").exists())
        self.assertEqual((saved / "refs/main").read_text(), first)
        for revision in (first, second):
            self.assertEqual((saved / "snapshots" / revision / "model.safetensors").read_bytes(), b"weights")
        shutil.rmtree(self.hub / "blobs")
        rows = json.loads(self.run_script("--list-backup", "--backup-dir", str(self.backup), "--format", "json").stdout)
        self.assertEqual({row["revision"] for row in rows}, {first, second})
        # Restore merges into an existing cache as well as copying to empty peers.
        self.model(revision="c" * 40)
        self.run_script("--restore", "org/model", "--backup-dir", str(self.backup), "-c", "--copy-parallel")
        for cache in (self.hub, self.peer1, self.peer2):
            self.assertEqual((cache / repo.name / "snapshots" / second / "common.safetensors").read_bytes(), b"weights")
            self.assertTrue((cache / repo.name / "snapshots" / ("c" * 40)).exists())

    def test_portable_backup_selected_revision_only(self):
        self.backup_without_symlinks("EOPNOTSUPP")
        first, second = "a" * 40, "b" * 40
        repo = self.model(shared=True)
        self.model(revision=second)
        snapshot = repo / "snapshots" / first
        (snapshot / "nested").mkdir()
        (snapshot / "nested/weights.safetensors").symlink_to("../../../blobs/" + first)
        (snapshot / "config.json").write_text('{}')
        metadata = repo / ".no_exist" / first
        metadata.mkdir(parents=True)
        (metadata / "optional.json").touch()
        self.run_script("--backup", "org/model", "--revision", first, "--backup-dir", str(self.backup),
                        "--delete", "--force")
        saved = self.backup / repo.name
        self.assertEqual([path.name for path in (saved / "snapshots").iterdir()], [first])
        self.assertFalse((saved / "blobs").exists())
        self.assertFalse((saved / "refs/main").exists())
        self.assertTrue((saved / ".no_exist" / first / "optional.json").is_file())
        self.assertFalse((repo / "snapshots" / first).exists())
        self.assertTrue((repo / "snapshots" / second).exists())
        shutil.rmtree(repo)
        shutil.rmtree(self.hub / "blobs")
        self.run_script("--restore", "org/model", "--backup-dir", str(self.backup))
        self.assertEqual((repo / "snapshots" / first / "nested/weights.safetensors").read_bytes(), b"weights")

    def test_failed_portable_backup_never_deletes_or_publishes(self):
        self.backup_without_symlinks()
        repo = self.model()
        self.env["TEST_FAIL_RSYNC"] = "1"
        self.run_script("--backup", "org/model", "--backup-dir", str(self.backup),
                        "--delete", "--force", success=False)
        self.assertTrue(repo.exists())
        self.assertFalse(any(self.backup.iterdir()))
        self.assertFalse(self.calls(action="rm"))

    def test_probe_io_error_aborts_instead_of_trying_portable_copy(self):
        self.backup_without_symlinks("EIO")
        repo = self.model()
        self.run_script("--backup", "org/model", "--backup-dir", str(self.backup),
                        "--delete", "--force", success=False)
        self.assertTrue(repo.exists())
        self.assertFalse(self.calls("rsync"))
        self.assertFalse(self.calls(action="rm"))
        self.assertFalse(any(self.backup.iterdir()))

    def remote_without_uvx(self, peer="peer1", downloader="curl", existing=None):
        # Isolate PATH completely so tests never discover or install real uv.
        remote_bin = self.root / peer / "bin"
        remote_bin.mkdir()
        remote_home = self.root / peer / "home with spaces"
        remote_home.mkdir()
        for tool in ("bash", "sh", "hostname", "mktemp", "rm", "mkdir", "cp", "chmod",
                     "find", "id", "realpath"):
            (remote_bin / tool).symlink_to(shutil.which(tool))
        (remote_bin / "python3").symlink_to(self.bin / "python3")
        if downloader:
            executable = remote_bin / downloader
            executable.write_text(f"#!{sys.executable}\n" + MOCK)
            executable.chmod(0o755)
        if existing:
            existing_uvx = remote_home / existing / "uvx"
            existing_uvx.parent.mkdir(parents=True)
            shutil.copyfile(self.bin / "uvx", existing_uvx)
            existing_uvx.chmod(0o755)
        remote_envs = json.loads(self.env.get("TEST_REMOTE_ENVS", "{}"))
        remote_envs[peer] = {"HOME": str(remote_home), "PATH": str(remote_bin),
                             "TEST_UVX_SOURCE": str(self.bin / "uvx")}
        self.env["TEST_REMOTE_ENVS"] = json.dumps(remote_envs)
        return remote_home

    def test_remote_uvx_outside_ssh_path_is_reused(self):
        for peer, location in (("peer1", ".local/bin"), ("peer2", ".cargo/bin")):
            self.remote_without_uvx(peer, existing=location)
            self.model(getattr(self, peer))
        rows = json.loads(self.run_script("--list", "-c", "--format", "json").stdout)
        self.assertEqual(rows[0]["nodes"], ["peer1", "peer2"])
        self.assertFalse(self.calls("curl"))
        self.assertFalse(self.calls("wget"))

    def test_missing_remote_uvx_installs_once_and_preserves_json_stdout(self):
        remote_home = self.remote_without_uvx()
        self.model(self.peer1)
        result = self.run_script("--list", "-c", "peer1", "--format", "json")
        self.assertEqual(json.loads(result.stdout)[0]["nodes"], ["peer1"])
        self.assertIn("Mock uv installer output", result.stderr)
        self.assertTrue((remote_home / ".local/bin/uvx").is_file())
        self.run_script("--list", "-c", "peer1", "--format", "csv")
        self.run_script("--delete", "org/model", "-c", "peer1", "--force")
        self.assertFalse((self.peer1 / "models--org--model").exists())
        self.assertEqual(len(self.calls("curl")), 1)

    def test_missing_remote_uvx_uses_wget_for_cleanup(self):
        self.remote_without_uvx(downloader="wget")
        self.model(self.peer1, referenced=False)
        self.run_script("--cleanup", "-c", "peer1", "--force")
        self.assertFalse((self.peer1 / "models--org--model").exists())
        self.assertEqual(len(self.calls("wget")), 1)
        self.assertFalse(self.calls("curl"))

    def test_uvx_download_failure_never_executes_partial_installer_or_deletes(self):
        remote_home = self.remote_without_uvx()
        self.model()
        self.model(self.peer1)
        self.env["TEST_INSTALL_DOWNLOAD_FAIL"] = "1"
        result = self.run_script("--delete", "org/model", "-c", "peer1", "--force", success=False)
        self.assertIn("Could not install uvx", result.stderr)
        self.assertNotIn("Mock uv installer output", result.stderr)
        self.assertFalse((remote_home / ".local/bin/uvx").exists())
        self.assertFalse(self.calls(action="rm"))

    def test_uvx_installer_failure_aborts_before_deletion(self):
        self.remote_without_uvx()
        self.model()
        self.model(self.peer1)
        self.env["TEST_INSTALL_FAIL"] = "1"
        result = self.run_script("--delete", "org/model", "-c", "peer1", "--force", success=False)
        self.assertIn("Could not install uvx", result.stderr)
        self.assertFalse(self.calls(action="rm"))

    def test_uvx_without_downloader_reports_actionable_error(self):
        self.remote_without_uvx(downloader=None)
        self.model(self.peer1)
        result = self.run_script("--list", "-c", "peer1", success=False)
        self.assertIn("Installing uv requires curl or wget", result.stderr)
        self.assertIn("Could not inventory tester@peer1", result.stderr)

    def test_cluster_inventory_remote_only_revisions_sort_and_formats(self):
        self.model(data=b"x" * 10)
        self.model(self.peer1, data=b"x" * 10)
        self.model(self.peer1, revision="b" * 40, data=b"x" * 20)
        self.model(self.peer2, model="other/remote", data=b"x" * 30)
        self.model(model="org/dataset", kind="datasets")
        result = self.run_script("--list", "-c", "--format", "json")
        rows = json.loads(result.stdout)
        self.assertEqual([r["size_gb"] for r in rows], [30e-9, 20e-9, 10e-9])
        self.assertEqual(rows[0]["model"], "other/remote")
        self.assertEqual(rows[0]["nodes"], ["peer2"])
        self.assertEqual(len(rows[2]["nodes"]), 2)
        for call in self.calls(action="ls"):
            self.assertEqual(call["args"][call["args"].index("--sort") + 1], "size:desc")
        local = json.loads(self.run_script("--list", "--format", "json").stdout)
        self.assertEqual(len(local), 1)
        self.assertNotIn("nodes", local[0])
        result = self.run_script("--list", "-c", "peer2,peer1", "--sort", "name:desc", "--format", "csv")
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        self.assertEqual(rows[0]["model"], "other/remote")
        self.assertIn("| Size (GB) |", self.run_script("--list", "--format", "md").stdout)
        self.assertIn("Size (GB)", self.run_script("--list").stdout)

    def test_empty_missing_cache_and_remote_failure(self):
        self.assertEqual(json.loads(self.run_script("--list", "--format", "json").stdout), [])
        shutil.rmtree(self.hub)
        self.model(self.peer1)
        rows = json.loads(self.run_script("--list", "-c", "--format", "json").stdout)
        self.assertEqual(rows[0]["nodes"], ["peer1"])
        self.env["TEST_FAIL_SSH"] = "peer2"
        self.run_script("--list", "-c", "--format", "json", success=False)
        self.run_script("--delete", "org/model", "-c", "--force", success=False)
        self.assertFalse(self.calls(action="rm"))

    def test_time_sort_uses_numeric_timestamps_across_nodes(self):
        first = self.model(model="org/first") / "blobs" / ("a" * 40)
        second = self.model(self.peer1, model="org/second") / "blobs" / ("a" * 40)
        os.utime(first, (100, 300))
        os.utime(second, (300, 100))
        for sort, expected in (("accessed", "org/second"), ("modified:desc", "org/first")):
            rows = json.loads(self.run_script("--list", "-c", "--sort", sort, "--format", "json").stdout)
            self.assertEqual(rows[0]["model"], expected)

    def test_discovery_uses_existing_download_enumeration(self):
        self.script = self.root / "hf-download.sh"
        shutil.copyfile(ROOT / "hf-download.sh", self.script)
        (self.root / "hf-cache.py").symlink_to(ROOT / "hf-cache.py")
        (self.root / "autodiscover.sh").write_text(
            f"source {shlex.quote(str(ROOT / 'autodiscover.sh'))}\n" + '''
DOTENV_COPY_HOSTS=""
detect_interfaces() { echo interfaces; }
detect_local_ip() { echo local-ip; }
detect_nodes() { echo nodes; }
detect_copy_hosts() { COPY_PEER_NODES=(peer2 peer1); }
''')
        self.model(self.peer1)
        self.model(self.peer2)
        result = self.run_script("--list", "-c", "--format", "json")
        self.assertEqual(json.loads(result.stdout)[0]["nodes"], ["peer2", "peer1"])
        self.assertIn("Autodiscovered copy hosts: peer2 peer1", result.stderr)

    def test_documented_download_argument_orders(self):
        self.model()
        for flags in (("-c", "org/model"), ("-c", "peer1,peer2", "org/model"),
                      ("org/model", "-c", "peer1", "--copy-parallel")):
            with self.subTest(flags=flags):
                self.run_script(*flags)
        downloads = [call for call in self.calls() if call["args"][:2] == ["hf", "download"]]
        self.assertEqual(len(downloads), 3)
        self.assertTrue(all(call["args"][-1] == "org/model" for call in downloads))

    def test_delete_confirmation_force_and_remote_only(self):
        repo = self.model(self.peer1)
        self.run_script("--delete", "org/model", "-c", input="no\n")
        self.assertTrue(repo.exists())
        self.assertFalse(self.calls(action="rm"))
        self.run_script("--delete", "org/model", "-c", "--force")
        self.assertFalse(repo.exists())
        self.assertEqual([c["node"] for c in self.calls(action="rm")], ["head", "peer1", "peer2"])

    def test_delete_yes_and_failure_propagation(self):
        local, peer = self.model(), self.model(self.peer2)
        self.env["TEST_FAIL_DELETE"] = "peer1"
        self.run_script("--delete", "org/model", "-c", input="yes\n", success=False)
        self.assertFalse(local.exists())
        self.assertFalse(peer.exists())

    def test_cleanup_preserves_refs_datasets_and_protected_hashes(self):
        for cache in (self.hub, self.peer1):
            self.model(cache)
            self.model(cache, revision="b" * 40, referenced=False)
            self.model(cache, model="other/dataset", kind="datasets", revision="c" * 40, referenced=False)
            self.model(cache, revision="c" * 40, referenced=False)
        self.run_script("--cleanup", "-c", input="n\n")
        self.assertFalse(self.calls(action="rm"))
        self.run_script("--cleanup", "-c", "--force")
        for cache in (self.hub, self.peer1):
            self.assertTrue((cache / ("models--org--model/snapshots/" + "a" * 40)).exists())
            self.assertFalse((cache / ("models--org--model/snapshots/" + "b" * 40)).exists())
            self.assertTrue((cache / ("models--org--model/snapshots/" + "c" * 40)).exists())
            self.assertTrue((cache / "datasets--other--dataset").exists())

    def test_backup_delete_list_restore_and_parallel_distribution(self):
        repo = self.model(shared=True)
        self.model(revision="b" * 40, referenced=False)
        self.model(self.peer1)
        self.run_script("--backup", "org/model", "--backup-dir", str(self.backup), "--delete", "-c", "--force")
        self.assertFalse(repo.exists())
        backup_repo = self.backup / repo.name
        self.assertFalse((backup_repo / ("blobs/" + "a" * 40)).is_symlink())
        self.assertTrue((backup_repo / ("snapshots/" + "a" * 40 + "/model.safetensors")).is_symlink())
        shutil.rmtree(self.hub / "blobs")
        listed = self.run_script("--list-backup", "--backup-dir", str(self.backup), "--format", "json")
        self.assertEqual(len(json.loads(listed.stdout)), 2)
        self.assertEqual(len(json.loads(self.run_script("--list-backup", "org/model", "--backup-dir", str(self.backup), "--format", "json").stdout)), 2)
        self.run_script("--restore", "org/model", "--backup-dir", str(self.backup), "-c", "--copy-parallel")
        for cache in (self.hub, self.peer1, self.peer2):
            for revision in ("a" * 40, "b" * 40):
                self.assertEqual((cache / repo.name / "snapshots" / revision / "model.safetensors").read_bytes(), b"weights")
        self.model(revision="c" * 40, referenced=False)
        self.run_script("--restore", "org/model", "--backup-dir", str(self.backup))
        self.assertTrue((repo / "snapshots" / ("c" * 40)).exists())

    def test_backup_selected_revision_preserves_links_and_only_needed_blobs(self):
        first, second = "a" * 40, "b" * 40
        repo = self.model(shared=True)
        self.model(revision=second)
        tag = repo / "refs/releases/stable"
        tag.parent.mkdir()
        tag.write_text(first)
        snapshot = repo / "snapshots" / first
        (snapshot / "nested").mkdir()
        (snapshot / "nested/weights.safetensors").symlink_to("../../../blobs/" + first)
        (snapshot / "config.json").write_text('{}')
        for revision in (first, second):
            metadata = repo / ".no_exist" / revision
            metadata.mkdir(parents=True)
            (metadata / "optional.json").touch()
        (repo / "trees").mkdir()
        (repo / "trees" / f"{first}.json").write_text('{}')
        self.run_script("--backup", "org/model", "--revision", "releases/stable", "--backup-dir", str(self.backup))
        saved = self.backup / repo.name
        self.assertEqual([p.name for p in (saved / "snapshots").iterdir()], [first])
        self.assertEqual([p.name for p in (saved / "blobs").iterdir()], [first])
        self.assertFalse((saved / "blobs" / first).is_symlink())
        self.assertTrue((saved / "snapshots" / first / "nested/weights.safetensors").is_symlink())
        self.assertEqual((saved / "refs/releases/stable").read_text(), first)
        self.assertFalse((saved / "refs/main").exists())
        self.assertFalse((saved / ".no_exist" / second).exists())
        self.assertTrue((saved / "trees" / f"{first}.json").exists())
        shutil.rmtree(repo)
        shutil.rmtree(self.hub / "blobs")
        self.run_script("--restore", "org/model", "--backup-dir", str(self.backup))
        self.assertEqual((repo / "snapshots" / first / "nested/weights.safetensors").read_bytes(), b"weights")

    def test_backup_detached_revision_by_unique_prefix(self):
        self.model(referenced=False)
        self.model(revision="b" * 40)
        self.run_script("--backup", "org/model", "--revision", "aaaaaaa", "--backup-dir", str(self.backup))
        rows = json.loads(self.run_script("--list-backup", "--backup-dir", str(self.backup), "--format", "json").stdout)
        self.assertEqual([row["revision"] for row in rows], ["a" * 40])

    def test_delete_selected_revision_on_cluster_retains_others(self):
        first, second = "a" * 40, "b" * 40
        for cache in (self.hub, self.peer1, self.peer2):
            self.model(cache, referenced=False)
            self.model(cache, revision=second)
        self.run_script("--delete", "org/model", "--revision", first, "-c", input="no\n")
        self.assertFalse(self.calls(action="rm"))
        result = self.run_script("--delete", "org/model", "--revision", first, "-c", input="yes\n")
        self.assertIn(f"Delete revision '{first}'", result.stderr)
        for cache in (self.hub, self.peer1, self.peer2):
            repo = cache / "models--org--model"
            self.assertFalse((repo / "snapshots" / first).exists())
            self.assertEqual((repo / "snapshots" / second / "model.safetensors").read_bytes(), b"weights")
            self.assertEqual((repo / "refs/main").read_text(), second)
        for call in self.calls(action="rm"):
            self.assertEqual(call["args"][3], first)

    def test_delete_revision_present_only_on_peer(self):
        self.model(revision="b" * 40)
        self.model(self.peer1)
        self.run_script("--delete", "org/model", "--revision", "aaaaaaa", "-c", "--force")
        self.assertEqual([call["node"] for call in self.calls(action="rm")], ["peer1"])
        self.assertTrue((self.hub / "models--org--model").exists())

    def test_delete_branch_resolves_cached_commit(self):
        repo = self.model()
        self.model(revision="b" * 40, referenced=False)
        self.run_script("--delete", "org/model", "--revision", "main", "--force")
        self.assertFalse((repo / "snapshots" / ("a" * 40)).exists())
        self.assertTrue((repo / "snapshots" / ("b" * 40)).exists())

    def test_unknown_and_ambiguous_revisions_never_delete(self):
        self.model()
        self.model(self.peer1, revision="b" * 40)
        self.run_script("--delete", "org/model", "--revision", "main", "-c", "--force", success=False)
        self.run_script("--delete", "org/model", "--revision", "missing", "-c", "--force", success=False)
        self.model(revision="a" * 39 + "b", referenced=False)
        self.run_script("--delete", "org/model", "--revision", "aaaaaaa", "--force", success=False)
        self.assertFalse(self.calls(action="rm"))

    def test_revision_hash_collision_aborts_entire_cluster_delete(self):
        self.model()
        self.model(self.peer1)
        self.model(self.peer1, model="other/dataset", kind="datasets")
        self.run_script("--delete", "org/model", "--revision", "a" * 40, "-c", "--force", success=False)
        self.assertFalse(self.calls(action="rm"))

    def test_backup_delete_revision_pins_heads_commit_and_preserves_peer_refs(self):
        first, second = "a" * 40, "b" * 40
        self.model()
        self.model(revision=second, referenced=False)
        self.model(self.peer1, referenced=False)
        self.model(self.peer1, revision=second)
        self.model(self.peer2, revision=second)
        self.run_script("--backup", "org/model", "--revision", "main", "--backup-dir", str(self.backup),
                        "--delete", "-c", "--force")
        for cache in (self.hub, self.peer1, self.peer2):
            repo = cache / "models--org--model"
            self.assertFalse((repo / "snapshots" / first).exists())
            self.assertTrue((repo / "snapshots" / second).exists())
        self.assertEqual((self.peer1 / "models--org--model/refs/main").read_text(), second)
        self.assertEqual([call["node"] for call in self.calls(action="rm")], ["head", "peer1"])
        saved = self.backup / "models--org--model"
        self.assertEqual([p.name for p in (saved / "snapshots").iterdir()], [first])

    def test_missing_backup_revision_does_not_publish_or_delete(self):
        self.model()
        self.run_script("--backup", "org/model", "--revision", "missing", "--backup-dir", str(self.backup),
                        "--delete", "--force", success=False)
        self.assertFalse(any(self.backup.iterdir()))
        self.assertFalse(self.calls(action="rm"))

    def test_failed_backup_never_deletes_or_publishes(self):
        repo = self.model()
        self.env["TEST_FAIL_RSYNC"] = "1"
        self.run_script("--backup", "org/model", "--backup-dir", str(self.backup), "--delete", "--force", success=False)
        self.assertTrue(repo.exists())
        self.assertFalse(any(self.backup.iterdir()))
        self.assertFalse(self.calls(action="rm"))

    def test_backup_refuses_deleting_remote_revision_not_backed_up(self):
        repo = self.model()
        self.model(self.peer1, revision="b" * 40)
        self.run_script("--backup", "org/model", "--backup-dir", str(self.backup), "--delete", "-c", "--force", success=False)
        self.assertTrue(repo.exists())
        self.assertFalse(self.calls(action="rm"))
        self.assertTrue((self.backup / repo.name).exists())

    def test_backup_delete_decline_keeps_backup_and_cache(self):
        repo = self.model()
        self.run_script("--backup", "org/model", "--backup-dir", str(self.backup), "--delete", "-c", input="no\n")
        self.assertTrue(repo.exists())
        self.assertTrue((self.backup / repo.name).exists())
        self.assertFalse(self.calls(action="rm"))

    def test_backup_refuses_deleting_extra_peer_files_in_same_revision(self):
        self.model()
        repo = self.model(self.peer1)
        (repo / "snapshots" / ("a" * 40) / "extra-shard.safetensors").write_bytes(b"extra")
        self.run_script("--backup", "org/model", "--backup-dir", str(self.backup), "--delete", "-c", "--force", success=False)
        self.assertFalse(self.calls(action="rm"))

    def test_backup_delete_rechecks_after_remote_ownership_preparation(self):
        self.model()
        self.model(self.peer1)
        self.env["TEST_NEW_FILE_AFTER_REPAIR"] = "peer1"
        self.run_script("--backup", "org/model", "--backup-dir", str(self.backup), "--delete", "-c", "--force", success=False)
        self.assertFalse(self.calls(action="rm"))

    def test_backup_delete_refuses_repositories_omitted_by_hf_scan(self):
        self.model()
        self.model(self.peer1)
        self.env["TEST_OMIT_REPO_NODE"] = "peer1"
        self.run_script("--backup", "org/model", "--backup-dir", str(self.backup), "--delete", "-c", "--force", success=False)
        self.assertFalse(self.calls(action="rm"))

    def test_existing_backup_and_unsafe_paths_are_preserved(self):
        repo = self.model()
        self.run_script("--backup", "org/model", "--backup-dir", str(self.backup))
        self.run_script("--backup", "org/model", "--backup-dir", str(self.backup), "--delete", "--force", success=False)
        self.assertTrue(repo.exists())
        self.run_script("--backup", "org/model", "--backup-dir", str(self.hub), success=False)
        self.run_script("--backup", "../model", "--backup-dir", str(self.backup), success=False)
        self.assertFalse(self.calls(action="rm"))

    def test_broken_backup_and_unsafe_restore_symlink_fail(self):
        repo = self.model(shared=True)
        (repo / "blobs" / ("a" * 40)).resolve().unlink()
        self.run_script("--backup", "org/model", "--backup-dir", str(self.backup), "--delete", "--force", success=False)
        self.assertFalse(self.calls(action="rm"))
        shutil.rmtree(repo)
        backup_repo = self.model(self.backup)
        repo.symlink_to(backup_repo, target_is_directory=True)
        self.run_script("--restore", "org/model", "--backup-dir", str(self.backup), success=False)

    def test_restore_copy_failure_returns_failure(self):
        self.model(self.backup)
        self.env["TEST_FAIL_COPY"] = "peer1"
        self.run_script("--restore", "org/model", "--backup-dir", str(self.backup), "-c", "--copy-parallel", success=False)
        self.assertTrue((self.peer2 / "models--org--model").exists())

    def test_invalid_flags_fail_before_cluster_or_cache_access(self):
        for args in (("--format",), ("--list", "--sort", "bogus"), ("--backup", "org/model"),
                     ("--restore", "org/model", "--delete"), ("--list", "--cleanup"),
                     ("--list", "--format", "yaml"), ("--typo",), ("--delete",)):
            with self.subTest(args=args):
                self.run_script(*args, success=False)
        self.assertFalse(self.calls())
        self.assertFalse(self.calls("ssh"))

    def test_invalid_revision_flags_fail_before_cache_access(self):
        for args in (("--delete", "org/model", "--revision"),
                     ("--delete", "org/model", "--revision", "../main"),
                     ("--list", "--revision", "main"),
                     ("--cleanup", "--revision", "main"),
                     ("org/model", "--revision", "main")):
            with self.subTest(args=args):
                self.run_script(*args, success=False)
        self.assertFalse(self.calls())
        self.assertFalse(self.calls("ssh"))


if __name__ == "__main__":
    unittest.main()
