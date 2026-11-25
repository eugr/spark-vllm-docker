#!/usr/bin/env python3
"""Exercise cache repair with mocked downloads, SSH, Docker, and sudo."""

import json
import os
from pathlib import Path
import pty
import select
import shutil
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
MOCK_TOOL = r'''#!/usr/bin/env python3
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
node = os.environ.get("TEST_NODE", "head")
state = Path(os.environ["TEST_STATE"])
settings = json.loads((state / "settings.json").read_text())
node_uid = str(settings.get("uids", {}).get(node, os.getuid()))
record = {"tool": tool, "args": args, "node": node, "tty": os.isatty(0)}
with (state / "calls.jsonl").open("a") as log:
    log.write(json.dumps(record) + "\n")

def fixed_marker(path):
    return state / (node + hashlib.sha256(path.encode()).hexdigest())

if tool == "find":
    assert args[1:] == ["!", "-uid", node_uid, "-print", "-quit"], args
    problem = settings.get("problems", {}).get(node, {}).get(args[0])
    if problem and not fixed_marker(args[0]).exists():
        if problem == "unreadable":
            sys.exit(1)
        print(args[0] + "/root-owned-file")
    else:
        # Fixtures belong to the test runner; nodes may have different mocked UIDs.
        os.execv(os.environ["TEST_REAL_FIND"],
                 ["find", args[0], "!", "-uid", str(os.getuid()), "-print", "-quit"])
elif tool == "id":
    if args == ["-un"]:
        print("local-owner" if node == "head" else "remote-owner")
    elif args == ["-u"]:
        print(node_uid)
    else:
        os.execv(os.environ["TEST_REAL_ID"], ["id", *args])
elif tool == "sudo":
    if settings.get("password"):
        if "-n" in args or not os.isatty(0):
            sys.exit("sudo: a password is required")
        print("TEST sudo password on " + node + ":", flush=True)
        assert input() == "test-password"
    if node in settings.get("sudo_fail", []):
        sys.exit(1)
    command = args[1:] if args[0] == "-n" else args
    assert command[:4] == ["chown", "-R", "-h", "--"], command
    if not settings.get("ineffective_repair"):
        fixed_marker(command[-1]).touch()
elif tool == "docker":
    # Cache repair must target this node even if a remote context is configured.
    assert args[:2] == ["--host", "unix:///var/run/docker.sock"], args
    args = args[2:]
    images = settings.get("docker_images", {}).get(node, [])
    if args[:2] == ["image", "inspect"]:
        assert len(args) == 3, args
        sys.exit(0 if args[2] in images and node not in settings.get("docker_unavailable", []) else 1)
    assert args[0] == "run", args
    assert "--rm" in args and "--pull=never" in args, args
    assert args[args.index("--user") + 1] == "0", args
    assert args[args.index("--entrypoint") + 1] == "chown", args
    image = args[args.index("--entrypoint") + 2]
    assert image in images, image
    assert args[-5:] == ["-R", "-h", "--", node_uid, "/hf-cache"], args
    mount = args[args.index("--mount") + 1]
    fields = dict(field.split("=", 1) for field in next(csv.reader([mount])))
    assert fields["type"] == "bind" and fields["dst"] == "/hf-cache", fields
    assert Path(fields["src"]).is_dir(), fields
    if image in settings.get("docker_fail", {}).get(node, []):
        sys.exit(1)
    if node not in settings.get("docker_ineffective", []):
        fixed_marker(fields["src"]).touch()
elif tool == "uvx":
    assert args == ["hf", "download", "org/model"], args
    Path(os.environ["TEST_HUB"], "models--org--model").mkdir(parents=True, exist_ok=True)
elif tool == "ssh":
    assert args[:2] == ["-o", "BatchMode=yes"], args
    tty_flag, target, command = args[2:]
    assert tty_flag in ("-t", "-nT"), args
    if tty_flag == "-t":
        assert os.isatty(0)
    peer = target.split("@", 1)[1]
    env = dict(os.environ, TEST_NODE=peer)
    kwargs = {} if tty_flag == "-t" else {"stdin": subprocess.DEVNULL}
    sys.exit(subprocess.run(["bash", "-c", command], env=env, **kwargs).returncode)
elif tool == "rsync":
    assert "-s" in args, args
else:
    sys.exit("Unexpected tool: " + tool)
'''


class CachePermissionsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="hf-permissions-")
        self.addCleanup(temporary.cleanup)
        self.fixture = Path(temporary.name)
        self.bin = self.fixture / "bin"
        self.bin.mkdir()
        self.cache = self.fixture / "home/.cache/huggingface"
        self.hub = self.cache / "hub"
        self.hub.mkdir(parents=True)
        self.config = self.fixture / "test.env"
        self.config.touch()
        self.log = self.fixture / "calls.jsonl"
        self.settings = {}
        self.env = {
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "HOME": str(self.fixture / "home"),
            "USER": "not-the-actual-owner",
            "TEST_STATE": str(self.fixture),
            "TEST_REAL_FIND": shutil.which("find"),
            "TEST_REAL_ID": shutil.which("id"),
        }
        for tool in ("find", "id", "sudo", "docker", "uvx", "ssh", "rsync"):
            path = self.bin / tool
            path.write_text(MOCK_TOOL)
            path.chmod(0o755)

    def problem(self, node="head", path=None, kind="foreign"):
        path = self.cache if path is None else path
        path.mkdir(parents=True, exist_ok=True)
        self.settings.setdefault("problems", {}).setdefault(node, {})[str(path)] = kind

    def run_download(self, *flags, success=True, terminal=False):
        (self.fixture / "settings.json").write_text(json.dumps(self.settings))
        self.log.write_text("")
        self.env["TEST_HUB"] = str(self.hub)
        command = ["bash", str(ROOT / "hf-download.sh"), "--config", str(self.config),
                   "org/model", *flags]
        if terminal:
            result_code, output = self.run_in_terminal(command)
        else:
            result = subprocess.run(command, cwd=self.fixture, env=self.env,
                                    stdin=subprocess.DEVNULL, capture_output=True,
                                    text=True, timeout=20)
            result_code, output = result.returncode, result.stdout + result.stderr
        self.assertEqual(result_code == 0, success, output)
        self.output = output
        self.calls = [json.loads(line) for line in self.log.read_text().splitlines()]

    def run_in_terminal(self, command):
        master, slave = pty.openpty()
        try:
            with subprocess.Popen(command, cwd=self.fixture, env=self.env,
                                  stdin=slave, stdout=slave, stderr=slave) as process:
                os.close(slave)
                slave = None
                output = b""
                answered = 0
                deadline = time.monotonic() + 20
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        process.kill()
                        self.fail("Sudo prompt hung: " + output.decode())
                    if select.select([master], [], [], 0.1)[0]:
                        try:
                            output += os.read(master, 65536)
                        except OSError:
                            break
                        prompts = output.count(b"TEST sudo password on ")
                        if prompts > answered:
                            os.write(master, b"test-password\n")
                            answered += 1
                process.wait(timeout=5)
                return process.returncode, output.decode()
        finally:
            os.close(master)
            if slave is not None:
                os.close(slave)

    def calls_for(self, tool):
        return [call for call in self.calls if call["tool"] == tool]

    def test_owned_and_missing_caches_skip_sudo(self):
        for exists in (True, False):
            with self.subTest(exists=exists):
                if not exists:
                    shutil.rmtree(self.cache)
                self.run_download()
                self.assertFalse(self.calls_for("sudo"))
                self.assertFalse(self.calls_for("docker"))
                self.assertFalse(self.calls_for("ssh"))

    def test_local_repair_precedes_download_and_uses_actual_owner(self):
        self.problem()
        self.run_download()
        repair = self.calls_for("sudo")[0]
        self.assertEqual(repair["args"], ["-n", "chown", "-R", "-h", "--",
                                          "local-owner", str(self.cache)])
        self.assertLess(self.calls.index(repair), self.calls.index(self.calls_for("uvx")[0]))
        # HF_HOME includes hub, so no duplicate scan is needed.
        self.assertEqual(len(self.calls_for("find")), 2)

    def docker_runs(self):
        return [call for call in self.calls_for("docker") if call["args"][2] == "run"]

    def test_docker_prefers_regular_image_and_skips_sudo(self):
        self.problem()
        self.settings["docker_images"] = {"head": ["vllm-node", "vllm-node-b12x"]}
        self.settings["password"] = True
        self.run_download()
        runs = self.docker_runs()
        self.assertEqual(len(runs), 1)
        self.assertIn("vllm-node", runs[0]["args"])
        self.assertFalse(self.calls_for("sudo"))
        self.assertLess(self.calls.index(runs[0]), self.calls.index(self.calls_for("uvx")[0]))

    def test_docker_selects_available_images_and_uid_on_each_node(self):
        for node in ("head", "peer1", "peer2"):
            self.problem(node)
        self.settings["docker_images"] = {
            "head": ["vllm-node-b12x"], "peer1": ["vllm-node"], "peer2": ["vllm-node-b12x"]}
        self.settings["uids"] = {"head": 1001, "peer1": 2002, "peer2": 3003}
        self.settings["password"] = True
        self.run_download("-c", "peer1,peer2", "--copy-parallel")
        runs = self.docker_runs()
        self.assertEqual([c["node"] for c in runs], ["head", "peer1", "peer2"])
        self.assertFalse(self.calls_for("sudo"))
        for call in runs:
            self.assertIn(self.settings["docker_images"][call["node"]][0], call["args"])
            self.assertEqual(call["args"][-2], str(self.settings["uids"][call["node"]]))
            self.assertLess(self.calls.index(call),
                            min(self.calls.index(c) for c in self.calls_for("rsync")))

    def test_failed_docker_image_tries_other_installed_image(self):
        self.problem()
        self.settings["docker_images"] = {"head": ["vllm-node", "vllm-node-b12x"]}
        self.settings["docker_fail"] = {"head": ["vllm-node"]}
        self.run_download()
        self.assertEqual(len(self.docker_runs()), 2)
        self.assertIn("vllm-node-b12x", self.docker_runs()[1]["args"])
        self.assertFalse(self.calls_for("sudo"))

    def test_unavailable_failed_or_ineffective_docker_falls_back_to_sudo(self):
        failures = [{"docker_unavailable": ["head"]},
                    {"docker_fail": {"head": ["vllm-node"]}},
                    {"docker_ineffective": ["head"]}]
        for index, failure in enumerate(failures):
            with self.subTest(failure=failure):
                self.cache = self.fixture / f"cache-{index}"
                self.hub = self.cache / "hub"
                self.env["HF_HOME"] = str(self.cache)
                self.settings = {"docker_images": {"head": ["vllm-node"]}, **failure}
                self.problem()
                self.run_download()
                self.assertEqual(len(self.calls_for("sudo")), 1)
                self.assertIn("trying sudo", self.output)

    def test_docker_failure_preserves_remote_sudo_password_prompt(self):
        self.problem()
        self.problem("peer1")
        self.settings["docker_images"] = {"head": ["vllm-node"], "peer1": ["vllm-node-b12x"]}
        self.settings["docker_fail"] = {"peer1": ["vllm-node-b12x"]}
        self.settings["password"] = True
        self.run_download("-c", "peer1", "--copy-parallel", terminal=True)
        self.assertEqual([c["node"] for c in self.calls_for("sudo")], ["peer1"])
        self.assertTrue(self.calls_for("sudo")[0]["tty"])

    def test_docker_and_sudo_failure_stop_download(self):
        self.problem()
        self.settings["docker_images"] = {"head": ["vllm-node"]}
        self.settings["docker_fail"] = {"head": ["vllm-node"]}
        self.settings["password"] = True
        self.run_download(success=False)
        self.assertFalse(self.calls_for("uvx"))

    def test_docker_mount_preserves_special_characters_in_remote_hub_path(self):
        self.hub = self.fixture / 'hub with "quotes", $(touch INJECTED); $literal'
        self.env["HF_HUB_CACHE"] = str(self.hub)
        self.env["DOCKER_HOST"] = "tcp://other-docker-host:2375"
        self.env["DOCKER_CONTEXT"] = "other-docker-host"
        self.problem("peer1", path=self.hub)
        self.settings["docker_images"] = {"peer1": ["vllm-node-b12x"]}
        self.run_download("-c", "peer1")
        self.assertEqual([c["node"] for c in self.docker_runs()], ["peer1"])
        self.assertFalse(self.calls_for("sudo"))
        self.assertFalse((self.fixture / "INJECTED").exists())

    def test_cache_environment_precedence_and_expansion(self):
        cases = [
            ({"XDG_CACHE_HOME": "$HOME/xdg"}, "home/xdg/huggingface", None),
            ({"XDG_CACHE_HOME": ""}, "huggingface", None),
            ({"XDG_CACHE_HOME": "ignored", "HF_HOME": "~/custom cache"},
             "home/custom cache", None),
            ({"HF_HOME": "relative cache"}, "relative cache", None),
            ({"HUGGINGFACE_HUB_CACHE": "~/legacy hub"}, None, "home/legacy hub"),
            ({"HUGGINGFACE_HUB_CACHE": "ignored", "HF_HUB_CACHE": "$HOME/new hub"},
             None, "home/new hub"),
        ]
        for variables, cache, hub in cases:
            with self.subTest(variables=variables):
                for key in ("HF_HOME", "XDG_CACHE_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
                    self.env.pop(key, None)
                self.env.update(variables)
                self.cache = self.fixture / (cache or "home/.cache/huggingface")
                self.hub = self.fixture / hub if hub else self.cache / "hub"
                target = self.hub if hub else self.cache
                self.problem(path=target)
                self.run_download()
                self.assertEqual(self.calls_for("sudo")[0]["args"][-1], str(target))
                self.assertIn(f"Model directory: {self.hub}/models--org--model", self.output)

    def test_override_repairs_hf_home_and_separate_hub(self):
        self.hub = self.fixture / "separate hub"
        self.env["HF_HUB_CACHE"] = str(self.hub)
        self.problem()
        self.problem(path=self.hub)
        self.run_download()
        self.assertEqual([c["args"][-1] for c in self.calls_for("sudo")],
                         [str(self.cache), str(self.hub)])

    def test_unreadable_cache_triggers_repair(self):
        self.problem(kind="unreadable")
        self.run_download()
        self.assertEqual(len(self.calls_for("sudo")), 1)

    def test_failed_or_ineffective_repair_stops_download(self):
        self.problem()
        for settings in ({"sudo_fail": ["head"]}, {"ineffective_repair": True}, {"password": True}):
            with self.subTest(settings=settings):
                self.settings = {"problems": self.settings["problems"], **settings}
                self.run_download(success=False)
                self.assertFalse(self.calls_for("uvx"))

    def test_parallel_copies_wait_for_all_remote_repairs(self):
        self.problem("peer1")
        self.problem("peer2")
        self.run_download("-c", "peer1,peer2", "-u", "remote-owner", "--copy-parallel")
        repairs = self.calls_for("sudo")
        copies = self.calls_for("rsync")
        self.assertEqual([c["node"] for c in repairs], ["peer1", "peer2"])
        self.assertEqual(len(copies), 2)
        for repair in repairs:
            self.assertEqual(repair["args"][-2:], ["remote-owner", str(self.cache)])
            self.assertLess(self.calls.index(repair), min(self.calls.index(c) for c in copies))
        for call in self.calls_for("ssh"):
            self.assertEqual(call["args"][2], "-nT")
            self.assertTrue(call["args"][3].startswith("remote-owner@"))

    def test_remote_failure_stops_all_copies(self):
        self.problem("peer2")
        for failure in ({"sudo_fail": ["peer2"]}, {"password": True}):
            with self.subTest(failure=failure):
                self.settings = {"problems": self.settings["problems"], **failure}
                self.run_download("-c", "peer1,peer2", "--copy-parallel", success=False)
                self.assertFalse(self.calls_for("rsync"))
                self.assertIn("Cache preparation failed on not-the-actual-owner@peer2", self.output)

    def test_remote_hub_override_is_repaired_and_used_for_copy(self):
        self.hub = self.fixture / "custom hub"
        self.env["HF_HUB_CACHE"] = str(self.hub)
        self.problem("peer1", path=self.hub)
        self.run_download("-c", "peer1")
        self.assertEqual(self.calls_for("sudo")[0]["args"][-1], str(self.hub))
        self.assertEqual(self.calls_for("rsync")[0]["args"][-1],
                         f"not-the-actual-owner@peer1:{self.hub}/models--org--model/")

    def test_saved_copy_hosts_are_checked_only_with_copy_flag(self):
        self.config.write_text("COPY_HOSTS=peer1,peer2\n")
        self.run_download()
        self.assertFalse(self.calls_for("ssh"))
        self.run_download("-c")
        self.assertEqual(len(self.calls_for("ssh")), 2)
        self.assertEqual(len(self.calls_for("rsync")), 2)
        self.assertFalse(self.calls_for("sudo"))

    def test_password_prompts_use_terminal_locally_and_on_each_peer(self):
        for node in ("head", "peer1", "peer2"):
            self.problem(node)
        self.settings["password"] = True
        self.run_download("-c", "peer1,peer2", "--copy-parallel", terminal=True)
        repairs = self.calls_for("sudo")
        self.assertEqual([c["node"] for c in repairs], ["head", "peer1", "peer2"])
        for call in repairs:
            self.assertTrue(call["tty"])
            self.assertNotIn("-n", call["args"])
        self.assertTrue(all(c["args"][2] == "-t" for c in self.calls_for("ssh")))

    def test_remote_paths_are_quoted_and_match_copy_destination(self):
        self.cache = self.fixture / "cache with 'quotes' $(touch INJECTED); $literal"
        self.hub = self.cache / "hub"
        self.env["HF_HOME"] = str(self.cache)
        self.problem("peer1")
        self.run_download("-c", "peer1")
        self.assertEqual(self.calls_for("sudo")[0]["args"][-1], str(self.cache))
        self.assertEqual(self.calls_for("rsync")[0]["args"][-1],
                         f"not-the-actual-owner@peer1:{self.hub}/models--org--model/")
        self.assertFalse((self.fixture / "INJECTED").exists())

    def test_symlinked_cache_root_is_repaired_at_target(self):
        target = self.fixture / "actual cache"
        shutil.move(self.cache, target)
        self.cache.symlink_to(target, target_is_directory=True)
        self.problem(path=target)
        self.run_download()
        self.assertEqual(self.calls_for("sudo")[0]["args"][-1], str(target))

    def test_unsafe_or_empty_cache_paths_stop_before_sudo(self):
        for path in ("/", self.env["HOME"], str(self.fixture), ""):
            with self.subTest(path=path):
                self.env["HF_HOME"] = path
                self.run_download(success=False)
                self.assertFalse(self.calls_for("sudo"))
                self.assertFalse(self.calls_for("uvx"))


if __name__ == "__main__":
    unittest.main()
