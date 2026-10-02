"""Patch vLLM's Glm5NextProcessor.from_pretrained so an HF repo id is resolved
to a local cached snapshot dir before the raw `open(join(model_path, ...))`.

Why: the GLM-5-Next multimodal path calls
`Glm5NextProcessor.from_pretrained(self.ctx.model_config.model)` with the raw HF
id (`zai-org/GLM-5.3-Flash`). transformers' AutoTokenizer and the
`get_image_processor_config` helper resolve that id via the hub, but the
video_processor read does a bare `open(os.path.join(model_path,
"processor_config.json"))` which never turns the id into a filesystem path, so it
raises:
    FileNotFoundError: 'zai-org/GLM-5.3-Flash/processor_config.json'
(It appears only when multimodal is enabled — remove --language-model-only.)
We resolve the id to the cached snapshot directory (tolerating an incomplete
snapshot, unlike snapshot_download(local_files_only=True), which rejects a
snapshot missing metadata files like .gitattributes/LICENSE/README.md).
"""

import os
import py_compile
import sys

PROC = ("/usr/local/lib/python3.12/dist-packages/"
        "vllm/transformers_utils/processors/glm5next.py")

MARKER = "# vllm-docker: resolve hf repo id -> local snapshot dir for processor"


def patch(path: str) -> None:
    src = open(path).read()
    if MARKER in src:
        print(f"[fix-glm53-mm-model-path] already patched: {path}")
        return

    anchor = "        model_path = pretrained_model_name_or_path\n"
    assert anchor in src, (
        f"[fix-glm53-mm-model-path] anchor not found in {path} (vllm version drift?)"
    )

    insertion = anchor + f"        {MARKER}\n" + (
        "        # An HF repo id is not a filesystem path, but the video_processor\n"
        "        # read below does a raw open(join(model_path, ...)). Resolve the id\n"
        "        # to the local cached snapshot dir here so that open finds\n"
        "        # processor_config.json regardless of whether the snapshot is complete.\n"
        "        if not os.path.isdir(model_path):\n"
        "            from huggingface_hub import try_to_load_from_cache\n"
        "            _resolved = try_to_load_from_cache(model_path, \"processor_config.json\")\n"
        "            if _resolved:\n"
        "                model_path = os.path.dirname(_resolved)\n"
    )

    open(path, "w").write(src.replace(anchor, insertion, 1))
    py_compile.compile(path, doraise=True)
    print(f"[fix-glm53-mm-model-path] patched: {path}")


if __name__ == "__main__":
    for p in sys.argv[1:] or [PROC]:
        patch(p)
