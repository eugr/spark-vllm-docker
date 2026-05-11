# Fix: ignore_keys_at_rope_validation is a list but transformers uses | (set union)
import re

path = "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/qwen3_5_moe.py"
with open(path) as f:
    content = f.read()

old = """kwargs["ignore_keys_at_rope_validation"] = [
            "mrope_section",
            "mrope_interleaved",
        ]"""

new = """kwargs["ignore_keys_at_rope_validation"] = {
            "mrope_section",
            "mrope_interleaved",
        }"""

content = content.replace(old, new)

with open(path, "w") as f:
    f.write(content)

print("Fixed ignore_keys_at_rope_validation: list -> set")
