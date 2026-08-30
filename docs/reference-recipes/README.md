# Reference recipes (not maintained for launch)

Ports of third-party GLM-5.3-Flash deployments that the production recipe
(`recipes/glm-5.3-flash-nvfp4.yaml`) was benchmarked against on the 4x Spark
cluster (2026-08-30). Kept for provenance and comparison, removed from
`recipes/` so the launchable surface stays the single validated config.
Measurements and full history: `docs/GLM53_FLASH.md`.

| File | Source | Status on our cluster |
| --- | --- | --- |
| glm-5.3-flash-nvfp4-bench.yaml | tonyd2wild launcher, verbatim flags | fully measured (52.5 tok/s, GSM8K 89, RULER 1.0 to 131K) |
| glm-5.3-flash-exl3.yaml | Mia-AiLab EXL3 kit | fully measured (49.0 tok/s, GSM8K 88, degrades past 120K) |
| glm-5.3-flash-nvfp4-mm.yaml | Mia-AiLab Dual-DGX-Spark port | NEVER booted here (container never built); author claims only |
