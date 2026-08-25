# Agent Instructions

These instructions apply to the entire repository.

## Repository

This project provides Bash and Python orchestration for running vLLM on one or
more NVIDIA DGX Spark systems. Work from the repository root and read `README.md`
for the public project overview.

## Choose One Guide

- **Use or operate the repository:** For host preparation, recipe selection,
  cluster discovery, image or model setup, recipe launches, and live-server
  verification, follow `docs/AGENT_RUNBOOK.md`.
- **Develop the repository:** For inspection, fixes, features, reviews, tests,
  or changes to scripts, recipes, mods, Dockerfiles, and documentation, follow
  `docs/AGENT_DEVELOPMENT.md`.
- **Both:** Follow the development guide first. Use the operational runbook
  afterward only when the user also requested a real build, download, or launch.

Read only the guide relevant to the task unless the work crosses that boundary.
A recipe `--dry-run` used to validate generated commands is development. A
non-dry recipe run, `--setup`, discovery, image preparation, model download, or
container launch is operation.

## Common Boundaries

- Inspect before changing repository, host, container, or cluster state.
- Preserve unrelated user changes and existing local configuration or artifacts.
- Do not expose credentials or `.env` contents in chat, logs, diffs, or commands.
- Operational tasks do not authorize source changes. Development tasks do not
  authorize real deployments. Perform both only when the user requests both.
- Do not prune, overwrite, stop, remove, or force-refresh existing resources
  unless the requested task requires it.
