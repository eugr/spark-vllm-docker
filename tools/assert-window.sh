#!/usr/bin/env bash
# Assert that a running vLLM server serves the context window we asked for.
#
# `max_model_len: auto` in a recipe becomes -1, and vLLM then resolves it with
# _auto_fit_max_model_len() -- a binary search for the longest sequence that
# fits the KV pool. The result varies with whatever memory happened to be free
# at boot, so a recipe can silently serve 833k when it was meant to serve 1M.
# Nothing in the API response says "this was reduced"; the only evidence is one
# INFO line in the boot log. This checks both sides:
#
#   1. the boot log contains no "Auto-fit max_model_len: reduced" line
#   2. /v1/models reports exactly the expected max_model_len
#
# Usage:
#   tools/assert-window.sh -e 1048576 -u vllm@deepseek-v4-flash-0731-1m.service
#   tools/assert-window.sh -e 1048576 -f /tmp/foreground-launch.log
#
# A foreground launch writes to a file rather than journald, hence -f.

set -uo pipefail

EXPECTED=1048576
UNIT=""
LOGFILE=""
BASE="${BASE:-https://localhost:8000}"
KEYFILE="${KEYFILE:-$HOME/.vllm-api-key}"

while getopts "e:u:f:b:k:h" opt; do
    case "$opt" in
        e) EXPECTED="$OPTARG" ;;
        u) UNIT="$OPTARG" ;;
        f) LOGFILE="$OPTARG" ;;
        b) BASE="$OPTARG" ;;
        k) KEYFILE="$OPTARG" ;;
        h|*) sed -n '2,20p' "$0"; exit 0 ;;
    esac
done

fail=0
note() { printf '%s\n' "$*"; }

# --- 1. boot log ------------------------------------------------------------
log_text=""
if [[ -n "$LOGFILE" ]]; then
    if [[ -r "$LOGFILE" ]]; then
        log_text="$(cat "$LOGFILE")"
    else
        note "WARN  log file not readable: $LOGFILE"
    fi
elif [[ -n "$UNIT" ]]; then
    # Only the current boot of the unit; an older restart's line is not evidence
    # about the server answering right now.
    since="$(systemctl show -p ActiveEnterTimestamp --value "$UNIT" 2>/dev/null)"
    if [[ -n "$since" ]]; then
        log_text="$(journalctl -u "$UNIT" --since "$since" 2>/dev/null)"
    else
        log_text="$(journalctl -u "$UNIT" -n 5000 2>/dev/null)"
    fi
fi

if [[ -n "$log_text" ]]; then
    if autofit="$(grep -aoE 'Auto-fit max_model_len: reduced from [0-9]+ to [0-9]+' <<<"$log_text" | tail -1)"; [[ -n "$autofit" ]]; then
        note "FAIL  $autofit"
        note "      the window was shrunk to fit KV; see the fallback ladder in"
        note "      recipes/deepseek-v4-flash-0731-1m.yaml"
        fail=1
    else
        note "ok    no auto-fit reduction in the boot log"
    fi
    grep -aoE 'GPU KV cache size: [0-9,]+ tokens' <<<"$log_text" | tail -1 | sed 's/^/      /'
    grep -aoE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' <<<"$log_text" | tail -1 | sed 's/^/      /'
else
    note "WARN  no boot log checked (pass -u UNIT or -f LOGFILE)"
fi

# --- 2. what the server advertises -----------------------------------------
key="${VLLM_API_KEY:-}"
if [[ -z "$key" && -r "$KEYFILE" ]]; then
    key="$(sed -n 's/^VLLM_API_KEY=//p' "$KEYFILE" | tr -d "\"'" | head -1)"
fi

served="$(curl -sk --max-time 30 -H "Authorization: Bearer $key" "$BASE/v1/models" \
    | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for m in d.get("data", []):
    print(m.get("id"), m.get("max_model_len"))' 2>/dev/null)"

if [[ -z "$served" ]]; then
    note "FAIL  could not read $BASE/v1/models (server down, or wrong key)"
    fail=1
else
    while read -r model len; do
        if [[ "$len" == "$EXPECTED" ]]; then
            note "ok    $model advertises max_model_len=$len"
        else
            note "FAIL  $model advertises max_model_len=$len, expected $EXPECTED"
            fail=1
        fi
    done <<<"$served"
fi

exit "$fail"
