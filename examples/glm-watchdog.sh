#!/bin/bash
# GLM-5.2 cluster watchdog.
# Probes the vLLM endpoint with a tiny completion every 2 minutes. After 3
# consecutive failures (~6+ min unresponsive), restarts the cluster via the
# recipe. Touch /tmp/glm-watchdog.pause to suspend (maintenance); it caps
# itself at 3 restarts per 6h window and then stands down, leaving
# /tmp/glm-watchdog.gave-up as the signal that a human is needed.

REPO=/home/rngo/code/spark-vllm-docker
RECIPE=glm-5.2-nvfp4
URL=http://localhost:8000/v1/chat/completions
LOG=/home/rngo/glm-watchdog.log
PAUSE=/tmp/glm-watchdog.pause
GAVEUP=/tmp/glm-watchdog.gave-up
RESTART_STAMPS=/tmp/glm-watchdog.restarts

FAILS=0
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
log "watchdog started (pid $$)"

probe() {
  curl -sf -m 120 -X POST "$URL" -H "Content-Type: application/json" \
    -d '{"model":"GLM-5.2-NVFP4","messages":[{"role":"user","content":"Reply with the single word ok."}],"max_tokens":5}' \
    >/dev/null 2>&1
}

recent_restarts() {
  [ -f "$RESTART_STAMPS" ] || { echo 0; return; }
  local now cutoff n=0
  now=$(date +%s); cutoff=$((now - 21600))
  while read -r ts; do [ "$ts" -gt "$cutoff" ] && n=$((n+1)); done < "$RESTART_STAMPS"
  echo "$n"
}

while true; do
  sleep 120
  [ -f "$PAUSE" ] && { FAILS=0; continue; }
  [ -f "$GAVEUP" ] && continue

  # Stand down unless this cluster is supposed to be running GLM:
  # - no vllm_node container on the head = intentionally stopped
  # - API answering with a different model = something else is serving
  if ! docker ps --format '{{.Names}}' | grep -q '^vllm_node$'; then
    FAILS=0; continue
  fi
  SERVED=$(curl -sf -m 10 http://localhost:8000/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1)
  if [ -n "$SERVED" ] && ! echo "$SERVED" | grep -q "GLM-5.2-NVFP4"; then
    FAILS=0; continue
  fi

  # Startup grace: a container younger than 30 min may still be loading
  # weights or compiling — don't count probes against it.
  STARTED=$(docker inspect -f '{{.State.StartedAt}}' vllm_node 2>/dev/null)
  if [ -n "$STARTED" ]; then
    AGE=$(( $(date +%s) - $(date -d "$STARTED" +%s 2>/dev/null || echo 0) ))
    if [ "$AGE" -lt 1800 ]; then
      FAILS=0; continue
    fi
  fi

  if probe; then
    [ "$FAILS" -gt 0 ] && log "recovered after $FAILS failed probe(s)"
    FAILS=0
    continue
  fi

  FAILS=$((FAILS+1))
  log "probe failed ($FAILS/3)"
  [ "$FAILS" -lt 3 ] && continue

  if [ "$(recent_restarts)" -ge 3 ]; then
    log "3 restarts in 6h already — standing down, human needed"
    touch "$GAVEUP"
    continue
  fi

  log "restarting cluster"
  date +%s >> "$RESTART_STAMPS"
  cd "$REPO" || { log "repo missing"; continue; }
  ./launch-cluster.sh stop >> "$LOG" 2>&1
  python3 run-recipe.py "$RECIPE" -d --ib-if rocep1s0f0 --eth-if enp1s0f0np0 >> "$LOG" 2>&1
  log "relaunch dispatched; grace period 25 min"
  sleep 1500   # cold compile headroom before probing again
  FAILS=0
done
