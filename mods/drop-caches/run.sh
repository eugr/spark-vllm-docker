#!/bin/bash

# This mod will drop the FS caches every minute - useful to unstuck Qwen3.5-397B or other similar models during loading

CMD='sync; echo 3 > /proc/sys/vm/drop_caches'
LOG="/tmp/drop_caches.log"
PIDFILE="/tmp/drop_caches.pid"

nohup bash -c '
  while true; do
    '"$CMD"' >> "'"$LOG"'" 2>&1
    sleep 60
  done
' >/dev/null 2>&1 &

echo $! > "$PIDFILE"
echo "Started drop_caches loop with PID $(cat "$PIDFILE"); log is available in $LOG"