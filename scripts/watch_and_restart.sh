#!/usr/bin/env bash
################################################################################
# File: restart_once.sh
# Purpose:  Wait until the target python process exits, then launch it *once*
#           and leave it running in the foreground.  The script then terminates.
################################################################################

source myenv/bin/activate
PY_INTERPRETER="/home/truongchu/Academic/Graduation_Thesis/Project/AI/ipen-agent/myenv/bin/python3.9"
PY_SCRIPT="/home/truongchu/Academic/Graduation_Thesis/Project/AI/ipen-agent/rl/train/pentest_env.py"

CMD="$PY_INTERPRETER $PY_SCRIPT"
POLL=10             # seconds between checks – adjust if you like

echo "Watcher started at $(date '+%F %T')"
echo "Waiting for: $CMD"

# ── 1️⃣ Wait until there is **no** matching process ───────────────────────────
while pgrep -f "$CMD" > /dev/null ; do
    sleep "$POLL"
done

# ── 2️⃣ Launch once and stay attached ────────────────────────────────────────
echo "$(date '+%F %T')  »  original run is over – starting new run..."
exec $CMD              # exec replaces the watcher; script ends when training ends
