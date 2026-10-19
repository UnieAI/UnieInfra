#!/bin/bash
# Profile one tokenspeed arm at c16: the torch profiler records PROF_STEPS steps
# from 20 s into the cell. Never read this cell's duration: exporting the trace
# stalls the server.
#   prof_arm.sh TAG IMAGE OUTDIR [extra serve args...]
set -u
TAG=$1; IMG=$2; OUT=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd); PORT=${PORT:-30204}
HOOK=$(mktemp)
cat > "$HOOK" <<H
sleep 20
curl -s localhost:$PORT/start_profile -H 'content-type: application/json' -d '{"output_dir":"/prof","num_steps":${PROF_STEPS:-120},"activities":["CPU","GPU"],"with_stack":false,"profile_id":"$TAG"}'
H
ARM_MOUNTS="-v $OUT:/prof" CONCS=16 PRE_CELL_HOOK=$HOOK "$HERE/arm_ts.sh" $TAG $IMG "$@"
