#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 UnieAI
#
# Copy this checkout's HEAD, and the wheels from build_wheels.sh if present, to
# the GB10. Run from your workstation, inside the repository:
#
#   unieinfra/gb10/sync.sh [ssh-host]        # default host: gb10-unieai
#
# Only committed files are sent (git archive), and legacy/ is left out: 204 MB
# of history the runtime never reads. Nothing on the GB10 is touched except
# ~/unieinfra-main/{src,wheels} (GB10_REMOTE_DIR).
set -euo pipefail

HOST="${1:-gb10-unieai}"
WHEELS="${GB10_WHEELS:-$HOME/.cache/unieinfra-main/wheels}"
REMOTE="${GB10_REMOTE_DIR:-unieinfra-main}"
REV="$(git rev-parse --short HEAD)"

if ! git diff --quiet HEAD; then
  echo "sync.sh: uncommitted changes are NOT sent; only HEAD ($REV) is." >&2
fi

git archive --format=tar HEAD -- . ':(exclude)legacy' |
  ssh "$HOST" "set -e
    rm -rf ~/$REMOTE/src.new
    mkdir -p ~/$REMOTE/src.new
    tar -x -C ~/$REMOTE/src.new
    echo $REV > ~/$REMOTE/src.new/.gb10-rev
    rm -rf ~/$REMOTE/src
    mv ~/$REMOTE/src.new ~/$REMOTE/src
    chmod +x ~/$REMOTE/src/unieinfra/gb10/*.sh"
echo "synced source $REV to $HOST:~/$REMOTE/src"

wrev="$(cat "$WHEELS/.gb10-rev" 2>/dev/null || echo norev)"
if compgen -G "$WHEELS/*.whl" >/dev/null && [ "$wrev" != "$REV" ]; then
  # run.sh installs whatever wheels it finds, so stale ones would silently
  # replace this source. Clear the remote set; run.sh then builds from source.
  ssh "$HOST" "rm -rf ~/$REMOTE/wheels"
  echo "sync.sh: wheels are from $wrev, source is $REV; not sending them" >&2
  echo "  (run build_wheels.sh, or let run.sh build from source on the GB10)" >&2
elif compgen -G "$WHEELS/*.whl" >/dev/null; then
  COPYFILE_DISABLE=1 tar -c -C "$WHEELS" . |
    ssh "$HOST" "set -e
      rm -rf ~/$REMOTE/wheels.new
      mkdir -p ~/$REMOTE/wheels.new
      tar -x -C ~/$REMOTE/wheels.new
      rm -rf ~/$REMOTE/wheels
      mv ~/$REMOTE/wheels.new ~/$REMOTE/wheels"
  echo "synced wheels $wrev ($(du -sh "$WHEELS" | cut -f1)) to $HOST:~/$REMOTE/wheels"
else
  echo "no wheels at $WHEELS; run build_wheels.sh to build them"
fi
