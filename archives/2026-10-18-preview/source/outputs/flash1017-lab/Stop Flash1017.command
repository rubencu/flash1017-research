#!/bin/zsh
set -eu
FLASH1017_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "$FLASH1017_ROOT/run.sh" -m flash1017.stop
