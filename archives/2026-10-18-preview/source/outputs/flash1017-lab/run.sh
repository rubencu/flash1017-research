#!/bin/zsh
set -eu
FLASH1017_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
FLASH1017_APP='/Applications/oMLX.app/Contents/Resources'
FLASH1017_PYTHON="$FLASH1017_APP/Python/cpython-3.11/bin/python3.11"
export PYTHONPATH="$FLASH1017_ROOT:$FLASH1017_ROOT/_runtime:$FLASH1017_APP:$FLASH1017_APP/Python/framework-mlx-base/lib/python3.11/site-packages"
export TOKENIZERS_PARALLELISM=false
exec "$FLASH1017_PYTHON" "$@"
