#!/bin/zsh
set -u
# Separate opt-in shortcut; the existing native shortcut is unchanged.
if ${ORIGINAL_LAB_ROOT}/outputs/flash1017-lab/run.sh ${ORIGINAL_LAB_ROOT}/work/serving-shortcut-v3/shortcut.py; then
  exit 0
else
  FLASH1017_SHORTCUT_EXIT=$?
  print ""
  print "Press Return to close this window."
  read -r FLASH1017_SHORTCUT_REPLY
  exit "$FLASH1017_SHORTCUT_EXIT"
fi
