#!/bin/zsh
# Let a trained policy play one note and report, from MIDI, which key it pressed. See scripts/keys_play.py.
#
#   scripts/keys_play.sh both mi
#   scripts/keys_play.sh text sol
#   scripts/keys_play.sh text10 do
#   scripts/keys_play.sh score
cd "$(dirname "$0")/.."
exec uv run --with-requirements scripts/requirements-keys.txt python scripts/keys_play.py "$@"
