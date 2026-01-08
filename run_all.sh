#!/usr/bin/env bash
set -e

# 같은 venv에서 실행하는 경우 필요 시:
# source /home/dam/manim/.venv/bin/activate

python ui.py &
UI_PID=$!

manimgl screwnet.py ScrewNetAxisMotionViz -p -ql &
MANIM_PID=$!

# Ctrl+C 시 두 프로세스 같이 종료
trap "kill $UI_PID $MANIM_PID 2>/dev/null || true" INT TERM

wait $MANIM_PID
kill $UI_PID 2>/dev/null || true
