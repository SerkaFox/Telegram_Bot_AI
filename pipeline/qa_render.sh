#!/bin/bash
# qa_render.sh NAME -> sheet (1 frame / 2 s), loudness, size, ASR transcript
F=/home/iaadmin/tg_comfy_bot/outputs/v2/renders/MEATZE_$1.mp4
S=/tmp/claude-1000/-home-iaadmin-tg-comfy-bot/606374ee-87ad-47e3-8a98-2f3268dde32c/scratchpad/render_$1.jpg
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $F); SZ=$(( $(stat -c %s $F) / 1000000 ))
ffmpeg -v error -y -i $F -vf "fps=0.5,scale=384:-1,tile=6x8" -frames:v 1 $S
L=$(ffmpeg -i $F -af ebur128 -f null - 2>&1 | grep -A1 "Integrated" | tail -1 | awk '{print $2}')
echo "$1: ${D}s ${SZ}MB ${L} LUFS sheet=$S"
/home/iaadmin/tg_comfy_bot/venv/bin/python /home/iaadmin/tg_comfy_bot/outputs/asr_check.py $F 2>/dev/null | grep "^\["
