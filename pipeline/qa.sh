#!/bin/bash
# qa.sh name... -> 12-frame sheets (evenly spaced) in scratchpad/qa2/
S=/tmp/claude-1000/-home-iaadmin-tg-comfy-bot/606374ee-87ad-47e3-8a98-2f3268dde32c/scratchpad/qa2; mkdir -p $S
for n in "$@"; do f=/home/iaadmin/tg_comfy_bot/outputs/v2/clips/$n.mp4
  N=$(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 $f); st=$(( N / 12 ))
  ffmpeg -v error -y -i $f -vf "select='not(mod(n\,$st))',scale=480:-1,tile=4x3" -frames:v 1 $S/$n.jpg; echo $S/$n.jpg
done
