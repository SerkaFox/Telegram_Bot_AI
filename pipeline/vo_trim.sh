#!/bin/bash
# trim leading/trailing silence of every wav in $1 into $1/trim (48k stereo), 0.12s head pad, 0.25s tail pad
mkdir -p "$1/trim"
for f in "$1"/*.wav; do
  o="$1/trim/$(basename "$f")"; [ -f "$o" ] && [ "$o" -nt "$f" ] && continue
  ffmpeg -v error -y -i "$f" -af "silenceremove=start_periods=1:start_threshold=-38dB:start_silence=0.05,areverse,silenceremove=start_periods=1:start_threshold=-38dB:start_silence=0.05,areverse,adelay=120|120,apad=pad_dur=0.25,aresample=48000" -ac 2 "$o"
done
