#!/bin/bash
# cb_fixloop.sh spec.json dir [rounds] — regen Chatterbox lines until the ASR gate passes (rate check ignored)
spec=$1; dir=$2; n=${3:-5}
for r in $(seq 1 $n); do
  bad=$(/home/iaadmin/tg_comfy_bot/venv/bin/python vo_check2.py $spec $dir 2>/dev/null | awk '/^BAD [0-9]/{ r=$2; w=$4; sub("w/s","",w); if (r<0.85 || w<1.4) print $5 }')
  [ -z "$bad" ] && { echo "round $r clean"; exit 0; }
  echo "round $r bad: $bad"
  for k in $bad; do rm -f $dir/$k.wav; done
  CB_F=v2/ref_f.wav CB_M=v2/ref_m.wav /home/iaadmin/chatter_venv/bin/python cb_tts.py $spec $dir cpu 2>&1 | grep "^VO"
done
