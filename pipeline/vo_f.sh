cd /home/iaadmin/tg_comfy_bot/outputs
for v in v1 v2 v3 v4 v5 v6 v7 v8 v9 v10; do
  CB_F=v2/ref_f.wav CB_M=v2/ref_m.wav /home/iaadmin/chatter_venv/bin/python cb_tts.py v2/${v}_script_cb.json v2/vo_$v cpu 2>&1 | grep ^VO
  ./cb_fixloop.sh v2/${v}_script_cb.json v2/vo_$v 4; ./vo_trim.sh v2/vo_$v; echo "== $v"
done
T=v2/vo_v1/trim; ffmpeg -v error -y -i $T/v1_w1.wav -i $T/v1_w2.wav -filter_complex "[0][1]concat=n=2:v=0:a=1" $T/v1_w12.wav
echo F DONE
