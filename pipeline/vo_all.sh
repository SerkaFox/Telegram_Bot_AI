cd /home/iaadmin/tg_comfy_bot/outputs
for v in v3 v4 v5 v6 v7 v8 v9 v10; do
  CB_F=v2/ref_f.wav CB_M=v2/ref_m.wav /home/iaadmin/chatter_venv/bin/python cb_tts.py v2/${v}_script_cb.json v2/vo_$v cpu 2>&1 | grep ^VO
  ./cb_fixloop.sh v2/${v}_script_cb.json v2/vo_$v 4
  ./vo_trim.sh v2/vo_$v
  echo "== $v done"
done
echo ALL VO DONE
