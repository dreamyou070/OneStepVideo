#srun -p big_suma_rtx3090 -q big_qos --gres=gpu:1 --pty bash -i --m 3 --is_teacher

python t2v_inference.py --m 5 \
  --start_num 260 --end_num 300
