#srun -p big_suma_rtx3090 -q big_qos --gres=gpu:1 --pty bash -i --m 3 --is_teacher

python t2v_inference.py --m 1 \
  --start_num 100 --end_num 140