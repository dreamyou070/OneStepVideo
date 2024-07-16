#python train_distill.py --config configs/training/v1/training.yaml \
# weight decay
# layer_dict_short = {0: 'down_0_0', 1: 'down_0_1',
 #                    2: 'down_1_0', 3: 'down_1_1',
 #                    4: 'down_2_0', 5: 'down_2_1',
 #                    6: 'down_3_0', 7: 'down_3_1',
 #                    8: 'mid',
 #                    9: 'up_0_0', 10: 'up_0_1', 11: 'up_0_2',
 #                    12: 'up_1_0', 13: 'up_1_1', 14: 'up_1_2',
 #                    15: 'up_2_0', 16: 'up_2_1', 17: 'up_2_2',
 #                    18: 'up_3_0', 19: 'up_3_1', 20: 'up_3_2',}
# If I use down_0_0, can it be ... ?
# # \

port_number=52360

accelerate launch --config_file ../gpu_config/gpu_0_config \
 --main_process_port $port_number \
 train_distill.py \
 --use_wandb \
 --seed 42 \
 --output_dir 'experiment' \
 --teacher_motion_model_dir "wangfuyun/AnimateLCM" \
 --pretrained_model_path "emilianJR/epiCRealism" \
 --sub_folder_name 'down_all_distill_weight_1.0_vlb_weight_1.0_loss_feature_weight_1.0' \
 --max_train_epoch 100 \
 --config configs/training/v1/training.yaml \
 --sample_n_frames 8 \
 --inference_step 6 \
 --num_frames 8 \
 --motion_control \
 --guidance_scale 2.0 \
 --skip_layers "['up_0_0','up_1_0','up_2_0','up_3_0']" \
 --csv_path "/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-csv/0_300.csv" \
 --video_folder "/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-partial-video" \
 --distill_weight 1.0 --vlb_weight 1.0 --loss_feature_weight 1.0 \
 --adam_weight_decay 0.01 --learning_rate 0.0000001