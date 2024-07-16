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
# #--motion_control \
port_number=50331
accelerate launch --config_file ../gpu_config/gpu_0_config \
 --main_process_port $port_number \
 test_distill.py \
 --sub_folder_name 'experiment/no_skip_lr_1e2_distill_loss_test' \
 --wandb \
 --sample_n_frames 8 \
 --inference_step 6 \
 --guidance_scale 2.0 --motion_control \
 --skip_layers "[]" \
 --csv_path "/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-csv/0_300.csv" \
 --video_folder "/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-partial-video" \
 --distill_weight 1.0 --vlb_weight 1.0 --loss_feature_weight 1.0 \
 --use_wandb --cfg_random_null_text \
 --start_num 260 --end_num 300 \
 --output_dir '/share0/dreamyou070/dreamyou070/OneStepVideo/experiment/no_skip_lr_1e2_distill_loss_vlb_loss_random_init' \
 --saved_epoch 8