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
# # --mixed_precision_training \
# only vlb loss ... 'lykon/dreamshaper-8-lcm', \
# all training from random "emilianJR/epiCRealism"

python train_test.py \
 --use_wandb \
 --seed 42 \
 --output_dir 'experiment' \
 --teacher_motion_model_dir "wangfuyun/AnimateLCM" \
 --pretrained_model_path "emilianJR/epiCRealism" \
 --sub_folder_name 'no_skip_lr_1e2_distill_loss_test2' \
 --max_train_epoch 100 \
 --config configs/training/v1/training.yaml \
 --sample_n_frames 1 \
 --inference_step 6 \
 --num_frames 1 \
 --motion_control \
 --guidance_scale 2.0 \
 --skip_layers " ['down_0_0', 'down_0_1','down_1_0','down_1_1','down_2_0','down_2_1',
 'down_3_0','down_3_1','mid','up_0_0','up_0_1','up_0_2','up_1_0','up_1_1','up_1_2',
 'up_2_0','up_2_1','up_2_2','up_3_0','up_3_1',]" \
 --random_init \
 --csv_path "/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-csv/0_300.csv" \
 --video_folder "/share0/dreamyou070/dreamyou070/MyData/video/webvid-10M/webvid-10M-partial-video" \
 --distill_weight 1.0 --vlb_weight 1.0 --loss_feature_weight 11.0 \
 --adam_weight_decay 0.01 --learning_rate 10

