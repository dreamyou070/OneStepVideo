# --motion_control --full_attention \
# --motion_control --window_attention --window_size 5
# --self_control # --window_attention --window_size 5
#layer_dict_short = {0: 'down_0_0', 1: 'down_0_1',
#              2: 'down_1_0', 3: 'down_1_1',
#              4: 'down_2_0', 5: 'down_2_1',
#              6: 'down_3_0', 20 : 'down_3_1',
#              7: 'mid',
#              8: 'up_0_0', 9: 'up_0_1', 10: 'up_0_2',
#              11: 'up_1_0', 12: 'up_1_1', 13: 'up_1_2',
#              14: 'up_2_0', 15: 'up_2_1', 16: 'up_2_2',
#              17: 'up_3_0', 18: 'up_3_1', 19: 'up_3_2'}

python i2v_inference_ex.py \
 --n_prompt "ImgFixerPre0.3, glowing face, bad proportions, blurry, blurred composition, low resolution, bad, ugly, bad composition, terrible, 3d, render, comic, manga, flat, watermark, signature, worst quality, low quality, normal quality, lowres, simple background, inaccurate limb, extra fingers, fewer fingers, missing fingers, extra arms, extra legs, inaccurate eyes, bad composition, bad anatomy, error, extra digit, fewer digits, cinnadust, cropped, low res, worst quality, low quality, normal quality, jpeg artifacts, extra digit, fewer digits, trademark, watermark, artist's name, username, signature, text, words, human, blurry, blurred composition, blurry foreground, blurry background" \
 --inference_steps 6 \
 --num_frames 16 --full_attention --motion_control --window_size 10 \
 --skip_layers "['down_3_0', 'down_3_1', 'mid', 'up_0_0','up_0_1','up_0_2']"