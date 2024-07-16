#--pretrained_model_path '/home/dreamyou070/.cache/huggingface/hub/models--Lykon--dreamshaper-7/snapshots/9b481047f77996efa025e75e03941dbf51f506ad' \

python batch_inference_self.py \
 --inference_config /share0/dreamyou070/dreamyou070/AnimateLCM/AnimateLCM/animatelcm_sd15/configs/inference-i2v.yaml \
 --config configs/batch_inference_i2v.yaml \
 --adapter_scale 0.6 --inference_steps 4 --cfg 1.5