import os

prompt_dir = r'prompts/filtered_caption.txt'
with open(prompt_dir, 'r', encoding='utf-8' ) as f:
    prompts = f.readlines()

prompt_100_dir = r'prompts/filtered_caption_100_5.txt'
val_num = 0
vals = []
with open(prompt_100_dir, 'a') as f:
    for prompt_idx, prompt in enumerate(prompts) :
        prompt = prompt.strip()
        if prompt != "" :
            val_num += 1
            if val_num <= 100 :
                f.write(f'{prompt}\n')
