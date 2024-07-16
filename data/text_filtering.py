import os

base_file = 'filtered_captions_val.txt'
with open(base_file, 'r', encoding = 'utf-8') as f:
    lines = f.readlines()

#filter_file = 'filtered_captions_val_100_300.txt'
#with open(filter_file, 'w', encoding = 'utf-8') as ff:
def make_txt_file_from_list (trg_list, trg_txt) :
    with open(trg_txt, 'w', encoding = 'utf-8') as ff:
        for line in trg_list:
            ff.write(line+'\n')

list_1, list_2, list_3, list_4, list_5 = [], [], [], [], []
iter = 0
for line in lines:
    line = line.strip()
    if line != "" :
        iter += 1
        if iter >= 100 and iter < 140 :
            list_1.append(line)
        elif iter >= 140 and iter < 180 :
            list_2.append(line)
        elif iter >= 180 and iter < 220 :
            list_3.append(line)
        elif iter >= 220 and iter < 260 :
            list_4.append(line)
        elif iter >= 260 and iter < 300 :
            list_5.append(line)

# [2] make file
make_txt_file_from_list(list_1, 'filtered_captions_val_100_140.txt')
make_txt_file_from_list(list_2, 'filtered_captions_val_140_180.txt')
make_txt_file_from_list(list_3, 'filtered_captions_val_180_220.txt')
make_txt_file_from_list(list_4, 'filtered_captions_val_220_260.txt')
make_txt_file_from_list(list_5, 'filtered_captions_val_260_300.txt')