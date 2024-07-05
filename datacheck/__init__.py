import os

base_dir = r'D:\3. 삼성 과제\samsung_project\dataset\haa500_v1_1\video'
classes = os.listdir(base_dir)
print(f' len of classes : {len(classes)}')
datas = []
for class_name in classes :
    class_dir = os.path.join(base_dir, class_name)
    images = os.listdir(class_dir)
    elem = [class_name, len(images)]
    datas.append(elem)
save_dir = r'D:\3. 삼성 과제\samsung_project\dataset\1_preparing_dataset.xlsx'
import pandas as pd
df = pd.DataFrame(datas, columns=['class', 'num_images'])
df.to_excel(save_dir, index=False)