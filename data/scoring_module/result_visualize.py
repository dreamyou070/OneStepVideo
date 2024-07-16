import os
import pandas as pd
import matplotlib.pyplot as plt
def main() :
    target_xlsx = 'final.xlsx'
    # read xlsx
    df = pd.read_excel(target_xlsx)
    header = df.columns
    content = df.loc[0]
    len_content = len(content)
    img_sims, fvds, txt_sims = [], [], []
    layers = []
    for i, item in enumerate(content) :

        if i < 63 :
            if i % 3 == 0 :
                img_sim = item
                img_sims.append(img_sim)
                layer = header[i].split('_image_sim')[0]
                layers.append(layer)
            elif i % 3 == 1 :
                fvd = item
                fvds.append(fvd)
            else :
                txt_sim = item
                txt_sims.append(txt_sim)
    # plotting
    plt.figure(figsize=(10, 5))
    plt.bar(layers, img_sims, color='b', label='Image similarity', alpha=0.5)
    # x label angle
    plt.xticks(rotation=30, ha='right')
    plt.title('Image similarity')
    plt.tight_layout()
    plt.savefig('image_similarity.png')
    plt.close()
    # [2] plot
    plt.figure(figsize=(10, 5))
    plt.bar(layers, fvds, color='r', label='FVD', alpha=0.5)
    # x label angle
    plt.xticks(rotation=30, ha='right')
    plt.title('FVD')
    plt.tight_layout()
    plt.savefig('fvd.png')
    plt.close()
    # [3] plot
    plt.figure(figsize=(10, 5))
    plt.bar(layers, txt_sims, color='g', label='Text similarity', alpha=0.5)
    # x label angle
    plt.xticks(rotation=30, ha='right')
    plt.title('Text similarity')
    plt.tight_layout()
    plt.savefig('text_similarity.png')

if __name__ == "__main__" :
    main()