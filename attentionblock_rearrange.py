import os
import matplotlib.pyplot as plt
def main() :
    """
    base_folder = r'D:\1.연구\[연구4] Video\실험결과\long_video\2024-06-30T18-45-26-infsteps_6_window_size_8_group_query'
    images = os.listdir(base_folder)
    for image in images :
        layer_name = image.split('_time')[0]
        layer_folder = os.path.join(r'D:\1.연구\[연구4] Video\실험결과\long_video', layer_name)
        if not os.path.exists(layer_folder) :
            os.makedirs(layer_folder)
        org_image = os.path.join(base_folder, image)
        new_image = os.path.join(layer_folder, image)
        os.rename(org_image, new_image)
    """

    # [1] down block
    base_folder = r'D:\1.연구\[연구4] Video\실험결과\long_video\attention/up'
    folders = os.listdir(base_folder)
    for folder in folders  :
        layer_name = folder.split('_time')[0]
        folder_dir = os.path.join(base_folder, folder)
        images = os.listdir(folder_dir)
        # fig size 조정
        fig, axes = plt.subplots(1, len(images), figsize=(7,2))
        #fig, axes = plt.subplots(1, len(images))
        for image in images :
            #if 'attn_map' in image :
            #    image_dir = os.path.join(folder_dir, image)
            #    os.remove(image_dir)
            name = os.path.splitext(image)[0]
            time = name.split('_time_')[1].split('.')[0]
            image_dir = os.path.join(folder_dir, image)
            img = plt.imread(image_dir)
            axes[images.index(image)].imshow(img)
            axes[images.index(image)].axis('off')
            # title
            axes[images.index(image)].set_title(f'{time}', fontsize=7)
        plt.tight_layout()
        plt.suptitle(f'{layer_name}', fontsize=10)
        plt.savefig(os.path.join(folder_dir, 'attn_map.png'))




if __name__ == "__main__" :
    main()