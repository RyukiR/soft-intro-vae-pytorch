import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images.append(np.array(img))
    return images

folder = "D:\\GitHub\\T2I-Adapter\\preprocessed_outputs\\trn_stim_data-sketch"
images = load_images_from_folder(folder)

# 将图片列表转化为numpy数组
images_np = np.array(images)

# 保存numpy数组到np文件
np.save('D:\\GitHub\\soft-intro-vae-pytorch\\soft_intro_vae\\data_preprocessor\\trn_stim_data-sketch.npy', images_np)
