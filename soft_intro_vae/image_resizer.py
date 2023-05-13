import numpy as np
import cv2

# 加载数据集
data = np.load('../Dataset/trn_stim_data-sketch.npy')

# 初始化一个空的列表来存放resize后的图像
resized_images = []

print(f'Original dataset shape: {data.shape}')

# 遍历数据集中的每一张图片
for i, image in enumerate(data):
    # print(f'Original image {i+1} shape: {image.shape}')
    
    # 使用cv2.resize函数将图片大小改为256*256
    resized_image = cv2.resize(image, (256, 256))
    
    # print(f'Resized image {i+1} shape: {resized_image.shape}')

    # 将resize后的图片添加到列表中
    resized_images.append(resized_image)

# 将列表转换为numpy数组
resized_images = np.array(resized_images)

print(f'Resized dataset shape: {resized_images.shape}')

# 如果你需要将处理后的数据保存为.npy文件，可以使用以下代码：
np.save('../Dataset/trn_stim_data-sketch-256.npy', resized_images)