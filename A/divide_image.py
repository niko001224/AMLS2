import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

images_folder = './cassava-leaf-disease-classification/images'  # 包含所有图片的文件夹
csv_file = './cassava-leaf-disease-classification/train.csv'  # CSV文件路径，假设列名分别为'image'和'label'
train_folder = './cassava-leaf-disease-classification/train_images'  # 训练集图片将被复制到这里
test_folder = './cassava-leaf-disease-classification/test_images'  # 测试集图片将被复制到这里

# 读取CSV文件
df = pd.read_csv(csv_file)

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# 确保目标文件夹存在
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 复制图片到训练集文件夹
for _, row in train_df.iterrows():
    source_path = os.path.join(images_folder, row['image_id'])
    dest_path = os.path.join(train_folder, row['image_id'])
    shutil.copy(source_path, dest_path)

# 复制图片到测试集文件夹
for _, row in test_df.iterrows():
    source_path = os.path.join(images_folder, row['image_id'])
    dest_path = os.path.join(test_folder, row['image_id'])
    shutil.copy(source_path, dest_path)

print("图片已成功分配到训练集和测试集文件夹。")
