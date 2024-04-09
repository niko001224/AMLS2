import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy

train_transforms = A.Compose([
    A.RandomResizedCrop(size=(384,384), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=1, fill_value=0, p=0.5),
    ToTensorV2(),
])

source_folder = './cassava-leaf-disease-classification/train_images'
output_folder = './cassava-leaf-disease-classification/enhance_train_images'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for img_name in os.listdir(source_folder):
    if img_name.endswith(('.png', '.jpg', '.jpeg')):
        
        img_path = os.path.join(source_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        
        augmented = train_transforms(image=img)
        augmented_img = augmented["image"]  
        
        # 由于增强后的图像是Tensor，需要转换回numpy数组，并从[0, 1]范围转换回[0, 255]
        augmented_img = augmented_img.permute(1, 2, 0).numpy() * 255
        augmented_img = augmented_img.astype('uint8')
        
        # 保存增强后的图像
        save_path = os.path.join(output_folder, img_name)
        cv2.imwrite(save_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))  # 将RGB转换回BGR进行保存