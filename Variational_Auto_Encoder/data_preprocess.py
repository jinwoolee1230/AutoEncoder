import os
from PIL import Image

train_file_path= "/Users/jinwoolee/Projects/AutoEncoder/dataset/train/"
processed_train_file_path= "/Users/jinwoolee/Projects/AutoEncoder/dataset/train_processed/"
validation_file_path= "/Users/jinwoolee/Projects/AutoEncoder/dataset/validation/"
processed_validation_file_path= "/Users/jinwoolee/Projects/AutoEncoder/dataset/validation_processed/"

def crop_center_square(pil_img):
    width, height = pil_img.size
    if width == height:
        return pil_img  # 이미지가 이미 정방형일 경우 그대로 반환

    if width > height:
        left = (width - height) // 2
        upper = 0
        right = left + height
        lower = height
    else:
        left = 0
        upper = (height - width) // 2
        right = width
        lower = upper + width
    pil_img= pil_img.crop((left, upper, right, lower))
    pil_img= pil_img.resize((400, 400))
    return pil_img

train_files = os.listdir(train_file_path)
validation_files = os.listdir(validation_file_path)

for image in train_files:
    img= Image.open(train_file_path+image)
    cropped_img = crop_center_square(img)
    cropped_img.save(processed_train_file_path+image)
    # cropped_img.show()
for image in validation_files:
    img= Image.open(validation_file_path+image)
    cropped_img = crop_center_square(img)
    cropped_img.save(processed_validation_file_path+image)
    # cropped_img.show()