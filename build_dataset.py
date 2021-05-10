import configparser
import glob
import math
import time
import numpy as np
import cv2
from tqdm import tqdm
import os
import h5py

config = configparser.ConfigParser()
config.read('config.ini')
builder_config = config['BUILDER']

def get_filenames(files_path):
    return [os.path.basename(x) for x in glob.glob(files_path)]

def crop_save(images_path, masks_path, result_images_path, result_masks_path, img_size):
    """
    Функція поділяє зображення на частини, фільтрує їх та зберігає фрагменти фото і масок на диску.

    Параметри:
    
    """
    skipped_num = 0
    saved_num = 0

    print(images_path)
    
    image_files = get_filenames(images_path +"*.tiff")
    print('Кількість файлів: {}'.format(len(image_files)))
    
    start_time = time.time()

    for image_filename in tqdm(image_files, total = len(image_files)):
       
        image = cv2.imread(images_path + image_filename)
        
        mask_path = masks_path + image_filename[:-1]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        counter = 0
        
        for r in range(0, image.shape[0], img_size):
            for c in range(0, image.shape[1], img_size):
                counter += 1
                
                cropped_image, cropped_mask = crop(image, mask, r, r + img_size, c, c + img_size)
                cropped_mask[cropped_mask>1] = 255

                if np.any(cropped_mask):
                    black_mask_num, white_mask_num = np.unique(cropped_mask, return_counts=True)[1]
                    black_image_num = np.sum(cropped_image == 0)    

                    if white_mask_num/black_mask_num < 0.01 or black_image_num/cropped_image.size > 0.1:
                        skipped_num += 1
                        continue
                    saved_num += 1
                    cv2.imwrite(result_images_path + str(counter) + '_' + image_filename, cropped_image)
                    cv2.imwrite(result_masks_path + str(counter) + '_' + image_filename, cropped_mask)
                
    print("Час: {}с.".format(round((time.time()-start_time), 2)))
    print("Пропущено зображень: ", skipped_num)
    print("Збережено зображень: ", saved_num)
    return saved_num

def crop(image, mask, r1, r2, c1, c2):
    result_image = np.zeros((img_size , img_size, 3), np.uint8)
    result_mask = np.zeros((img_size , img_size), np.uint8)

    cropped_image = image[r1:r2, c1:c2]
    cropped_mask = mask[r1:r2, c1:c2]

    result_image[:cropped_image.shape[0], :cropped_image.shape[1]] += cropped_image
    result_mask[:cropped_mask.shape[0], :cropped_mask.shape[1]] += cropped_mask

    return(result_image, result_mask)

def save_to_h5py(img_data, mask_data, save_path):
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('images', data=img_data, compression="gzip", compression_opts=4)
        f.create_dataset('masks', data=mask_data, compression="gzip", compression_opts=4)

if __name__ == "__main__":
    root = builder_config['root']
    test_files_percent = float(builder_config['test_files_percent'])
    img_size = int(builder_config['img_size'])
    save_num = int(builder_config['save_num'])

    images_path = root + builder_config['images_subfolder']
    masks_path = root + builder_config['masks_subfolder']
    result_images_path = root + builder_config['result_images_subfolder']
    result_masks_path = root + builder_config['result_masks_subfolder']

    if not os.path.exists(result_images_path):
        os.mkdir(result_images_path)
    
    if not os.path.exists(result_masks_path):
        os.mkdir(result_masks_path)

    save_num = crop_save(images_path, masks_path, result_images_path, result_masks_path, img_size)

    test_num = int(test_files_percent * save_num)
    train_num = save_num - test_num

    print("Collecting train images...")
    img_np = np.array([cv2.imread(path) for path in glob.glob(result_images_path + '*.tiff')[:train_num]])
    print(img_np.shape)

    print("Collecting train masks...")
    mas_np = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in glob.glob(result_masks_path + '*.tiff')[:train_num]])

    print("Saving training images and masks to h5py...\n")
    save_to_h5py(img_np, mas_np, 'train.hdf5')

    del img_np
    del mas_np

    print("Collecting test images...")
    img_np = np.array([cv2.imread(path) for path in glob.glob(result_images_path + '*.tiff')][train_num:save_num])
    print(img_np.shape)

    print("Collecting test masks...")
    mas_np = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in glob.glob(result_masks_path + '*.tiff')][train_num:save_num])

    print("Saving test images and masks to h5py...\n")
    save_to_h5py(img_np, mas_np, 'test.hdf5')

    del img_np
    del mas_np