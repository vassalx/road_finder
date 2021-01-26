import configparser
import glob
import math
import time
import numpy as np
import cv2
from tqdm import tqdm
import os

def get_filenames(files_path):
    return [os.path.basename(x) for x in glob.glob(files_path)]

def split_to_test_and_train(root_path, images_path, masks_path, test_split=0.3):
    image_filenames = get_filenames(images_path +"*.tiff")
    test_set_size = int(test_split * len(image_filenames))
    
    train_dir = root_path + "Train/"
    test_dir = root_path + "Test/"

    train_image_dir = train_dir+"Images/"
    train_mask_dir = train_dir+"Masks/"
    test_image_dir = test_dir+"Images/"
    test_mask_dir = test_dir+"Masks/"
    
    for i in range(test_set_size):
        filename = image_filenames[i]
        os.renames(images_path + filename, test_image_dir + filename)
        os.renames(masks_path + filename, test_mask_dir + filename)
    
    for i in range(test_set_size, len(image_filenames)):
        filename = image_filenames[i]
        os.renames(images_path + filename, train_image_dir + filename)
        os.renames(masks_path + filename, train_mask_dir + filename)
            
    print("Train-Test-Split COMPLETED.\nNUMBER OF IMAGES IN TRAIN SET:{}\nNUMBER OF IMAGES IN TEST SET: {}".format(len(image_filenames)-test_set_size, test_set_size))
    print("\nTrain Directory:", train_dir)
    print("Test Directory:", test_dir)

def crop_and_save(images_path, masks_path, result_images_path, result_masks_path, img_size):
    print("Dataset Build Started!")
    
    skipped_num = 0

    print(images_path)
    
    image_files = get_filenames(images_path +"*.tiff")
    print('Number of files: {}'.format(len(image_files)))
    
    start_time = time.time()

    for image_filename in tqdm(image_files, total = len(image_files)):
       
        image_path = images_path + image_filename
        image = cv2.imread(image_path)
        
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
                    
                    if white_mask_num/black_mask_num < 0.01:
                        skipped_num += 1
                        continue

                    white_image_num = np.sum(cropped_image == 255)

                    if white_image_num/cropped_image.size > 0.1:
                        skipped_num += 1
                        continue

                    cv2.imwrite(result_images_path + str(counter) + '_' + image_filename, cropped_image)
                    cv2.imwrite(result_masks_path + str(counter) + '_' + image_filename, cropped_mask)

    print("complete: {} seconds.\images in {}\masks in{}".format(round((time.time()-start_time), 2), result_images_path, result_masks_path))
    print("\nNumber of skipped images: {}".format(skipped_num))

def crop(image, mask, r1, r2, c1, c2):
    result_image = np.zeros((img_size , img_size, 3), np.uint8)
    result_mask = np.zeros((img_size , img_size), np.uint8)

    cropped_image = image[r1:r2, c1:c2]
    cropped_mask = mask[r1:r2, c1:c2]

    result_image[:cropped_image.shape[0], :cropped_image.shape[1]] += cropped_image
    result_mask[:cropped_mask.shape[0], :cropped_mask.shape[1]] += cropped_mask

    return(result_image, result_mask)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    builder_config = config['BUILDER']

    root = builder_config['root']
    test_files_percent = float(builder_config['test_files_percent'])
    img_size = int(builder_config['img_size'])

    images_path = root + builder_config['images_subfolder']
    masks_path = root + builder_config['masks_subfolder']
    result_images_path = root + builder_config['result_images_subfolder']
    result_masks_path = root + builder_config['result_masks_subfolder']

    if not os.path.exists(result_images_path):
        os.mkdir(result_images_path)
    
    if not os.path.exists(result_masks_path):
        os.mkdir(result_masks_path)

    crop_and_save(images_path, masks_path, result_images_path, result_masks_path, img_size)
    split_to_test_and_train(root, result_images_path, result_masks_path, test_files_percent)