# ======================================== # 
# This Program Is Used To Increase Dataset #
# ======================================== # 

import glob
import cv2
import os

# Filename format
filename = "dataset_"

# The path for images
save_path_img = "training_dataset/images/"
# The path for masks road
save_path_msk = "training_dataset/masks/road/"
# The path for masks land
save_path_inv = "training_dataset/masks/land/"
# Get the number of total dataset
tot_dataset = len(glob.glob(os.path.join("training_dataset/images", "*.jpg")))

# Read all files in the images path
for path in glob.glob(os.path.join("training_dataset/images", "*.jpg")):
    number = ""
    i = ""
    for char in path:
        if char.isdigit():
            i += char
        elif char == '.':
            number = i
        else:
            i = ""
    
    # Range of dataset that we want to flip or rotate
    if int(number) > 199:
        image = cv2.imread(path)
        mask = cv2.imread("training_dataset/masks/road/dataset_" + str(number) + ".jpg", 0)

        # Uncomment for flipping horizontally
        flipped_img = cv2.flip(image, 1)
        flipped_msk = cv2.flip(mask, 1)

        # # Uncomment for flipping vertically
        # flipped_img = cv2.flip(image, 0)
        # flipped_msk = cv2.flip(mask, 0)

        # # Uncomment for rotating 90 degrees
        # flipped_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # flipped_msk = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

        # Invert masks road for masks land
        invert_msk = cv2.bitwise_not(flipped_msk)

        # Write the file into the given path
        cv2.imwrite(save_path_img + filename + str(tot_dataset) + ".jpg", flipped_img)
        cv2.imwrite(save_path_msk + filename + str(tot_dataset) + ".jpg", flipped_msk)
        cv2.imwrite(save_path_inv + filename + str(tot_dataset) + ".jpg", invert_msk)

        tot_dataset += 1