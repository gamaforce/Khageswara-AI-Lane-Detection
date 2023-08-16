# =============================================== #
# This Program Is Used To Convert Video To Frames #
# =============================================== #

import cv2
import math
import os
import sys
import glob

video_num = 3
output_path = os.path.join("training_dataset/images")
tot_dataset = len(glob.glob(os.path.join(output_path, "*.jpg")))
interval = 1

if not os.path.exists(output_path):
    path = os.path.join("image/", "video"+str(video_num))
    os.mkdir(path)
    
    mask_path = os.path.join("image/video"+str(video_num)+"/", "masks")
    image_path = os.path.join("image/video"+str(video_num)+"/", "images")
    os.mkdir(mask_path)
    os.mkdir(image_path)
    

file_path = os.path.join("video/input/input_video_" + str(video_num) + ".mp4")
cap = cv2.VideoCapture(file_path)
frameRate = cap.get(5)

i = tot_dataset - 1

while cap.isOpened():
    frameId = cap.get(1) 
    ret, frame = cap.read()
    if not ret:
        break
    if frameId % (math.floor(frameRate) * interval) == 0:
        i += 1
        output_filename = "dataset_" + str(i) + ".jpg"
        output_file_path= os.path.join(output_path, output_filename)
        
        print(output_file_path)
        cv2.imwrite(output_file_path, frame)

cap.release()
