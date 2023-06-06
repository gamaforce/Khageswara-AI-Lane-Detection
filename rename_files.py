import glob
import os

new_name = "dataset_"
folder_path = "training_dataset/masks/road"

for path in glob.glob(os.path.join(folder_path, "*.jpg")):
    number = ""
    i = ""
    for char in path:
        if char.isdigit():
            i += char
        elif char == '.':
            number = i
        else:
            i = ""

    os.rename(path, folder_path + "/" + new_name + str(number) + ".jpg")