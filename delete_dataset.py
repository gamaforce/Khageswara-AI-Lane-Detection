import os

land_path = "training_dataset/masks/land"
road_path = "training_dataset/masks/road"
img_path = "training_dataset/images"

def delete_files(folder_path):
    i = 0
    x = 0
    y = 0

    paths = []
    all_files = []
    same_files = []

    for dir_path, dir_names, file_names in os.walk(folder_path):
        if x == 0:
            all_files = file_names
            for dir_path_img, dir_names_img, file_names_img in os.walk(img_path):
                if y == 0:
                    for files in file_names:
                        for files_img in file_names_img:
                            if files_img == files:
                                i+=1
                                same_files.append(files)
                                break
                y+=1
        x+=1    

    temp3 = [x for x in all_files if x not in same_files]

    print("Total Files: ",len(all_files))
    print("Same Files",len(same_files))
    print("Deleted Files: ",len(temp3))

    for path in temp3:
        os.remove(folder_path+"/"+path)

delete_files(road_path)
delete_files(land_path)