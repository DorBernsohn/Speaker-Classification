import os
import cv2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing import image


def video_to_frame(df, path_to_videos, path_to_save):
    '''
    the function takes a path to access videos and a path to create a folders of frames
    :param df: data frame
    :param path_to_videos: path
    :param path_to_save: path
    :return: data frame of frame names and class name
    '''
    # path_to_save = r"C:\Users\dbernsohn\Documents\Python Scripts\video_classification\train_1"
    # path_to_videos = r"C:\Users\dbernsohn\Documents\projects\SVW\Videos"
    file_name_list = []
    file_class_list = []
    for i in tqdm(range(df.shape[0])):
        count = 0
        videoFile = path_to_videos + os.sep + str(df['Genre'][i]) + os.sep + str(df['FileName'][i])
        cap = cv2.VideoCapture(videoFile)
        frameRate = cap.get(5)
        x=1
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                # storing the frames in a new folder named train_1
                if not os.path.isdir(path_to_save):
                    os.mkdir(path_to_save)
                if not os.path.isdir(path_to_save + os.sep + str(df['Genre'][i])):
                    os.mkdir(path_to_save + os.sep + str(df['Genre'][i]))
                file_path = path_to_save + os.sep + df['Genre'][i]
                file_name = str(df['FileName'][i]) + "_frame%d.jpg" % count;count+=1
                file_name_list.append(file_name)
                file_class_list.append(str(df['Genre'][i]))
                cv2.imwrite(file_path + os.sep + file_name, frame)

        cap.release()
    frame_df = pd.DataFrame({'Image': file_name_list, 'Class': file_class_list})
    return frame_df


def create_row_pixels(df, folder_name):
    '''
    the function turn images into numpy array of pixels
    :param df: data frame
    :return: numpy array
    '''
    train_image = []
    labels = []
    for i in tqdm(range(df.shape[0])):
        # # loading the image and keeping the target size as (224,224,3)
        # img = image.load_img(folder_name + os.sep + df['Class'][i] + os.sep + df['Image'][i], target_size=(224, 224, 3))
        # # converting it to array
        # img = image.img_to_array(img)
        # # normalizing the pixel value
        # img = img / 255
        #######
        image = cv2.imread(folder_name + os.sep + df['Class'][i] + os.sep + df['Image'][i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        train_image.append(image)
        labels.append(df['Class'][i])
        #######
        # appending the image to the train_image list
        # train_image.append(img)

    # converting the list to numpy array
    print('# of images: ', len(train_image))
    # X = np.array(train_image)

    # shape of the array
    return train_image, labels